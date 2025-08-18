# ==============================================================================
# SCRIPT: fall_detector.py
#
# ROLE: The Sentry (Real-time Detection) - DEMO-READY HYBRID VERSION
#
# SUMMARY:
# This script uses a GStreamer pipeline and the Hailo AI processor to detect
# falls from a live video feed. This version incorporates a hybrid detection
# model to be both robust for real-world use and reliable for demonstrations.
#
# HOW IT WORKS (HYBRID LOGIC):
# An alert is triggered if EITHER of the following conditions are met:
#
# 1. THE "SUDDEN FALL":
#    a) Detects a high downward velocity to flag a *potential* fall.
#    b) Confirms the fall only if the person then remains in a horizontal
#       posture for a short duration (`POST_FALL_DWELL_FRAMES`).
#
# 2. THE "SLOW SLUMP" / "DEMO TRIGGER":
#    a) Detects that a person has been in a continuous horizontal posture
#       for a longer, specified duration (`PROLONGED_LIE_DOWN_FRAMES`),
#       regardless of their initial velocity.
#
# This dual approach ensures both realistic, complex falls and deliberate
# "lie down" events (useful for demos) are detected reliably.
#
# OUTPUTS:
# - A snapshot image (.jpg) of the fall event.
# - A data file (.json) containing fall metrics for the `analyst.py` script.
# ==============================================================================

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import datetime
import json
import time
from dotenv import load_dotenv
from typing import Dict, List

# --- Hailo App Imports ---
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp

# Load environment variables from the .env file
load_dotenv()

# --- Configuration for Detection Logic ---
# --- You can tune these values in your .env file to adjust sensitivity ---

# The downward velocity required to trigger a fall 'monitoring' state.
VELOCITY_THRESHOLD = float(os.getenv('VELOCITY_THRESHOLD', 0.08)) 

# The posture metric threshold. If the smoothed vertical spread of keypoints
# is less than this, the person is considered horizontal. (Range: 0.0 to 1.0)
VERTICAL_SPREAD_THRESHOLD = float(os.getenv('VERTICAL_SPREAD_THRESHOLD', 0.35))

# The smoothing factor for the posture metric (Exponential Moving Average).
# A smaller value means more smoothing but slower reaction time.
EMA_ALPHA = float(os.getenv('EMA_ALPHA', 0.1))

# How many consecutive frames a person must remain horizontal *after* a high-velocity
# event to confirm a fall.
POST_FALL_DWELL_FRAMES = int(os.getenv('POST_FALL_DWELL_FRAMES', 45)) # ~1.5 seconds at 30fps

# ==============================================================================
# NEW DEMO-FRIENDLY CONFIGURATION
# ==============================================================================
# How many consecutive frames a person must be horizontal to trigger a "prolonged
# lie down" alert, even without a prior high-velocity event.
# Make this longer than POST_FALL_DWELL_FRAMES. This is your main demo trigger.
PROLONGED_LIE_DOWN_FRAMES = int(os.getenv('PROLONGED_LIE_DOWN_FRAMES', 90)) # ~3 seconds at 30fps
# ==============================================================================

# How many consecutive frames a person must be upright to be considered 'recovered'.
RECOVERY_FRAMES_THRESHOLD = int(os.getenv('RECOVERY_FRAMES_THRESHOLD', 15))

# How many frames a track can be lost for before being deleted.
STALE_TRACK_THRESHOLD_FRAMES = 150

# The IoU threshold to re-identify a person who has fallen.
RECOVERY_BBOX_IOU_THRESHOLD = 0.5

# Grace period at startup to allow the system to stabilise.
STARTUP_GRACE_PERIOD_SECONDS = 10 

# --- Directory Setup ---
SNAPSHOT_DIR = "fall_snapshots"
ANALYSIS_QUEUE_DIR = "analysis_queue"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(ANALYSIS_QUEUE_DIR, exist_ok=True)

class user_app_callback_class(app_callback_class):
    """Stores persistent application state."""
    def __init__(self):
        super().__init__()
        self.person_states: Dict[int, dict] = {}
        self.fallen_person_recovery_zones: list = []
        self.start_time = time.time()

# ==============================================================================
# REFACTORED LOGIC: The monolithic callback is now broken into focused functions
# for clarity, robustness, and easier maintenance.
# ==============================================================================

def calculate_stable_posture(points: List[hailo.HailoPoint], state: dict) -> tuple[bool, float]:
    """
    Calculates a stable posture metric based on the vertical spread of keypoints,
    smoothed over time using an Exponential Moving Average (EMA). This is far more
    robust than using a bounding box aspect ratio, as it is resilient to model
    flickering.
    """
    if not points:
        return False, state.get('smoothed_vertical_spread', 1.0)

    all_y_coords = [p.y() for p in points]
    vertical_spread = max(all_y_coords) - min(all_y_coords)
    last_smoothed_spread = state.get('smoothed_vertical_spread', vertical_spread)
    new_smoothed_spread = (EMA_ALPHA * vertical_spread) + ((1 - EMA_ALPHA) * last_smoothed_spread)
    state['smoothed_vertical_spread'] = new_smoothed_spread
    is_horizontal = new_smoothed_spread < VERTICAL_SPREAD_THRESHOLD
    
    return is_horizontal, new_smoothed_spread

def update_person_metrics(state: dict, torso_center_y: float, current_frame_num: int) -> float:
    """Calculates the vertical velocity of a person's torso."""
    frame_diff = current_frame_num - state['last_frame_num']
    vertical_velocity = 0
    if 0 < frame_diff < 5:
        y_diff = torso_center_y - state['last_y']
        if y_diff > 0:
            vertical_velocity = y_diff / frame_diff
    return vertical_velocity

def process_person_state(state: dict, track_id: int, vertical_velocity: float, is_horizontal: bool, current_frame_num: int, posture_metric: float, frame: np.ndarray, bbox_dims: tuple, user_data: user_app_callback_class):
    """
    The core state machine for fall detection. This implements the hybrid logic
    for both sudden falls and prolonged "lie down" events.
    """
    fall_detected = False

    # --- State: STANDING ---
    # The person is considered upright. We look for a high-velocity event OR a prolonged horizontal state.
    if state['fall_status'] == 'standing':
        state['recovery_frame_count'] = 0

        # Update the continuous horizontal dwell counter
        if is_horizontal:
            state['horizontal_dwell_count'] += 1
        else:
            state['horizontal_dwell_count'] = 0 # Reset if not horizontal

        # --- Condition 1: High downward velocity detected (The "Sudden Fall") ---
        if vertical_velocity > VELOCITY_THRESHOLD:
            print(f"  - [Info] High velocity for ID {track_id}. Entering monitoring state.")
            state['fall_status'] = 'monitoring'
            state['potential_fall_start_frame'] = current_frame_num
        
        # --- Condition 2: Prolonged horizontal state detected (The "Demo Trigger / Slow Slump") ---
        elif state['horizontal_dwell_count'] > PROLONGED_LIE_DOWN_FRAMES:
            print(f"  - [Info] Prolonged horizontal state for ID {track_id}. Confirming fall.")
            state['fall_status'] = 'fallen'
            fall_detected = True # Trigger the fall directly

    # --- State: MONITORING ---
    # A high-velocity event occurred. We now check for a sustained horizontal posture.
    elif state['fall_status'] == 'monitoring':
        dwell_duration = current_frame_num - state.get('potential_fall_start_frame', current_frame_num)
        
        # Condition 1b: Person is horizontal and has remained so for the dwell time.
        if is_horizontal and dwell_duration >= POST_FALL_DWELL_FRAMES:
            state['fall_status'] = 'fallen'
            fall_detected = True
        
        # Timeout: If they don't become horizontal after a while, they have recovered.
        elif dwell_duration > POST_FALL_DWELL_FRAMES:
            print(f"  - [Info] ID {track_id} recovered after high velocity. Resetting.")
            state['fall_status'] = 'standing'
            
    # --- State: FALLEN ---
    # The person is confirmed on the ground. We now monitor for recovery.
    elif state['fall_status'] == 'fallen':
        if not is_horizontal:
            state['recovery_frame_count'] += 1
            if state['recovery_frame_count'] > RECOVERY_FRAMES_THRESHOLD:
                print(f"  - [Info] ID {track_id} has recovered. Resetting status.")
                state['fall_status'] = 'standing'
                state['alert_sent'] = False
                state['horizontal_dwell_count'] = 0 # Reset dwell count on recovery
                user_data.fallen_person_recovery_zones = [p for p in user_data.fallen_person_recovery_zones if p['track_id'] != track_id]
        else:
            state['recovery_frame_count'] = 0

    # --- Trigger Alert (if a new fall was detected) ---
    if fall_detected and not state['alert_sent']:
        print(f"!!! FALL EVENT CONFIRMED for ID {track_id} !!!")
        fall_event_data = {
            "track_id": track_id, 
            "timestamp_obj": datetime.datetime.now(), 
            "velocity": vertical_velocity, 
            "posture_metric": posture_metric,
            "snapshot_path": None
        }
        trigger_analysis_job(frame, fall_event_data, bbox_dims)
        user_data.fallen_person_recovery_zones.append({'track_id': track_id, 'bbox': bbox_dims})
        state['alert_sent'] = True

def app_callback(pad, info, user_data: user_app_callback_class):
    """Main GStreamer callback function, called for each frame."""
    if time.time() - user_data.start_time < STARTUP_GRACE_PERIOD_SECONDS:
        return Gst.PadProbeReturn.OK

    buffer = info.get_buffer()
    if buffer is None: return Gst.PadProbeReturn.OK

    user_data.increment()
    current_frame_num = user_data.get_count()
    frame = None
    width, height = 0, 0
    if user_data.use_frame:
        format, width, height = get_caps_from_pad(pad)
        if all((format, width, height)):
            frame = get_numpy_from_buffer(buffer, format, width, height)

    active_ids_in_frame = set()
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    for detection in detections:
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if track: active_ids_in_frame.add(track[0].get_id())
    
    stale_ids = [
        track_id for track_id, state in user_data.person_states.items()
        if track_id not in active_ids_in_frame and state['fall_status'] != 'fallen' and
           (current_frame_num - state['last_frame_num'] > STALE_TRACK_THRESHOLD_FRAMES)
    ]
    for track_id in stale_ids: del user_data.person_states[track_id]

    keypoints_map = get_keypoints()
    for detection in detections:
        if detection.get_label() != "person": continue
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if not track: continue
        track_id = track[0].get_id()
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks: continue
        
        points = landmarks[0].get_points()
        bbox = detection.get_bbox()
        bbox_dims = (bbox.xmin(), bbox.ymin(), bbox.width(), bbox.height())
        torso_center_y = (points[keypoints_map['left_hip']].y() + points[keypoints_map['right_hip']].y()) / 2.0

        if track_id not in user_data.person_states:
            is_recovered_person = False
            for i, fallen_person in enumerate(user_data.fallen_person_recovery_zones):
                if calculate_iou(bbox_dims, fallen_person['bbox']) > RECOVERY_BBOX_IOU_THRESHOLD:
                    user_data.person_states[track_id] = user_data.person_states.pop(fallen_person['track_id'])
                    user_data.fallen_person_recovery_zones[i]['track_id'] = track_id
                    is_recovered_person = True
                    break
            if not is_recovered_person:
                # --- MODIFIED STATE INITIALISATION ---
                user_data.person_states[track_id] = {
                    'last_y': torso_center_y, 
                    'last_frame_num': current_frame_num, 
                    'fall_status': 'standing', 
                    'alert_sent': False, 
                    'potential_fall_start_frame': 0,
                    'recovery_frame_count': 0,
                    'horizontal_dwell_count': 0, # <-- ADDED for prolonged lie-down detection
                    'smoothed_vertical_spread': 1.0 
                }
        
        state = user_data.person_states[track_id]

        is_horizontal, posture_metric = calculate_stable_posture(points, state)
        vertical_velocity = update_person_metrics(state, torso_center_y, current_frame_num)
        process_person_state(state, track_id, vertical_velocity, is_horizontal, current_frame_num, posture_metric, frame, bbox_dims, user_data)
        
        state['last_y'] = torso_center_y
        state['last_frame_num'] = current_frame_num

        if frame is not None:
            draw_detection_on_frame(frame, bbox, track_id, state['fall_status'], width, height)
    
    user_data.alert_active = any(s['fall_status'] == 'fallen' for s in user_data.person_states.values())
    if frame is not None:
        if user_data.alert_active: draw_alert_banner(frame, width)
        user_data.set_frame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
    return Gst.PadProbeReturn.OK

def trigger_analysis_job(frame: np.ndarray, fall_data: dict, bbox_dims: tuple):
    """Creates a snapshot and a JSON job file for the analyst process."""
    print("--- Fall detected! Triggering analysis job. ---")
    timestamp_obj = fall_data['timestamp_obj']
    base_filename = f"job_{timestamp_obj.strftime('%Y%m%d_%H%M%S')}_{fall_data['track_id']}"
    
    snapshot_path = None
    if frame is not None:
        snapshot_filename = f"{base_filename}.jpg"
        temp_path = os.path.join(SNAPSHOT_DIR, snapshot_filename)
        try:
            cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            snapshot_path = temp_path
            print(f"  - Snapshot saved: {snapshot_path}")
        except Exception as e:
            print(f"  - Warning: Could not save snapshot. Reason: {e}")

    job_data = {
        "track_id": fall_data['track_id'], 
        "timestamp_iso": timestamp_obj.isoformat(), 
        "velocity": fall_data['velocity'], 
        "posture_metric": fall_data['posture_metric'],
        "snapshot_path": snapshot_path
    }
    
    job_path = os.path.join(ANALYSIS_QUEUE_DIR, f"{base_filename}.json")
    try:
        with open(job_path, 'w') as f:
            json.dump(job_data, f, indent=4)
        print(f"  - Analysis job created: {job_path}")
    except Exception as e:
        print(f"  - CRITICAL ERROR: Could not write job file! {e}")

# --- Unchanged Helper Functions ---

def get_keypoints():
    """Returns a dictionary mapping keypoint names to their indices."""
    return {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
    }

def calculate_iou(boxA, boxB):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    intersection_area = max(0, xB - xA) * max(0, yB - yA)
    if intersection_area == 0: return 0
    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union_area = float(boxA_area + boxB_area - intersection_area)
    return intersection_area / union_area

def draw_detection_on_frame(frame, bbox, track_id, status, width, height):
    """Draws a bounding box and status text for a detected person."""
    xmin = int(bbox.xmin() * width)
    ymin = int(bbox.ymin() * height)
    xmax = int((bbox.xmin() + bbox.width()) * width)
    ymax = int((bbox.ymin() + bbox.height()) * height)
    color_map = {'standing': (0, 255, 0), 'monitoring': (0, 255, 255), 'fallen': (0, 0, 255)}
    box_color = color_map.get(status, (255, 255, 255))
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
    cv2.putText(frame, f"ID {track_id}: {status}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

def draw_alert_banner(frame, width):
    """Draws a semi-transparent red banner at the top of the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 255), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    text = "FALL DETECTED!"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 1.5, 2)
    text_x = (width - text_size[0]) // 2
    cv2.putText(frame, text, (text_x, 55), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 2)

if __name__ == "__main__":
    print("--- Smart Fall Detector v4.1 (Sentry - Demo-Ready Hybrid Logic) ---")
    print("Starting GStreamer pipeline... Press Ctrl+C to exit.")
    user_data = user_app_callback_class()
    user_data.use_frame = True
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()