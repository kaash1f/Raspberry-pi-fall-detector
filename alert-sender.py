# ==============================================================================
# SCRIPT: alert_sender.py
#
# ROLE: The Messenger (Real-time Alerter)
#
# SUMMARY:
# This is a lightweight script designed to run continuously in the background.
# Its only job is to watch the 'analysis_queue' for new incident files,
# send an immediate notification via Telegram, and then move the job file
# to a 'processed_jobs' directory for later, offline analysis.
#
# This script does NOT use the Ollama LLM, ensuring minimal resource usage
# and maximum reliability for the critical task of sending an alert.
# ==============================================================================
import os
import json
import time
import requests
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

ANALYSIS_QUEUE_DIR = "analysis_queue"
PROCESSED_JOBS_DIR = "processed_jobs" # <-- New directory for hand-off
POLLING_INTERVAL_SECONDS = 3 # Check for new jobs frequently

def create_simple_alert_message(job_data: dict) -> str:
    """Creates a basic, templated alert message without using an LLM."""
    timestamp = job_data.get('timestamp_iso', 'N/A')
    track_id = job_data.get('track_id', 'N/A')
    
    # Simple, clear, and fast message generation
    message = (
        f"🚨 IMMEDIATE FALL ALERT 🚨\n\n"
        f"A potential fall has been detected and requires your attention.\n\n"
        f"Person ID: {track_id}\n"
        f"Time: {timestamp}\n\n"
        f"A detailed report will be generated later. A snapshot is attached if available."
    )
    return message

def send_telegram_text_alert(message: str):
    """Sends a text-only message to Telegram."""
    if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        print("  - Skipping Telegram: Credentials not set.")
        return False
    api_url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
    try:
        response = requests.post(api_url, data=payload, timeout=10)
        response.raise_for_status()
        print("  - Success: Telegram text-only alert sent.")
        return True
    except Exception as e:
        print(f"  - Warning: Could not send Telegram text alert: {e}")
        return False

def send_telegram_photo_alert(caption: str, image_path: str):
    """Sends a photo with a caption to Telegram."""
    if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        print("  - Skipping Telegram: Credentials not set.")
        return False
    api_url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto'
    try:
        with open(image_path, 'rb') as photo_file:
            files = {'photo': photo_file}
            payload = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption}
            response = requests.post(api_url, data=payload, files=files, timeout=15)
            response.raise_for_status()
            print("  - Success: Telegram photo alert sent.")
            return True
    except Exception as e:
        print(f"  - Warning: Could not send Telegram photo alert: {e}")
        return False

def process_alert_job(job_path: str):
    """
    Reads a job, sends a Telegram alert, and moves the job for later processing.
    """
    job_basename = os.path.basename(job_path)
    print(f"\nNew incident detected: {job_basename}")

    try:
        with open(job_path, 'r') as f:
            job_data = json.load(f)

        # 1. Generate the simple, non-LLM message
        alert_message = create_simple_alert_message(job_data)

        # 2. Attempt to send the alert via Telegram
        snapshot_path = job_data.get("snapshot_path")
        if snapshot_path and os.path.exists(snapshot_path):
            send_telegram_photo_alert(alert_message, snapshot_path)
        else:
            send_telegram_text_alert(alert_message)
        
        # 3. Move the file for the report generator
        destination_path = os.path.join(PROCESSED_JOBS_DIR, job_basename)
        os.rename(job_path, destination_path)
        print(f"  - Job file moved to '{destination_path}' for daily reporting.")

    except json.JSONDecodeError as e:
        print(f"  - ERROR: Could not read job file {job_basename}. It may be corrupt. Error: {e}")
    except Exception as e:
        print(f"  - FATAL ERROR processing {job_basename}: {e}")
        # We leave the file in the queue for manual inspection

if __name__ == "__main__":
    print("--- Real-time Alert Sender v1.0 ---")
    
    os.makedirs(ANALYSIS_QUEUE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_JOBS_DIR, exist_ok=True)
    
    if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        print("\nNOTE: Telegram credentials not set. Alerts will not be sent.")
    
    print(f"\nWatching for jobs in '{ANALYSIS_QUEUE_DIR}'... Press Ctrl+C to exit.")
    
    while True:
        try:
            job_files = [f for f in os.listdir(ANALYSIS_QUEUE_DIR) if f.endswith('.json')]
            if job_files:
                # Process the oldest job first
                job_to_process = os.path.join(ANALYSIS_QUEUE_DIR, sorted(job_files)[0])
                process_alert_job(job_to_process)
            else:
                time.sleep(POLLING_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\nAlert Sender shutting down.")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")
            time.sleep(POLLING_INTERVAL_SECONDS * 2)