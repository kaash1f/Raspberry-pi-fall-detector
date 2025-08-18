# AI-Powered Fall Detection System for Edge Devices

## 1. Project Overview

This project implements a robust, real-time fall detection system designed to run on a resource-constrained edge device like a Raspberry Pi, augmented with a Hailo AI accelerator.

The core mission is to provide immediate, reliable fall alerts while avoiding system instability caused by resource-heavy tasks. It achieves this by decoupling the system into three distinct, cooperating processes:

*   **The Sentry (`fall_detector.py`):** A high-performance, real-time script that uses the Hailo processor to analyse a video feed, detect falls based on a hybrid logic model, and create a small "incident job file".
*   **The Messenger (`alert_sender.py`):** A lightweight, continuously running script that watches for new incident files. Its sole purpose is to send an immediate, templated alert to Telegram, ensuring the notification is delivered with minimal delay and resource usage.
*   **The Analyst (`report_generator.py`):** A resource-intensive script that uses a local Large Language Model (Ollama) to analyse all incidents. It is designed to be run on-demand or on a daily schedule, producing a consolidated, human-readable report without interfering with real-time detection.

## 2. System Architecture & Data Flow

The system is designed to be resilient. A failure in the reporting stage will not prevent a real-time alert from being sent.

1.  **Detection:** `fall_detector.py` detects a fall event.
    *   **Output:** Creates a snapshot image in `fall_snapshots/`.
    *   **Output:** Creates a `job_... .json` file in `analysis_queue/`.

2.  **Alerting:** `alert_sender.py` notices the new `.json` file.
    *   **Action:** Sends a simple, instant alert to Telegram with the snapshot.
    *   **Action:** Moves the `.json` file from `analysis_queue/` to `processed_jobs/`.

3.  **Reporting (On-Demand/Scheduled):** `report_generator.py` is run.
    *   **Action:** Scans all `.json` files in `processed_jobs/`.
    *   **Action:** Uses the Ollama LLM to generate a single summary report.
    *   **Output:** Saves a detailed `.md` report file to `daily_reports/`.
    *   **Action:** Moves the processed `.json` files to `archived_jobs/` to prevent re-processing.