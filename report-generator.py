# ==============================================================================
# SCRIPT: report_generator.py
#
# ROLE: The Analyst (On-Demand Report Generator)
#
# SUMMARY:
# This is a resource-intensive script intended to be run on-demand or on a
# schedule (e.g., once daily). It first verifies the Ollama service is running,
# then scans for processed incident files, uses the LLM to generate a single,
# consolidated summary report, and archives the source files.
#
# The final report is saved locally on the device for review.
# ==============================================================================
import os
import json
import sys  # Import the sys module to allow for a clean exit
import ollama
from dotenv import load_dotenv
from datetime import datetime

# --- Configuration ---
load_dotenv()
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'gemma3:1b-it-qat')

PROCESSED_JOBS_DIR = "processed_jobs"
REPORTS_DIR = "daily_reports"
ARCHIVE_DIR = "archived_jobs"

def check_ollama_service():
    """
    Performs a pre-flight check to ensure the Ollama service is running and reachable.
    Returns True if the service is active, False otherwise.
    """
    print("--- Checking Ollama Service Status ---")
    try:
        # ollama.list() is a lightweight, low-resource command that will fail
        # if the service is not running or is otherwise unreachable.
        ollama.list()
        print("✅ Ollama service is running and responsive.")
        return True
    except Exception:
        # We catch a broad exception, as the underlying error could be a
        # connection refusal, timeout, or other network issue.
        print("\n❌ CRITICAL ERROR: Could not connect to the Ollama service.")
        print("   Please ensure the Ollama background service is running on your device.")
        print("   You can check its status with: sudo systemctl status ollama")
        print("   You can start it with:         sudo systemctl start ollama")
        return False

def generate_summary_report(incidents: list) -> str:
    """Generates a single, consolidated report for multiple incidents using an LLM."""
    if not incidents:
        return "No new incidents to report."

    # Build a detailed string containing all incident data
    incident_details = ""
    for i, data in enumerate(incidents):
        incident_details += (
            f"\n--- Incident {i+1} ---\n"
            f"- Person ID: {data.get('track_id')}\n"
            f"- Timestamp: {data.get('timestamp_iso')}\n"
            f"- Downward Velocity Metric: {data.get('velocity', 0):.2f}\n"
            f"- Final Posture Metric: {data.get('posture_metric', 0):.2f}\n"
            f"- Snapshot Available: {'Yes' if data.get('snapshot_path') else 'No'}\n"
        )
    
    prompt = f"""
    You are a professional safety and security analyst. You are compiling a daily summary report
    based on a list of fall detection events from an automated monitoring system.

    Your task is to create a clear, well-structured report summarising the events below.
    - Start with an executive summary (e.g., "Total of X incidents reported").
    - Then, list each incident with a brief, professional description.
    - Maintain a factual and objective tone.

    Here is the raw data for the reporting period:
    {incident_details}

    Generate the full summary report now.
    """
    
    try:
        print(f"  - Asking '{OLLAMA_MODEL}' to analyse {len(incidents)} incidents...")
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': prompt}])
        report = response['message']['content']
        print("  - LLM analysis complete.")
        return report.strip()
    except Exception as e:
        print(f"  - WARNING: LLM report generation failed. Reason: {e}")
        fallback_report = (
            f"LLM ANALYSIS FAILED\n\n"
            f"The AI model could not be reached. Raw incident data is provided below:\n{incident_details}"
        )
        return fallback_report

if __name__ == "__main__":
    print("--- On-Demand Incident Report Generator v1.1 (with Health Check) ---")
    
    # --- Pre-flight check for the Ollama service ---
    # If the service isn't running, print an error and exit the script.
    if not check_ollama_service():
        sys.exit(1) # Exit with a non-zero status code to indicate an error

    os.makedirs(PROCESSED_JOBS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    try:
        job_files = [f for f in os.listdir(PROCESSED_JOBS_DIR) if f.endswith('.json')]
        
        if not job_files:
            print("\nNo new incidents found in the processed queue. Exiting.")
            sys.exit(0)
            
        print(f"\nFound {len(job_files)} new incidents to include in the report.")

        all_incidents = []
        for filename in job_files:
            job_path = os.path.join(PROCESSED_JOBS_DIR, filename)
            try:
                with open(job_path, 'r') as f:
                    all_incidents.append(json.load(f))
            except Exception as e:
                print(f"  - Warning: Could not read or parse {filename}. Skipping. Error: {e}")

        if not all_incidents:
            print("No valid incident data could be read. Exiting.")
            sys.exit(0)
            
        summary_report_text = generate_summary_report(all_incidents)

        report_filename = f"incident_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path = os.path.join(REPORTS_DIR, report_filename)
        try:
            with open(report_path, 'w') as f:
                f.write(summary_report_text)
            print(f"\nSUCCESS: Report saved to '{report_path}'")
        except Exception as e:
            print(f"  - CRITICAL ERROR: Could not save the final report! Error: {e}")
            sys.exit(1)
            
        print("Archiving processed job files...")
        for filename in job_files:
            source_path = os.path.join(PROCESSED_JOBS_DIR, filename)
            dest_path = os.path.join(ARCHIVE_DIR, filename)
            try:
                os.rename(source_path, dest_path)
            except Exception as e:
                print(f"  - Warning: Could not archive {filename}. Error: {e}")
        
        print("Report generation complete.")

    except Exception as e:
        print(f"An unexpected error occurred during report generation: {e}")
        sys.exit(1)