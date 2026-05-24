import time
import schedule
import pandas as pd
import requests
import os
from data_prep import prepare_data
from model_training import train_and_save_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Dataset.csv")
API_URL = os.environ.get("API_URL", "http://localhost:8000/reload-model")

def run_pipeline():
    print("Starting Nightly MLOps Pipeline...")
    
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}. Skipping.")
        return
        
    print("1. Ingesting new daily rows...")
    df = pd.read_csv(DATA_PATH)
    
    print("2. Running data preparation (Date engineering, encoding, dropping leakage)...")
    cleaned_df = prepare_data(df)
    
    print("3. Checking for Data Drift...")
    from drift_detection import detect_drift
    
    baseline_path = os.path.join(BASE_DIR, "models", "baseline_data.csv")
    needs_retraining = True
    
    if os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)
        # Using the last 500 rows to simulate 'recent' data
        recent_df = cleaned_df.tail(500)
        
        # Drop target for drift detection
        if 'Days_Overdue_Delay' in baseline_df.columns:
            baseline_features = baseline_df.drop(columns=['Days_Overdue_Delay'])
        else:
            baseline_features = baseline_df
            
        recent_features = recent_df.drop(columns=['Days_Overdue_Delay'], errors='ignore')
        
        drift_detected = detect_drift(baseline_features, recent_features, threshold=0.2)
        needs_retraining = drift_detected
    else:
        print("No baseline data found. Initial training required.")

    success = True
    if needs_retraining:
        print("4. Retraining model...")
        success = train_and_save_model(cleaned_df)
    else:
        print("4. Skipping retraining. Model is stable.")

    if success and needs_retraining:
        print("5. Notifying API to reload model...")
        try:
            response = requests.post(API_URL)
            print(f"API Response: {response.json()}")
        except Exception as e:
            print(f"Failed to notify API: {e}")

if __name__ == "__main__":
    # Run once at startup
    run_pipeline()
    
    # Schedule to run every day at midnight
    schedule.every().day.at("00:00").do(run_pipeline)
    
    print("Scheduler is active. Waiting for next job...")
    while True:
        schedule.run_pending()
        time.sleep(60)
