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
    
    # In a fully mature setup, we'd check for drift here by evaluating recent rows.
    # For now, we simulate retraining to keep the model fresh.
    print("3. Retraining model...")
    success = train_and_save_model(cleaned_df)
    
    if success:
        print("4. Notifying API to reload model...")
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
