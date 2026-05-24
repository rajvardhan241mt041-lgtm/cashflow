import pandas as pd
import numpy as np

def calculate_psi(expected, actual, buckets=10):
    """Calculates Population Stability Index for a single feature."""
    # Handle single values or empty arrays
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    # Use percentiles to create bins
    bins = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    bins[0] -= 0.0001
    bins[-1] += 0.0001
    bins = np.unique(bins) # Ensure bins are unique
    
    if len(bins) < 2:
        return 0.0

    # Calculate frequencies in each bin
    expected_percents = np.histogram(expected, bins)[0] / len(expected)
    actual_percents = np.histogram(actual, bins)[0] / len(actual)
    
    # Replace 0 with a small value to avoid division by zero or log(0)
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    # PSI Formula
    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi_value

def detect_drift(baseline_df: pd.DataFrame, recent_df: pd.DataFrame, threshold=0.2) -> bool:
    """
    Calculates PSI for all numeric columns.
    Returns True if any feature crosses the PSI threshold.
    """
    numeric_cols = baseline_df.select_dtypes(include=[np.number]).columns
    
    drift_detected = False
    print("\n--- PSI Data Drift Check ---")
    for col in numeric_cols:
        if col in recent_df.columns:
            psi = calculate_psi(baseline_df[col].dropna(), recent_df[col].dropna())
            if psi > threshold:
                print(f"HIGH DRIFT DETECTED in '{col}' -> PSI: {psi:.4f} (Threshold: {threshold})")
                drift_detected = True
            elif psi > (threshold / 2):
                print(f"Warning in '{col}' -> PSI: {psi:.4f}")
    
    if drift_detected:
        print("Overall Status: DRIFT DETECTED. Model retraining triggered.")
    else:
        print("Overall Status: NO DRIFT. Current model is stable.")
        
    return drift_detected
