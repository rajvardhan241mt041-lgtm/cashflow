import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

LEAKAGE_COLS = ['Clearing_date', 'Clearing_doc', 'Delay_Bins', 'DelayFlag', 'Weekday_clearing', 'Weekday_clearnum']

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans data, engineers dates, and removes data leakage columns.
    """
    # 1. Drop Leakage columns
    cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 2. Date Engineering
    date_cols = ['Doc_Date', 'Posting_Date', 'Net_Due_Date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    if 'Net_Due_Date' in df.columns and 'Doc_Date' in df.columns:
        df['days_to_pay_terms'] = (df['Net_Due_Date'] - df['Doc_Date']).dt.days

    if 'Net_Due_Date' in df.columns:
        df['due_month'] = df['Net_Due_Date'].dt.month
        df['due_dayofweek'] = df['Net_Due_Date'].dt.dayofweek
    
    # Drop original date columns
    df = df.drop(columns=[c for c in date_cols if c in df.columns])

    # 3. Categorical Encoding
    cat_cols = ['Payment_Method_description', 'Region', 'City', 'Customer_Age_Year_Bins']
    # If the user has other string columns, encode them as well
    for col in df.select_dtypes(include=['object']).columns:
        if col not in cat_cols:
            cat_cols.append(col)

    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            # Convert to string to avoid mixed type errors
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
    # Save encoders if needed for inference mapping later
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(encoders, os.path.join(model_dir, "encoders.joblib"))

    return df
