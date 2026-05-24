import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import shap
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from rules_engine import get_recommendation

app = FastAPI(title="Cash Flow Forecasting API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "latest_xgboost_model.json")
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "encoders.joblib")

model = xgb.Booster()
encoders = {}

def load_model_and_encoders():
    global encoders
    model_loaded = False
    if os.path.exists(MODEL_PATH):
        model.load_model(MODEL_PATH)
        model_loaded = True
    
    if os.path.exists(ENCODERS_PATH):
        try:
            encoders = joblib.load(ENCODERS_PATH)
        except Exception as e:
            print(f"Error loading encoders: {e}")
        
    return model_loaded

# Attempt to load model and encoders at startup
load_model_and_encoders()

class InvoiceData(BaseModel):
    Cust_Num: float = 5039221094.0
    Document_No: float = 91225037841.0
    Amount: float = 15000.0
    Age_Of_Customer_Months: float = 24.0
    No_of_orders_by_customer: float = 10.0
    Rank_of_order_by_customer: float = 5.0
    Doc_Date: str = "2026-05-24"
    Net_Due_Date: str = "2026-06-24"
    Payment_Method_description: str = "No Payment Method"
    Region: str = "AA111"
    City: str = "AA22"
    Customer_Name: str = "AA00"
    Zipcode: str = "AX0012"
    Payment_Term: float = 30.0

@app.post("/reload-model")
def reload_model_endpoint():
    """Endpoint for hot-swapping the model."""
    success = load_model_and_encoders()
    if success:
        return {"message": "Successfully hot-swapped to the newest model and encoders!"}
    raise HTTPException(status_code=404, detail="Model file not found")

@app.post("/predict")
def predict_invoice(data: InvoiceData):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model is not yet trained or loaded.")
        
    # Reload encoders if they weren't loaded yet
    global encoders
    if not encoders and os.path.exists(ENCODERS_PATH):
        try:
            encoders = joblib.load(ENCODERS_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load encoders: {e}")

    # 1. Parse dates and calculate engineered features
    try:
        doc_dt = datetime.strptime(data.Doc_Date, "%Y-%m-%d")
        due_dt = datetime.strptime(data.Net_Due_Date, "%Y-%m-%d")
    except ValueError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid date format. Expected YYYY-MM-DD. Error: {str(e)}"
        )

    days_to_pay_terms = (due_dt - doc_dt).days
    due_month = due_dt.month
    due_dayofweek = due_dt.weekday()  # Monday = 0, Sunday = 6
    
    # 2. Binning
    age_years = data.Age_Of_Customer_Months / 12.0
    
    # Customer Age Year Bins
    if age_years < 2:
        age_bin_str = '0 to 2'
    elif age_years < 4:
        age_bin_str = '2 to 4'
    elif age_years < 6:
        age_bin_str = '4 to 6'
    elif age_years < 8:
        age_bin_str = '6 to 8'
    elif age_years < 10:
        age_bin_str = '8 to 10'
    else:
        age_bin_str = '10+'

    # Amount Bins
    if data.Amount < 50000:
        amt_bin_str = '0 to 50K'
    elif data.Amount < 100000:
        amt_bin_str = '50K to 100K'
    elif data.Amount < 150000:
        amt_bin_str = '100K to 150K'
    elif data.Amount < 200000:
        amt_bin_str = '150K to 200K'
    else:
        amt_bin_str = '200K+'

    # Payment Term Bins
    if data.Payment_Term < 16:
        pt_bin_str = '0 to 15'
    elif data.Payment_Term < 31:
        pt_bin_str = '16 to 30'
    elif data.Payment_Term < 61:
        pt_bin_str = '31 to 60'
    elif data.Payment_Term < 181:
        pt_bin_str = '61 to 180'
    else:
        pt_bin_str = '180+'

    # Weekday Due String (e.g., 'Mon', 'Tue')
    weekday_due_str = due_dt.strftime('%a')

    # Weekday Due 1 (numeric representation: Mon=2, Tue=3, ..., Sun=1)
    weekday_due_1 = due_dt.isoweekday() % 7 + 1
    
    # Quarter Clearing (1, 2, 3, or 4 based on Net_Due_Date month)
    quarter_clearing = (due_dt.month - 1) // 3 + 1

    # Helper function for safe label encoding
    def safe_encode(col_name: str, val: str, default_class: str = None) -> float:
        val_str = str(val).strip()
        if col_name in encoders:
            le = encoders[col_name]
            if val_str in le.classes_:
                return float(le.transform([val_str])[0])
            if default_class and default_class in le.classes_:
                return float(le.transform([default_class])[0])
            # Default to first class index if completely unknown
            return 0.0
        return 0.0

    # 3. Encode categoricals safely
    encoded_payment_method = safe_encode('Payment_Method_description', data.Payment_Method_description, 'No Payment Method')
    encoded_region = safe_encode('Region', data.Region)
    encoded_city = safe_encode('City', data.City)
    encoded_customer_name = safe_encode('Customer_Name', data.Customer_Name)
    encoded_zipcode = safe_encode('Zipcode', data.Zipcode)
    encoded_age_bin = safe_encode('Customer_Age_Year_Bins', age_bin_str, '10+')
    encoded_amount_bin = safe_encode('Amount_Bins', amt_bin_str, '0 to 50K')
    encoded_payment_term_bin = safe_encode('Payment_Term_Bins', pt_bin_str, '16 to 30')
    encoded_weekday_due = safe_encode('Weekday_due', weekday_due_str)

    # 4. Construct complete features row matching the 22 expected features
    features_dict = {
        'Cust_Num': float(data.Cust_Num),
        'Payment_Method_description': encoded_payment_method,
        'Document_No': float(data.Document_No),
        'Amount': float(data.Amount),
        'Amount_Bins': encoded_amount_bin,
        'Zipcode': encoded_zipcode,
        'Region': encoded_region,
        'City': encoded_city,
        'Customer_Name': encoded_customer_name,
        'Age_Of_Customer_Months': float(data.Age_Of_Customer_Months),
        'Age_Of_Customer_Year': float(age_years),
        'Customer_Age_Year_Bins': encoded_age_bin,
        'Payment_Term': float(data.Payment_Term),
        'Payment_Term_Bins': encoded_payment_term_bin,
        'No_of_orders_by_customer': float(data.No_of_orders_by_customer),
        'Rank_of_order_by_customer': float(data.Rank_of_order_by_customer),
        'Weekday_due': encoded_weekday_due,
        'Quarter_clearing': float(quarter_clearing),
        'Weekday_due.1': float(weekday_due_1),
        'days_to_pay_terms': float(days_to_pay_terms),
        'due_month': float(due_month),
        'due_dayofweek': float(due_dayofweek)
    }

    df_features = pd.DataFrame([features_dict])
    
    # Reorder columns to match model's expected features
    if model.feature_names:
        df_features = df_features[model.feature_names]
        
    dmatrix = xgb.DMatrix(df_features)
    
    # 5. Prediction
    predicted_delay_arr = model.predict(dmatrix)
    predicted_delay = float(predicted_delay_arr[0])
    
    # 6. Next Best Action (Rules Engine)
    action = get_recommendation(predicted_delay, data.Amount, data.Age_Of_Customer_Months)
    
    # 7. SHAP Explainability
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_features)
        
        feature_names = df_features.columns.tolist()
        shap_dict = dict(zip(feature_names, shap_values[0]))
        sorted_shap = sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)
        top_3_reasons = [{"feature": k, "impact": float(v)} for k, v in sorted_shap[:3]]
    except Exception as e:
        top_3_reasons = [{"error": str(e)}]

    return {
        "Predicted_Delay": predicted_delay,
        "Next_Best_Action": action,
        "Top_SHAP_Reasons": top_3_reasons
    }

