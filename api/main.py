import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import shap
import pandas as pd
from typing import List, Dict, Any
from rules_engine import get_recommendation

app = FastAPI(title="Cash Flow Forecasting API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "latest_xgboost_model.json")
model = xgb.Booster()

def load_model():
    if os.path.exists(MODEL_PATH):
        model.load_model(MODEL_PATH)
        return True
    return False

# Attempt to load model at startup
load_model()

class InvoiceData(BaseModel):
    Amount: float
    Age_Of_Customer_Months: float
    features: Dict[str, Any]

@app.post("/reload-model")
def reload_model_endpoint():
    """Endpoint for hot-swapping the model."""
    success = load_model()
    if success:
        return {"message": "Successfully hot-swapped to the newest model!"}
    raise HTTPException(status_code=404, detail="Model file not found")

@app.post("/predict")
def predict_invoice(data: InvoiceData):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model is not yet trained or loaded.")
        
    # Convert input to DataFrame for XGBoost and SHAP
    df_features = pd.DataFrame([data.features])
    
    dmatrix = xgb.DMatrix(df_features)
    
    # 1. Prediction
    predicted_delay_arr = model.predict(dmatrix)
    predicted_delay = float(predicted_delay_arr[0])
    
    # 2. Next Best Action (Rules Engine)
    action = get_recommendation(predicted_delay, data.Amount, data.Age_Of_Customer_Months)
    
    # 3. SHAP Explainability
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
