"""
main.py — FastAPI Model Serving & Cash Flow Scoring API
========================================================
SAD V1 compliant. Provides:
    POST /score-invoice  — Full inference pipeline with SHAP + rules engine
    POST /reload-model   — Hot-swap to latest Production model from MLflow
    GET  /model-info     — Returns current model version and MLflow metadata
    GET  /health         — Health check (required by docker-compose healthcheck)

Model Loading Strategy:
    1. At startup, attempt to load from MLflow Model Registry (Production stage).
    2. If MLflow is unavailable, fall back to local 'models/' flat files.
    3. /reload-model re-executes step 1 without restarting the container.

Author: MLOps Factory
PEP-8 compliant.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import joblib
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from rules_engine import Recommendation, get_recommendation

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Cash Flow Forecasting & Recommendation API",
    description=(
        "SAD V1 compliant scoring API. "
        "Predicts Days_Overdue_Delay for AR invoices using XGBoost + MLflow."
    ),
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "models"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REGISTERED_MODEL_NAME = "cashflow_xgb"

# Paths for flat-file fallback
FALLBACK_MODEL_PATH = os.path.join(MODEL_DIR, "latest_xgboost_model.json")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.joblib")

# ---------------------------------------------------------------------------
# Global model state — mutated by load_production_model() and /reload-model
# ---------------------------------------------------------------------------
_model: Optional[Any] = None           # XGBRegressor or xgb.Booster
_encoders: Dict[str, Any] = {}
_model_source: str = "none"            # "mlflow" or "fallback"
_model_version: Optional[str] = None
_model_run_id: Optional[str] = None
_model_loaded_at: Optional[datetime] = None


# ===========================================================================
# MODEL LOADING
# ===========================================================================

def load_production_model() -> bool:
    """
    Load the Production model from the MLflow Model Registry.
    Falls back to the local flat-file model if MLflow is unreachable.

    Returns:
        bool: True if a model was successfully loaded, False otherwise.
    """
    global _model, _encoders, _model_source, _model_version
    global _model_run_id, _model_loaded_at

    # ------------------------------------------------------------------
    # Attempt 1: MLflow Model Registry
    # ------------------------------------------------------------------
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/Production"
        loaded = mlflow.xgboost.load_model(model_uri)

        # Retrieve version metadata from the registry
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.get_latest_versions(
            REGISTERED_MODEL_NAME, stages=["Production"]
        )
        if versions:
            _model_version = versions[0].version
            _model_run_id = versions[0].run_id
        else:
            _model_version = "unknown"
            _model_run_id = "unknown"

        _model = loaded
        _model_source = "mlflow"
        _model_loaded_at = datetime.utcnow()

        logger.info(
            "Loaded Production model from MLflow registry — version=%s, run_id=%s",
            _model_version, _model_run_id
        )

    except Exception as mlflow_exc:
        logger.warning(
            "Could not load model from MLflow (%s). Trying local fallback...",
            mlflow_exc
        )

        # --------------------------------------------------------------
        # Attempt 2: Local flat-file fallback (xgb.Booster)
        # --------------------------------------------------------------
        if os.path.exists(FALLBACK_MODEL_PATH):
            booster = xgb.Booster()
            booster.load_model(FALLBACK_MODEL_PATH)
            _model = booster
            _model_source = "fallback"
            _model_version = "local"
            _model_run_id = "N/A"
            _model_loaded_at = datetime.utcnow()
            logger.info("Loaded fallback model from '%s'", FALLBACK_MODEL_PATH)
        else:
            logger.error(
                "No model available. MLflow failed and no local fallback found at '%s'.",
                FALLBACK_MODEL_PATH
            )
            return False

    # ------------------------------------------------------------------
    # Load encoders (always from disk — shared between worker and API)
    # ------------------------------------------------------------------
    if os.path.exists(ENCODERS_PATH):
        _encoders = joblib.load(ENCODERS_PATH)
        logger.info("Loaded %d categorical encoders from '%s'", len(_encoders), ENCODERS_PATH)
    else:
        logger.warning("Encoders file not found at '%s'. Categorical encoding disabled.", ENCODERS_PATH)

    return True


# Load model at startup
@app.on_event("startup")
async def startup_event():
    """FastAPI startup hook — load model immediately on container start."""
    load_production_model()


# ===========================================================================
# FEATURE ENGINEERING — mirrors feature_store.py for inference parity
# ===========================================================================

# Categorical columns that must be label-encoded at inference time
CATEGORICAL_COLUMNS = [
    "Payment_Method_description", "Region", "City", "Customer_Age_Year_Bins",
    "Amount_Bins", "Zipcode", "Customer_Name", "Payment_Term_Bins", "Weekday_due",
]


def _bin_amount(amount: float) -> str:
    """Return the Amount_Bins label matching training data conventions."""
    if amount < 50_000:
        return "0 to 50K"
    elif amount < 100_000:
        return "50K to 100K"
    elif amount < 150_000:
        return "100K to 150K"
    elif amount < 200_000:
        return "150K to 200K"
    return "200K+"


def _bin_payment_term(term_days: float) -> str:
    """Return the Payment_Term_Bins label matching training data conventions."""
    if term_days < 16:
        return "0 to 15"
    elif term_days < 31:
        return "16 to 30"
    elif term_days < 61:
        return "31 to 60"
    elif term_days < 181:
        return "61 to 180"
    return "180+"


def _bin_customer_age_years(months: float) -> str:
    """Return the Customer_Age_Year_Bins label matching training data."""
    years = months / 12.0
    if years < 2:
        return "0 to 2"
    elif years < 4:
        return "2 to 4"
    elif years < 6:
        return "4 to 6"
    elif years < 8:
        return "6 to 8"
    elif years < 10:
        return "8 to 10"
    return "10+"


def _safe_encode(col_name: str, value: str, encoders: dict) -> float:
    """
    Encode a categorical value using the fitted LabelEncoder.
    Unknown categories fall back to index 0 (safe degradation).
    """
    le = encoders.get(col_name)
    if le is None:
        return 0.0
    val_str = str(value).strip()
    if val_str in le.classes_:
        return float(le.transform([val_str])[0])
    return 0.0


def build_feature_row(invoice: "InvoiceScoreRequest", encoders: dict) -> pd.DataFrame:
    """
    Convert a raw InvoiceScoreRequest into a single-row feature dataframe
    matching the schema expected by the trained XGBoost model.

    This function replicates the transformations in feature_store.engineer_features()
    so inference is fully consistent with training.

    Args:
        invoice: Validated Pydantic request model.
        encoders: Dict of fitted LabelEncoders keyed by column name.

    Returns:
        pd.DataFrame: Single-row feature matrix ready for xgb.DMatrix.
    """
    # Parse dates
    try:
        doc_dt = datetime.strptime(invoice.Doc_Date, "%Y-%m-%d")
        due_dt = datetime.strptime(invoice.Net_Due_Date, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format (expected YYYY-MM-DD): {exc}"
        )

    # Date-derived features
    days_to_pay_terms = (due_dt - doc_dt).days
    due_month = due_dt.month
    due_dayofweek = due_dt.weekday()   # Monday=0
    due_quarter = (due_dt.month - 1) // 3 + 1
    due_year = due_dt.year
    quarter_clearing = due_quarter     # Proxy for quarter at prediction time
    weekday_due_1 = (due_dt.weekday() + 1) % 7 + 1

    # Weekday name for encoding
    weekday_name = due_dt.strftime("%a")  # "Mon", "Tue", …

    # Derived bin strings
    age_bin_str = _bin_customer_age_years(invoice.Age_Of_Customer_Months)
    amount_bin_str = _bin_amount(invoice.Amount)
    payment_term_bin_str = _bin_payment_term(invoice.Payment_Term)

    # Categorical encodings (safe — unknown → 0)
    features = {
        "Cust_Num": float(invoice.Cust_Num),
        "Payment_Method_description": _safe_encode(
            "Payment_Method_description", invoice.Payment_Method_description, encoders
        ),
        "Document_No": float(invoice.Document_No),
        "Amount": float(invoice.Amount),
        "Amount_Bins": _safe_encode("Amount_Bins", amount_bin_str, encoders),
        "Zipcode": _safe_encode("Zipcode", invoice.Zipcode, encoders),
        "Region": _safe_encode("Region", invoice.Region, encoders),
        "City": _safe_encode("City", invoice.City, encoders),
        "Customer_Name": _safe_encode("Customer_Name", invoice.Customer_Name, encoders),
        "Age_Of_Customer_Months": float(invoice.Age_Of_Customer_Months),
        "Age_Of_Customer_Year": invoice.Age_Of_Customer_Months / 12.0,
        "Customer_Age_Year_Bins": _safe_encode(
            "Customer_Age_Year_Bins", age_bin_str, encoders
        ),
        "Payment_Term": float(invoice.Payment_Term),
        "Payment_Term_Bins": _safe_encode(
            "Payment_Term_Bins", payment_term_bin_str, encoders
        ),
        "No_of_orders_by_customer": float(invoice.No_of_orders_by_customer),
        "Rank_of_order_by_customer": float(invoice.Rank_of_order_by_customer),
        "Weekday_due": _safe_encode("Weekday_due", weekday_name, encoders),
        "Quarter_clearing": float(quarter_clearing),
        "Weekday_due.1": float(weekday_due_1),
        "days_to_pay_terms": float(days_to_pay_terms),
        "due_month": float(due_month),
        "due_dayofweek": float(due_dayofweek),
        # Additional engineered features present in training data
        "due_quarter": float(due_quarter),
        "due_year": float(due_year),
    }

    return pd.DataFrame([features])


# ===========================================================================
# REQUEST / RESPONSE SCHEMAS
# ===========================================================================

class InvoiceScoreRequest(BaseModel):
    """Raw invoice data submitted for scoring. Mirrors the raw ERP schema."""
    Cust_Num: float = Field(default=5039221094.0, description="Customer number")
    Document_No: float = Field(default=91225037841.0, description="Invoice document number")
    Amount: float = Field(default=15000.0, description="Invoice amount in currency units")
    Age_Of_Customer_Months: float = Field(default=24.0, description="Customer relationship age in months")
    No_of_orders_by_customer: float = Field(default=10.0, description="Total historical orders by customer")
    Rank_of_order_by_customer: float = Field(default=5.0, description="Chronological rank of this invoice")
    Doc_Date: str = Field(default="2026-05-24", description="Invoice document date (YYYY-MM-DD)")
    Net_Due_Date: str = Field(default="2026-06-24", description="Net payment due date (YYYY-MM-DD)")
    Payment_Method_description: str = Field(default="No Payment Method")
    Region: str = Field(default="AA111")
    City: str = Field(default="AA22")
    Customer_Name: str = Field(default="AA00")
    Zipcode: str = Field(default="AX0012")
    Payment_Term: float = Field(default=30.0, description="Contractual payment term in days")


class SHAPReason(BaseModel):
    """A single feature's contribution to the predicted delay."""
    feature: str
    impact_days: float
    direction: str  # "increases" or "decreases"


class InvoiceScoreResponse(BaseModel):
    """Structured scoring response returned to the caller."""
    Predicted_Delay: float
    Expected_Payment_Date: str
    Next_Best_Action: str
    Priority: int
    Reason: str
    Top_SHAP_Reasons: List[SHAPReason]
    Model_Version: Optional[str]
    Model_Source: str


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    registered_model_name: str
    model_version: Optional[str]
    model_source: str
    run_id: Optional[str]
    loaded_at: Optional[str]
    mlflow_tracking_uri: str


# ===========================================================================
# ENDPOINTS
# ===========================================================================

@app.get("/health", tags=["Infrastructure"])
def health_check():
    """Health check endpoint required by Docker healthcheck config."""
    return {"status": "healthy", "model_loaded": _model is not None}


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Infrastructure"])
def model_info():
    """Return metadata about the currently loaded Production model."""
    return ModelInfoResponse(
        registered_model_name=REGISTERED_MODEL_NAME,
        model_version=_model_version,
        model_source=_model_source,
        run_id=_model_run_id,
        loaded_at=_model_loaded_at.isoformat() if _model_loaded_at else None,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    )


@app.post("/reload-model", tags=["Infrastructure"])
def reload_model():
    """
    Hot-swap the serving model to the latest Production version from MLflow.
    Called automatically by the Prefect flow after a successful training run.
    """
    success = load_production_model()
    if success:
        return {
            "message": "Model reloaded successfully.",
            "model_version": _model_version,
            "model_source": _model_source,
            "run_id": _model_run_id,
        }
    raise HTTPException(status_code=503, detail="Failed to load any model.")


@app.post("/score-invoice", response_model=InvoiceScoreResponse, tags=["Scoring"])
def score_invoice(invoice: InvoiceScoreRequest):
    """
    Full inference pipeline for a single invoice.

    Performs:
        1. Feature engineering (date math, binning, label encoding).
        2. XGBoost prediction (Days_Overdue_Delay).
        3. SHAP explainability (Top 3 feature drivers).
        4. Business rules engine (Next-Best-Action).
        5. Expected payment date calculation.

    Returns:
        InvoiceScoreResponse with all fields populated.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="No model is loaded. Wait for the worker to complete its first training run."
        )

    # ------------------------------------------------------------------
    # 1. Build feature row
    # ------------------------------------------------------------------
    df_features = build_feature_row(invoice, _encoders)

    # Align columns to model's expected feature names
    if hasattr(_model, "feature_names_in_"):
        # MLflow XGBRegressor path
        expected_cols = list(_model.feature_names_in_)
    elif hasattr(_model, "feature_names") and _model.feature_names:
        # xgb.Booster (fallback) path
        expected_cols = _model.feature_names
    else:
        expected_cols = df_features.columns.tolist()

    # Add any missing columns as NaN (XGBoost handles them natively)
    for col in expected_cols:
        if col not in df_features.columns:
            df_features[col] = np.nan

    # Keep only expected columns in correct order
    df_features = df_features[expected_cols]

    # ------------------------------------------------------------------
    # 2. Predict
    # ------------------------------------------------------------------
    try:
        if _model_source == "mlflow":
            # MLflow model returns ndarray directly
            predicted_delay = float(_model.predict(df_features)[0])
        else:
            # Booster needs DMatrix
            dmatrix = xgb.DMatrix(df_features)
            predicted_delay = float(_model.predict(dmatrix)[0])
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    # ------------------------------------------------------------------
    # 3. Expected Payment Date
    #    = Net_Due_Date + Predicted_Delay (rounded to nearest day)
    # ------------------------------------------------------------------
    due_dt = datetime.strptime(invoice.Net_Due_Date, "%Y-%m-%d")
    expected_payment_date = due_dt + timedelta(days=round(predicted_delay))
    expected_payment_date_str = expected_payment_date.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # 4. SHAP Explainability
    # ------------------------------------------------------------------
    shap_reasons: List[SHAPReason] = []
    try:
        # Use the underlying Booster for SHAP (works for both model types)
        if _model_source == "mlflow":
            booster = _model.get_booster()
        else:
            booster = _model

        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(df_features)

        # Build sorted (abs desc) list of feature impacts
        shap_series = pd.Series(
            shap_values[0],
            index=df_features.columns
        ).sort_values(key=abs, ascending=False)

        for feat, impact in shap_series.head(3).items():
            shap_reasons.append(SHAPReason(
                feature=feat,
                impact_days=round(float(impact), 3),
                direction="increases" if impact > 0 else "decreases",
            ))
    except Exception as shap_exc:
        logger.warning("SHAP computation failed: %s", shap_exc)
        shap_reasons = []

    # ------------------------------------------------------------------
    # 5. Business Rules Engine
    # ------------------------------------------------------------------
    recommendation: Recommendation = get_recommendation(
        predicted_delay=predicted_delay,
        amount=invoice.Amount,
        age_of_customer_months=invoice.Age_Of_Customer_Months,
    )

    logger.info(
        "Scored invoice %s — delay=%.2f days, action=%s, expected_date=%s",
        invoice.Document_No, predicted_delay,
        recommendation.action, expected_payment_date_str
    )

    return InvoiceScoreResponse(
        Predicted_Delay=round(predicted_delay, 2),
        Expected_Payment_Date=expected_payment_date_str,
        Next_Best_Action=recommendation.action,
        Priority=recommendation.priority,
        Reason=recommendation.reason,
        Top_SHAP_Reasons=shap_reasons,
        Model_Version=_model_version,
        Model_Source=_model_source,
    )


# ---------------------------------------------------------------------------
# Backward-compatible /predict alias
# ---------------------------------------------------------------------------
@app.post("/predict", include_in_schema=False)
def predict_alias(invoice: InvoiceScoreRequest):
    """Backward-compatible alias for /score-invoice."""
    return score_invoice(invoice)
