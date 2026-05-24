"""
flows.py — Prefect Nightly MLOps Orchestration
===============================================
SAD V1 compliant. Implements a Prefect @flow that runs nightly:
    1. Ingest raw ERP data → engineer features → save to PostgreSQL Feature Store.
    2. Load the latest snapshot and run baseline evaluation on current model.
    3. Check for performance drift (recent MAE > baseline + 20%).
    4. If drift detected → train new XGBoost model with MLflow tracking.
    5. Promote new model to Production in the MLflow registry.
    6. Notify the FastAPI service to hot-swap the active model.

Schedule: Every night at 02:00 UTC (configurable via PREFECT_CRON env var).

Author: MLOps Factory
PEP-8 compliant.
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from mlflow.tracking import MlflowClient
from prefect import flow, task, get_run_logger
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

import feature_store
import model_training

# ---------------------------------------------------------------------------
# Module-level logging (Prefect tasks use get_run_logger() internally)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_URL", "http://api:8000")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REGISTERED_MODEL_NAME = model_training.REGISTERED_MODEL_NAME

# Drift threshold: retrain if recent MAE exceeds baseline by this factor
DRIFT_THRESHOLD_FACTOR = 1.20   # 20% degradation triggers retraining

# Baseline MAE: the expected MAE from the last good production model.
# This value is updated after every successful training run.
BASELINE_MAE_ENV = "BASELINE_MODEL_MAE"


# ===========================================================================
# TASK 1 — Feature Ingestion
# ===========================================================================

@task(
    name="ingest-features",
    retries=2,
    retry_delay_seconds=30,
    description="Load raw ERP data, engineer features, and save to PostgreSQL.",
)
def ingest_features(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Full feature store ingest pipeline:
        1. Load raw CSV from disk.
        2. Drop leakage columns (SAD V1 anti-leakage protocol).
        3. Engineer all features (dates, bins, label encoding).
        4. Persist a point-in-time snapshot to PostgreSQL.

    Returns:
        pd.DataFrame: The engineered feature dataframe.
    """
    run_logger = get_run_logger()
    run_logger.info("Task: ingest-features started.")

    df = feature_store.run_full_ingest(data_path)

    run_logger.info(
        "Feature ingest complete: %d rows × %d columns saved to PostgreSQL.",
        len(df), len(df.columns)
    )
    return df


# ===========================================================================
# TASK 2 — Load Feature Snapshot
# ===========================================================================

@task(
    name="load-snapshot",
    description="Load the latest feature snapshot from the PostgreSQL Feature Store.",
)
def load_snapshot() -> pd.DataFrame:
    """
    Load the most recent feature snapshot from PostgreSQL for training and
    drift evaluation.

    Returns:
        pd.DataFrame: Latest engineered feature snapshot.
    """
    run_logger = get_run_logger()
    run_logger.info("Task: load-snapshot started.")

    engine = feature_store.connect_db()
    df = feature_store.load_feature_snapshot(engine, latest_only=True)

    run_logger.info("Loaded %d rows from feature snapshot.", len(df))
    return df


# ===========================================================================
# TASK 3 — Drift Detection
# ===========================================================================

@task(
    name="check-drift",
    description="Evaluate current model performance and detect degradation.",
)
def check_drift(df: pd.DataFrame) -> Tuple[bool, float]:
    """
    Evaluate the current Production model against the latest feature snapshot
    and check whether performance has degraded beyond the drift threshold.

    Strategy:
        - Load the Production model from MLflow (or fallback file).
        - Score a held-out sample from the latest snapshot.
        - Compare MAE to the stored baseline MAE.
        - If current_mae > baseline_mae * DRIFT_THRESHOLD_FACTOR → drift detected.

    Args:
        df: Latest engineered feature dataframe.

    Returns:
        Tuple[bool, float]: (drift_detected: bool, current_mae: float)
    """
    run_logger = get_run_logger()
    run_logger.info("Task: check-drift started.")

    target_col = model_training.TARGET_COLUMN
    if target_col not in df.columns:
        run_logger.warning(
            "Target column '%s' not found. Skipping drift check. Forcing retrain.",
            target_col
        )
        return True, float("inf")

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    # Sample up to 5000 rows for efficiency
    sample_idx = np.random.choice(len(df), size=min(len(df), 5000), replace=False)
    X_sample = X.iloc[sample_idx]
    y_sample = y.iloc[sample_idx]

    # Try to load Production model from MLflow
    model = None
    try:
        import mlflow.xgboost
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/Production"
        model = mlflow.xgboost.load_model(model_uri)
        run_logger.info("Loaded Production model from MLflow for drift evaluation.")
    except Exception as exc:
        run_logger.warning(
            "Could not load MLflow model (%s). Trying local fallback...", exc
        )
        model_dir = os.getenv("MODEL_DIR", "/app/models")
        fallback_path = os.path.join(model_dir, "latest_xgboost_model.json")
        if os.path.exists(fallback_path):
            booster = xgb.Booster()
            booster.load_model(fallback_path)
            # Wrap Booster in DMatrix for prediction
            dmatrix = xgb.DMatrix(X_sample)
            preds = booster.predict(dmatrix)
            current_mae = float(np.mean(np.abs(y_sample - preds)))
        else:
            run_logger.warning("No model available for drift check. Forcing retrain.")
            return True, float("inf")

    if model is not None:
        preds = model.predict(X_sample)
        current_mae = float(np.mean(np.abs(y_sample - preds)))

    # Retrieve baseline MAE from environment
    baseline_mae = float(os.getenv(BASELINE_MAE_ENV, "999"))
    drift_ratio = current_mae / max(baseline_mae, 0.001)

    run_logger.info(
        "Drift check — current MAE: %.4f | baseline MAE: %.4f | ratio: %.2f (threshold: %.2f)",
        current_mae, baseline_mae, drift_ratio, DRIFT_THRESHOLD_FACTOR
    )

    drift_detected = drift_ratio > DRIFT_THRESHOLD_FACTOR
    if drift_detected:
        run_logger.warning(
            "DRIFT DETECTED: MAE degraded by %.1f%% (threshold: %.1f%%)",
            (drift_ratio - 1) * 100,
            (DRIFT_THRESHOLD_FACTOR - 1) * 100,
        )
    else:
        run_logger.info("No drift detected. Model performance is within acceptable range.")

    return drift_detected, current_mae


# ===========================================================================
# TASK 4 — Train New Model
# ===========================================================================

@task(
    name="train-model",
    retries=1,
    retry_delay_seconds=60,
    description="Train a new XGBoost model with MLflow tracking.",
)
def train_model(df: pd.DataFrame) -> Tuple[float, float, str]:
    """
    Trigger a full model training run via model_training.py.

    Args:
        df: Engineered feature dataframe from the Feature Store.

    Returns:
        Tuple[float, float, str]: (val_mae, val_rmse, run_id)
    """
    run_logger = get_run_logger()
    run_logger.info("Task: train-model started. Rows: %d", len(df))

    val_mae, val_rmse, run_id = model_training.train_and_register_model(df)

    run_logger.info(
        "Training complete — MAE: %.4f | RMSE: %.4f | run_id: %s",
        val_mae, val_rmse, run_id
    )
    return val_mae, val_rmse, run_id


# ===========================================================================
# TASK 5 — Notify API (Hot-swap Model)
# ===========================================================================

@task(
    name="notify-api",
    retries=3,
    retry_delay_seconds=10,
    description="Notify the FastAPI service to reload the Production model.",
)
def notify_api() -> bool:
    """
    Hit the FastAPI /reload-model endpoint so the serving layer immediately
    loads the newly promoted Production model from the MLflow registry.

    Returns:
        bool: True if the API acknowledged the reload, False otherwise.
    """
    run_logger = get_run_logger()
    endpoint = f"{API_BASE_URL}/reload-model"
    run_logger.info("Notifying API at %s...", endpoint)

    try:
        response = requests.post(endpoint, timeout=30)
        if response.status_code == 200:
            run_logger.info("API model reload confirmed: %s", response.json())
            return True
        else:
            run_logger.warning(
                "API returned unexpected status %d: %s",
                response.status_code, response.text
            )
            return False
    except requests.RequestException as exc:
        run_logger.error("Failed to notify API: %s", exc)
        return False


# ===========================================================================
# MAIN FLOW — Nightly MLOps Pipeline
# ===========================================================================

@flow(
    name="nightly-cashflow-pipeline",
    description=(
        "SAD V1 nightly MLOps pipeline. "
        "Ingests data → checks drift → retrains if needed → promotes model → notifies API."
    ),
)
def nightly_cashflow_pipeline(
    data_path: Optional[str] = None,
    force_retrain: bool = False,
) -> dict:
    """
    Nightly orchestration flow. Runs all five tasks in sequence.

    Args:
        data_path: Optional override for the raw CSV path.
        force_retrain: If True, skip drift check and always retrain.

    Returns:
        dict: Summary of the pipeline run with key metrics.
    """
    run_logger = get_run_logger()
    run_logger.info("=== Nightly Cash Flow MLOps Pipeline Starting ===")

    # ------------------------------------------------------------------
    # Task 1: Ingest features into PostgreSQL
    # ------------------------------------------------------------------
    ingested_df = ingest_features(data_path)

    # ------------------------------------------------------------------
    # Task 2: Load the snapshot back for evaluation
    # ------------------------------------------------------------------
    snapshot_df = load_snapshot()

    # ------------------------------------------------------------------
    # Task 3: Check for performance drift
    # ------------------------------------------------------------------
    retrain_needed = force_retrain
    current_mae = None

    if not force_retrain:
        drift_detected, current_mae = check_drift(snapshot_df)
        retrain_needed = drift_detected

    # ------------------------------------------------------------------
    # Task 4: Conditionally retrain
    # ------------------------------------------------------------------
    run_id = None
    new_mae = None
    new_rmse = None

    if retrain_needed:
        run_logger.info("Retraining triggered (force=%s, drift=%s).", force_retrain, retrain_needed)
        new_mae, new_rmse, run_id = train_model(snapshot_df)

        # ------------------------------------------------------------------
        # Task 5: Notify the API to hot-swap to the new Production model
        # ------------------------------------------------------------------
        notify_api()

        run_logger.info(
            "Pipeline complete — new model promoted. MAE: %.4f | RMSE: %.4f | run_id: %s",
            new_mae, new_rmse, run_id
        )
    else:
        run_logger.info(
            "Pipeline complete — no retraining needed. Current MAE: %.4f",
            current_mae
        )

    run_logger.info("=== Nightly Cash Flow MLOps Pipeline Finished ===")

    return {
        "retrained": retrain_needed,
        "current_mae": current_mae,
        "new_mae": new_mae,
        "new_rmse": new_rmse,
        "run_id": run_id,
    }


# ===========================================================================
# ENTRY POINT — Deploy or run immediately
# ===========================================================================

if __name__ == "__main__":
    import sys

    if "--serve" in sys.argv:
        # Deploy with nightly cron schedule (02:00 UTC)
        cron_expr = os.getenv("PREFECT_CRON", "0 2 * * *")
        print(f"Deploying nightly pipeline with schedule: '{cron_expr}'")
        nightly_cashflow_pipeline.serve(
            name="nightly-cashflow-pipeline",
            cron=cron_expr,
        )
    else:
        # Run immediately (used for first-run bootstrapping or manual trigger)
        print("Running pipeline immediately (one-shot)...")
        result = nightly_cashflow_pipeline(force_retrain=True)
        print(f"Pipeline result: {result}")
