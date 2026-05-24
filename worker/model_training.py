"""
model_training.py — XGBoost Training with MLflow Tracking & SHAP Explainability
================================================================================
SAD V1 compliant. Implements:
    - Temporal train/validation split (no random leakage)
    - XGBRegressor with native NaN handling (no imputation)
    - MLflow experiment tracking: params, MAE, RMSE, SHAP artifact
    - Model registration to MLflow Model Registry as 'cashflow_xgb'
    - Automatic promotion to Production stage

Author: MLOps Factory
PEP-8 compliant.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error

matplotlib.use("Agg")   # Non-interactive backend for Docker/server environments

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COLUMN = "Days_Overdue_Delay"
DATE_SORT_COLUMN = "due_year"          # Used for temporal ordering after engineering
VALIDATION_MONTHS = 3                  # Most recent N months held out as validation
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "cashflow_forecasting"
REGISTERED_MODEL_NAME = "cashflow_xgb"

# XGBoost hyperparameters — tuned for tabular AR data
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "objective": "reg:squarederror",
    "tree_method": "hist",            # Efficient for large datasets
    "missing": np.nan,                # Native NaN handling — NO imputation needed
    "random_state": 42,
    "n_jobs": -1,
}


# ===========================================================================
# TEMPORAL SPLIT — No random splitting to prevent future data contamination
# ===========================================================================

def temporal_train_val_split(
    df: pd.DataFrame,
    validation_months: int = VALIDATION_MONTHS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a time-based train/validation split.

    Strategy:
        - Sort all rows chronologically by their 'due_month' + 'due_year' features.
        - Assign the most recent `validation_months` months to the validation set.
        - Everything older goes to the training set.

    This mirrors production conditions: the model is always trained on historical
    data and evaluated on the most recent, unseen invoices.

    Args:
        df: Engineered feature dataframe (must contain 'due_month' and 'due_year').
        validation_months: Number of most-recent months to hold out. Default: 3.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, val_df)

    Raises:
        ValueError: If date columns are missing or the split produces an empty set.
    """
    if "due_month" not in df.columns or "due_year" not in df.columns:
        raise ValueError(
            "Dataframe must have 'due_month' and 'due_year' columns for temporal split. "
            "Ensure feature_store.engineer_features() has been called."
        )

    # Reconstruct a sortable period column (YYYYMM integer)
    df = df.copy()
    df["_period"] = df["due_year"] * 100 + df["due_month"]

    # Determine the cutoff period
    max_period = df["_period"].max()
    max_year = max_period // 100
    max_month = max_period % 100

    # Roll back 'validation_months' months
    cutoff_date = datetime(max_year, max_month, 1) - timedelta(days=validation_months * 30)
    cutoff_period = cutoff_date.year * 100 + cutoff_date.month

    train_df = df[df["_period"] < cutoff_period].drop(columns=["_period"])
    val_df = df[df["_period"] >= cutoff_period].drop(columns=["_period"])

    if train_df.empty:
        raise ValueError(
            f"Training set is empty after temporal split at period {cutoff_period}. "
            "Try reducing validation_months."
        )
    if val_df.empty:
        raise ValueError(
            f"Validation set is empty. Max period in data: {max_period}. "
            "No data falls within the recent {validation_months} months."
        )

    logger.info(
        "Temporal split: train=%d rows (periods <%d) | val=%d rows (periods >=%d)",
        len(train_df), cutoff_period, len(val_df), cutoff_period
    )
    return train_df, val_df


# ===========================================================================
# SHAP EXPLAINABILITY — Generate and log SHAP summary plot
# ===========================================================================

def generate_shap_summary(
    model: xgb.XGBRegressor,
    X_val: pd.DataFrame,
    artifact_dir: str,
) -> dict:
    """
    Compute SHAP values for the validation set using TreeExplainer and
    produce a bar-chart summary plot saved to `artifact_dir`.

    Args:
        model: Fitted XGBRegressor model.
        X_val: Validation feature matrix (pd.DataFrame).
        artifact_dir: Directory path to save the SHAP summary PNG.

    Returns:
        dict: Mean absolute SHAP values per feature (for MLflow metric logging).
    """
    logger.info("Computing SHAP values on %d validation rows...", len(X_val))

    # Use a sample for speed if the validation set is large
    sample_size = min(len(X_val), 2000)
    X_sample = X_val.sample(n=sample_size, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP per feature — used as metrics in MLflow
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_sample.columns,
    ).sort_values(ascending=False)

    # Save summary bar plot as artifact
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    shap_plot_path = os.path.join(artifact_dir, "shap_summary.png")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(shap_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("SHAP summary plot saved to '%s'", shap_plot_path)
    return mean_abs_shap.to_dict(), shap_plot_path


# ===========================================================================
# CORE TRAINING FUNCTION
# ===========================================================================

def train_and_register_model(df: pd.DataFrame) -> Tuple[float, float, str]:
    """
    Full training pipeline:
        1. Temporal train/val split.
        2. Separate features (X) from target (y).
        3. Train XGBRegressor with native NaN handling.
        4. Evaluate: compute MAE and RMSE on validation set.
        5. Generate SHAP summary and plot.
        6. Log everything to MLflow (params, metrics, model, SHAP artifact).
        7. Register model and promote to Production.

    Args:
        df: Engineered feature dataframe from the Feature Store.

    Returns:
        Tuple[float, float, str]: (val_mae, val_rmse, mlflow_run_id)
    """
    # ------------------------------------------------------------------
    # Configure MLflow
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(
        "MLflow tracking URI: %s | Experiment: %s",
        MLFLOW_TRACKING_URI, EXPERIMENT_NAME
    )

    # ------------------------------------------------------------------
    # Temporal split
    # ------------------------------------------------------------------
    train_df, val_df = temporal_train_val_split(df, VALIDATION_MONTHS)

    feature_cols = [c for c in train_df.columns if c != TARGET_COLUMN]
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COLUMN]

    logger.info(
        "Feature matrix: %d features | Train: %d rows | Val: %d rows",
        len(feature_cols), len(X_train), len(X_val)
    )

    # ------------------------------------------------------------------
    # MLflow Run
    # ------------------------------------------------------------------
    with mlflow.start_run(run_name=f"cashflow_xgb_{datetime.utcnow():%Y%m%d_%H%M%S}") as run:
        run_id = run.info.run_id
        logger.info("MLflow run started: run_id=%s", run_id)

        # Log dataset metadata
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("validation_months", VALIDATION_MONTHS)
        mlflow.log_param("split_strategy", "temporal_by_due_date")

        # Log XGBoost hyperparameters
        mlflow.log_params(XGBOOST_PARAMS)

        # --------------------------------------------------------------
        # Train XGBRegressor
        # Note: missing=np.nan instructs XGBoost to handle NaNs natively
        # (uses the default 'go left' direction during tree building).
        # NO mean/zero imputation is performed.
        # --------------------------------------------------------------
        logger.info("Training XGBRegressor (%d estimators)...", XGBOOST_PARAMS["n_estimators"])
        model = xgb.XGBRegressor(**XGBOOST_PARAMS)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        logger.info("Training complete.")

        # --------------------------------------------------------------
        # Validation metrics
        # --------------------------------------------------------------
        val_preds = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_preds)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))

        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_rmse", val_rmse)
        logger.info("Validation — MAE: %.4f days | RMSE: %.4f days", val_mae, val_rmse)

        # --------------------------------------------------------------
        # SHAP Explainability
        # --------------------------------------------------------------
        tmp_artifact_dir = os.path.join(MODEL_DIR, "tmp_artifacts", run_id)
        shap_metrics, shap_plot_path = generate_shap_summary(model, X_val, tmp_artifact_dir)

        # Log top-10 SHAP importance scores as MLflow metrics
        for feat, importance in list(shap_metrics.items())[:10]:
            safe_key = f"shap_{feat[:40].replace(' ', '_')}"  # MLflow key limit
            mlflow.log_metric(safe_key, round(importance, 6))

        # Log SHAP plot as an artifact
        mlflow.log_artifact(shap_plot_path, artifact_path="shap")

        # Store feature names as a tag for the API to reference
        mlflow.set_tag("feature_columns", ",".join(feature_cols))
        mlflow.set_tag("target_column", TARGET_COLUMN)

        # --------------------------------------------------------------
        # Log the trained model to the MLflow Model Registry
        # --------------------------------------------------------------
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_val.head(1),
        )
        logger.info(
            "Model logged to MLflow registry as '%s'", REGISTERED_MODEL_NAME
        )

        # Also save a local fallback copy for environments without MLflow
        os.makedirs(MODEL_DIR, exist_ok=True)
        fallback_path = os.path.join(MODEL_DIR, "latest_xgboost_model.json")
        model.save_model(fallback_path)
        logger.info("Fallback model saved locally to '%s'", fallback_path)

    # ------------------------------------------------------------------
    # Promote the newly registered model version to Production
    # ------------------------------------------------------------------
    _promote_model_to_production(run_id)

    return val_mae, val_rmse, run_id


# ===========================================================================
# MODEL PROMOTION
# ===========================================================================

def _promote_model_to_production(run_id: str) -> None:
    """
    Find the MLflow model version associated with `run_id` and transition
    it to the 'Production' stage, archiving any previous Production version.

    Args:
        run_id: MLflow run ID of the training run that logged the model.
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    try:
        # Retrieve all versions of the registered model
        versions = client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}'"
        )

        # Find the version matching this run
        target_version = None
        for v in versions:
            if v.run_id == run_id:
                target_version = v
                break

        if target_version is None:
            logger.warning(
                "Could not find model version for run_id=%s in registry '%s'. "
                "Skipping promotion.",
                run_id, REGISTERED_MODEL_NAME
            )
            return

        # Archive all current Production versions
        for v in versions:
            if v.current_stage == "Production":
                client.transition_model_version_stage(
                    name=REGISTERED_MODEL_NAME,
                    version=v.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )
                logger.info(
                    "Archived previous Production model: version=%s", v.version
                )

        # Promote new version
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=target_version.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(
            "Model '%s' version %s promoted to Production (run_id=%s)",
            REGISTERED_MODEL_NAME, target_version.version, run_id
        )

    except Exception as exc:
        # Non-fatal: if MLflow registry is temporarily unavailable, log the
        # error and continue — the model was still saved locally.
        logger.error("Failed to promote model to Production: %s", exc)
