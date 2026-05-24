"""
feature_store.py — PostgreSQL Feature Store for Cash Flow Forecasting
=====================================================================
SAD V1 compliant. Implements:
    - Anti-leakage column dropping
    - Date-based feature engineering
    - Categorical label encoding (saved to disk for inference parity)
    - Point-in-time snapshot persistence to PostgreSQL

Author: MLOps Factory
PEP-8 compliant.
"""

import os
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

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

# These columns are derived from post-payment events and MUST be dropped to
# prevent data leakage (they would not be available at prediction time).
LEAKAGE_COLUMNS = [
    "Clearing_date",
    "Clearing_doc",
    "Delay_Bins",
    "DelayFlag",
    "Weekday_clearing",
    "Weekday_clearnum",
]

# Columns that will be label-encoded (their encoders are persisted to disk
# so the API can replicate the exact same encoding at inference time).
CATEGORICAL_COLUMNS = [
    "Payment_Method_description",
    "Region",
    "City",
    "Customer_Age_Year_Bins",
    "Amount_Bins",
    "Zipcode",
    "Customer_Name",
    "Payment_Term_Bins",
    "Weekday_due",
]

# Date columns requiring datetime parsing
DATE_COLUMNS = ["Doc_Date", "Posting_Date", "Net_Due_Date"]

# PostgreSQL table name for storing feature snapshots
SNAPSHOT_TABLE = "feature_snapshots"

# Path where fitted encoders are persisted for inference parity
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.joblib")


# ===========================================================================
# DATABASE CONNECTION
# ===========================================================================

def connect_db() -> "sqlalchemy.engine.Engine":
    """
    Create and return a SQLAlchemy engine connected to the PostgreSQL
    Feature Store using the DATABASE_URL environment variable.

    Returns:
        sqlalchemy.engine.Engine: Active database engine.

    Raises:
        EnvironmentError: If DATABASE_URL is not set.
        SQLAlchemyError: If the connection cannot be established.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise EnvironmentError(
            "DATABASE_URL environment variable is not set. "
            "Example: postgresql://user:pass@host:5432/dbname"
        )
    try:
        engine = create_engine(db_url, pool_pre_ping=True)
        # Test connectivity immediately
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("PostgreSQL connection established: %s", db_url.split("@")[-1])
        return engine
    except SQLAlchemyError as exc:
        logger.error("Failed to connect to PostgreSQL: %s", exc)
        raise


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """
    Load raw ERP invoice data from a CSV file.

    Args:
        path: Full path to the CSV file. Defaults to the DATA_PATH env var
              or '/app/data/Dataset.csv'.

    Returns:
        pd.DataFrame: Raw, unprocessed invoice dataframe.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    data_path = path or os.getenv("DATA_PATH", "/app/data/Dataset.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    logger.info(
        "Loaded raw data: %d rows × %d columns from '%s'",
        len(df), len(df.columns), data_path
    )
    return df


# ===========================================================================
# ANTI-LEAKAGE — DROP FUTURE-LOOKING COLUMNS
# ===========================================================================

def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all columns that would constitute data leakage — i.e., columns that
    are only available AFTER payment has been collected and therefore cannot
    be known at prediction time.

    Columns dropped (SAD V1 spec):
        - Clearing_date    → the actual payment date (future info)
        - Clearing_doc     → the clearing document number (future info)
        - Delay_Bins       → bucketed version of the target (leakage)
        - DelayFlag        → binary flag derived from the target (leakage)
        - Weekday_clearing → weekday of clearing date (future info)
        - Weekday_clearnum → numeric weekday of clearing (future info)

    Args:
        df: Raw dataframe.

    Returns:
        pd.DataFrame: Dataframe with leakage columns removed.
    """
    cols_present = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_present)
    logger.info("Dropped %d leakage columns: %s", len(cols_present), cols_present)
    return df


# ===========================================================================
# FEATURE ENGINEERING
# ===========================================================================

def engineer_features(df: pd.DataFrame, fit_encoders: bool = True) -> pd.DataFrame:
    """
    Apply all feature engineering transformations to the dataframe.

    Steps performed:
        1. Parse date columns to datetime.
        2. Create time-based features from Net_Due_Date.
        3. Compute days_to_pay_terms = Net_Due_Date - Doc_Date.
        4. Create Age_Of_Customer_Year from Age_Of_Customer_Months.
        5. Label-encode categorical columns (fitting or transforming only).
        6. Drop the original raw date columns.
        7. Drop rows with missing target (Days_Overdue_Delay).

    Args:
        df: Dataframe with leakage columns already removed.
        fit_encoders: If True, fit new LabelEncoders on the data and save them
                      to disk. Set to False at inference time (use saved encoders).

    Returns:
        pd.DataFrame: Fully engineered feature dataframe ready for ML training.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Parse date columns
    # ------------------------------------------------------------------
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            n_nulls = df[col].isna().sum()
            if n_nulls > 0:
                logger.warning("Column '%s': %d dates could not be parsed.", col, n_nulls)

    # ------------------------------------------------------------------
    # 2. Date-derived features
    # ------------------------------------------------------------------
    if "Net_Due_Date" in df.columns:
        df["due_month"] = df["Net_Due_Date"].dt.month
        df["due_dayofweek"] = df["Net_Due_Date"].dt.dayofweek  # Monday=0
        df["due_quarter"] = df["Net_Due_Date"].dt.quarter
        df["due_year"] = df["Net_Due_Date"].dt.year

    # ------------------------------------------------------------------
    # 3. days_to_pay_terms — the contractual payment window
    # ------------------------------------------------------------------
    if "Net_Due_Date" in df.columns and "Doc_Date" in df.columns:
        df["days_to_pay_terms"] = (df["Net_Due_Date"] - df["Doc_Date"]).dt.days
        logger.info(
            "Engineered 'days_to_pay_terms': mean=%.1f, min=%d, max=%d",
            df["days_to_pay_terms"].mean(),
            df["days_to_pay_terms"].min(),
            df["days_to_pay_terms"].max(),
        )

    # ------------------------------------------------------------------
    # 4. Customer age in years
    # ------------------------------------------------------------------
    if "Age_Of_Customer_Months" in df.columns:
        df["Age_Of_Customer_Year"] = df["Age_Of_Customer_Months"] / 12.0

    # ------------------------------------------------------------------
    # 5. Categorical label encoding
    # ------------------------------------------------------------------
    encoders = {}
    if not fit_encoders and os.path.exists(ENCODERS_PATH):
        # Inference mode: load pre-fitted encoders from disk
        encoders = joblib.load(ENCODERS_PATH)
        logger.info("Loaded %d encoders from '%s'", len(encoders), ENCODERS_PATH)

    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).fillna("__UNKNOWN__")

        if fit_encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            # Safe encoding: unknown categories map to index 0
            le = encoders.get(col)
            if le is not None:
                mask_known = df[col].isin(le.classes_)
                df.loc[mask_known, col] = le.transform(df.loc[mask_known, col])
                df.loc[~mask_known, col] = 0
                df[col] = df[col].astype(int)
            else:
                logger.warning("No encoder found for column '%s'. Setting to 0.", col)
                df[col] = 0

    # Persist newly fitted encoders
    if fit_encoders and encoders:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(encoders, ENCODERS_PATH)
        logger.info("Saved %d encoders to '%s'", len(encoders), ENCODERS_PATH)

    # ------------------------------------------------------------------
    # 6. Drop raw date columns (now replaced by numeric features)
    # ------------------------------------------------------------------
    df = df.drop(columns=[c for c in DATE_COLUMNS if c in df.columns])

    # ------------------------------------------------------------------
    # 7. Drop rows with no target variable
    # ------------------------------------------------------------------
    if "Days_Overdue_Delay" in df.columns:
        before = len(df)
        df = df.dropna(subset=["Days_Overdue_Delay"])
        dropped = before - len(df)
        if dropped > 0:
            logger.warning("Dropped %d rows with missing target 'Days_Overdue_Delay'.", dropped)

    logger.info(
        "Feature engineering complete: %d rows × %d columns",
        len(df), len(df.columns)
    )
    return df


# ===========================================================================
# POSTGRESQL FEATURE SNAPSHOT — WRITE
# ===========================================================================

def save_feature_snapshot(
    df: pd.DataFrame,
    engine: "sqlalchemy.engine.Engine",
    snapshot_label: str | None = None,
) -> None:
    """
    Persist the engineered feature dataframe as a versioned point-in-time
    snapshot to the PostgreSQL Feature Store.

    A 'snapshot_ts' timestamp column and a 'snapshot_label' column are added
    to uniquely identify each run. Rows are appended to the table, preserving
    full history for reproducible training.

    Args:
        df: Engineered feature dataframe (output of engineer_features).
        engine: Active SQLAlchemy engine connected to the feature store.
        snapshot_label: Optional string label (e.g. "2026-05-24"). Defaults
                        to ISO timestamp of now.

    Raises:
        SQLAlchemyError: If the write operation fails.
    """
    df = df.copy()
    df["snapshot_ts"] = datetime.utcnow()
    df["snapshot_label"] = snapshot_label or datetime.utcnow().isoformat()

    try:
        rows_written = df.to_sql(
            name=SNAPSHOT_TABLE,
            con=engine,
            if_exists="append",   # Append to preserve history
            index=False,
            method="multi",       # Batch inserts for performance
            chunksize=1000,
        )
        logger.info(
            "Wrote %d rows to PostgreSQL table '%s' (label='%s')",
            len(df), SNAPSHOT_TABLE, df["snapshot_label"].iloc[0]
        )
    except SQLAlchemyError as exc:
        logger.error("Failed to write feature snapshot to PostgreSQL: %s", exc)
        raise


# ===========================================================================
# POSTGRESQL FEATURE SNAPSHOT — READ
# ===========================================================================

def load_feature_snapshot(
    engine: "sqlalchemy.engine.Engine",
    latest_only: bool = True,
) -> pd.DataFrame:
    """
    Load feature snapshot(s) from the PostgreSQL Feature Store.

    Args:
        engine: Active SQLAlchemy engine.
        latest_only: If True (default), returns only rows from the most
                     recent snapshot_label. Set to False to return all history.

    Returns:
        pd.DataFrame: Feature snapshot ready for model training.

    Raises:
        ValueError: If the feature_snapshots table is empty or does not exist.
    """
    try:
        if latest_only:
            query = f"""
                SELECT * FROM {SNAPSHOT_TABLE}
                WHERE snapshot_label = (
                    SELECT MAX(snapshot_label) FROM {SNAPSHOT_TABLE}
                )
            """
        else:
            query = f"SELECT * FROM {SNAPSHOT_TABLE} ORDER BY snapshot_ts ASC"

        df = pd.read_sql(query, engine)

        if df.empty:
            raise ValueError(
                f"Feature snapshot table '{SNAPSHOT_TABLE}' is empty. "
                "Run the ingest pipeline first."
            )

        # Drop metadata columns added during snapshot save
        df = df.drop(columns=["snapshot_ts", "snapshot_label"], errors="ignore")

        logger.info(
            "Loaded %d rows × %d columns from feature snapshot (latest_only=%s)",
            len(df), len(df.columns), latest_only
        )
        return df

    except Exception as exc:
        logger.error("Failed to load feature snapshot: %s", exc)
        raise


# ===========================================================================
# CONVENIENCE WRAPPER — Full Ingest Pipeline
# ===========================================================================

def run_full_ingest(data_path: str | None = None) -> pd.DataFrame:
    """
    Execute the full feature store ingest pipeline in one call:
        1. Load raw CSV data.
        2. Drop leakage columns.
        3. Engineer features (fit new encoders).
        4. Save snapshot to PostgreSQL.

    Args:
        data_path: Optional path to raw CSV. Defaults to DATA_PATH env var.

    Returns:
        pd.DataFrame: The engineered feature dataframe that was stored.
    """
    logger.info("=== Feature Store Ingest Pipeline Starting ===")

    # Step 1: Load
    df_raw = load_raw_data(data_path)

    # Step 2: Anti-leakage
    df_clean = drop_leakage_columns(df_raw)

    # Step 3: Feature engineering (fits + saves encoders)
    df_features = engineer_features(df_clean, fit_encoders=True)

    # Step 4: Persist to PostgreSQL
    engine = connect_db()
    save_feature_snapshot(df_features, engine)

    logger.info("=== Feature Store Ingest Pipeline Complete ===")
    return df_features
