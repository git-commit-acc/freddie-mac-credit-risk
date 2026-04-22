"""pipeline_spark.py - Master PySpark pipeline for Freddie Mac Credit Risk"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def get_spark_session():
    return SparkSession.builder \
        .appName("Freddie_Mac_Credit_Risk") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "2g") \
        .getOrCreate()


def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{int(time.time())}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)]
    )
    return log_file


logger = logging.getLogger(__name__)


# def stage_ingest(spark, config, year_start: int, year_end: int) -> dict:
#     from ingestion.spark_loader import (
#         load_origination, load_servicing, clean_origination, clean_servicing, save_dataframe
#     )

#     orig_path = os.path.join(config.paths.parquet_dir, "origination")
#     svcg_path = os.path.join(config.paths.parquet_dir, "servicing")

#     # Check if already exists
#     if os.path.exists(orig_path) and os.path.exists(svcg_path):
#         logger.info("Parquet cache found, skipping ingestion.")
#         return {"orig_path": orig_path, "svcg_path": svcg_path, "cached": True}

#     logger.info("Loading origination for %d-%d...", year_start, year_end)
#     orig_raw = load_origination(spark, config.paths.raw_dir, year_start, year_end)
#     orig_clean = clean_origination(orig_raw)
#     save_dataframe(orig_clean, orig_path)
#     logger.info("Origination saved: %d rows", orig_clean.count())

#     logger.info("Loading servicing for %d-%d...", year_start, year_end)
#     svcg_raw = load_servicing(spark, config.paths.raw_dir, year_start, year_end)
#     svcg_clean = clean_servicing(svcg_raw)
#     save_dataframe(svcg_clean, svcg_path)
#     logger.info("Servicing saved: %d rows", svcg_clean.count())

#     return {"orig_path": orig_path, "svcg_path": svcg_path, "n_orig": orig_clean.count(), "n_svcg": svcg_clean.count()}


# def stage_targets(spark, config) -> dict:
#     from ingestion.spark_loader import load_dataframe
#     from features.spark_targets import add_event_flags, add_rolling_delinquency_features, build_target_12m, select_snapshot

#     snapshot_path = os.path.join(config.paths.feature_dir, "snapshot")

#     if os.path.exists(snapshot_path):
#         logger.info("Snapshot found, skipping target construction.")
#         return {"snapshot_path": snapshot_path, "cached": True}

#     svcg_path = os.path.join(config.paths.parquet_dir, "servicing")
#     if not os.path.exists(svcg_path):
#         raise RuntimeError("Servicing not found. Run 'ingest' first.")

#     logger.info("Loading servicing data...")
#     svcg_df = spark.read.parquet(svcg_path)

#     logger.info("Adding event flags...")
#     svcg_df = add_event_flags(svcg_df)

#     logger.info("Computing rolling features...")
#     svcg_df = add_rolling_delinquency_features(svcg_df, windows=config.features.rolling_windows)

#     logger.info("Building targets...")
#     svcg_df = build_target_12m(svcg_df, window_months=config.target.window_months)

#     logger.info("Selecting snapshot...")
#     snapshot_df = select_snapshot(svcg_df, min_obs_months=config.target.min_obs_months)

#     snapshot_df.write.mode("overwrite").parquet(snapshot_path)

#     n_total = snapshot_df.count()
#     n_pos = snapshot_df.filter(col("TARGET_12M") == 1).count()

#     return {"snapshot_path": snapshot_path, "n_loans": n_total, "n_defaults": n_pos, "default_rate_pct": round(100.0 * n_pos / max(n_total, 1), 2)}


# def stage_features(spark, config) -> dict:
#     from ingestion.spark_loader import load_dataframe
#     from features.spark_engineering import (
#         add_origination_features, add_derived_origination_features,
#         add_vintage_features, add_missing_indicators, add_dynamic_features, impute_features
#     )

#     modeling_path = os.path.join(config.paths.feature_dir, "modeling")

#     if os.path.exists(modeling_path):
#         logger.info("Modeling data found, skipping feature engineering.")
#         return {"modeling_path": modeling_path, "cached": True}

#     orig_path = os.path.join(config.paths.parquet_dir, "origination")
#     snapshot_path = os.path.join(config.paths.feature_dir, "snapshot")

#     if not os.path.exists(orig_path) or not os.path.exists(snapshot_path):
#         raise RuntimeError("Missing origination or snapshot. Run previous stages.")

#     logger.info("Loading data...")
#     orig_df = spark.read.parquet(orig_path)
#     snapshot_df = spark.read.parquet(snapshot_path)

#     logger.info("Adding origination features...")
#     orig_df = add_origination_features(orig_df)
#     orig_df = add_derived_origination_features(orig_df)
#     orig_df = add_vintage_features(orig_df)
#     orig_df = add_missing_indicators(orig_df)

#     logger.info("Joining snapshot with origination...")
#     modeling_df = snapshot_df.join(
#         orig_df.select("LOAN_SEQUENCE_NUMBER", *[c for c in orig_df.columns if c != "LOAN_SEQUENCE_NUMBER"]),
#         on="LOAN_SEQUENCE_NUMBER",
#         how="inner"
#     )

#     logger.info("Adding dynamic features...")
#     modeling_df = add_dynamic_features(modeling_df)

#     logger.info("Imputing missing values...")
#     modeling_df = impute_features(modeling_df)

#     modeling_df.write.mode("overwrite").parquet(modeling_path)
#     logger.info("Modeling data saved: %d rows, %d cols", modeling_df.count(), len(modeling_df.columns))

#     return {"modeling_path": modeling_path, "n_rows": modeling_df.count()}


# def stage_split(spark, config) -> dict:
#     from features.spark_splitting import create_splits
#     from ingestion.spark_loader import save_dataframe

#     split_dir = config.paths.split_dir
#     oos_train_path = os.path.join(split_dir, "oos_train")

#     if os.path.exists(oos_train_path):
#         logger.info("Split parquets found, skipping split stage.")
#         return {"split_dir": split_dir, "cached": True}

#     modeling_path = os.path.join(config.paths.feature_dir, "modeling")
#     if not os.path.exists(modeling_path):
#         raise RuntimeError("Modeling data not found. Run 'features' first.")

#     modeling_df = spark.read.parquet(modeling_path)
#     splits = create_splits(modeling_df, config)

#     os.makedirs(split_dir, exist_ok=True)

#     for split_name, (train, val, test) in splits.items():
#         if train is not None:
#             train.write.mode("overwrite").parquet(os.path.join(split_dir, f"{split_name}_train"))
#         if val is not None:
#             val.write.mode("overwrite").parquet(os.path.join(split_dir, f"{split_name}_val"))
#         if test is not None:
#             test.write.mode("overwrite").parquet(os.path.join(split_dir, f"{split_name}_test"))

#     return {"split_dir": split_dir}
def stage_ingest(spark, config, year_start: int, year_end: int) -> dict:
    from ingestion.spark_loader import (
        load_origination, load_servicing, clean_origination, clean_servicing, save_dataframe
    )

    orig_path = os.path.join(config.paths.parquet_dir, "origination")
    svcg_path = os.path.join(config.paths.parquet_dir, "servicing")

    # Check if already exists
    if os.path.exists(orig_path) and os.path.exists(svcg_path):
        logger.info("Parquet cache found, skipping ingestion.")
        return {"orig_path": orig_path, "svcg_path": svcg_path, "cached": True}

    logger.info("Loading origination for %d-%d...", year_start, year_end)
    orig_raw = load_origination(spark, config.paths.raw_dir, year_start, year_end)
    orig_clean = clean_origination(orig_raw)
    save_dataframe(orig_clean, orig_path)
    logger.info("Origination saved: %d rows", orig_clean.count())

    logger.info("Loading servicing for %d-%d...", year_start, year_end)
    svcg_raw = load_servicing(spark, config.paths.raw_dir, year_start, year_end)
    svcg_clean = clean_servicing(svcg_raw)
    save_dataframe(svcg_clean, svcg_path)
    logger.info("Servicing saved: %d rows", svcg_clean.count())

    return {"orig_path": orig_path, "svcg_path": svcg_path, "n_orig": orig_clean.count(), "n_svcg": svcg_clean.count()}

def stage_targets(spark, config) -> dict:
    from features.spark_targets import (
        add_event_flags, add_rolling_delinquency_features, 
        build_target_12m, select_snapshot
    )
    from ingestion.spark_loader import load_dataframe, save_dataframe, to_spark_path

    snapshot_path = os.path.join(config.paths.feature_dir, "snapshot")

    if os.path.exists(snapshot_path):
        logger.info("Snapshot found, skipping target construction.")
        return {"snapshot_path": snapshot_path, "cached": True}

    svcg_path = os.path.join(config.paths.parquet_dir, "servicing")
    if not os.path.exists(svcg_path):
        raise RuntimeError(f"Servicing not found at {svcg_path}. Run 'ingest' first.")

    logger.info("Loading servicing data from %s...", svcg_path)
    
    # Load using Spark with proper path
    spark_path = to_spark_path(svcg_path)
    svcg_df = spark.read.parquet(spark_path)
    
    logger.info("Servicing data loaded: %d rows", svcg_df.count())

    logger.info("Adding event flags...")
    svcg_df = add_event_flags(svcg_df)

    logger.info("Computing rolling features...")
    svcg_df = add_rolling_delinquency_features(svcg_df, windows=config.features.rolling_windows)

    logger.info("Building targets...")
    svcg_df = build_target_12m(svcg_df, window_months=config.target.window_months)

    logger.info("Selecting snapshot...")
    snapshot_df = select_snapshot(svcg_df, min_obs_months=config.target.min_obs_months)

    # Save snapshot
    save_dataframe(snapshot_df, snapshot_path)

    n_total = snapshot_df.count()
    n_pos = snapshot_df.filter(col("TARGET_12M") == 1).count()
    n_neg = n_total - n_pos
    
    logger.info("Snapshot created: %d loans, %d defaults (%.2f%%)", 
                n_total, n_pos, 100.0 * n_pos / max(n_total, 1))

    return {
        "snapshot_path": snapshot_path, 
        "n_loans": n_total, 
        "n_defaults": n_pos,
        "n_non_defaults": n_neg,
        "default_rate_pct": round(100.0 * n_pos / max(n_total, 1), 2)
    }
# def stage_features(spark, config) -> dict:
#     from features.spark_engineering import (
#         add_origination_features, add_derived_origination_features,
#         add_vintage_features, add_missing_indicators, add_dynamic_features, impute_features
#     )
#     from ingestion.spark_loader import to_spark_path, save_dataframe

#     modeling_path = os.path.join(config.paths.feature_dir, "modeling")

#     if os.path.exists(modeling_path):
#         logger.info("Modeling data found, skipping feature engineering.")
#         return {"modeling_path": modeling_path, "cached": True}

#     orig_path = os.path.join(config.paths.parquet_dir, "origination")
#     snapshot_path = os.path.join(config.paths.feature_dir, "snapshot")

#     if not os.path.exists(orig_path):
#         raise RuntimeError(f"Origination not found at {orig_path}. Run 'ingest' first.")
#     if not os.path.exists(snapshot_path):
#         raise RuntimeError(f"Snapshot not found at {snapshot_path}. Run 'targets' first.")

#     logger.info("Loading origination data...")
#     orig_path_spark = to_spark_path(orig_path)
#     orig_df = spark.read.parquet(orig_path_spark)
#     logger.info("Origination loaded: %d rows", orig_df.count())

#     logger.info("Loading snapshot data...")
#     snapshot_path_spark = to_spark_path(snapshot_path)
#     snapshot_df = spark.read.parquet(snapshot_path_spark)
#     logger.info("Snapshot loaded: %d rows", snapshot_df.count())

#     logger.info("Adding origination features...")
#     orig_df = add_origination_features(orig_df)
#     orig_df = add_derived_origination_features(orig_df)
#     orig_df = add_vintage_features(orig_df)
#     orig_df = add_missing_indicators(orig_df)

#     logger.info("Joining snapshot with origination...")
#     # Select columns to join (exclude LOAN_SEQUENCE_NUMBER from orig_df to avoid duplicate)
#     join_cols = [c for c in orig_df.columns if c != "LOAN_SEQUENCE_NUMBER"]
#     modeling_df = snapshot_df.join(
#         orig_df.select("LOAN_SEQUENCE_NUMBER", *join_cols),
#         on="LOAN_SEQUENCE_NUMBER",
#         how="inner"
#     )

#     logger.info("Adding dynamic features...")
#     modeling_df = add_dynamic_features(modeling_df)

#     logger.info("Imputing missing values...")
#     modeling_df = impute_features(modeling_df)

#     logger.info("Saving modeling data...")
#     save_dataframe(modeling_df, modeling_path)
    
#     logger.info("Modeling data saved: %d rows, %d cols", modeling_df.count(), len(modeling_df.columns))

#     return {"modeling_path": modeling_path, "n_rows": modeling_df.count()}

def stage_features(spark, config) -> dict:
    """Simplified feature engineering - convert to pandas for joins"""
    import pandas as pd
    
    modeling_path = os.path.join(config.paths.feature_dir, "modeling")

    if os.path.exists(modeling_path):
        logger.info("Modeling data found, skipping feature engineering.")
        return {"modeling_path": modeling_path, "cached": True}

    orig_path = os.path.join(config.paths.parquet_dir, "origination")
    snapshot_path = os.path.join(config.paths.feature_dir, "snapshot")

    if not os.path.exists(orig_path):
        raise RuntimeError(f"Origination not found at {orig_path}. Run 'ingest' first.")
    if not os.path.exists(snapshot_path):
        raise RuntimeError(f"Snapshot not found at {snapshot_path}. Run 'targets' first.")

    logger.info("Loading origination data as Pandas...")
    # Read parquet as pandas (easier for joins)
    orig_df = pd.read_parquet(orig_path)
    logger.info("Origination loaded: %d rows", len(orig_df))

    logger.info("Loading snapshot data as Pandas...")
    snapshot_df = pd.read_parquet(snapshot_path)
    logger.info("Snapshot loaded: %d rows", len(snapshot_df))

    logger.info("Adding origination features...")
    # Add features using pandas (simpler)
    from features.spark_engineering import add_origination_features_pandas, add_vintage_features_pandas
    
    orig_df = add_origination_features_pandas(orig_df)
    orig_df = add_vintage_features_pandas(orig_df)
    
    # Add missing indicators
    for col_name in ["CREDIT_SCORE", "ORIG_DTI", "ORIG_LTV", "ORIG_CLTV", "MI_PCT"]:
        if col_name in orig_df.columns:
            orig_df[f"{col_name}_MISSING"] = orig_df[col_name].isna().astype(int)

    logger.info("Joining snapshot with origination...")
    modeling_df = snapshot_df.merge(
        orig_df[["LOAN_SEQUENCE_NUMBER"] + [c for c in orig_df.columns if c != "LOAN_SEQUENCE_NUMBER"]],
        on="LOAN_SEQUENCE_NUMBER",
        how="inner"
    )
    logger.info("After join: %d rows", len(modeling_df))

    logger.info("Adding dynamic features...")
    # Add dynamic features
    modeling_df["loan_age_months"] = modeling_df["LOAN_AGE"].astype(float)
    modeling_df["loan_age_ratio"] = modeling_df["LOAN_AGE"] / modeling_df["ORIG_LOAN_TERM"].replace(0, 1)
    modeling_df["pct_life_remaining"] = modeling_df["REM_MONTHS_MATURITY"] / modeling_df["ORIG_LOAN_TERM"].replace(0, 1)
    modeling_df["upb_pct_remaining"] = modeling_df["CURR_ACTUAL_UPB"] / modeling_df["ORIG_UPB"].replace(0, 1)
    modeling_df["rate_spread"] = modeling_df["CURR_INTEREST_RATE"] - modeling_df["ORIG_INTEREST_RATE"]
    modeling_df["was_modified"] = modeling_df.get("MODIFIED", 0).fillna(0).astype(int)

    # Create age buckets
    modeling_df["loan_age_bucket"] = pd.cut(
        modeling_df["LOAN_AGE"],
        bins=[0, 12, 24, 36, 60, 120, 10000],
        labels=["0_12m", "12_24m", "24_36m", "36_60m", "60_120m", "120m_plus"]
    ).astype(str)
    modeling_df["loan_age_bucket"] = modeling_df["loan_age_bucket"].fillna("unknown")

    logger.info("Imputing missing values...")
    # Fill numeric with -1
    for col_name in config.features.numeric_features:
        if col_name in modeling_df.columns:
            modeling_df[col_name] = modeling_df[col_name].fillna(-1)
    
    # Fill categorical with UNKNOWN
    for col_name in config.features.categorical_features:
        if col_name in modeling_df.columns:
            modeling_df[col_name] = modeling_df[col_name].fillna("UNKNOWN").astype(str)

    logger.info("Saving modeling data...")
    os.makedirs(modeling_path, exist_ok=True)
    modeling_df.to_parquet(modeling_path, index=False)
    
    logger.info("Modeling data saved: %d rows, %d cols", len(modeling_df), len(modeling_df.columns))

    return {"modeling_path": modeling_path, "n_rows": len(modeling_df)}


def add_origination_features_pandas(df):
    """Pandas version of origination features"""
    # Credit score buckets
    df["credit_score_bucket"] = pd.cut(
        df["CREDIT_SCORE"],
        bins=[0, 580, 620, 660, 720, 760, 900],
        labels=["deep_subprime", "subprime", "near_prime", "prime", "prime_plus", "super_prime"],
        right=False
    ).astype(str)
    df["credit_score_bucket"] = df["credit_score_bucket"].fillna("unknown")

    # LTV buckets
    df["ltv_bucket"] = pd.cut(
        df["ORIG_LTV"],
        bins=[0, 60, 70, 80, 90, 97, 999],
        labels=["lte60", "60_70", "70_80", "80_90", "90_97", "gt97"],
        right=True
    ).astype(str)
    df["ltv_bucket"] = df["ltv_bucket"].fillna("unknown")

    df["is_high_ltv"] = (df["ORIG_LTV"] > 80).astype(float)
    df["is_high_ltv"] = df["is_high_ltv"].where(df["ORIG_LTV"].notna(), np.nan)

    # DTI buckets
    df["dti_bucket"] = pd.cut(
        df["ORIG_DTI"],
        bins=[0, 28, 36, 43, 50, 100],
        labels=["conservative", "moderate", "standard", "elevated", "high"],
        right=True
    ).astype(str)
    df["dti_bucket"] = df["dti_bucket"].fillna("unknown")

    df["is_high_dti"] = (df["ORIG_DTI"] > 43).astype(float)
    df["is_high_dti"] = df["is_high_dti"].where(df["ORIG_DTI"].notna(), np.nan)

    # MI flag
    df["has_mi"] = ((df["MI_PCT"].notna()) & (df["MI_PCT"] > 0)).astype(int)

    # Log transform
    df["log_orig_upb"] = np.log1p(df["ORIG_UPB"].fillna(0))

    # LTV * DTI interaction
    ltv_filled = df["ORIG_LTV"].fillna(df["ORIG_LTV"].median())
    dti_filled = df["ORIG_DTI"].fillna(df["ORIG_DTI"].median())
    df["ltv_dti_interaction"] = ltv_filled * dti_filled

    return df

def read_parquet_partitioned(path: str) -> pd.DataFrame:
    """Read Spark-style partitioned parquet directory"""
    import glob
    if os.path.isdir(path):
        parquet_files = glob.glob(os.path.join(path, "*.parquet"))
        if parquet_files:
            return pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    return pd.read_parquet(path)

def add_vintage_features_pandas(df):
    """Pandas version of vintage features"""
    yr = df["orig_year"]
    
    df["econ_regime"] = "unknown"
    df.loc[yr.between(1999, 2003), "econ_regime"] = "expansion_dotcom"
    df.loc[yr.between(2004, 2007), "econ_regime"] = "bubble_peak"
    df.loc[yr.between(2008, 2009), "econ_regime"] = "gfc_crisis"
    df.loc[yr.between(2010, 2012), "econ_regime"] = "post_gfc"
    df.loc[yr.between(2013, 2019), "econ_regime"] = "recovery"
    
    df["is_bubble_vintage"] = yr.between(2004, 2007).astype(int)
    df["is_gfc_vintage"] = yr.between(2008, 2009).astype(int)
    df["is_post_gfc"] = (yr >= 2010).astype(int)
    
    return df

def stage_split(spark, config) -> dict:
    """Create OOS and OOT splits"""
    from features.spark_splitting import create_splits
    from ingestion.spark_loader import to_spark_path
    import pandas as pd

    split_dir = config.paths.split_dir
    oos_train_path = os.path.join(split_dir, "oos_train")

    if os.path.exists(oos_train_path):
        logger.info("Split parquets found, skipping split stage.")
        return {"split_dir": split_dir, "cached": True}

    # Use the correct modeling path from config
    modeling_path = config.paths.feature_dir  # This is the directory path
    modeling_file = os.path.join(modeling_path, "modeling.parquet")  # Check if file exists
    
    # Check both possible locations
    if not os.path.exists(modeling_file):
        # Try the directory version
        modeling_dir = modeling_path
        if os.path.exists(modeling_dir) and os.path.isdir(modeling_dir):
            # Check if it has parquet files
            import glob
            parquet_files = glob.glob(os.path.join(modeling_dir, "*.parquet"))
            if parquet_files:
                modeling_file = modeling_dir
            else:
                raise RuntimeError(f"Modeling data not found at {modeling_path}")
        else:
            raise RuntimeError(f"Modeling data not found at {modeling_path}")
    
    logger.info(f"Loading modeling data from {modeling_file}...")
    
    # Load as pandas (simpler)
    modeling_df = pd.read_parquet(modeling_file)
    logger.info("Modeling data loaded: %d rows", len(modeling_df))
    
    # Create splits using pandas (simpler)
    from features.spark_splitting import create_splits_pandas
    splits = create_splits_pandas(modeling_df, config)

    os.makedirs(split_dir, exist_ok=True)

    for split_name, (train, val, test) in splits.items():
        if train is not None:
            train_path = os.path.join(split_dir, f"{split_name}_train.parquet")
            train.to_parquet(train_path, index=False)
            logger.info(f"Saved {split_name}_train: {len(train)} rows")
        if val is not None:
            val_path = os.path.join(split_dir, f"{split_name}_val.parquet")
            val.to_parquet(val_path, index=False)
            logger.info(f"Saved {split_name}_val: {len(val)} rows")
        if test is not None:
            test_path = os.path.join(split_dir, f"{split_name}_test.parquet")
            test.to_parquet(test_path, index=False)
            logger.info(f"Saved {split_name}_test: {len(test)} rows")

    return {"split_dir": split_dir}

def stage_train(config) -> dict:
    from models.spark_trainer import train_all_models
    from sklearn.preprocessing import OrdinalEncoder
    import pickle

    split_dir = config.paths.split_dir
    oot_train_path = os.path.join(split_dir, "oot_train")
    oos_train_path = os.path.join(split_dir, "oos_train")
    oos_val_path = os.path.join(split_dir, "oos_val")

    if os.path.exists(oot_train_path):
        train_df = pd.read_parquet(oot_train_path)
        logger.info("Using OOT train split.")
    elif os.path.exists(oos_train_path):
        train_df = pd.read_parquet(oos_train_path)
        logger.info("Using OOS train split.")
    else:
        raise RuntimeError("No training split found.")

    if os.path.exists(oos_val_path):
        val_df = pd.read_parquet(oos_val_path)
    else:
        val_df = train_df.sample(frac=0.2, random_state=42)
        train_df = train_df.drop(val_df.index)

    num_cols = [c for c in config.features.numeric_features if c in train_df.columns]
    cat_cols = [c for c in config.features.categorical_features if c in train_df.columns]
    all_feat = num_cols + cat_cols

    logger.info("Features: %d numeric + %d categorical = %d", len(num_cols), len(cat_cols), len(all_feat))

    # Encode categoricals
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
    train_df[cat_cols] = encoder.fit_transform(train_df[cat_cols].fillna("UNKNOWN").astype(str))
    val_df[cat_cols] = encoder.transform(val_df[cat_cols].fillna("UNKNOWN").astype(str))

    X_train = train_df[all_feat].fillna(-1)
    y_train = train_df["TARGET_12M"].astype(int)
    X_val = val_df[all_feat].fillna(-1)
    y_val = val_df["TARGET_12M"].astype(int)

    model_results = train_all_models(X_train, y_train, X_val, y_val, all_feat, config, config.paths.model_dir)

    # Save encoder and feature list
    with open(os.path.join(config.paths.model_dir, "encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)
    with open(os.path.join(config.paths.model_dir, "feature_cols.pkl"), "wb") as f:
        pickle.dump(all_feat, f)

    for name in model_results:
        model_results[name]["encoder"] = encoder

    return {"model_results": model_results, "feature_cols": all_feat, "models_trained": list(model_results.keys())}


def stage_evaluate(config, train_result: dict) -> dict:
    from validation.spark_evaluator import compute_all_metrics, generate_comparison_table
    import pickle

    model_results = train_result["model_results"]
    feature_cols = train_result["feature_cols"]

    with open(os.path.join(config.paths.model_dir, "encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)

    cat_cols = [c for c in config.features.categorical_features if c in feature_cols]

    all_metrics = []

    for split_name in ["oos", "oot"]:
        test_path = os.path.join(config.paths.split_dir, f"{split_name}_test")
        if not os.path.exists(test_path):
            continue

        test_df = pd.read_parquet(test_path)
        test_df = test_df[test_df["TARGET_12M"].notna()].copy()

        if len(test_df) == 0:
            continue

        test_df[cat_cols] = encoder.transform(test_df[cat_cols].fillna("UNKNOWN").astype(str))
        X_test = test_df[feature_cols].fillna(-1)
        y_test = test_df["TARGET_12M"].astype(int)

        for model_name, info in model_results.items():
            model = info.get("model")
            if model is None:
                continue
            try:
                y_score = model.predict_proba(X_test)[:, 1]
                m = compute_all_metrics(y_test.values, y_score, label=f"{model_name}/{split_name}")
                m["model"] = model_name
                m["split"] = split_name
                all_metrics.append(m)
            except Exception as e:
                logger.error("Eval failed for %s on %s: %s", model_name, split_name, e)

    comparison_df = generate_comparison_table(all_metrics)
    os.makedirs(config.paths.report_dir, exist_ok=True)
    comparison_df.to_csv(os.path.join(config.paths.report_dir, "model_comparison.csv"), index=False)

    return {"comparison_df": comparison_df, "all_metrics": all_metrics}


ALL_STAGES = ["ingest", "targets", "features", "split", "train", "evaluate"]


class CreditRiskPipeline:
    def __init__(self, config=None):
        from config.settings import CFG
        self.config = config or CFG
        self.spark = None

    def run(self, stages=None, year_start: int = 1999, year_end: int = 2012):
        stages = stages or ALL_STAGES
        results = {}

        log_file = setup_logging(self.config.paths.log_dir)
        logger.info("=" * 60)
        logger.info("FREDDIE MAC CREDIT RISK PIPELINE (PySpark) v%s", self.config.project_version)
        logger.info("Stages: %s | Years: %d-%d", stages, year_start, year_end)
        logger.info("=" * 60)

        self.spark = get_spark_session()
        t_total = time.time()

        for stage in stages:
            if stage not in ALL_STAGES:
                continue

            logger.info("\n--- STAGE: %s ---", stage.upper())
            t0 = time.time()

            try:
                if stage == "ingest":
                    results["ingest"] = stage_ingest(self.spark, self.config, year_start, year_end)
                elif stage == "targets":
                    results["targets"] = stage_targets(self.spark, self.config)
                elif stage == "features":
                    results["features"] = stage_features(self.spark, self.config)
                elif stage == "split":
                    results["split"] = stage_split(self.spark, self.config)
                elif stage == "train":
                    results["train"] = stage_train(self.config)
                elif stage == "evaluate":
                    if "train" not in results:
                        raise RuntimeError("'train' stage must run before 'evaluate'")
                    results["evaluate"] = stage_evaluate(self.config, results["train"])

                logger.info("Stage '%s' done in %.1fs", stage, time.time() - t0)

            except Exception as e:
                logger.error("Stage '%s' FAILED: %s", stage, e, exc_info=True)
                results[f"{stage}_error"] = str(e)
                break

        logger.info("\nTotal time: %.1fs", time.time() - t_total)
        self.spark.stop()
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", default=None)
    parser.add_argument("--start-year", type=int, default=1999)
    parser.add_argument("--end-year", type=int, default=2012)
    args = parser.parse_args()

    run_stages = ALL_STAGES if args.stages is None else args.stages
    pipeline = CreditRiskPipeline()
    results = pipeline.run(stages=run_stages, year_start=args.start_year, year_end=args.end_year)

    print("\n" + "=" * 50)
    print("PIPELINE SUMMARY")
    print("=" * 50)
    for stage in run_stages:
        if f"{stage}_error" in results:
            print(f"  FAILED  {stage}")
        elif stage in results:
            print(f"  OK      {stage}")