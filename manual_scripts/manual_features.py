"""manual_features.py - Run feature engineering manually"""
import pandas as pd
import numpy as np
import os
import glob

print("=" * 60)
print("Manual Feature Engineering")
print("=" * 60)

# Paths
snapshot_dir = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\data\features\snapshot"
orig_dir = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\data\parquet\origination"
output_file = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\data\features\modeling.parquet"

# Check if directories exist
print(f"\nChecking paths...")
print(f"Snapshot dir exists: {os.path.exists(snapshot_dir)}")
print(f"Origination dir exists: {os.path.exists(orig_dir)}")

if not os.path.exists(snapshot_dir):
    print(f"ERROR: Snapshot directory not found at {snapshot_dir}")
    print("Please run targets stage first:")
    print("python pipeline_spark.py --stages targets")
    exit(1)

if not os.path.exists(orig_dir):
    print(f"ERROR: Origination directory not found at {orig_dir}")
    print("Please run ingest stage first:")
    print("python pipeline_spark.py --stages ingest")
    exit(1)

print("\nLoading snapshot...")
snapshot_files = glob.glob(os.path.join(snapshot_dir, "*.parquet"))
print(f"Found {len(snapshot_files)} snapshot parquet files")
snapshot_df = pd.concat([pd.read_parquet(f) for f in snapshot_files], ignore_index=True)
print(f"Snapshot: {len(snapshot_df)} rows")

print("\nLoading origination...")
orig_files = glob.glob(os.path.join(orig_dir, "*.parquet"))
print(f"Found {len(orig_files)} origination parquet files")
orig_df = pd.concat([pd.read_parquet(f) for f in orig_files], ignore_index=True)
print(f"Origination: {len(orig_df)} rows")

print("\nMerging on LOAN_SEQUENCE_NUMBER...")
modeling_df = snapshot_df.merge(orig_df, on="LOAN_SEQUENCE_NUMBER", how="inner")
print(f"Merged: {len(modeling_df)} rows")

print("\nAdding basic features...")
modeling_df["loan_age_months"] = modeling_df["LOAN_AGE"].astype(float)
modeling_df["is_high_ltv"] = (modeling_df["ORIG_LTV"] > 80).astype(float)
modeling_df["is_high_ltv"] = modeling_df["is_high_ltv"].where(modeling_df["ORIG_LTV"].notna(), -1)
modeling_df["is_high_dti"] = (modeling_df["ORIG_DTI"] > 43).astype(float)
modeling_df["is_high_dti"] = modeling_df["is_high_dti"].where(modeling_df["ORIG_DTI"].notna(), -1)
modeling_df["has_mi"] = ((modeling_df["MI_PCT"].notna()) & (modeling_df["MI_PCT"] > 0)).astype(int)

# Add missing indicators
for col_name in ["CREDIT_SCORE", "ORIG_DTI", "ORIG_LTV", "ORIG_CLTV", "MI_PCT"]:
    if col_name in modeling_df.columns:
        modeling_df[f"{col_name}_MISSING"] = modeling_df[col_name].isna().astype(int)

print("\nSaving modeling data...")
os.makedirs(os.path.dirname(output_file), exist_ok=True)
modeling_df.to_parquet(output_file, index=False)
print(f"Saved to {output_file}")
print(f"Shape: {modeling_df.shape}")
print(f"Memory usage: {modeling_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "=" * 60)
print("SUCCESS! Now run:")
print("python pipeline_spark.py --stages split")
print("python pipeline_spark.py --stages train")
print("python pipeline_spark.py --stages evaluate")
print("=" * 60)