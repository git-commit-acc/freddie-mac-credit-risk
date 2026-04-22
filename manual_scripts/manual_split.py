"""manual_split.py - Create train/test splits manually"""
import pandas as pd
import numpy as np
import os

print("=" * 60)
print("Manual Split Creation")
print("=" * 60)

# Load modeling data
modeling_file = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\data\features\modeling.parquet"
print(f"\nLoading modeling data from {modeling_file}...")
df = pd.read_parquet(modeling_file)
print(f"Loaded {len(df)} rows")

# Filter to labeled rows
df_labeled = df[df["TARGET_12M"].notna()].copy()
print(f"Labeled rows: {len(df_labeled)}")
print(f"Default rate: {100 * df_labeled['TARGET_12M'].mean():.2f}%")

# Create OOS split (random)
print("\nCreating OOS split...")
np.random.seed(42)

# Use hash of loan sequence number for deterministic split
loan_hashes = df_labeled["LOAN_SEQUENCE_NUMBER"].apply(lambda x: hash(str(x)) % 100)
train_cut = 70  # 70% train
val_cut = 85    # 15% val, 15% test

oos_train = df_labeled[loan_hashes < train_cut].copy()
oos_val = df_labeled[(loan_hashes >= train_cut) & (loan_hashes < val_cut)].copy()
oos_test = df_labeled[loan_hashes >= val_cut].copy()

print(f"  OOS Train: {len(oos_train)} rows")
print(f"  OOS Val: {len(oos_val)} rows")
print(f"  OOS Test: {len(oos_test)} rows")

# Create OOT split (time-based)
print("\nCreating OOT split...")
oot_train = df_labeled[df_labeled["orig_year"].between(1999, 2007)].copy()
oot_test = df_labeled[df_labeled["orig_year"].between(2008, 2012)].copy()

print(f"  OOT Train (1999-2007): {len(oot_train)} rows")
print(f"  OOT Test (2008-2012): {len(oot_test)} rows")

# Save splits
split_dir = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\data\splits"
os.makedirs(split_dir, exist_ok=True)

print("\nSaving splits...")
oos_train.to_parquet(os.path.join(split_dir, "oos_train.parquet"), index=False)
oos_val.to_parquet(os.path.join(split_dir, "oos_val.parquet"), index=False)
oos_test.to_parquet(os.path.join(split_dir, "oos_test.parquet"), index=False)
oot_train.to_parquet(os.path.join(split_dir, "oot_train.parquet"), index=False)
oot_test.to_parquet(os.path.join(split_dir, "oot_test.parquet"), index=False)

print("All splits saved!")

# Also save as CSV for inspection (optional)
# oos_train.to_csv(os.path.join(split_dir, "oos_train.csv"), index=False)

print("\n" + "=" * 60)
print("SUCCESS! Now run:")
print("python pipeline_spark.py --stages train")
print("python pipeline_spark.py --stages evaluate")
print("=" * 60)