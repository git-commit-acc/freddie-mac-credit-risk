import pandas as pd
import os
import glob

# Read the snapshot parquet files
snapshot_dir = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\data\features\snapshot"
snapshot_files = glob.glob(os.path.join(snapshot_dir, "*.parquet"))
snapshot_df = pd.concat([pd.read_parquet(f) for f in snapshot_files], ignore_index=True)

print(f"Snapshot loaded: {len(snapshot_df)} rows")
print(f"Columns: {snapshot_df.columns.tolist()[:10]}")
print(f"TARGET_12M distribution:\n{snapshot_df['TARGET_12M'].value_counts()}")