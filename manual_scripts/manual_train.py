"""manual_train.py - Train models manually"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import OrdinalEncoder

print("=" * 60)
print("Manual Model Training")
print("=" * 60)

# Load splits
split_dir = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\data\splits"
train_df = pd.read_parquet(os.path.join(split_dir, "oot_train.parquet"))
val_df = pd.read_parquet(os.path.join(split_dir, "oos_val.parquet"))

print(f"Train: {len(train_df)} rows, Default rate: {train_df['TARGET_12M'].mean():.2%}")
print(f"Val: {len(val_df)} rows, Default rate: {val_df['TARGET_12M'].mean():.2%}")

# Feature columns
from config.settings import CFG
num_cols = [c for c in CFG.features.numeric_features if c in train_df.columns]
cat_cols = [c for c in CFG.features.categorical_features if c in train_df.columns]
all_feat = num_cols + cat_cols

print(f"\nFeatures: {len(all_feat)} total")

# Encode categoricals
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
train_df[cat_cols] = encoder.fit_transform(train_df[cat_cols].fillna("UNKNOWN").astype(str))
val_df[cat_cols] = encoder.transform(val_df[cat_cols].fillna("UNKNOWN").astype(str))

X_train = train_df[all_feat].fillna(-1)
y_train = train_df["TARGET_12M"].astype(int)
X_val = val_df[all_feat].fillna(-1)
y_val = val_df["TARGET_12M"].astype(int)

# Calculate imbalance ratio
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
ir = n_neg / n_pos
print(f"\nImbalance ratio: {ir:.1f}")

# Train XGBoost
print("\n" + "=" * 60)
print("Training XGBoost...")
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=ir,
    random_state=42,
    eval_metric="auc"
)

xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

# Save model
model_dir = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\data\models"
os.makedirs(model_dir, exist_ok=True)
with open(os.path.join(model_dir, "xgboost_model.pkl"), "wb") as f:
    pickle.dump(xgb_model, f)

# Evaluate
from sklearn.metrics import roc_auc_score
y_pred = xgb_model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print(f"XGBoost Validation AUC: {auc:.4f}")

print("\n" + "=" * 60)
print("SUCCESS! Models saved to:", model_dir)
print("=" * 60)