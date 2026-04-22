"""manual_train_all.py - Train all models and evaluate"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, classification_report

print("=" * 60)
print("Manual Training - All Models")
print("=" * 60)

# Load splits
split_dir = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\data\splits"
train_df = pd.read_parquet(os.path.join(split_dir, "oot_train.parquet"))
val_df = pd.read_parquet(os.path.join(split_dir, "oos_val.parquet"))
test_df = pd.read_parquet(os.path.join(split_dir, "oot_test.parquet"))

print(f"Train: {len(train_df)} rows, Default rate: {train_df['TARGET_12M'].mean():.2%}")
print(f"Val: {len(val_df)} rows, Default rate: {val_df['TARGET_12M'].mean():.2%}")
print(f"Test: {len(test_df)} rows, Default rate: {test_df['TARGET_12M'].mean():.2%}")

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
test_df[cat_cols] = encoder.transform(test_df[cat_cols].fillna("UNKNOWN").astype(str))

X_train = train_df[all_feat].fillna(-1)
y_train = train_df["TARGET_12M"].astype(int)
X_val = val_df[all_feat].fillna(-1)
y_val = val_df["TARGET_12M"].astype(int)
X_test = test_df[all_feat].fillna(-1)
y_test = test_df["TARGET_12M"].astype(int)

# Calculate imbalance ratio
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
ir = n_neg / n_pos
print(f"Imbalance ratio: {ir:.1f}")

model_dir = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\data\models"
os.makedirs(model_dir, exist_ok=True)

results = {}

# 1. XGBoost (already trained)
print("\n" + "=" * 60)
print("Loading XGBoost...")
with open(os.path.join(model_dir, "xgboost_model.pkl"), "rb") as f:
    xgb_model = pickle.load(f)
y_pred_xgb = xgb_model.predict_proba(X_test)[:, 1]
auc_xgb = roc_auc_score(y_test, y_pred_xgb)
results["XGBoost"] = auc_xgb
print(f"XGBoost Test AUC: {auc_xgb:.4f}")

# 2. LightGBM
print("\n" + "=" * 60)
print("Training LightGBM...")
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=ir,
    random_state=42,
    verbose=-1
)

lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
y_pred_lgb = lgb_model.predict_proba(X_test)[:, 1]
auc_lgb = roc_auc_score(y_test, y_pred_lgb)
results["LightGBM"] = auc_lgb
print(f"LightGBM Test AUC: {auc_lgb:.4f}")

# Save LightGBM
with open(os.path.join(model_dir, "lightgbm_model.pkl"), "wb") as f:
    pickle.dump(lgb_model, f)

# 3. Random Forest
print("\n" + "=" * 60)
print("Training Random Forest...")
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=30,
    class_weight={0: 1.0, 1: ir},
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, y_pred_rf)
results["Random Forest"] = auc_rf
print(f"Random Forest Test AUC: {auc_rf:.4f}")

# Save Random Forest
with open(os.path.join(model_dir, "rf_model.pkl"), "wb") as f:
    pickle.dump(rf_model, f)

# 4. Logistic Regression
print("\n" + "=" * 60)
print("Training Logistic Regression...")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

lr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        class_weight={0: 1.0, 1: ir},
        C=0.1,
        max_iter=500,
        random_state=42
    ))
])

lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict_proba(X_test)[:, 1]
auc_lr = roc_auc_score(y_test, y_pred_lr)
results["Logistic Regression"] = auc_lr
print(f"Logistic Regression Test AUC: {auc_lr:.4f}")

# Save Logistic Regression
with open(os.path.join(model_dir, "lr_model.pkl"), "wb") as f:
    pickle.dump(lr_model, f)

# 5. Ensemble (weighted average)
print("\n" + "=" * 60)
print("Creating Ensemble...")

# Get validation AUCs for weights
y_val_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
y_val_pred_lgb = lgb_model.predict_proba(X_val)[:, 1]
y_val_pred_rf = rf_model.predict_proba(X_val)[:, 1]
y_val_pred_lr = lr_model.predict_proba(X_val)[:, 1]

auc_val_xgb = roc_auc_score(y_val, y_val_pred_xgb)
auc_val_lgb = roc_auc_score(y_val, y_val_pred_lgb)
auc_val_rf = roc_auc_score(y_val, y_val_pred_rf)
auc_val_lr = roc_auc_score(y_val, y_val_pred_lr)

print(f"Validation AUCs:")
print(f"  XGBoost: {auc_val_xgb:.4f}")
print(f"  LightGBM: {auc_val_lgb:.4f}")
print(f"  Random Forest: {auc_val_rf:.4f}")
print(f"  Logistic Regression: {auc_val_lr:.4f}")

# Weight by validation AUC
total_auc = auc_val_xgb + auc_val_lgb + auc_val_rf + auc_val_lr
weights = {
    "xgboost": auc_val_xgb / total_auc,
    "lightgbm": auc_val_lgb / total_auc,
    "rf": auc_val_rf / total_auc,
    "lr": auc_val_lr / total_auc
}

print(f"\nEnsemble weights: {weights}")

# Ensemble prediction
y_pred_ensemble = (
    weights["xgboost"] * y_pred_xgb +
    weights["lightgbm"] * y_pred_lgb +
    weights["rf"] * y_pred_rf +
    weights["lr"] * y_pred_lr
)
auc_ensemble = roc_auc_score(y_test, y_pred_ensemble)
results["Ensemble"] = auc_ensemble
print(f"Ensemble Test AUC: {auc_ensemble:.4f}")

# Save ensemble
class WeightedEnsemble:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
    def predict_proba(self, X):
        combined = np.zeros(len(X))
        for name, model in self.models.items():
            combined += self.weights[name] * model.predict_proba(X)[:, 1]
        return np.column_stack([1 - combined, combined])

ensemble_model = WeightedEnsemble(
    {"xgboost": xgb_model, "lightgbm": lgb_model, "rf": rf_model, "lr": lr_model},
    weights
)

with open(os.path.join(model_dir, "ensemble_model.pkl"), "wb") as f:
    pickle.dump(ensemble_model, f)

# Save encoder and feature columns
with open(os.path.join(model_dir, "encoder.pkl"), "wb") as f:
    pickle.dump(encoder, f)
with open(os.path.join(model_dir, "feature_cols.pkl"), "wb") as f:
    pickle.dump(all_feat, f)

# Summary
print("\n" + "=" * 60)
print("FINAL RESULTS - Test AUC")
print("=" * 60)
for model, auc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {model:<20}: {auc:.4f}")

print("\n" + "=" * 60)
print("SUCCESS! All models saved to:", model_dir)
print("=" * 60)