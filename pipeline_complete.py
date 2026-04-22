"""pipeline_complete.py - Complete integrated pipeline"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-5s | %(message)s')
logger = logging.getLogger(__name__)

from config.settings import CFG

# Define WeightedEnsemble class
class WeightedEnsemble:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
    def predict_proba(self, X):
        combined = np.zeros(len(X))
        for name, model in self.models.items():
            combined += self.weights[name] * model.predict_proba(X)[:, 1]
        return np.column_stack([1 - combined, combined])


def stage_ingest():
    """Stage 1: Load raw data and save as parquet"""
    from ingestion.spark_loader import get_spark_session, load_origination, load_servicing, clean_origination, clean_servicing
    
    logger.info("=" * 60)
    logger.info("STAGE 1: INGEST")
    logger.info("=" * 60)
    
    spark = get_spark_session()
    
    orig_path = os.path.join(CFG.paths.parquet_dir, "origination")
    svcg_path = os.path.join(CFG.paths.parquet_dir, "servicing")
    
    if os.path.exists(orig_path) and os.path.exists(svcg_path):
        logger.info("Parquet cache found, skipping ingestion.")
        spark.stop()
        return
    
    logger.info("Loading origination...")
    orig_raw = load_origination(spark, CFG.paths.raw_dir, 1999, 2012)
    orig_clean = clean_origination(orig_raw)
    orig_clean.write.mode("overwrite").parquet(orig_path)
    logger.info(f"Origination saved: {orig_clean.count()} rows")
    
    logger.info("Loading servicing...")
    svcg_raw = load_servicing(spark, CFG.paths.raw_dir, 1999, 2012)
    svcg_clean = clean_servicing(svcg_raw)
    svcg_clean.write.mode("overwrite").parquet(svcg_path)
    logger.info(f"Servicing saved: {svcg_clean.count()} rows")
    
    spark.stop()
    logger.info("Ingest complete!")


def stage_targets():
    """Stage 2: Build targets and snapshot"""
    from features.spark_targets import add_event_flags, add_rolling_delinquency_features, build_target_12m, select_snapshot
    from ingestion.spark_loader import get_spark_session
    
    logger.info("=" * 60)
    logger.info("STAGE 2: TARGETS")
    logger.info("=" * 60)
    
    spark = get_spark_session()
    
    snapshot_file = os.path.join(CFG.paths.feature_dir, "snapshot.parquet")
    
    if os.path.exists(snapshot_file):
        logger.info("Snapshot found, skipping target construction.")
        spark.stop()
        return
    
    svcg_path = os.path.join(CFG.paths.parquet_dir, "servicing")
    svcg_df = spark.read.parquet(svcg_path)
    logger.info(f"Servicing loaded: {svcg_df.count()} rows")
    
    svcg_df = add_event_flags(svcg_df)
    svcg_df = add_rolling_delinquency_features(svcg_df, windows=[3, 6, 12])
    svcg_df = build_target_12m(svcg_df, window_months=12)
    snapshot_df = select_snapshot(svcg_df, min_obs_months=6)
    
    # Convert to pandas and save
    snapshot_pd = snapshot_df.toPandas()
    os.makedirs(CFG.paths.feature_dir, exist_ok=True)
    snapshot_pd.to_parquet(snapshot_file, index=False)
    
    logger.info(f"Snapshot saved: {len(snapshot_pd)} rows, default rate: {100 * snapshot_pd['TARGET_12M'].mean():.2f}%")
    spark.stop()


def stage_features():
    """Stage 3: Feature engineering"""
    logger.info("=" * 60)
    logger.info("STAGE 3: FEATURES")
    logger.info("=" * 60)
    
    modeling_file = os.path.join(CFG.paths.feature_dir, "modeling.parquet")
    
    if os.path.exists(modeling_file):
        logger.info("Modeling data found, skipping feature engineering.")
        return
    
    # Load data
    orig_path = os.path.join(CFG.paths.parquet_dir, "origination")
    snapshot_file = os.path.join(CFG.paths.feature_dir, "snapshot.parquet")
    
    # Read parquet (Spark format with part files)
    import glob
    orig_files = glob.glob(os.path.join(orig_path, "*.parquet"))
    orig_df = pd.concat([pd.read_parquet(f) for f in orig_files], ignore_index=True)
    snapshot_df = pd.read_parquet(snapshot_file)
    
    logger.info(f"Origination: {len(orig_df)} rows, Snapshot: {len(snapshot_df)} rows")
    
    # Merge
    modeling_df = snapshot_df.merge(orig_df, on="LOAN_SEQUENCE_NUMBER", how="inner")
    logger.info(f"Merged: {len(modeling_df)} rows")
    
    # Add basic features
    modeling_df["loan_age_months"] = modeling_df["LOAN_AGE"].astype(float)
    modeling_df["is_high_ltv"] = (modeling_df["ORIG_LTV"] > 80).astype(float)
    modeling_df["is_high_dti"] = (modeling_df["ORIG_DTI"] > 43).astype(float)
    modeling_df["has_mi"] = ((modeling_df["MI_PCT"].notna()) & (modeling_df["MI_PCT"] > 0)).astype(int)
    
    # Add missing indicators
    for col in ["CREDIT_SCORE", "ORIG_DTI", "ORIG_LTV", "ORIG_CLTV", "MI_PCT"]:
        if col in modeling_df.columns:
            modeling_df[f"{col}_MISSING"] = modeling_df[col].isna().astype(int)
    
    # Fill missing values
    for col in CFG.features.numeric_features:
        if col in modeling_df.columns:
            modeling_df[col] = modeling_df[col].fillna(-1)
    
    for col in CFG.features.categorical_features:
        if col in modeling_df.columns:
            modeling_df[col] = modeling_df[col].fillna("UNKNOWN").astype(str)
    
    modeling_df.to_parquet(modeling_file, index=False)
    logger.info(f"Modeling data saved: {len(modeling_df)} rows, {len(modeling_df.columns)} cols")


def stage_split():
    """Stage 4: Create train/test splits"""
    logger.info("=" * 60)
    logger.info("STAGE 4: SPLIT")
    logger.info("=" * 60)
    
    modeling_file = os.path.join(CFG.paths.feature_dir, "modeling.parquet")
    df = pd.read_parquet(modeling_file)
    df_labeled = df[df["TARGET_12M"].notna()].copy()
    
    logger.info(f"Labeled rows: {len(df_labeled)}, default rate: {100 * df_labeled['TARGET_12M'].mean():.2f}%")
    
    # OOS split
    np.random.seed(42)
    loan_hashes = df_labeled["LOAN_SEQUENCE_NUMBER"].apply(lambda x: hash(str(x)) % 100)
    
    oos_train = df_labeled[loan_hashes < 70].copy()
    oos_val = df_labeled[(loan_hashes >= 70) & (loan_hashes < 85)].copy()
    oos_test = df_labeled[loan_hashes >= 85].copy()
    
    # OOT split
    oot_train = df_labeled[df_labeled["orig_year"].between(1999, 2007)].copy()
    oot_test = df_labeled[df_labeled["orig_year"].between(2008, 2012)].copy()
    
    os.makedirs(CFG.paths.split_dir, exist_ok=True)
    
    oos_train.to_parquet(os.path.join(CFG.paths.split_dir, "oos_train.parquet"), index=False)
    oos_val.to_parquet(os.path.join(CFG.paths.split_dir, "oos_val.parquet"), index=False)
    oos_test.to_parquet(os.path.join(CFG.paths.split_dir, "oos_test.parquet"), index=False)
    oot_train.to_parquet(os.path.join(CFG.paths.split_dir, "oot_train.parquet"), index=False)
    oot_test.to_parquet(os.path.join(CFG.paths.split_dir, "oot_test.parquet"), index=False)
    
    logger.info(f"Splits saved: OOT train={len(oot_train)}, OOT test={len(oot_test)}")


def stage_train():
    """Stage 5: Train all models"""
    logger.info("=" * 60)
    logger.info("STAGE 5: TRAIN")
    logger.info("=" * 60)
    
    # Load splits
    train_df = pd.read_parquet(os.path.join(CFG.paths.split_dir, "oot_train.parquet"))
    val_df = pd.read_parquet(os.path.join(CFG.paths.split_dir, "oos_val.parquet"))
    
    # Get feature columns
    num_cols = [c for c in CFG.features.numeric_features if c in train_df.columns]
    cat_cols = [c for c in CFG.features.categorical_features if c in train_df.columns]
    all_feat = num_cols + cat_cols
    
    logger.info(f"Features: {len(num_cols)} numeric + {len(cat_cols)} categorical = {len(all_feat)}")
    
    # Encode categoricals
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
    train_df[cat_cols] = encoder.fit_transform(train_df[cat_cols].fillna("UNKNOWN").astype(str))
    val_df[cat_cols] = encoder.transform(val_df[cat_cols].fillna("UNKNOWN").astype(str))
    
    X_train = train_df[all_feat].fillna(-1)
    y_train = train_df["TARGET_12M"].astype(int)
    X_val = val_df[all_feat].fillna(-1)
    y_val = val_df["TARGET_12M"].astype(int)
    
    # Calculate imbalance ratio
    ir = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    logger.info(f"Imbalance ratio: {ir:.1f}")
    
    os.makedirs(CFG.paths.model_dir, exist_ok=True)
    
    # Train XGBoost
    logger.info("Training XGBoost...")
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=ir,
        random_state=42, eval_metric="auc"
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Train LightGBM
    logger.info("Training LightGBM...")
    import lightgbm as lgb
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=ir,
        random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    # Train Random Forest
    logger.info("Training Random Forest...")
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=30,
        class_weight={0: 1.0, 1: ir}, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Train Logistic Regression
    logger.info("Training Logistic Regression...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight={0: 1.0, 1: ir}, C=0.1, max_iter=500, random_state=42))
    ])
    lr_model.fit(X_train, y_train)
    
    # Calculate validation AUCs for ensemble weights
    y_val_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
    y_val_pred_lgb = lgb_model.predict_proba(X_val)[:, 1]
    y_val_pred_rf = rf_model.predict_proba(X_val)[:, 1]
    y_val_pred_lr = lr_model.predict_proba(X_val)[:, 1]
    
    auc_xgb = roc_auc_score(y_val, y_val_pred_xgb)
    auc_lgb = roc_auc_score(y_val, y_val_pred_lgb)
    auc_rf = roc_auc_score(y_val, y_val_pred_rf)
    auc_lr = roc_auc_score(y_val, y_val_pred_lr)
    
    logger.info(f"Validation AUCs: XGB={auc_xgb:.4f}, LGB={auc_lgb:.4f}, RF={auc_rf:.4f}, LR={auc_lr:.4f}")
    
    # Create ensemble
    total_auc = auc_xgb + auc_lgb + auc_rf + auc_lr
    weights = {
        "xgboost": auc_xgb / total_auc,
        "lightgbm": auc_lgb / total_auc,
        "rf": auc_rf / total_auc,
        "lr": auc_lr / total_auc
    }
    
    ensemble_model = WeightedEnsemble(
        {"xgboost": xgb_model, "lightgbm": lgb_model, "rf": rf_model, "lr": lr_model},
        weights
    )
    
    # Save all models and artifacts
    with open(os.path.join(CFG.paths.model_dir, "xgboost_model.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)
    with open(os.path.join(CFG.paths.model_dir, "lightgbm_model.pkl"), "wb") as f:
        pickle.dump(lgb_model, f)
    with open(os.path.join(CFG.paths.model_dir, "rf_model.pkl"), "wb") as f:
        pickle.dump(rf_model, f)
    with open(os.path.join(CFG.paths.model_dir, "lr_model.pkl"), "wb") as f:
        pickle.dump(lr_model, f)
    with open(os.path.join(CFG.paths.model_dir, "ensemble_model.pkl"), "wb") as f:
        pickle.dump(ensemble_model, f)
    with open(os.path.join(CFG.paths.model_dir, "encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)
    with open(os.path.join(CFG.paths.model_dir, "feature_cols.pkl"), "wb") as f:
        pickle.dump(all_feat, f)
    
    logger.info(f"All models saved to {CFG.paths.model_dir}")


def stage_evaluate():
    """Stage 6: Evaluate models"""
    logger.info("=" * 60)
    logger.info("STAGE 6: EVALUATE")
    logger.info("=" * 60)
    
    # Load test data
    test_df = pd.read_parquet(os.path.join(CFG.paths.split_dir, "oot_test.parquet"))
    
    # Load artifacts
    with open(os.path.join(CFG.paths.model_dir, "encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    with open(os.path.join(CFG.paths.model_dir, "feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    
    # Load models
    models = {}
    for name in ["xgboost_model.pkl", "lightgbm_model.pkl", "rf_model.pkl", "lr_model.pkl", "ensemble_model.pkl"]:
        with open(os.path.join(CFG.paths.model_dir, name), "rb") as f:
            model_name = name.replace("_model.pkl", "").capitalize()
            if name == "ensemble_model.pkl":
                model_name = "Ensemble"
            elif name == "xgboost_model.pkl":
                model_name = "XGBoost"
            elif name == "lightgbm_model.pkl":
                model_name = "LightGBM"
            elif name == "rf_model.pkl":
                model_name = "Random Forest"
            elif name == "lr_model.pkl":
                model_name = "Logistic Regression"
            models[model_name] = pickle.load(f)
    
    # Prepare test data
    cat_cols = [c for c in CFG.features.categorical_features if c in feature_cols]
    test_df_enc = test_df.copy()
    test_df_enc[cat_cols] = encoder.transform(test_df_enc[cat_cols].fillna("UNKNOWN").astype(str))
    X_test = test_df_enc[feature_cols].fillna(-1)
    y_test = test_df_enc["TARGET_12M"].astype(int)
    
    # Evaluate
    results = []
    for name, model in models.items():
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        gini = 2 * auc - 1
        pos_scores = y_pred[y_test == 1]
        neg_scores = y_pred[y_test == 0]
        ks_stat = ks_2samp(pos_scores, neg_scores).statistic if len(pos_scores) > 0 else 0
        
        results.append({"Model": name, "AUC": auc, "Gini": gini, "KS": ks_stat})
        logger.info(f"{name}: AUC={auc:.4f}, Gini={gini:.4f}, KS={ks_stat:.4f}")
    
    results_df = pd.DataFrame(results).sort_values("AUC", ascending=False)
    os.makedirs(CFG.paths.report_dir, exist_ok=True)
    results_df.to_csv(os.path.join(CFG.paths.report_dir, "model_comparison.csv"), index=False)
    logger.info(f"Results saved to {CFG.paths.report_dir}")
    
    # Plot ROC curves
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    colors = ['#1E3A5F', '#E63946', '#2A9D8F', '#E9C46A', '#264653']
    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        plt.plot(fpr, tpr, lw=2, color=colors[i % len(colors)], label=f'{name} (AUC={auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - OOT Test')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.paths.report_dir, 'roc_curves.png'), dpi=150)
    plt.close()
    logger.info("ROC curves saved")


def run_pipeline(stages=None):
    """Run selected stages"""
    stages_order = ['ingest', 'targets', 'features', 'split', 'train', 'evaluate']
    
    if stages is None:
        stages = stages_order
    
    for stage in stages:
        if stage not in stages_order:
            logger.warning(f"Unknown stage: {stage}")
            continue
        
        try:
            if stage == 'ingest':
                stage_ingest()
            elif stage == 'targets':
                stage_targets()
            elif stage == 'features':
                stage_features()
            elif stage == 'split':
                stage_split()
            elif stage == 'train':
                stage_train()
            elif stage == 'evaluate':
                stage_evaluate()
        except Exception as e:
            logger.error(f"Stage '{stage}' failed: {e}")
            raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", default=None)
    args = parser.parse_args()
    
    run_pipeline(args.stages)