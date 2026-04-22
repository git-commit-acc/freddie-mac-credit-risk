"""manual_evaluate.py - Evaluate all models and generate reports"""
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix
)
from scipy.stats import ks_2samp

# Define WeightedEnsemble class (same as in trainer)
class WeightedEnsemble:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
    
    def predict_proba(self, X):
        combined = np.zeros(len(X))
        for name, model in self.models.items():
            combined += self.weights[name] * model.predict_proba(X)[:, 1]
        return np.column_stack([1 - combined, combined])

print("=" * 60)
print("Model Evaluation")
print("=" * 60)

# Load data
split_dir = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\data\splits"
test_df = pd.read_parquet(os.path.join(split_dir, "oot_test.parquet"))
print(f"Test data: {len(test_df)} rows, Default rate: {test_df['TARGET_12M'].mean():.2%}")

# Load models and artifacts
model_dir = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\data\models"
with open(os.path.join(model_dir, "encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)
with open(os.path.join(model_dir, "feature_cols.pkl"), "rb") as f:
    feature_cols = pickle.load(f)

# Load all models
models = {}
model_files = ["xgboost_model.pkl", "lightgbm_model.pkl", "rf_model.pkl", "lr_model.pkl", "ensemble_model.pkl"]
model_names = ["XGBoost", "LightGBM", "Random Forest", "Logistic Regression", "Ensemble"]

for name, fname in zip(model_names, model_files):
    path = os.path.join(model_dir, fname)
    if os.path.exists(path):
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
        print(f"Loaded: {name}")
    else:
        print(f"Missing: {name}")

# Prepare test data
from config.settings import CFG
cat_cols = [c for c in CFG.features.categorical_features if c in feature_cols]
test_df_enc = test_df.copy()
test_df_enc[cat_cols] = encoder.transform(test_df_enc[cat_cols].fillna("UNKNOWN").astype(str))
X_test = test_df_enc[feature_cols].fillna(-1)
y_test = test_df_enc["TARGET_12M"].astype(int)

# Evaluate each model
results = []
print("\n" + "=" * 60)
print("Evaluation Results - OOT Test Set")
print("=" * 60)

for name, model in models.items():
    try:
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        
        # Calculate Gini = 2*AUC - 1
        gini = 2 * auc - 1
        
        # Calculate KS statistic
        pos_scores = y_pred[y_test == 1]
        neg_scores = y_pred[y_test == 0]
        ks_stat, _ = ks_2samp(pos_scores, neg_scores)
        
        results.append({
            "Model": name,
            "AUC": auc,
            "Gini": gini,
            "KS": ks_stat
        })
        print(f"{name:<20} AUC: {auc:.4f} | Gini: {gini:.4f} | KS: {ks_stat:.4f}")
    except Exception as e:
        print(f"{name:<20} Error: {e}")

results_df = pd.DataFrame(results).sort_values("AUC", ascending=False)
print("\n" + "=" * 60)
print("Ranking:")
print(results_df.to_string(index=False))

# Save results
report_dir = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\data\reports"
os.makedirs(report_dir, exist_ok=True)
results_df.to_csv(os.path.join(report_dir, "model_comparison.csv"), index=False)
print(f"\nResults saved to {report_dir}/model_comparison.csv")

# Plot ROC Curves
print("\nGenerating ROC Curves...")
plt.figure(figsize=(10, 8))
colors = ['#1E3A5F', '#E63946', '#2A9D8F', '#E9C46A', '#264653']

for i, (name, model) in enumerate(models.items()):
    try:
        y_pred = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f'{name} (AUC={auc:.4f})')
    except Exception as e:
        print(f"Could not plot {name}: {e}")

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - OOT Test Set')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(report_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"ROC curves saved to {report_dir}/roc_curves.png")

# Feature Importance (for XGBoost)
print("\n" + "=" * 60)
print("Top 15 Features - XGBoost")
print("=" * 60)
xgb_model = models.get("XGBoost")
if xgb_model and hasattr(xgb_model, 'feature_importances_'):
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(importance.head(15).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = importance.head(15)
    plt.barh(top_features['feature'], top_features['importance'], color='#1E3A5F')
    plt.xlabel('Importance')
    plt.title('XGBoost - Top 15 Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Feature importance saved to {report_dir}/feature_importance.png")
else:
    print("XGBoost model not found or missing feature_importances_")

# Credit Score Distribution
print("\n" + "=" * 60)
print("Credit Score Distribution")
print("=" * 60)

# Import scorer functions
from models.scorer import probability_to_score, score_distribution_report

ensemble = models.get("Ensemble")
if ensemble:
    try:
        y_pred = ensemble.predict_proba(X_test)[:, 1]
        scores = probability_to_score(y_pred)
        
        # Score distribution by outcome
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(scores[y_test == 0], bins=30, alpha=0.6, color='#2A9D8F', label='Non-Default', density=True)
        axes[0].hist(scores[y_test == 1], bins=30, alpha=0.6, color='#E63946', label='Default', density=True)
        axes[0].set_xlabel('Credit Score (300-900)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Score Distribution by Outcome')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Default rate by bucket
        dist_df = score_distribution_report(scores, y_test.values)
        if 'default_rate_pct' in dist_df.columns:
            axes[1].bar(dist_df['score_bucket'], dist_df['default_rate_pct'], color='#E63946', edgecolor='black', alpha=0.8)
            axes[1].set_xlabel('Score Bucket')
            axes[1].set_ylabel('Default Rate (%)')
            axes[1].set_title('Default Rate by Score Bucket')
            axes[1].tick_params(axis='x', rotation=30)
            for bar, val in zip(axes[1].patches, dist_df['default_rate_pct']):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'score_distribution.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save score distribution table
        dist_df.to_csv(os.path.join(report_dir, 'score_distribution.csv'), index=False)
        print(f"\nScore distribution saved to {report_dir}/score_distribution.csv")
        print(dist_df.to_string(index=False))
    except Exception as e:
        print(f"Error generating score distribution: {e}")
else:
    print("Ensemble model not found")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE!")
print(f"All reports saved to: {report_dir}")
print("=" * 60)