"""models/spark_trainer.py - Train models using pandas (Spark for data, sklearn for training)"""

import logging
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List

logger = logging.getLogger(__name__)


def train_xgboost(X_train, y_train, X_val, y_val, ir: float, config=None, output_dir: str = "data/models"):
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("Run: pip install xgboost")

    from config.settings import CFG
    cfg = (config or CFG).models.xgb_params.copy()

    n_estimators = cfg.pop("n_estimators", 300)
    eval_metric = cfg.pop("eval_metric", "auc")

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        scale_pos_weight=ir,
        eval_metric=eval_metric,
        early_stopping_rounds=30,
        **cfg
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    fi = pd.DataFrame({
        "feature": X_train.columns.tolist(),
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "xgboost_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)

    return {"model": model, "feature_importances": fi, "model_path": path}


def train_lightgbm(X_train, y_train, X_val, y_val, ir: float, config=None, output_dir: str = "data/models"):
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("Run: pip install lightgbm")

    from config.settings import CFG
    cfg = (config or CFG).models.lgbm_params.copy()

    n_estimators = cfg.pop("n_estimators", 300)

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        scale_pos_weight=ir,
        **cfg
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=50)
    ])

    fi = pd.DataFrame({
        "feature": X_train.columns.tolist(),
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    path = os.path.join(output_dir, "lightgbm_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)

    return {"model": model, "feature_importances": fi, "model_path": path}


def train_random_forest(X_train, y_train, ir: float, config=None, output_dir: str = "data/models"):
    from sklearn.ensemble import RandomForestClassifier
    from config.settings import CFG
    cfg = (config or CFG).models.rf_params.copy()

    model = RandomForestClassifier(
        class_weight={0: 1.0, 1: ir},
        **cfg
    )
    model.fit(X_train, y_train)

    fi = pd.DataFrame({
        "feature": X_train.columns.tolist(),
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    path = os.path.join(output_dir, "rf_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)

    return {"model": model, "feature_importances": fi, "model_path": path}


def train_logistic_regression(X_train, y_train, ir: float, config=None, output_dir: str = "data/models"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from config.settings import CFG
    cfg = (config or CFG).models.lr_params.copy()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight={0: 1.0, 1: ir}, **cfg))
    ])
    model.fit(X_train, y_train)

    path = os.path.join(output_dir, "lr_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)

    coefs = model.named_steps["lr"].coef_[0]
    fi = pd.DataFrame({
        "feature": X_train.columns.tolist(),
        "coefficient": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    return {"model": model, "feature_importances": fi, "model_path": path}


class WeightedEnsemble:
    def __init__(self, models: dict, weights: dict):
        self.models = models
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}

    def predict_proba(self, X):
        combined = np.zeros(len(X))
        for name, model in self.models.items():
            w = self.weights.get(name, 0)
            if w == 0:
                continue
            proba = model.predict_proba(X)[:, 1]
            combined += w * proba
        return np.column_stack([1 - combined, combined])

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def build_ensemble(model_results: dict, X_val, y_val, output_dir: str = "data/models"):
    from sklearn.metrics import roc_auc_score

    models = {}
    aucs = {}

    for name, result in model_results.items():
        model = result.get("model")
        if model is None:
            continue
        try:
            proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba)
            models[name] = model
            aucs[name] = auc
            logger.info("  %s val AUC = %.4f", name, auc)
        except Exception as e:
            logger.warning("Could not evaluate %s: %s", name, e)

    if not models:
        raise RuntimeError("No models available for ensemble.")

    ensemble = WeightedEnsemble(models, aucs)

    path = os.path.join(output_dir, "ensemble_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(ensemble, f)

    return {"model": ensemble, "component_aucs": aucs, "model_path": path}


def train_all_models(X_train, y_train, X_val, y_val, feature_cols: List[str],
                      config=None, output_dir: str = "data/models") -> Dict[str, dict]:
    os.makedirs(output_dir, exist_ok=True)

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    ir = float(n_neg) / max(n_pos, 1.0)
    logger.info("Training with IR=%.1f (neg=%d, pos=%d)", ir, n_neg, n_pos)

    results = {}

    try:
        results["xgboost"] = train_xgboost(X_train, y_train, X_val, y_val, ir, config, output_dir)
    except Exception as e:
        logger.error("XGBoost failed: %s", e)

    try:
        results["lightgbm"] = train_lightgbm(X_train, y_train, X_val, y_val, ir, config, output_dir)
    except Exception as e:
        logger.error("LightGBM failed: %s", e)

    try:
        results["random_forest"] = train_random_forest(X_train, y_train, ir, config, output_dir)
    except Exception as e:
        logger.error("Random Forest failed: %s", e)

    try:
        results["logistic_regression"] = train_logistic_regression(X_train, y_train, ir, config, output_dir)
    except Exception as e:
        logger.error("Logistic Regression failed: %s", e)

    if len(results) >= 2:
        try:
            results["ensemble"] = build_ensemble(results, X_val, y_val, output_dir)
        except Exception as e:
            logger.error("Ensemble failed: %s", e)

    return results