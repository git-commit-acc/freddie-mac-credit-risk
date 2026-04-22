"""validation/spark_evaluator.py - Evaluation metrics"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List

logger = logging.getLogger(__name__)


def compute_auc_roc(y_true, y_score) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def compute_auc_pr(y_true, y_score) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, y_score))


def compute_ks_statistic(y_true, y_score) -> float:
    from scipy.stats import ks_2samp
    pos_scores = y_score[np.array(y_true) == 1]
    neg_scores = y_score[np.array(y_true) == 0]
    ks_stat, _ = ks_2samp(pos_scores, neg_scores)
    return float(ks_stat)


def compute_gini(auc_roc: float) -> float:
    return 2.0 * auc_roc - 1.0


def compute_brier_score(y_true, y_score) -> float:
    from sklearn.metrics import brier_score_loss
    return float(brier_score_loss(y_true, y_score))


def compute_all_metrics(y_true, y_score, threshold: float = 0.5, label: str = "") -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= threshold).astype(int)

    auc = compute_auc_roc(y_true, y_score)

    m = {
        "label": label,
        "n_total": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "default_rate_pct": float(round(100.0 * y_true.mean(), 2)),
        "auc_roc": round(auc, 4),
        "gini": round(compute_gini(auc), 4),
        "ks_stat": round(compute_ks_statistic(y_true, y_score), 4),
        "auc_pr": round(compute_auc_pr(y_true, y_score), 4),
        "brier_score": round(compute_brier_score(y_true, y_score), 5),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
    }

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        m.update({"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)})

    return m


def generate_comparison_table(all_metrics: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(all_metrics)
    show_cols = ["model", "split", "n_total", "default_rate_pct", "auc_roc", "gini", "ks_stat", "auc_pr", "f1"]
    df = df[[c for c in show_cols if c in df.columns]]
    return df.sort_values(["split", "auc_roc"], ascending=[True, False]).reset_index(drop=True)