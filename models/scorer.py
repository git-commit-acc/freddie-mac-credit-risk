"""models/scorer.py - Convert probability to credit score (300-900)"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_SCORE = 600
BASE_ODDS = 19
PDO = 40
SCORE_MIN = 300
SCORE_MAX = 900


def probability_to_score(proba: np.ndarray) -> np.ndarray:
    """Convert predicted default probability to 300-900 score"""
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    log_odds = np.log((1.0 - proba) / proba)
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    raw_score = offset + factor * log_odds
    score = np.clip(np.round(raw_score), SCORE_MIN, SCORE_MAX).astype(int)
    return score


def score_to_risk_bucket(score: np.ndarray) -> pd.Series:
    """Assign risk label to each score"""
    labels = pd.cut(
        score,
        bins=[300, 580, 620, 660, 720, 760, 900],
        labels=["Very High Risk", "High Risk", "Medium-High Risk", 
                "Medium Risk", "Low-Medium Risk", "Low Risk"],
        right=True,
        include_lowest=True
    )
    return labels.astype(str)


def add_scores_to_dataframe(df: pd.DataFrame, proba_col: str = "pred_proba",
                             score_col: str = "credit_score_pred") -> pd.DataFrame:
    """Add score column and risk bucket to dataframe"""
    df = df.copy()
    df[score_col] = probability_to_score(df[proba_col].values)
    df["risk_bucket"] = score_to_risk_bucket(df[score_col].values)
    return df


def score_distribution_report(scores: np.ndarray, y_true: np.ndarray = None) -> pd.DataFrame:
    """Show score distribution across buckets"""
    buckets = pd.cut(
        scores,
        bins=[300, 580, 620, 660, 720, 760, 900],
        labels=["300-580", "580-620", "620-660", "660-720", "720-760", "760-900"],
        right=True,
        include_lowest=True
    )

    report_df = pd.DataFrame({"score_bucket": buckets, "score": scores})

    if y_true is not None:
        report_df["default"] = y_true

    summary = report_df.groupby("score_bucket", observed=False).agg(
        n_loans=("score", "count"),
        mean_score=("score", "mean"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    )

    if y_true is not None:
        dr = report_df.groupby("score_bucket", observed=False)["default"].mean()
        summary["default_rate_pct"] = (dr * 100).round(2)

    summary["pct_of_total"] = (summary["n_loans"] / summary["n_loans"].sum() * 100).round(1)
    return summary.reset_index()