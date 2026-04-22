"""
config/settings.py
All constants for the Freddie Mac SFLLD pipeline - PySpark version.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict

DEFAULT_RAW_DIR = r"D:\Projects\Major Project\dataset\raw_extracted"

# Origination columns (0-indexed)
ORIG_COLS_IDX: List[int] = [0, 1, 3, 5, 8, 9, 10, 11, 12, 14, 16, 19, 21, 24]

ORIG_COLS_NAMES: List[str] = [
    "CREDIT_SCORE", "FIRST_PAYMENT_DATE", "MATURITY_DATE", "MI_PCT",
    "ORIG_CLTV", "ORIG_DTI", "ORIG_UPB", "ORIG_LTV", "ORIG_INTEREST_RATE",
    "PPM_FLAG", "PROPERTY_STATE", "LOAN_SEQUENCE_NUMBER",
    "ORIG_LOAN_TERM", "SERVICER_NAME",
]

# Servicing columns (0-indexed)
SVCG_COLS_IDX: List[int] = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]

SVCG_COLS_NAMES: List[str] = [
    "LOAN_SEQUENCE_NUMBER", "MONTHLY_RPT_PRD", "CURR_ACTUAL_UPB",
    "DLQCY_STATUS", "LOAN_AGE", "REM_MONTHS_MATURITY",
    "MOD_FLAG", "ZERO_BAL_CD", "ZERO_BAL_EFF_DATE", "CURR_INTEREST_RATE",
]

ORIG_SENTINELS: Dict[str, list] = {
    "CREDIT_SCORE": [9999], "MI_PCT": [999], "ORIG_CLTV": [999],
    "ORIG_DTI": [999], "ORIG_LTV": [999],
}

SERIOUS_DELINQ_CODES: List[str] = [
    "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
    "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "RA",
]

LOSS_ZBC: List[str] = ["02", "03", "09", "15", "16"]
PREPAY_ZBC: str = "01"


# @dataclass
# class PathConfig:
#     raw_dir:       str = DEFAULT_RAW_DIR
#     parquet_dir:   str = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\data\parquet"
#     feature_dir:   str = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\data\features"
#     model_dir:     str = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\data\models"
#     report_dir:    str = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\data\reports"
#     log_dir:       str = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\logs"
#     split_dir:     str = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\data\splits"

#     def create_all(self):
#         for p in vars(self).values():
#             if isinstance(p, str):
#                 os.makedirs(p, exist_ok=True)
@dataclass
class PathConfig:
    raw_dir: str = DEFAULT_RAW_DIR
    # Use the spark project directory as base
    base_data_dir: str = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark\data"
    
    @property
    def parquet_dir(self) -> str:
        return os.path.join(self.base_data_dir, "parquet")
    
    @property
    def feature_dir(self) -> str:
        return os.path.join(self.base_data_dir, "features")
    
    @property
    def model_dir(self) -> str:
        return os.path.join(self.base_data_dir, "models")
    
    @property
    def report_dir(self) -> str:
        return os.path.join(self.base_data_dir, "reports")
    
    @property
    def log_dir(self) -> str:
        return os.path.join(self.base_data_dir, "logs")
    
    @property
    def split_dir(self) -> str:
        return os.path.join(self.base_data_dir, "splits")
    
    def create_all(self):
        for attr in ['parquet_dir', 'feature_dir', 'model_dir', 'report_dir', 'log_dir', 'split_dir']:
            p = getattr(self, attr)
            if isinstance(p, str):
                os.makedirs(p, exist_ok=True)

@dataclass
class TargetConfig:
    window_months:    int = 12
    min_obs_months:   int = 6


@dataclass
class FeatureConfig:
    rolling_windows: List[int] = field(default_factory=lambda: [3, 6, 12])

    numeric_features: List[str] = field(default_factory=lambda: [
        "CREDIT_SCORE", "ORIG_LTV", "ORIG_CLTV", "ORIG_DTI", "ORIG_INTEREST_RATE",
        "ORIG_UPB", "ORIG_LOAN_TERM", "MI_PCT", "log_orig_upb", "ltv_dti_interaction",
        "credit_ltv_ratio", "est_monthly_payment", "is_high_ltv", "is_high_dti",
        "has_mi", "loan_age_months", "loan_age_ratio", "pct_life_remaining",
        "upb_pct_remaining", "rate_spread", "was_modified", "max_delinq_3m",
        "mean_delinq_3m", "n_delinq_months_3m", "max_delinq_6m", "mean_delinq_6m",
        "n_delinq_months_6m", "n_serious_delinq_6m", "max_delinq_12m",
        "mean_delinq_12m", "n_delinq_months_12m", "n_serious_delinq_12m",
        "payment_streak", "delinq_trend_6m", "orig_year", "is_bubble_vintage",
        "is_gfc_vintage", "is_post_gfc", "CREDIT_SCORE_MISSING", "ORIG_DTI_MISSING",
        "ORIG_LTV_MISSING", "ORIG_CLTV_MISSING",
    ])

    categorical_features: List[str] = field(default_factory=lambda: [
        "PROPERTY_STATE", "credit_score_bucket", "ltv_bucket",
        "dti_bucket", "loan_age_bucket", "econ_regime",
    ])


@dataclass
class SplitConfig:
    oos_seed:        int = 42
    oos_train_ratio: float = 0.70
    oos_val_ratio:   float = 0.15
    oot_train_years: tuple = (1999, 2007)
    oot_test_years:  tuple = (2008, 2012)


@dataclass
class ModelConfig:
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 20,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
    })
    lgbm_params: dict = field(default_factory=lambda: {
        "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
    })
    rf_params: dict = field(default_factory=lambda: {
        "n_estimators": 200, "max_depth": 8, "min_samples_leaf": 30, "random_state": 42,
    })
    lr_params: dict = field(default_factory=lambda: {
        "C": 0.1, "max_iter": 500, "random_state": 42,
    })


@dataclass
class Config:
    paths:    PathConfig = field(default_factory=PathConfig)
    target:   TargetConfig = field(default_factory=TargetConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    splits:   SplitConfig = field(default_factory=SplitConfig)
    models:   ModelConfig = field(default_factory=ModelConfig)
    project_name: str = "Freddie_Mac_Credit_Risk"
    project_version: str = "4.0.0"

    def __post_init__(self):
        self.paths.create_all()


CFG = Config()