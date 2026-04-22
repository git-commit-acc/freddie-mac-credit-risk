"""features/spark_engineering.py - Feature engineering"""

from pyspark.sql import DataFrame
import pandas as pd
import numpy as np


from pyspark.sql.functions import (
    col, when, lit, round, log1p, coalesce, expr, upper, lower, trim, length,
    substring, isnan, isnull, abs as spark_abs
)
from pyspark.sql.types import IntegerType, DoubleType



def add_origination_features_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Pandas version of origination features"""
    # Credit score buckets
    df["credit_score_bucket"] = pd.cut(
        df["CREDIT_SCORE"],
        bins=[0, 580, 620, 660, 720, 760, 900],
        labels=["deep_subprime", "subprime", "near_prime", "prime", "prime_plus", "super_prime"],
        right=False
    ).astype(str)
    df["credit_score_bucket"] = df["credit_score_bucket"].fillna("unknown")

    # LTV buckets
    df["ltv_bucket"] = pd.cut(
        df["ORIG_LTV"],
        bins=[0, 60, 70, 80, 90, 97, 999],
        labels=["lte60", "60_70", "70_80", "80_90", "90_97", "gt97"],
        right=True
    ).astype(str)
    df["ltv_bucket"] = df["ltv_bucket"].fillna("unknown")

    df["is_high_ltv"] = (df["ORIG_LTV"] > 80).astype(float)
    df["is_high_ltv"] = df["is_high_ltv"].where(df["ORIG_LTV"].notna(), np.nan)

    # DTI buckets
    df["dti_bucket"] = pd.cut(
        df["ORIG_DTI"],
        bins=[0, 28, 36, 43, 50, 100],
        labels=["conservative", "moderate", "standard", "elevated", "high"],
        right=True
    ).astype(str)
    df["dti_bucket"] = df["dti_bucket"].fillna("unknown")

    df["is_high_dti"] = (df["ORIG_DTI"] > 43).astype(float)
    df["is_high_dti"] = df["is_high_dti"].where(df["ORIG_DTI"].notna(), np.nan)

    # MI flag
    df["has_mi"] = ((df["MI_PCT"].notna()) & (df["MI_PCT"] > 0)).astype(int)

    # PPM flag
    df["is_ppm"] = (df["PPM_FLAG"].str.upper() == "Y").astype(int)

    # Log transform
    df["log_orig_upb"] = np.log1p(df["ORIG_UPB"].fillna(0))

    # LTV * DTI interaction
    ltv_filled = df["ORIG_LTV"].fillna(df["ORIG_LTV"].median())
    dti_filled = df["ORIG_DTI"].fillna(df["ORIG_DTI"].median())
    df["ltv_dti_interaction"] = ltv_filled * dti_filled

    # Credit score / LTV ratio
    df["credit_ltv_ratio"] = np.where(
        (df["ORIG_LTV"].notna()) & (df["ORIG_LTV"] > 0) & df["CREDIT_SCORE"].notna(),
        df["CREDIT_SCORE"] / df["ORIG_LTV"],
        np.nan
    )

    # Estimated monthly payment
    r = df["ORIG_INTEREST_RATE"] / 1200.0
    n = df["ORIG_LOAN_TERM"]
    p = df["ORIG_UPB"]
    
    with np.errstate(divide="ignore", invalid="ignore"):
        pmt_num = p * r * np.power(1 + r, n)
        pmt_den = np.power(1 + r, n) - 1
        df["est_monthly_payment"] = np.where(
            (r > 0) & (pmt_den > 0) & p.notna(),
            pmt_num / pmt_den,
            np.nan
        )

    return df




def add_derived_origination_features(df: DataFrame) -> DataFrame:
    # Log transform
    df = df.withColumn("log_orig_upb", log1p(coalesce(col("ORIG_UPB"), lit(0))))

    # LTV * DTI interaction - use approximate quantiles for median
    # For large datasets, use approxQuantile
    stats = df.select(
        expr("percentile_approx(ORIG_LTV, 0.5)").alias("median_ltv"),
        expr("percentile_approx(ORIG_DTI, 0.5)").alias("median_dti")
    ).collect()
    
    median_ltv = stats[0]["median_ltv"] if stats[0]["median_ltv"] is not None else 80.0
    median_dti = stats[0]["median_dti"] if stats[0]["median_dti"] is not None else 36.0

    df = df.withColumn("ltv_filled", coalesce(col("ORIG_LTV"), lit(median_ltv)))
    df = df.withColumn("dti_filled", coalesce(col("ORIG_DTI"), lit(median_dti)))
    df = df.withColumn("ltv_dti_interaction", col("ltv_filled") * col("dti_filled"))
    df = df.drop("ltv_filled", "dti_filled")

    # Credit score / LTV ratio
    df = df.withColumn("credit_ltv_ratio",
        when(col("ORIG_LTV").isNotNull() & (col("ORIG_LTV") > 0) & col("CREDIT_SCORE").isNotNull(),
             col("CREDIT_SCORE") / col("ORIG_LTV"))
        .otherwise(lit(None)))

    # Estimated monthly payment
    df = df.withColumn("est_monthly_payment",
        when((col("ORIG_INTEREST_RATE") > 0) & (col("ORIG_LOAN_TERM") > 0) & col("ORIG_UPB").isNotNull(),
             col("ORIG_UPB") * (col("ORIG_INTEREST_RATE") / 1200) *
             expr("pow(1 + ORIG_INTEREST_RATE/1200, ORIG_LOAN_TERM)") /
             (expr("pow(1 + ORIG_INTEREST_RATE/1200, ORIG_LOAN_TERM)") - 1))
        .otherwise(lit(None)))

    return df


def add_vintage_features_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Pandas version of vintage features"""
    yr = df["orig_year"]
    
    df["econ_regime"] = "unknown"
    df.loc[yr.between(1999, 2003), "econ_regime"] = "expansion_dotcom"
    df.loc[yr.between(2004, 2007), "econ_regime"] = "bubble_peak"
    df.loc[yr.between(2008, 2009), "econ_regime"] = "gfc_crisis"
    df.loc[yr.between(2010, 2012), "econ_regime"] = "post_gfc"
    df.loc[yr.between(2013, 2019), "econ_regime"] = "recovery"
    
    df["is_bubble_vintage"] = yr.between(2004, 2007).astype(int)
    df["is_gfc_vintage"] = yr.between(2008, 2009).astype(int)
    df["is_post_gfc"] = (yr >= 2010).astype(int)
    
    return df


def add_missing_indicators(df: DataFrame) -> DataFrame:
    for col_name in ["CREDIT_SCORE", "ORIG_DTI", "ORIG_LTV", "ORIG_CLTV", "MI_PCT"]:
        if col_name in df.columns:
            df = df.withColumn(f"{col_name}_MISSING",
                when(col(col_name).isNull(), 1).otherwise(0).cast("int"))
    return df


def add_dynamic_features(df: DataFrame) -> DataFrame:
    df = df.withColumn("loan_age_months", col("LOAN_AGE").cast("double"))
    df = df.withColumn("loan_age_ratio",
        when(col("ORIG_LOAN_TERM") > 0, col("LOAN_AGE") / col("ORIG_LOAN_TERM"))
        .otherwise(lit(None)))

    df = df.withColumn("loan_age_bucket",
        when(col("LOAN_AGE") <= 12, "0_12m")
        .when(col("LOAN_AGE") <= 24, "12_24m")
        .when(col("LOAN_AGE") <= 36, "24_36m")
        .when(col("LOAN_AGE") <= 60, "36_60m")
        .when(col("LOAN_AGE") <= 120, "60_120m")
        .otherwise("120m_plus"))

    df = df.withColumn("pct_life_remaining",
        when(col("ORIG_LOAN_TERM") > 0, col("REM_MONTHS_MATURITY") / col("ORIG_LOAN_TERM"))
        .otherwise(lit(None)))

    df = df.withColumn("upb_pct_remaining",
        when(col("ORIG_UPB") > 0, col("CURR_ACTUAL_UPB") / col("ORIG_UPB"))
        .otherwise(lit(None)))

    df = df.withColumn("rate_spread", col("CURR_INTEREST_RATE") - col("ORIG_INTEREST_RATE"))
    df = df.withColumn("was_modified", coalesce(col("MODIFIED"), lit(0)).cast("int"))

    return df


def impute_features(df: DataFrame) -> DataFrame:
    """Fill nulls with -1 for numeric, UNKNOWN for categorical"""
    from config.settings import CFG

    for col_name in CFG.features.numeric_features:
        if col_name in df.columns:
            df = df.withColumn(col_name, coalesce(col(col_name).cast("double"), lit(-1.0)))

    for col_name in CFG.features.categorical_features:
        if col_name in df.columns:
            df = df.withColumn(col_name, coalesce(col(col_name).cast("string"), lit("UNKNOWN")))

    return df