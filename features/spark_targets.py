"""features/spark_targets.py - Target construction with window functions"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, max as spark_max, sum as spark_sum, lag, lead, lit, coalesce, 
    row_number, min as spark_min, count
)
from pyspark.sql.window import Window
from config.settings import SERIOUS_DELINQ_CODES, LOSS_ZBC, PREPAY_ZBC


def add_event_flags(df: DataFrame) -> DataFrame:
    df = df.withColumn("IS_90DPD", when(col("DLQCY_STATUS").isin(SERIOUS_DELINQ_CODES), 1).otherwise(0))
    df = df.withColumn("IS_ZBC_LOSS", when(col("ZERO_BAL_CD").isin(LOSS_ZBC), 1).otherwise(0))
    df = df.withColumn("IS_PREPAY", when(col("ZERO_BAL_CD") == PREPAY_ZBC, 1).otherwise(0))
    df = df.withColumn("IS_DEFAULT_EVENT", (col("IS_90DPD") + col("IS_ZBC_LOSS") > 0).cast("int"))
    return df


def add_rolling_delinquency_features(df: DataFrame, windows=(3, 6, 12)) -> DataFrame:
    """Window-based rolling features - efficient in Spark"""
    
    dlq_num = col("DLQCY_STATUS_NUM").cast("double")
    
    for k in windows:
        # Define window for this specific k
        w = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("month_idx").rowsBetween(-k + 1, 0)
        
        df = df.withColumn(f"max_delinq_{k}m", spark_max(coalesce(dlq_num, lit(0))).over(w))
        df = df.withColumn(f"mean_delinq_{k}m", 
                          (spark_sum(coalesce(dlq_num, lit(0))).over(w) / lit(k)).cast("double"))
        df = df.withColumn(f"n_delinq_months_{k}m",
            spark_sum(when(col("DLQCY_STATUS_NUM") > 0, 1).otherwise(0)).over(w))

        if k >= 6:
            df = df.withColumn(f"n_serious_delinq_{k}m",
                spark_sum(col("IS_90DPD")).over(w))
    
    # Payment streak - count consecutive current months
    df = df.withColumn("_is_current", when(coalesce(dlq_num, lit(0)) == 0, 1).otherwise(0))
    
    # Use row number difference method for streak calculation
    w_streak = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("month_idx")
    df = df.withColumn("_row_num", row_number().over(w_streak))
    df = df.withColumn("_delinq_row", 
        when(col("_is_current") == 0, col("_row_num")).otherwise(lit(None)))
    df = df.withColumn("_last_delinq", 
        spark_max("_delinq_row").over(w_streak.rowsBetween(Window.unboundedPreceding, 0)))
    df = df.withColumn("payment_streak", 
        when(col("_last_delinq").isNull(), col("_row_num"))
        .otherwise(col("_row_num") - col("_last_delinq")))
    
    df = df.drop("_is_current", "_row_num", "_delinq_row", "_last_delinq")
    
    # Delinquency trend: current level vs 5 months ago
    w_lag = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("month_idx")
    df = df.withColumn("_lag5", lag(coalesce(dlq_num, lit(0)), 5).over(w_lag))
    df = df.withColumn("delinq_trend_6m", coalesce(dlq_num, lit(0)) - coalesce(col("_lag5"), lit(0)))
    df = df.drop("_lag5")
    
    return df


def build_target_12m(df: DataFrame, window_months: int = 12) -> DataFrame:
    """Forward-looking target using window max over future rows"""
    
    # Sort within each loan
    w = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy(col("month_idx").desc())
    
    # Look ahead by taking max of next N rows (reverse order trick)
    # First, create reverse order index
    df = df.withColumn("_rev_idx", row_number().over(w))
    
    # Define forward window (looking at rows with smaller rev_idx = future)
    w_fwd = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("_rev_idx").rowsBetween(0, window_months)
    
    # Max of default event in future window
    df = df.withColumn("_next_default", 
        spark_max(col("IS_DEFAULT_EVENT")).over(w_fwd))
    
    # Check if prepay happens within window
    df = df.withColumn("_fwd_prepay", 
        spark_max(col("IS_PREPAY")).over(w_fwd))
    
    # Count rows in window to know if we have enough future data
    df = df.withColumn("_fwd_count", 
        count("*").over(w_fwd))
    
    # Assign target
    df = df.withColumn("TARGET_12M",
        when(col("_next_default") == 1, 1)
        .when((col("_fwd_prepay") == 1) & (col("_fwd_count") < window_months), lit(None))
        .otherwise(0))
    
    return df.drop("_rev_idx", "_next_default", "_fwd_prepay", "_fwd_count")


def select_snapshot(df: DataFrame, min_obs_months: int = 6) -> DataFrame:
    """Select latest row per loan with sufficient history"""
    
    # First, ensure we have enough history by looking at row number
    w = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("month_idx")
    df = df.withColumn("row_num", row_number().over(w))
    df = df.filter(col("row_num") > min_obs_months)
    
    # Keep latest row per loan (highest month_idx)
    w2 = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy(col("month_idx").desc())
    df = df.withColumn("rn", row_number().over(w2))
    df = df.filter(col("rn") == 1).drop("rn", "row_num")
    
    return df