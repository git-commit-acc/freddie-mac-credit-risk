"""ingestion/spark_loader.py - Fixed Windows path handling"""

import os
import re
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, lit, trim, upper, substring, length, coalesce
)
from pyspark.sql.types import DoubleType, IntegerType, ShortType

from config.settings import ORIG_COLS_IDX, ORIG_COLS_NAMES, SVCG_COLS_IDX, SVCG_COLS_NAMES, ORIG_SENTINELS


def get_spark_session(app_name="Freddie_Mac_Pipeline"):
    """Create Spark session for local Windows filesystem"""
    from pyspark.sql import SparkSession
    
    return SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "2g") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .config("spark.hadoop.fs.defaultFS", "file:///") \
        .getOrCreate()


def to_spark_path(path: str) -> str:
    """Convert Windows path to Spark-compatible file:/// format"""
    # Replace backslashes with forward slashes
    path = path.replace("\\", "/")
    # Remove any existing file:// prefix
    path = re.sub(r'^file:/+', '', path)
    # Add file:/// prefix
    return f"file:///{path}"


def discover_sample_files(raw_dir: str, year_start: int, year_end: int):
    """Discover sample files in directory"""
    orig_paths = []
    svcg_paths = []
    
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    
    for fname in sorted(os.listdir(raw_dir)):
        m = re.match(r"sample_orig_(\d{4})\.txt$", fname, re.IGNORECASE)
        if m and year_start <= int(m.group(1)) <= year_end:
            orig_paths.append(os.path.join(raw_dir, fname))
        
        m = re.match(r"sample_svcg_(\d{4})\.txt$", fname, re.IGNORECASE)
        if m and year_start <= int(m.group(1)) <= year_end:
            svcg_paths.append(os.path.join(raw_dir, fname))
    
    return orig_paths, svcg_paths


def load_origination(spark, raw_dir: str, year_start: int, year_end: int) -> DataFrame:
    """Load origination files - Windows compatible"""
    orig_paths, _ = discover_sample_files(raw_dir, year_start, year_end)
    
    if not orig_paths:
        raise FileNotFoundError(f"No origination files found for years {year_start}-{year_end}")
    
    # Convert Windows paths to Spark format
    spark_paths = [to_spark_path(p) for p in orig_paths]
    
    print(f"Loading files: {spark_paths}")  # Debug output
    
    # Read all files
    df = spark.read \
        .option("sep", "|") \
        .option("header", "false") \
        .option("mode", "PERMISSIVE") \
        .option("nullValue", "") \
        .csv(spark_paths)
    
    # Select needed columns by position
    cols_to_select = [f"_c{i}" for i in range(len(df.columns))]
    selected_cols = [cols_to_select[i] for i in ORIG_COLS_IDX]
    df = df.select(*selected_cols)
    
    # Rename columns
    for old, new in zip(selected_cols, ORIG_COLS_NAMES):
        df = df.withColumnRenamed(old, new)
    
    return df


def load_servicing(spark, raw_dir: str, year_start: int, year_end: int) -> DataFrame:
    """Load servicing files - Windows compatible"""
    _, svcg_paths = discover_sample_files(raw_dir, year_start, year_end)
    
    if not svcg_paths:
        raise FileNotFoundError(f"No servicing files found for years {year_start}-{year_end}")
    
    # Convert Windows paths to Spark format
    spark_paths = [to_spark_path(p) for p in svcg_paths]
    
    print(f"Loading servicing files: {len(spark_paths)} files")  # Debug output
    
    df = spark.read \
        .option("sep", "|") \
        .option("header", "false") \
        .option("mode", "PERMISSIVE") \
        .option("nullValue", "") \
        .csv(spark_paths)
    
    cols_to_select = [f"_c{i}" for i in range(len(df.columns))]
    selected_cols = [cols_to_select[i] for i in SVCG_COLS_IDX]
    df = df.select(*selected_cols)
    
    for old, new in zip(selected_cols, SVCG_COLS_NAMES):
        df = df.withColumnRenamed(old, new)
    
    return df


def clean_origination(df: DataFrame) -> DataFrame:
    """Clean origination data"""
    
    # Convert to numeric types with proper null handling
    numeric_cols = ["CREDIT_SCORE", "ORIG_CLTV", "ORIG_DTI", "ORIG_LOAN_TERM", 
                    "MI_PCT", "ORIG_UPB", "ORIG_LTV", "ORIG_INTEREST_RATE"]
    
    for c in numeric_cols:
        if c in df.columns:
            df = df.withColumn(c, when(col(c) == "", None).otherwise(col(c)))
            df = df.withColumn(c, col(c).cast(DoubleType()))
    
    # Replace sentinels with NULL
    for col_name, sentinels in ORIG_SENTINELS.items():
        if col_name in df.columns:
            for sentinel in sentinels:
                df = df.withColumn(col_name, 
                    when(col(col_name) == sentinel, None).otherwise(col(col_name)))
    
    # Parse origination year from FIRST_PAYMENT_DATE
    df = df.withColumn("FIRST_PAYMENT_DATE", trim(col("FIRST_PAYMENT_DATE")))
    df = df.withColumn("orig_year", 
        when(length(col("FIRST_PAYMENT_DATE")) >= 4,
             substring(col("FIRST_PAYMENT_DATE"), 1, 4).cast(IntegerType()))
        .otherwise(None))
    
    df = df.withColumn("orig_month",
        when(length(col("FIRST_PAYMENT_DATE")) >= 6,
             substring(col("FIRST_PAYMENT_DATE"), 5, 2).cast(IntegerType()))
        .otherwise(None))
    
    # Clean string columns
    df = df.withColumn("PPM_FLAG", trim(upper(col("PPM_FLAG"))))
    df = df.withColumn("PROPERTY_STATE", trim(upper(col("PROPERTY_STATE"))))
    df = df.withColumn("LOAN_SEQUENCE_NUMBER", trim(col("LOAN_SEQUENCE_NUMBER")))
    df = df.withColumn("SERVICER_NAME", trim(col("SERVICER_NAME")))
    
    # Filter out null loan IDs
    df = df.filter(col("LOAN_SEQUENCE_NUMBER").isNotNull() & (col("LOAN_SEQUENCE_NUMBER") != ""))
    
    # Apply valid ranges
    df = df.withColumn("CREDIT_SCORE", 
        when(col("CREDIT_SCORE").between(300, 850), col("CREDIT_SCORE")).otherwise(None))
    df = df.withColumn("MI_PCT",
        when(col("MI_PCT").between(0, 55), col("MI_PCT")).otherwise(None))
    df = df.withColumn("ORIG_CLTV",
        when(col("ORIG_CLTV").between(1, 998), col("ORIG_CLTV")).otherwise(None))
    df = df.withColumn("ORIG_DTI",
        when(col("ORIG_DTI").between(0, 65), col("ORIG_DTI")).otherwise(None))
    df = df.withColumn("ORIG_LTV",
        when(col("ORIG_LTV").between(1, 998), col("ORIG_LTV")).otherwise(None))
    
    return df


def clean_servicing(df: DataFrame) -> DataFrame:
    """Clean servicing data"""
    
    # Trim all string columns
    df = df.withColumn("LOAN_SEQUENCE_NUMBER", trim(col("LOAN_SEQUENCE_NUMBER")))
    df = df.withColumn("MONTHLY_RPT_PRD", trim(col("MONTHLY_RPT_PRD")))
    df = df.withColumn("DLQCY_STATUS", trim(col("DLQCY_STATUS")))
    df = df.withColumn("MOD_FLAG", trim(upper(col("MOD_FLAG"))))
    df = df.withColumn("ZERO_BAL_CD", trim(col("ZERO_BAL_CD")))
    
    # Parse reporting period
    df = df.withColumn("rpt_year",
        when(length(col("MONTHLY_RPT_PRD")) >= 4,
             substring(col("MONTHLY_RPT_PRD"), 1, 4).cast(IntegerType()))
        .otherwise(None))
    
    df = df.withColumn("rpt_month",
        when(length(col("MONTHLY_RPT_PRD")) >= 6,
             substring(col("MONTHLY_RPT_PRD"), 5, 2).cast(IntegerType()))
        .otherwise(None))
    
    # Create month index
    df = df.withColumn("month_idx", 
        when(col("rpt_year").isNotNull() & col("rpt_month").isNotNull(),
             col("rpt_year") * 12 + col("rpt_month"))
        .otherwise(None))
    
    # Numeric conversions
    numeric_cols = ["CURR_ACTUAL_UPB", "LOAN_AGE", "REM_MONTHS_MATURITY", "CURR_INTEREST_RATE"]
    for c in numeric_cols:
        if c in df.columns:
            df = df.withColumn(c, when(col(c) == "", None).otherwise(col(c)))
            df = df.withColumn(c, col(c).cast(DoubleType()))
    
    # Convert RA to 90 for numeric delinquency
    df = df.withColumn("DLQCY_STATUS_NUM",
        when(col("DLQCY_STATUS") == "RA", 90.0)
        .when(col("DLQCY_STATUS").rlike("^\\d+$"), col("DLQCY_STATUS").cast(DoubleType()))
        .otherwise(None))
    
    # Modification flag
    df = df.withColumn("MODIFIED", 
        when(col("MOD_FLAG").isin("Y", "P"), 1).otherwise(0).cast(ShortType()))
    
    # Clean zero balance codes
    df = df.withColumn("ZERO_BAL_CD",
        when(col("ZERO_BAL_CD").isin("", "nan", "None", "null"), lit(None))
        .otherwise(col("ZERO_BAL_CD")))
    
    # Filter null loan IDs
    df = df.filter(col("LOAN_SEQUENCE_NUMBER").isNotNull() & (col("LOAN_SEQUENCE_NUMBER") != ""))
    
    return df

def save_dataframe(df: DataFrame, path: str, mode="overwrite"):
    """Save DataFrame to Parquet - ensures directory exists"""
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    spark_path = to_spark_path(path)
    df.write.mode(mode).parquet(spark_path)


def load_dataframe(spark, path: str):
    """Load DataFrame from Parquet - reads from directory"""
    # Check if path is a directory (Spark saves as directory with part files)
    if os.path.exists(path) and os.path.isdir(path):
        spark_path = to_spark_path(path)
        return spark.read.parquet(spark_path)
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")