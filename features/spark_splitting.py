"""features/spark_splitting.py - Data splits"""

from pyspark.sql import DataFrame
import pandas as pd
import numpy as np
from pyspark.sql.functions import col, rand, when, hash as spark_hash, lit
from pyspark.sql.types import IntegerType


def split_oos(df: DataFrame, target_col: str = "TARGET_12M",
              train_ratio: float = 0.70, val_ratio: float = 0.15,
              seed: int = 42) -> tuple:
    """Random split using hash of loan ID"""
    df = df.filter(col(target_col).isNotNull())

    # Use hash of loan sequence number for deterministic split
    df = df.withColumn("_split_bucket",
        (spark_hash(col("LOAN_SEQUENCE_NUMBER")) % lit(100)).cast(IntegerType()))

    train_cut = int(train_ratio * 100)
    val_cut = int((train_ratio + val_ratio) * 100)

    train = df.filter(col("_split_bucket") < train_cut).drop("_split_bucket")
    val = df.filter((col("_split_bucket") >= train_cut) & (col("_split_bucket") < val_cut)).drop("_split_bucket")
    test = df.filter(col("_split_bucket") >= val_cut).drop("_split_bucket")

    return train, val, test


def split_oot(df: DataFrame, target_col: str = "TARGET_12M",
              train_year_range=(1999, 2007), test_year_range=(2008, 2012)) -> tuple:
    """Time-based split"""
    df = df.filter(col(target_col).isNotNull())

    train = df.filter(col("orig_year").between(train_year_range[0], train_year_range[1]))
    test = df.filter(col("orig_year").between(test_year_range[0], test_year_range[1]))

    return train, test

def create_splits(df: DataFrame, config=None):
    """Create both splits"""
    from config.settings import CFG
    cfg = (config or CFG).splits

    df_clean = df.filter(col("TARGET_12M").isNotNull())

    splits = {}
    oos_train, oos_val, oos_test = split_oos(
        df_clean, train_ratio=cfg.oos_train_ratio,
        val_ratio=cfg.oos_val_ratio, seed=cfg.oos_seed)
    splits["oos"] = (oos_train, oos_val, oos_test)

    oot_train, oot_test = split_oot(df_clean,
        train_year_range=cfg.oot_train_years, test_year_range=cfg.oot_test_years)
    splits["oot"] = (oot_train, None, oot_test)

    return splits

def create_splits_pandas(df: pd.DataFrame, config=None):
    """Create both splits using pandas"""
    from config.settings import CFG
    cfg = (config or CFG).splits

    df_clean = df[df["TARGET_12M"].notna()].copy()

    splits = {}
    
    # OOS split
    np.random.seed(cfg.oos_seed)
    buckets = df_clean["LOAN_SEQUENCE_NUMBER"].apply(lambda x: hash(str(x) + str(cfg.oos_seed)) % 100)
    
    train_cut = int(cfg.oos_train_ratio * 100)
    val_cut = int((cfg.oos_train_ratio + cfg.oos_val_ratio) * 100)
    
    oos_train = df_clean[buckets < train_cut].reset_index(drop=True)
    oos_val = df_clean[(buckets >= train_cut) & (buckets < val_cut)].reset_index(drop=True)
    oos_test = df_clean[buckets >= val_cut].reset_index(drop=True)
    splits["oos"] = (oos_train, oos_val, oos_test)
    
    # OOT split
    oot_train = df_clean[df_clean["orig_year"].between(*cfg.oot_train_years)].reset_index(drop=True)
    oot_test = df_clean[df_clean["orig_year"].between(*cfg.oot_test_years)].reset_index(drop=True)
    splits["oot"] = (oot_train, None, oot_test)
    
    return splits