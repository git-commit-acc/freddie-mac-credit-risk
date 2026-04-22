"""utils/io_utils.py - File I/O helpers"""

import logging
import os
import pickle
import pandas as pd

logger = logging.getLogger(__name__)


def save_dataframe(df: pd.DataFrame, path: str, fmt: str = "parquet"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False, compression="snappy")
    elif fmt == "csv":
        df.to_csv(path, index=False)
    logger.info("Saved %d rows -> %s", len(df), path)


def load_dataframe(path: str, fmt: str = "parquet") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if fmt == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)