from typing import Union, Optional
import logging

import pandas as pd
import numpy as np

def cumsum_to_previous(x: pd.Series) -> pd.Series:
    """
    Cumulative sum to the previous value in the group, assuming the dataframe is sorted in ascending order of timestamp.
    """
    return x.shift(periods=1).cumsum()

def bool_to_float(x: Optional[bool]) -> np.float:
    """
    Apply row-wise to turn a nullable bool series (pd.Series with dtype object)
    to a pd.Series with dtype float.
    """
    if pd.isna(x):
        return np.nan
    else:
        return float(x)

def get_logger() -> logging.Logger:
    """
    Get logger instance.
    """
    level = logging.INFO
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers = []
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
