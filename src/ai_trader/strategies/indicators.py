"""Shared technical indicators, as plain functions for use with Backtesting.py's self.I().

Each takes a sequence of values and returns a pandas Series/array of the same length
(leading values are NaN until the lookback window fills). No TA-Lib dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def SMA(values, n: int):
    """Simple moving average."""
    return pd.Series(values).rolling(n).mean()


def rolling_std(values, n: int):
    return pd.Series(values).rolling(n).std()


def RSI(values, n: int = 14):
    """Relative Strength Index (0-100). >70 = overbought, <30 = oversold (classic levels)."""
    s = pd.Series(values, dtype="float64")
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    # When there were no losses in the window, RSI is 100 by definition.
    return rsi.fillna(100).where(loss.notna(), other=np.nan)


def rolling_max(values, n: int):
    return pd.Series(values).rolling(n).max()


def rolling_min(values, n: int):
    return pd.Series(values).rolling(n).min()
