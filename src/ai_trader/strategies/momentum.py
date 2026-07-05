"""Baseline momentum strategy for Backtesting.py.

Deliberately simple and ML-free: a dual moving-average crossover. When the fast SMA
crosses above the slow SMA, go long; when it crosses below, close. This is the
honest baseline every fancier idea (including sentiment) must beat out-of-sample.
"""
from __future__ import annotations

from backtesting import Strategy
from backtesting.lib import crossover


def SMA(values, n):
    """Simple moving average over the last `n` values."""
    import pandas as pd

    return pd.Series(values).rolling(n).mean()


class MomentumCross(Strategy):
    fast = 20
    slow = 50

    def init(self):
        close = self.data.Close
        self.sma_fast = self.I(SMA, close, self.fast)
        self.sma_slow = self.I(SMA, close, self.slow)

    def next(self):
        if crossover(self.sma_fast, self.sma_slow):
            self.position.close()
            self.buy()
        elif crossover(self.sma_slow, self.sma_fast):
            self.position.close()
