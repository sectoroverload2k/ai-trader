import numpy as np
import pandas as pd

from ai_trader.strategies.indicators import RSI, SMA, rolling_max, rolling_min


def test_sma_matches_manual_average():
    s = SMA([1, 2, 3, 4, 5], 3)
    assert np.isnan(s.iloc[1])           # not enough data yet
    assert s.iloc[2] == 2.0              # (1+2+3)/3
    assert s.iloc[4] == 4.0              # (3+4+5)/3


def test_rsi_stays_within_bounds():
    rng = np.random.default_rng(0)
    prices = 100 + np.cumsum(rng.normal(0, 1, 200))
    rsi = RSI(prices, 14).dropna()
    assert rsi.between(0, 100).all()


def test_rsi_is_100_on_monotonic_rise():
    # Straight uptrend => no losses => RSI pinned near 100.
    rsi = RSI(list(range(1, 50)), 14).dropna()
    assert (rsi > 99).all()


def test_rolling_max_min():
    vals = [3, 1, 4, 1, 5, 9, 2]
    assert rolling_max(vals, 3).iloc[5] == 9
    assert rolling_min(vals, 3).iloc[3] == 1
