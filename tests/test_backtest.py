import numpy as np
import pandas as pd

from ai_trader.backtest import run_backtest
from ai_trader.strategies.momentum import MomentumCross


def _synthetic_prices(n=300, seed=42):
    """A gently trending random walk with OHLCV columns for Backtesting.py."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.01, n)  # slight upward drift
    close = 100 * np.exp(np.cumsum(steps))
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    vol = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx
    )


def test_run_backtest_returns_populated_result():
    prices = _synthetic_prices()
    result = run_backtest(prices, MomentumCross)

    assert result.num_trades >= 0
    assert isinstance(result.beats_benchmark, bool)
    # Benchmark comparison is wired up and self-consistent.
    assert result.beats_benchmark == (result.strategy_return_pct > result.buy_hold_return_pct)
    assert "Buy & hold" in result.summary()


def test_run_backtest_rejects_empty_frame():
    import pytest

    with pytest.raises(ValueError):
        run_backtest(pd.DataFrame(), MomentumCross)
