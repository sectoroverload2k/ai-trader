import numpy as np
import pandas as pd
import pytest

from ai_trader.backtest import run_backtest
from ai_trader.strategies.breakout import DonchianBreakout
from ai_trader.strategies.buy_and_hold import BuyAndHold
from ai_trader.strategies.mean_reversion import BollingerMeanReversion, RSIMeanReversion
from ai_trader.strategies.momentum import MomentumCross

ALL_STRATEGIES = [
    BuyAndHold,
    MomentumCross,
    DonchianBreakout,
    RSIMeanReversion,
    BollingerMeanReversion,
]


def _prices(n=400, seed=0, drift=0.0003, vol=0.012):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.002, n)),
            "High": close * (1 + rng.uniform(0, 0.01, n)),
            "Low": close * (1 - rng.uniform(0, 0.01, n)),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=idx,
    )


@pytest.mark.parametrize("strat", ALL_STRATEGIES)
def test_every_strategy_runs_and_reports_benchmark(strat):
    result = run_backtest(_prices(), strat)
    assert result.num_trades >= 0
    assert isinstance(result.beats_benchmark, bool)
    # The benchmark comparison must be internally consistent.
    assert result.beats_benchmark == (result.strategy_return_pct > result.buy_hold_return_pct)


def test_buy_and_hold_tracks_the_benchmark():
    # A buy-and-hold strategy should land very close to the buy & hold return.
    result = run_backtest(_prices(), BuyAndHold)
    assert result.num_trades <= 1
    assert abs(result.strategy_return_pct - result.buy_hold_return_pct) < 5.0


def test_mean_reversion_trades_on_a_mean_reverting_series():
    # Oscillating (not trending) series so the RSI strategy actually finds dips.
    n = 400
    t = np.arange(n)
    close = 100 + 10 * np.sin(t / 8.0)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close,
         "Volume": np.full(n, 1_000_000)},
        index=idx,
    )
    result = run_backtest(prices, RSIMeanReversion)
    assert result.num_trades > 0  # oscillation should trigger oversold entries
