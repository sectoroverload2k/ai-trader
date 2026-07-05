import numpy as np
import pandas as pd

from ai_trader.backtest import walk_forward
from ai_trader.strategies.buy_and_hold import BuyAndHold
from ai_trader.strategies.momentum import MomentumCross


def _prices(n=600, seed=5):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * (1 + rng.uniform(0, 0.01, n)),
            "Low": close * (1 - rng.uniform(0, 0.01, n)),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=idx,
    )


def test_walk_forward_produces_per_window_results():
    wf = walk_forward(_prices(), MomentumCross, n_windows=5, strategy_name="mom")
    assert wf.strategy_name == "mom"
    assert wf.n_windows == 5
    assert len(wf.per_window) == 5
    assert 0.0 <= wf.beat_rate <= 1.0
    assert wf.windows_beating_benchmark <= wf.n_windows


def test_walk_forward_skips_windows_too_small_for_indicators():
    # 100 bars into 5 windows = 20 bars each, all below min_bars=60 -> nothing runs.
    wf = walk_forward(_prices(n=100), MomentumCross, n_windows=5, min_bars=60)
    assert wf.n_windows == 0
    assert wf.per_window == []


def test_buy_and_hold_return_tracks_the_benchmark():
    # A buy-and-hold strategy's mean return should sit very close to the benchmark's
    # (small gap from entry timing + commission), never wildly above or below it.
    wf = walk_forward(_prices(), BuyAndHold, n_windows=5)
    assert abs(wf.mean_strategy_return_pct - wf.mean_benchmark_return_pct) < 5.0
    assert "beat buy-and-hold" in wf.summary()
