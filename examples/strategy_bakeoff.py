"""Bake-off: run every strategy through walk-forward and rank them honestly.

    python examples/strategy_bakeoff.py

For each strategy we report how often it beat buy-and-hold across several out-of-sample
windows. The ranking key is that beat-rate, NOT raw return — a strategy that wins one
window and loses four is worse than one that quietly wins three. Expect buy-and-hold to
be hard to beat; that's the honest, well-documented reality, not a bug.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_trader.backtest import walk_forward
from ai_trader.config import load_settings
from ai_trader.data import get_price_history
from ai_trader.strategies.breakout import DonchianBreakout
from ai_trader.strategies.buy_and_hold import BuyAndHold
from ai_trader.strategies.mean_reversion import BollingerMeanReversion, RSIMeanReversion
from ai_trader.strategies.momentum import MomentumCross

STRATEGIES = {
    "Buy & Hold": BuyAndHold,
    "Momentum (SMA cross)": MomentumCross,
    "Donchian breakout": DonchianBreakout,
    "RSI mean reversion": RSIMeanReversion,
    "Bollinger mean reversion": BollingerMeanReversion,
}
SYMBOL = "SPY"
START = datetime(2018, 1, 1, tzinfo=timezone.utc)
N_WINDOWS = 6


def main() -> int:
    settings = load_settings()
    if not settings.is_configured:
        print("No Alpaca keys found. See README for setup.")
        return 1

    print(f"Fetching {SYMBOL} daily bars ...")
    prices = get_price_history(SYMBOL, start=START)
    print(f"  {len(prices)} bars, split into {N_WINDOWS} walk-forward windows\n")

    rows = []
    for name, strat in STRATEGIES.items():
        wf = walk_forward(prices, strat, n_windows=N_WINDOWS, strategy_name=name)
        rows.append(wf)

    # Rank by beat-rate, then by mean return over benchmark.
    rows.sort(
        key=lambda r: (r.beat_rate, r.mean_strategy_return_pct - r.mean_benchmark_return_pct),
        reverse=True,
    )

    print(f"{'Strategy':26s} {'Beat B&H':>10s} {'Mean ret':>10s} {'Benchmark':>10s} {'Sharpe':>7s}")
    print("-" * 68)
    for r in rows:
        print(
            f"{r.strategy_name:26s} "
            f"{r.windows_beating_benchmark}/{r.n_windows:>1d} ({r.beat_rate:>3.0%}) "
            f"{r.mean_strategy_return_pct:>9.2f}% "
            f"{r.mean_benchmark_return_pct:>9.2f}% "
            f"{r.mean_sharpe:>7.2f}"
        )

    print(
        "\nRanked by how often each beat buy-and-hold across windows. If buy-and-hold "
        "tops the table, that IS the finding — most active strategies don't beat it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
