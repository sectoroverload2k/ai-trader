"""Backtest the momentum baseline and compare it to buy-and-hold.

    python examples/run_backtest.py

Requires Alpaca keys (for the historical price data). The whole point of the output
is the last line: does the strategy actually beat just buying and holding?
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_trader.backtest import run_backtest
from ai_trader.config import load_settings
from ai_trader.data import get_price_history
from ai_trader.strategies.momentum import MomentumCross

SYMBOL = "SPY"


def main() -> int:
    settings = load_settings()
    if not settings.is_configured:
        print("No Alpaca keys found. Copy .env.example to .env and add your paper keys.")
        return 1

    # ~5 years of daily bars.
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    bars = get_price_history(SYMBOL, start=start)
    print(f"Fetched {len(bars)} daily bars for {SYMBOL}\n")

    result = run_backtest(bars, MomentumCross)
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
