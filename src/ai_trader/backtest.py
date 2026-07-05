"""Honest-measurement backtest harness.

Wraps Backtesting.py and, crucially, ALWAYS reports the strategy against a
buy-and-hold benchmark. A strategy that "makes money" but underperforms simply
buying and holding the same stock has no edge — the benchmark is the whole point.

We model a small commission/slippage cost even though Alpaca is commission-free,
because real fills suffer spread + slippage and ignoring that is the #1 way
backtests lie.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from backtesting import Backtest


@dataclass
class BacktestResult:
    strategy_return_pct: float
    buy_hold_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    num_trades: int
    beats_benchmark: bool
    raw: object  # full Backtesting.py stats Series, for deeper inspection

    def summary(self) -> str:
        verdict = "BEATS" if self.beats_benchmark else "does NOT beat"
        return (
            f"Strategy return:   {self.strategy_return_pct:6.2f}%\n"
            f"Buy & hold return: {self.buy_hold_return_pct:6.2f}%\n"
            f"Sharpe ratio:      {self.sharpe:6.2f}\n"
            f"Max drawdown:      {self.max_drawdown_pct:6.2f}%\n"
            f"Trades:            {self.num_trades}\n"
            f"=> Strategy {verdict} buy-and-hold benchmark."
        )


def run_backtest(
    price_history: pd.DataFrame,
    strategy,
    cash: float = 10_000,
    commission: float = 0.001,  # 0.1% to proxy spread/slippage, NOT Alpaca fees
) -> BacktestResult:
    """Backtest `strategy` on `price_history` (OHLCV) and compare to buy-and-hold."""
    if price_history.empty:
        raise ValueError("price_history is empty — fetch data first.")

    bt = Backtest(price_history, strategy, cash=cash, commission=commission)
    stats = bt.run()

    strat_ret = float(stats["Return [%]"])
    bh_ret = float(stats["Buy & Hold Return [%]"])

    return BacktestResult(
        strategy_return_pct=strat_ret,
        buy_hold_return_pct=bh_ret,
        sharpe=float(stats["Sharpe Ratio"]),
        max_drawdown_pct=float(stats["Max. Drawdown [%]"]),
        num_trades=int(stats["# Trades"]),
        beats_benchmark=strat_ret > bh_ret,
        raw=stats,
    )
