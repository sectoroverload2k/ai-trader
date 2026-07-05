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

    bt = Backtest(
        price_history,
        strategy,
        cash=cash,
        commission=commission,
        finalize_trades=True,  # close any open position at the end so stats are complete
    )
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


def split_by_fraction(
    df: pd.DataFrame, train_frac: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split. The test slice is your out-of-sample window.

    We never tune on the test slice — that's the whole point of holding it out.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    n = int(len(df) * train_frac)
    return df.iloc[:n], df.iloc[n:]


def compare_strategies(
    price_history: pd.DataFrame, strategies: dict[str, object], **kwargs
) -> dict[str, BacktestResult]:
    """Run several strategies on the same data and return their results by name.

    Every result already carries the buy-and-hold benchmark, so you can see at a
    glance whether any strategy actually beats simply holding the stock.
    """
    return {name: run_backtest(price_history, strat, **kwargs) for name, strat in strategies.items()}


@dataclass
class WalkForwardResult:
    strategy_name: str
    n_windows: int
    windows_beating_benchmark: int
    mean_strategy_return_pct: float
    mean_benchmark_return_pct: float
    mean_sharpe: float
    per_window: list  # list[BacktestResult]

    @property
    def beat_rate(self) -> float:
        """Fraction of windows where the strategy beat buy-and-hold (0.5 = coin flip)."""
        return self.windows_beating_benchmark / self.n_windows if self.n_windows else float("nan")

    def summary(self) -> str:
        return (
            f"{self.strategy_name}: beat buy-and-hold in "
            f"{self.windows_beating_benchmark}/{self.n_windows} windows "
            f"({self.beat_rate:.0%}); mean return {self.mean_strategy_return_pct:+.2f}% "
            f"vs {self.mean_benchmark_return_pct:+.2f}% benchmark; "
            f"mean Sharpe {self.mean_sharpe:.2f}"
        )


def walk_forward(
    price_history: pd.DataFrame,
    strategy,
    n_windows: int = 5,
    min_bars: int = 60,
    strategy_name: str | None = None,
    **kwargs,
) -> WalkForwardResult:
    """Evaluate a strategy across consecutive out-of-sample windows.

    The series is cut into `n_windows` contiguous chunks and the strategy is backtested
    on each independently. Reporting per-window results — and how often the strategy
    beats buy-and-hold — is far more honest than a single split: one good window is
    easy to cherry-pick, a strategy that wins most windows is harder to fake.

    Windows shorter than `min_bars` (too few bars for the indicators to warm up) are
    skipped rather than producing garbage.
    """
    name = strategy_name or getattr(strategy, "__name__", "strategy")
    n = len(price_history)
    size = n // n_windows if n_windows else n

    results: list[BacktestResult] = []
    for i in range(n_windows):
        start = i * size
        end = n if i == n_windows - 1 else (i + 1) * size
        window = price_history.iloc[start:end]
        if len(window) < min_bars:
            continue
        try:
            results.append(run_backtest(window, strategy, **kwargs))
        except Exception:
            # A window where the strategy can't run (e.g. no valid bars) is skipped,
            # not fatal — the other windows still tell us something.
            continue

    if not results:
        return WalkForwardResult(name, 0, 0, float("nan"), float("nan"), float("nan"), [])

    beating = sum(1 for r in results if r.beats_benchmark)
    mean = lambda xs: sum(xs) / len(xs)
    return WalkForwardResult(
        strategy_name=name,
        n_windows=len(results),
        windows_beating_benchmark=beating,
        mean_strategy_return_pct=mean([r.strategy_return_pct for r in results]),
        mean_benchmark_return_pct=mean([r.buy_hold_return_pct for r in results]),
        mean_sharpe=mean([r.sharpe for r in results]),
        per_window=results,
    )
