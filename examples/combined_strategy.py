"""Does adding news sentiment to momentum beat price-alone — out-of-sample?

    pip install -r requirements.txt -r requirements-sentiment.txt
    python examples/combined_strategy.py

Pulls paginated news + prices, scores sentiment, then backtests THREE things on a
held-out test window: buy & hold, price-only momentum, and sentiment-gated momentum.
The honest question is whether the sentiment gate beats price-only on data it never
saw. Usually it doesn't by much — and that's a real answer.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_trader.backtest import compare_strategies, split_by_fraction
from ai_trader.config import load_settings
from ai_trader.data import get_all_news, get_price_history
from ai_trader.experiment import attach_sentiment, daily_sentiment
from ai_trader.strategies.momentum import MomentumCross
from ai_trader.strategies.sentiment_momentum import SentimentMomentum

SYMBOL = "AAPL"
START = datetime(2022, 1, 1, tzinfo=timezone.utc)


def main() -> int:
    settings = load_settings()
    if not settings.is_configured:
        print("No Alpaca keys found. See README for setup.")
        return 1

    print(f"Fetching prices + news for {SYMBOL} ...")
    prices = get_price_history(SYMBOL, start=START)
    news = get_all_news(SYMBOL, start=START, max_items=1500)
    print(f"  {len(prices)} price days, {len(news)} news items. Scoring sentiment ...")

    sent = daily_sentiment(news)
    prices = attach_sentiment(prices, sent)

    # Hold out the most recent 30% as out-of-sample. We tune nothing on it.
    _, test = split_by_fraction(prices, train_frac=0.7)
    print(f"  out-of-sample window: {len(test)} days\n")

    results = compare_strategies(
        test,
        {
            "Price-only momentum": MomentumCross,
            "Sentiment-gated momentum": SentimentMomentum,
        },
    )

    bh = next(iter(results.values())).buy_hold_return_pct
    print(f"Buy & hold (benchmark):      {bh:7.2f}%\n")
    for name, r in results.items():
        edge = r.strategy_return_pct - bh
        verdict = "beats" if edge > 0 else "loses to"
        print(
            f"{name:28s} {r.strategy_return_pct:7.2f}%  "
            f"(Sharpe {r.sharpe:4.2f}, {r.num_trades} trades, {verdict} benchmark by {abs(edge):.2f} pts)"
        )

    print(
        "\nRead this honestly: a tiny difference between the two strategies is noise, "
        "not signal. One held-out window is a hint, not proof — walk-forward across many "
        "windows is the next rigor step."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
