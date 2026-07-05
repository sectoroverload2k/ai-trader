"""Measure whether news sentiment predicts the next day's move, for a few tickers.

    pip install -r requirements.txt -r requirements-sentiment.txt
    python examples/sentiment_experiment.py

Requires Alpaca keys (prices + news) and FinBERT (torch). This is the honest test of
the project's core question — expect a weak/near-zero relationship, and read the
hit-rate as "50% = no edge."
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_trader.config import load_settings
from ai_trader.data import get_news, get_price_history
from ai_trader.experiment import build_dataset, daily_sentiment, evaluate, format_report

SYMBOLS = ["AAPL", "TSLA", "NVDA"]
START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def main() -> int:
    settings = load_settings()
    if not settings.is_configured:
        print("No Alpaca keys found. See README for setup.")
        return 1

    for symbol in SYMBOLS:
        print(f"\nFetching data for {symbol} ...")
        prices = get_price_history(symbol, start=START)
        # Alpaca News API caps at 50 per call; page a bit for more coverage.
        news = get_news(symbol, start=START, limit=50)
        print(f"  {len(prices)} price days, {len(news)} news items. Scoring sentiment ...")

        sent = daily_sentiment(news)             # FinBERT scores each item
        dataset = build_dataset(prices, sent)
        stats = evaluate(dataset)
        print(format_report(symbol, stats))

    print(
        "Reminder: near-zero correlation and ~50% hit-rate is the expected, honest "
        "result. A big number here almost always means look-ahead or too little data."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
