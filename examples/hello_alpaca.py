"""Smoke test: connect to your Alpaca PAPER account, fetch prices + news, print them.

Run this first to confirm your keys work end-to-end:

    pip install -r requirements.txt
    cp .env.example .env      # then edit .env with your paper keys
    python examples/hello_alpaca.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make `src/` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_trader.config import load_settings
from ai_trader.data import get_news, get_price_history

SYMBOL = "AAPL"


def main() -> int:
    settings = load_settings()
    if not settings.is_configured:
        print("No Alpaca keys found. Copy .env.example to .env and add your paper keys.")
        return 1

    print(f"Mode: {'PAPER' if settings.paper else 'LIVE'} (should be PAPER)\n")

    start = datetime.now(timezone.utc) - timedelta(days=30)

    print(f"Last few daily bars for {SYMBOL}:")
    bars = get_price_history(SYMBOL, start=start)
    print(bars.tail(5).to_string())

    print(f"\nRecent news for {SYMBOL}:")
    for item in get_news(SYMBOL, start=start, limit=5):
        print(f"  [{item['timestamp']}] {item['headline']}")

    print("\nOK — connection works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
