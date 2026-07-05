"""Data layer: historical price bars and news, fetched from Alpaca.

Uses the official `alpaca-py` SDK. Imports are done lazily inside functions so the
rest of the package (config, strategy logic, backtests on cached data) can be used
without the SDK installed or credentials configured.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .config import Settings, load_settings


def _require_configured(settings: Settings) -> None:
    if not settings.is_configured:
        raise RuntimeError(
            "Alpaca API keys are not set. Copy .env.example to .env and fill in your "
            "PAPER-trading keys from https://alpaca.markets/ (see README)."
        )


def get_price_history(
    symbol: str,
    start: datetime,
    end: datetime | None = None,
    timeframe: str = "1Day",
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of OHLCV bars indexed by timestamp.

    Columns: Open, High, Low, Close, Volume  (capitalized for Backtesting.py).
    """
    settings = settings or load_settings()
    _require_configured(settings)

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tf_map = {
        "1Day": TimeFrame.Day,
        "1Hour": TimeFrame.Hour,
        "1Min": TimeFrame.Minute,
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    }
    tf = tf_map.get(timeframe, TimeFrame.Day)

    client = StockHistoricalDataClient(settings.api_key, settings.secret_key)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start,
        end=end or datetime.now(timezone.utc),
    )
    bars = client.get_stock_bars(req).df
    if bars.empty:
        return bars

    # get_stock_bars returns a multi-index (symbol, timestamp); flatten to one symbol.
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")

    bars = bars.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    return bars[["Open", "High", "Low", "Close", "Volume"]]


def get_news(
    symbol: str,
    start: datetime,
    end: datetime | None = None,
    limit: int = 50,
    settings: Settings | None = None,
) -> list[dict]:
    """Return a list of news items for a symbol.

    Each item: {"timestamp", "headline", "summary", "url"}.
    Sentiment is intentionally NOT computed here — see sentiment.py. Keeping fetch
    and scoring separate makes it easy to cache raw news and re-score later.
    """
    settings = settings or load_settings()
    _require_configured(settings)

    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    client = NewsClient(settings.api_key, settings.secret_key)
    req = NewsRequest(
        symbols=symbol,
        start=start,
        end=end or datetime.now(timezone.utc),
        limit=limit,
    )
    news = client.get_news(req)
    items = getattr(news, "data", {}).get("news", []) if hasattr(news, "data") else news.news

    out: list[dict] = []
    for a in items:
        out.append(
            {
                "timestamp": getattr(a, "created_at", None),
                "headline": getattr(a, "headline", "") or "",
                "summary": getattr(a, "summary", "") or "",
                "url": getattr(a, "url", "") or "",
            }
        )
    return out
