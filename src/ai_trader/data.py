"""Data layer: historical price bars and news, fetched from Alpaca.

Uses the official `alpaca-py` SDK. Imports are done lazily inside functions so the
rest of the package (config, strategy logic, backtests on cached data) can be used
without the SDK installed or credentials configured.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
    feed: str = "iex",
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of OHLCV bars indexed by timestamp.

    Columns: Open, High, Low, Close, Volume  (capitalized for Backtesting.py).

    `feed` defaults to "iex". Alpaca's free data plan cannot query the consolidated
    "sip" feed (or the last 15 minutes of it) and returns HTTP 403 if you try, so IEX
    is the correct default for a free account. IEX is a single-exchange feed — fine
    for daily-bar research; upgrade the data plan later if you need full-tape data.
    """
    settings = settings or load_settings()
    _require_configured(settings)

    from alpaca.data.enums import DataFeed
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
    feed_enum = DataFeed.SIP if feed.lower() == "sip" else DataFeed.IEX

    # The free plan also can't read the most recent 15 min; pull `end` back to be safe.
    default_end = datetime.now(timezone.utc) - timedelta(minutes=16)

    client = StockHistoricalDataClient(settings.api_key, settings.secret_key)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start,
        end=end or default_end,
        feed=feed_enum,
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


def _news_item_to_dict(a) -> dict:
    """Normalize one Alpaca news article object into a plain dict."""
    return {
        "timestamp": getattr(a, "created_at", None),
        "headline": getattr(a, "headline", "") or "",
        "summary": getattr(a, "summary", "") or "",
        "url": getattr(a, "url", "") or "",
    }


def _extract_news_items(resp) -> list:
    """Pull the list of article objects out of an Alpaca NewsSet response."""
    if hasattr(resp, "data") and isinstance(getattr(resp, "data"), dict):
        return resp.data.get("news", [])
    return getattr(resp, "news", []) or []


def _paginate_news(fetch_page, max_items: int) -> list[dict]:
    """Loop a page-fetcher until we hit `max_items` or run out of pages.

    `fetch_page(page_token) -> (items: list[dict], next_token: str | None)`.
    Pure/injectable so pagination can be unit-tested without the network.
    """
    out: list[dict] = []
    token = None
    while len(out) < max_items:
        items, token = fetch_page(token)
        if not items:
            break
        out.extend(items)
        if not token:
            break
    return out[:max_items]


def get_news(
    symbol: str,
    start: datetime,
    end: datetime | None = None,
    limit: int = 50,
    settings: Settings | None = None,
) -> list[dict]:
    """Return up to `limit` news items for a symbol (single page).

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
    return [_news_item_to_dict(a) for a in _extract_news_items(client.get_news(req))]


def get_all_news(
    symbol: str,
    start: datetime,
    end: datetime | None = None,
    max_items: int = 1000,
    settings: Settings | None = None,
) -> list[dict]:
    """Page through Alpaca's news API to gather up to `max_items` articles.

    The API returns 50 per page; this follows `next_page_token` until it runs out
    or `max_items` is reached — needed to get a sample large enough to measure.
    """
    settings = settings or load_settings()
    _require_configured(settings)

    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    client = NewsClient(settings.api_key, settings.secret_key)
    end = end or datetime.now(timezone.utc)

    def fetch_page(page_token):
        req = NewsRequest(
            symbols=symbol, start=start, end=end, limit=50, page_token=page_token
        )
        resp = client.get_news(req)
        items = [_news_item_to_dict(a) for a in _extract_news_items(resp)]
        return items, getattr(resp, "next_page_token", None)

    return _paginate_news(fetch_page, max_items)
