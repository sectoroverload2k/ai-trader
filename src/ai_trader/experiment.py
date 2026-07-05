"""Does news sentiment predict the NEXT day's price move?

This is the honest test of the project's central question. For each trading day we:
  1. score that day's news sentiment,
  2. line it up against the return from that day's close to the NEXT day's close,
  3. measure whether sentiment has any predictive relationship.

Look-ahead is avoided by construction: sentiment for day D uses only news timestamped
on day D, and it is matched to the D -> D+1 return (which is not known on day D).

The expected result (see RESEARCH.md) is a weak or near-zero relationship. That is a
finding, not a failure — the point is to measure it honestly rather than assume an edge.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd


def daily_sentiment(
    news_items: list[dict],
    score_fn: Callable[[str], float] | None = None,
    use: str = "summary",
) -> pd.Series:
    """Average sentiment per calendar day.

    `score_fn(text) -> float in [-1, 1]` defaults to FinBERT (imported lazily so this
    module stays usable — and testable — without torch installed).
    Returns a Series indexed by `datetime.date`.
    """
    if score_fn is None:
        from .sentiment import score_text as score_fn  # lazy: avoids importing torch

    rows: list[tuple] = []
    for item in news_items:
        ts = item.get("timestamp")
        if ts is None:
            continue
        date = ts.date() if hasattr(ts, "date") else pd.to_datetime(ts).date()
        text = item.get(use) or item.get("headline") or ""
        rows.append((date, score_fn(text)))

    if not rows:
        return pd.Series(dtype=float, name="sentiment")

    df = pd.DataFrame(rows, columns=["date", "score"])
    out = df.groupby("date")["score"].mean()
    out.name = "sentiment"
    return out


def build_dataset(prices: pd.DataFrame, daily_sent: pd.Series) -> pd.DataFrame:
    """Join daily sentiment to each day's NEXT-day close-to-close return.

    `prices` is an OHLCV frame indexed by timestamp (as returned by data.get_price_history).
    Output columns: sentiment, next_return  (one row per day that has both).
    """
    close = prices["Close"].copy()
    # Collapse the timestamp index to plain dates so it aligns with sentiment's date index.
    close.index = pd.to_datetime(close.index).date

    # pct_change() gives the D-1 -> D return at index D; shift(-1) moves it so index D
    # holds the D -> D+1 return, i.e. the thing we're trying to predict on day D.
    next_return = close.pct_change().shift(-1)
    next_return.name = "next_return"

    df = pd.concat([daily_sent, next_return], axis=1, join="inner").dropna()
    return df


def evaluate(df: pd.DataFrame) -> dict:
    """Summarize the predictive relationship between sentiment and next-day return.

    Returns correlations plus a directional hit-rate (on days with non-zero sentiment):
    how often the sign of sentiment matched the sign of the next-day move. ~0.50 means
    no directional edge.
    """
    if len(df) < 3:
        return {"n": len(df), "note": "not enough data to evaluate"}

    nonzero = df[df["sentiment"] != 0]
    if len(nonzero):
        hit_rate = float(((nonzero["sentiment"] > 0) == (nonzero["next_return"] > 0)).mean())
    else:
        hit_rate = float("nan")

    # Spearman = Pearson of the ranks. Computed directly to avoid a scipy dependency.
    spearman = float(df["sentiment"].rank().corr(df["next_return"].rank()))

    return {
        "n": int(len(df)),
        "n_directional": int(len(nonzero)),
        "pearson": float(df["sentiment"].corr(df["next_return"])),
        "spearman": spearman,
        "hit_rate": hit_rate,
        "mean_next_return": float(df["next_return"].mean()),
    }


def format_report(symbol: str, stats: dict) -> str:
    if "note" in stats:
        return f"{symbol}: {stats['note']} (n={stats['n']})"
    return (
        f"{symbol}\n"
        f"  days analyzed:        {stats['n']}\n"
        f"  days with news:       {stats['n_directional']}\n"
        f"  Pearson corr:         {stats['pearson']:+.3f}   (0 = no linear relationship)\n"
        f"  Spearman corr:        {stats['spearman']:+.3f}   (0 = no rank relationship)\n"
        f"  directional hit-rate: {stats['hit_rate']:.1%}   (50% = coin flip, no edge)\n"
    )
