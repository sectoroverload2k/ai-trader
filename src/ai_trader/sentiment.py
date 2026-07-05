"""News sentiment scoring with FinBERT.

FinBERT (ProsusAI/finbert) is a BERT model fine-tuned on financial text. It runs
locally and is free. The model is loaded lazily and cached on first use.

Realistic expectations (see RESEARCH.md): sentiment is a *weak* signal. Company-
specific news content predicts better than headlines, and broad sector news adds
noise. Always measure whether adding sentiment actually beats a price-only baseline
on out-of-sample data before believing it.
"""
from __future__ import annotations

from functools import lru_cache

# FinBERT outputs three classes; map to a single score in [-1, 1].
_LABEL_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


@lru_cache(maxsize=1)
def _load_pipeline():
    """Load and cache the FinBERT sentiment pipeline (heavy: pulls in torch)."""
    from transformers import pipeline

    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        truncation=True,
        max_length=512,
    )


def score_text(text: str) -> float:
    """Return a sentiment score in [-1, 1] for a single piece of text.

    Score = P(positive) - P(negative), so magnitude reflects confidence.
    Empty text returns 0.0 (neutral).
    """
    text = (text or "").strip()
    if not text:
        return 0.0

    clf = _load_pipeline()
    # return_all_scores gives probabilities for all three classes.
    results = clf(text, top_k=None)
    probs = {r["label"].lower(): r["score"] for r in results}
    return probs.get("positive", 0.0) - probs.get("negative", 0.0)


def score_news(news_items: list[dict], use: str = "summary") -> float:
    """Aggregate sentiment across news items into one score in [-1, 1].

    `use` selects the text field: "summary" (content — predicts better) or
    "headline". Falls back to headline when the summary is empty.
    Returns the mean score, or 0.0 when there is no news.
    """
    if not news_items:
        return 0.0

    scores = []
    for item in news_items:
        text = item.get(use) or item.get("headline") or ""
        scores.append(score_text(text))
    return sum(scores) / len(scores) if scores else 0.0
