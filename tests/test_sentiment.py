import pytest

from ai_trader.sentiment import score_news


def test_score_news_averages_items_without_loading_finbert(monkeypatch):
    # Patch the per-text scorer so the test never imports torch/FinBERT.
    import ai_trader.sentiment as sentiment_mod

    monkeypatch.setattr(sentiment_mod, "score_text", lambda t: 1.0 if "up" in t else -1.0)

    items = [{"summary": "stock up"}, {"summary": "stock down"}, {"summary": "up up"}]
    # (1 + -1 + 1) / 3
    assert score_news(items) == pytest.approx(1.0 / 3)


def test_score_news_empty_is_neutral():
    assert score_news([]) == 0.0


def test_score_news_falls_back_to_headline(monkeypatch):
    import ai_trader.sentiment as sentiment_mod

    seen = []
    monkeypatch.setattr(sentiment_mod, "score_text", lambda t: seen.append(t) or 0.0)

    score_news([{"summary": "", "headline": "big news"}], use="summary")
    assert seen == ["big news"]  # empty summary -> used headline
