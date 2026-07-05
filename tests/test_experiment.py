from datetime import datetime, timezone

import pandas as pd
import pytest

from ai_trader.experiment import (
    attach_sentiment,
    build_dataset,
    daily_sentiment,
    evaluate,
)


def _news(day, text):
    return {"timestamp": datetime(2024, 1, day, 12, 0, tzinfo=timezone.utc), "summary": text}


def test_daily_sentiment_groups_and_averages_by_date():
    # Injected scorer: +1 if "good" in text, -1 if "bad", else 0.
    def fake(text):
        return 1.0 if "good" in text else (-1.0 if "bad" in text else 0.0)

    news = [_news(1, "good"), _news(1, "bad"), _news(2, "good")]
    s = daily_sentiment(news, score_fn=fake)
    assert s.loc[datetime(2024, 1, 1).date()] == 0.0  # (+1 -1)/2
    assert s.loc[datetime(2024, 1, 2).date()] == 1.0


def test_daily_sentiment_empty():
    assert daily_sentiment([], score_fn=lambda t: 0.0).empty


def test_build_dataset_uses_next_day_return_no_lookahead():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    prices = pd.DataFrame({"Close": [100.0, 110.0, 99.0]}, index=idx)
    sent = pd.Series(
        {idx[0].date(): 0.5, idx[1].date(): -0.5, idx[2].date(): 0.9}, name="sentiment"
    )
    df = build_dataset(prices, sent)
    # Day1 next_return = (110-100)/100 = 0.10; Day2 = (99-110)/110 ≈ -0.10.
    # Day3 has no next day -> dropped.
    assert len(df) == 2
    assert df.loc[idx[0].date(), "next_return"] == pytest.approx(0.10)
    assert df.loc[idx[1].date(), "next_return"] == pytest.approx(-0.10)


def test_evaluate_perfect_predictor_gives_full_hit_rate():
    # Sentiment sign always matches next-day return sign.
    df = pd.DataFrame(
        {
            "sentiment": [0.8, -0.4, 0.6, -0.9],
            "next_return": [0.02, -0.01, 0.03, -0.02],
        }
    )
    stats = evaluate(df)
    assert stats["hit_rate"] == 1.0
    assert stats["pearson"] > 0


def test_evaluate_handles_too_little_data():
    df = pd.DataFrame({"sentiment": [0.1], "next_return": [0.01]})
    assert "note" in evaluate(df)


def test_attach_sentiment_aligns_and_fills_missing_with_zero():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    prices = pd.DataFrame(
        {"Open": [1, 2, 3], "High": [1, 2, 3], "Low": [1, 2, 3],
         "Close": [1, 2, 3], "Volume": [1, 1, 1]},
        index=idx,
    )
    sent = pd.Series({idx[0].date(): 0.7, idx[2].date(): -0.3}, name="sentiment")
    out = attach_sentiment(prices, sent)
    assert list(out["Sentiment"]) == [0.7, 0.0, -0.3]  # day 2 had no news -> neutral
    assert len(out) == len(prices)  # never changes row count
