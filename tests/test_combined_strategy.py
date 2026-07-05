import numpy as np
import pandas as pd
import pytest

from ai_trader.backtest import compare_strategies, run_backtest, split_by_fraction
from ai_trader.experiment import attach_sentiment
from ai_trader.strategies.momentum import MomentumCross
from ai_trader.strategies.sentiment_momentum import SentimentMomentum


def _synthetic_prices(n=300, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.002, n)),
            "High": close * (1 + rng.uniform(0, 0.01, n)),
            "Low": close * (1 - rng.uniform(0, 0.01, n)),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=idx,
    )


def test_split_by_fraction_is_chronological():
    df = _synthetic_prices(100)
    train, test = split_by_fraction(df, 0.7)
    assert len(train) == 70 and len(test) == 30
    assert train.index.max() < test.index.min()  # no overlap, time-ordered


def test_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        split_by_fraction(_synthetic_prices(10), 1.5)


def test_sentiment_gated_strategy_runs_and_compares():
    prices = _synthetic_prices()
    rng = np.random.default_rng(7)
    sent = pd.Series(rng.uniform(-1, 1, len(prices)), index=pd.to_datetime(prices.index).date)
    prices = attach_sentiment(prices, sent)

    results = compare_strategies(
        prices,
        {"price": MomentumCross, "sentiment": SentimentMomentum},
    )
    assert set(results) == {"price", "sentiment"}
    # Both share the same underlying data, so the buy & hold benchmark must match.
    assert results["price"].buy_hold_return_pct == pytest.approx(
        results["sentiment"].buy_hold_return_pct
    )


def test_gating_never_takes_more_trades_than_ungated():
    # With the same crossovers, a non-negative-sentiment gate can only remove entries.
    prices = _synthetic_prices(seed=3)
    rng = np.random.default_rng(3)
    sent = pd.Series(rng.uniform(-1, 1, len(prices)), index=pd.to_datetime(prices.index).date)
    prices = attach_sentiment(prices, sent)

    ungated = run_backtest(prices, MomentumCross)
    gated = run_backtest(prices, SentimentMomentum)
    assert gated.num_trades <= ungated.num_trades
