"""Sentiment-gated momentum strategy.

Same moving-average-crossover entry as the plain momentum baseline, but a long is
only taken when recent news sentiment is not negative. The whole point is to test —
out-of-sample — whether this gate actually improves on price-alone, or just adds
noise and trades less. Expect it to be a wash more often than not.

Reads a `Sentiment` column (see experiment.attach_sentiment). Backtesting.py fills
orders on the next bar, so acting on bar D's sentiment is not look-ahead.
"""
from __future__ import annotations

from backtesting import Strategy
from backtesting.lib import crossover

from .momentum import SMA


class SentimentMomentum(Strategy):
    fast = 20
    slow = 50
    sent_threshold = 0.0  # require sentiment >= this to go long

    def init(self):
        close = self.data.Close
        self.sma_fast = self.I(SMA, close, self.fast)
        self.sma_slow = self.I(SMA, close, self.slow)

    def next(self):
        sentiment = self.data.Sentiment[-1]
        if crossover(self.sma_fast, self.sma_slow) and sentiment >= self.sent_threshold:
            self.position.close()
            self.buy()
        elif crossover(self.sma_slow, self.sma_fast):
            self.position.close()
