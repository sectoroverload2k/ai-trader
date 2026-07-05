"""Mean-reversion strategies — the counterpart to momentum.

The bet: prices that stretch far from their recent average tend to snap back. This
family tends to have a higher Sharpe but relies on precise entries and gets destroyed
in strong trends (it keeps "buying the dip" all the way down). Measure it out-of-sample
against buy-and-hold like everything else.
"""
from __future__ import annotations

from backtesting import Strategy

from .indicators import RSI, SMA, rolling_std


class RSIMeanReversion(Strategy):
    """Buy when oversold (RSI < low), exit when it recovers (RSI > high)."""

    rsi_period = 14
    low = 30
    high = 70

    def init(self):
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)

    def next(self):
        if not self.position and self.rsi[-1] < self.low:
            self.buy()
        elif self.position and self.rsi[-1] > self.high:
            self.position.close()


class BollingerMeanReversion(Strategy):
    """Buy when price closes below the lower band; exit when it reverts to the mean."""

    period = 20
    k = 2.0  # band width in standard deviations

    def init(self):
        close = self.data.Close
        self.mid = self.I(SMA, close, self.period)
        self.std = self.I(rolling_std, close, self.period)

    def next(self):
        price = self.data.Close[-1]
        lower = self.mid[-1] - self.k * self.std[-1]
        if not self.position and price < lower:
            self.buy()
        elif self.position and price > self.mid[-1]:
            self.position.close()
