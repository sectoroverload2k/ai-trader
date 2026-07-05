"""Donchian-channel breakout — a classic trend-following approach.

Buy when price breaks above the highest high of the last N days; exit when it breaks
below the lowest low. This is the core of the famous "Turtle Traders" system. Catches
big trends but whipsaws in sideways markets.
"""
from __future__ import annotations

from backtesting import Strategy

from .indicators import rolling_max, rolling_min


class DonchianBreakout(Strategy):
    entry_period = 20
    exit_period = 10

    def init(self):
        # Shift by one bar so the channel is the prior N-day extreme, never including
        # the current bar in its own breakout test (avoids look-ahead).
        self.upper = self.I(
            lambda h: rolling_max(h, self.entry_period).shift(1), self.data.High
        )
        self.lower = self.I(
            lambda l: rolling_min(l, self.exit_period).shift(1), self.data.Low
        )

    def next(self):
        price = self.data.Close[-1]
        if not self.position and price > self.upper[-1]:
            self.buy()
        elif self.position and price < self.lower[-1]:
            self.position.close()
