"""Buy-and-hold as an actual strategy object.

It buys on the first bar and never sells. Including it in a bake-off makes the
benchmark a first-class competitor instead of just a reference line — and it is the
thing almost every active strategy fails to beat.
"""
from __future__ import annotations

from backtesting import Strategy


class BuyAndHold(Strategy):
    def init(self):
        pass

    def next(self):
        if not self.position:
            self.buy()
