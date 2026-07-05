# ai-trader

A **paper-trading** (simulated, no real money) research platform for experimenting with
AI/algorithmic stock-trading ideas on top of [Alpaca](https://alpaca.markets/) — including
feeding **news sentiment** into strategies.

> ⚠️ **Read [`RESEARCH.md`](./RESEARCH.md) first.** This is a *learning and honest-measurement*
> project, not a get-rich system. Markets are mostly efficient; ~80% of backtested strategies
> fail live; "predict stocks from news with 98% accuracy" claims are almost always overfitting
> or data leakage. Realistic edge from sentiment is small. The goal is to measure honestly, not
> to fool ourselves. **Nothing here is investment advice. Paper trading only.**

## What's in here

```
RESEARCH.md                 The landscape, strategies, and design philosophy (start here)
requirements.txt            Python dependencies
.env.example                Template for your Alpaca PAPER keys (copy to .env)
src/ai_trader/
  config.py                 Loads Alpaca keys from the environment
  data.py                   Fetch historical price bars + news from Alpaca
  sentiment.py              Score news sentiment with FinBERT (local, free)
  backtest.py               Backtest harness — ALWAYS compares vs. buy & hold
  strategies/momentum.py    Simple moving-average-crossover baseline (no ML)
examples/
  hello_alpaca.py           Smoke test: connect, pull prices + news, print
  run_backtest.py           Run the momentum baseline and compare to buy & hold
```

## Setup

1. **Get free paper-trading keys.** Sign up at <https://alpaca.markets/>, switch the dashboard to
   **Paper Trading**, and generate API keys. Paper trading uses fake money — you cannot lose real
   money with these keys.

2. **Install and configure:**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env         # then edit .env and paste your paper keys
   ```
   (FinBERT/torch in `requirements.txt` is a large download; you can skip it until you use the
   sentiment module.)

3. **Confirm it works:**
   ```bash
   python examples/hello_alpaca.py     # prints recent AAPL bars + headlines
   python examples/run_backtest.py     # backtests the momentum baseline vs. buy & hold
   ```

## Roadmap (see RESEARCH.md §5)

1. ✅ Data layer (prices + news)
2. ✅ Honest backtest harness (vs. buy & hold benchmark)
3. ✅ Baseline momentum strategy (no ML)
4. ✅ Sentiment scoring (FinBERT)
5. ⬜ Combined strategy: add sentiment as a signal, measure if it beats price-only *out-of-sample*
6. ⬜ Live paper-trading loop on Alpaca
7. ⬜ Risk layer: position sizing + stop-losses

**Golden rule:** build the honest measurement before the model. Beat buy-and-hold out-of-sample,
or it's noise.
