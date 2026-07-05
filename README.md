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

First, **get free paper-trading keys.** Sign up at <https://alpaca.markets/>, switch the dashboard
to **Paper Trading**, and generate **Trading API** keys. Paper trading uses fake money — you cannot
lose real money with these keys. You need three values:

| Name | Value |
|---|---|
| `ALPACA_API_KEY` | your paper API key id |
| `ALPACA_API_SECRET` | your paper secret key |
| `ALPACA_API_ENDPOINT` | `https://paper-api.alpaca.markets` |

Then pick **one** way to run:

### Option A — GitHub Actions (no local setup, works from a phone)

1. Add the three values above as **repository secrets**
   (repo → Settings → Secrets and variables → Actions → New repository secret).
2. Go to the **Actions** tab → **Trade Check (paper)** → **Run workflow**, pick `smoke` (or
   `backtest`), and run. Output appears in the run logs.

> Note: GitHub Actions is great for on-demand runs (smoke test, backtests, a scheduled news scan),
> but it is **not** meant to host an always-on live trading loop. That comes later, on a small
> always-on machine.

### Option B — your own computer

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env             # then edit .env with your paper keys
python examples/hello_alpaca.py  # prints recent AAPL bars + headlines
python examples/run_backtest.py  # backtests the momentum baseline vs. buy & hold
```

For the news-sentiment module, also `pip install -r requirements-sentiment.txt` (large; pulls in
torch).

**Safety:** the code treats anything other than the paper endpoint as live and refuses to place
orders there. Read-only checks (prices/news/backtests) run regardless.

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
