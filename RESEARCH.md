# AI Stock Trading — Research & Design Notes

> Status: research baseline (July 2026). Everything here targets **paper (simulated) trading**
> only. Nothing in this repo is investment advice.

## 0. The reality check (read this first)

- **~80% of strategies that look profitable in a backtest fail in live markets.** Backtests
  overstate performance because of overfitting, look-ahead bias, and ignored costs/slippage.
- Rigorous studies find ML price-prediction accuracy is often **not statistically different
  from random chance**, with trading returns "at market or worse" — consistent with markets
  being *mostly* efficient (prices already reflect known information).
- Papers/blogs claiming **"98% accuracy predicting stocks from news" are almost always
  data leakage or overfitting.** Honest edge from sentiment is *small* — nudging a 50% win
  rate to maybe 52–54%, not 98%.
- Therefore the goal of this project is a **learning + honest-measurement platform**, not a
  money machine. Success = "we correctly measured whether an edge exists," not "we got rich."

## 1. Broker / execution layer — Alpaca

Chosen because it is commission-free, has a first-class Python SDK (`alpaca-py`), free and
unlimited **paper trading**, and — conveniently — a **built-in free news API** (Benzinga-sourced,
~130 articles/day, history back to 2015). One platform gives us prices *and* news.

- Paper trading base URL: `https://paper-api.alpaca.markets`
- Never trade live until a strategy has survived months of paper trading + out-of-sample tests.

## 2. Popular free frameworks (landscape)

| Tool | Role | Notes |
|---|---|---|
| Alpaca | Broker + data + news | Our execution & data layer |
| Backtesting.py | Backtester | Simplest; our starting point |
| Backtrader (~22k★) | Backtester | Full-featured, event-driven |
| QuantConnect / LEAN | Cloud platform | Most complete free ecosystem, steeper curve |
| Zipline-reloaded | Research backtester | Used in the standard ML-for-trading book |
| Freqtrade | Crypto bot | Only if we go crypto |
| NautilusTrader | HFT-grade platform | Overkill for now |

## 3. Strategies that actually have evidence

- **Momentum / trend-following** — buy what's rising. ~12% annual excess return in academic
  studies (3–12 month lookbacks); Sharpe 0.5–0.8; but 20–40% drawdowns in choppy markets.
- **Mean reversion** — buy dips below a moving average expecting a bounce. Higher Sharpe
  (0.8–1.2), smaller/more frequent wins, needs precise execution.
- Neither is magic. **Risk management (position sizing + stop-losses) matters more than the
  entry signal.** Build that from day one.

## 4. News / media sentiment — what's realistic

Legitimate active research, with these consistent findings:

- News **content** predicts better than **headlines**.
- **Company-specific** news helps; mixing in broad sector news adds noise and *hurts* accuracy.
- Standard stack: score sentiment with **FinBERT** (finance-tuned, runs locally, free) or an LLM,
  then feed the score alongside price/technical features into a model (often an LSTM).
- Free news+sentiment sources: **Alpaca News API** (easiest — same platform), **Marketaux**,
  **Finnhub**, **Alpha Vantage** (all have free tiers, tag articles by ticker, sentiment −1..+1).

Realistic outcome: a *small* edge if any. The valuable skill is measuring it **without fooling
ourselves** (strict out-of-sample splits, no look-ahead).

## 5. Target architecture (build in this order)

1. **Data layer** — prices + news from Alpaca.
2. **Backtest + honest-measurement harness** — build this BEFORE fancy models.
3. **Baseline strategy** — plain momentum / mean reversion, no ML.
4. **Sentiment layer** — FinBERT scoring of news.
5. **Combined strategy** — add sentiment as an extra signal; measure if it beats price-alone.
6. **Execution** — Alpaca **paper** account only.
7. **Risk layer** — position sizing + stop-losses throughout.

Golden rule: **build the honest measurement harness before the model.** Most people do it
backwards and fool themselves.

## 6. Honest-measurement checklist

- [ ] Out-of-sample / walk-forward test (never tune on your test period).
- [ ] Include commissions=0 (Alpaca) BUT model slippage + spread.
- [ ] Compare every strategy against a **buy-and-hold benchmark** (usually SPY).
- [ ] Report Sharpe, max drawdown, and % of time underwater — not just total return.
- [ ] No look-ahead: a bar's decision may only use data available *before* that bar closed.
- [ ] Paper trade for months before even discussing real money.

## Sources

- Alpaca — https://alpaca.markets/ , News API — https://alpaca.markets/learn/sentiment-analysis-with-news-api-and-transformers
- Backtesting.py — https://kernc.github.io/backtesting.py/
- best-of-algorithmic-trading — https://github.com/merovinh/best-of-algorithmic-trading
- QuantConnect — https://www.quantconnect.com/
- Momentum vs mean reversion — https://www.stephentwomey.com/blog/momentum-vs-mean-reversion/
- Sentiment + ensemble learning (2025) — https://www.sciencedirect.com/science/article/pii/S2215016125001062
- LLM-based news + market data (2026) — https://www.sciencedirect.com/science/article/pii/S0020025526006420
- FinBERT-LSTM — https://arxiv.org/pdf/2211.07392
- Marketaux — https://www.marketaux.com/ , Finnhub — https://finnhub.io/docs/api/news-sentiment , Alpha Vantage — https://www.alphavantage.co/
- Weak-form efficiency + ML (accuracy ≈ chance) — https://arxiv.org/pdf/1909.05151
