# Zurich Datathon 2026 — Team "Uetlibytes"

**Top-5 finish** in the Hudson River Trading-sponsored market-close-prediction challenge at the Zurich Datathon 2026.

## Approach

Drift-preserving risk-parity position sizing that combines a directional prior with a mean-reversion fade and a news-sentiment signal:

```
pos        = clip(1 + scaler * (-K * fh + W * bmb_recent), -2, 2)
scaler     = clip(sigma_ref / max(sigma, floor * sigma_ref), 0.5, 2.0)
sigma      = std(diff(log(close[0:50])))
bmb_recent = sum_h sign(h) * exp(-(49 - bar_ix_h) / tau)
```

Key move: **only the directional component is volatility-scaled**. That preserves the +1 positive-drift prior (the dominant edge in the training distribution) while risk-parity-sizing the fade + sentiment bet (the noisier overlay). Sentiment is extracted from news headlines via ~30 curated bullish / bearish regex patterns.

Core pipeline: [`src/pipeline.py`](./src/pipeline.py) (~140 LOC).

## Result

Cross-validated Sharpe (5-fold contiguous CV, per-fold `sigma_ref` to avoid leakage):

| | Mean | Worst fold |
|---|---:|---:|
| **This submission (V1b)** | **3.128** | **2.182** |
| Baseline | 3.119 | 2.021 |

Beats the baseline on both mean and worst fold.

Unit tests: [`tests/test_pipeline.py`](./tests/test_pipeline.py). CV driver: [`scripts/cv.py`](./scripts/cv.py). Final presentation slides: [`Uetlibytes-Presentation.pdf`](./Uetlibytes-Presentation.pdf).

---

## Original challenge description

# Zurich Datathon 2026: Simulated Market Close Prediction

## Challenge

Each data file contains many sessions. Each session simulates a single synthetic stock trading over a number of time bars. You are given OHLC (Open, High, Low, Close) bar data and a mix of news headlines for each session.

- Training sessions: you see all bars.
- Test sessions: you see only the first half of the trading session.

Your task: Decide how much stock to buy/sell half-way through each session.

## Submission format

A CSV with two columns:

```
session,target_position
500,42.5
501,-56.92043
...
```

At the half-way point of each session, the buyer/seller buys/sells `target_position` shares of the stock at the close price of that bar; at the end of the session, they close out at the close price of the last bar.

## Metric

```
pnl_i  = target_position_i * (close_price_end_i / close_price_halfway_i - 1)
sharpe = np.mean(pnl_i) / np.std(pnl_i) * 16
```

Final score = the Sharpe ratio above.

## Data

Files follow: `data/{data_kind}_{seen/unseen}_{train/public_test/private_test}.parquet`

OHLC files contain `session`, `bar_ix`, `open`, `high`, `low`, `close`. Headline files contain `session`, `bar_ix`, `headline`. Prices normalized to begin at 1. Headlines and companies are fictional.
