"""
Feature engineering pipeline.

Builds a session-level feature matrix from raw OHLC bars and headlines.
Every function must work identically on train and test data (no target leakage).

Usage (from run_pipeline.py or notebooks):
    from src.features import build_features, get_target, get_halfway_close

    features = build_features(bars_seen, headlines_seen, include_tier2=True)
    target   = get_target(bars_seen, bars_unseen)          # train only
    halfway  = get_halfway_close(bars_seen)                # for Sharpe calc
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Tier 1 — OHLC features (verified signal, build in Phase 1)
# ---------------------------------------------------------------------------

def _ohlc_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute per-session OHLC features from seen bars (bar_ix 0-49).

    Returns DataFrame indexed by ``session`` with columns:
      - first_half_return  (r=-0.069, p=0.028)  close[49]/close[0] - 1
      - ret_from_30        (r=-0.081, p=0.010)  close[49]/close[30] - 1
      - volatility         (r=+0.072, p=0.023)  std of log(close/open)
      - avg_hl_range       (r=+0.078, p=0.013)  mean of (high-low)/open
      - up_ratio           (r=-0.059, p=0.061)  fraction of bars where close > open
      - close_rel_range    (r=-0.055, p=0.083)  (close[49] - session_low) / (session_high - session_low)
    """
    # TODO: Compute first_half_return = close at bar 49 / close at bar 0 - 1
    # TODO: Compute ret_from_30 = close at bar 49 / close at bar 30 - 1
    # TODO: Compute volatility = std of log(close/open) per session
    # TODO: Compute avg_hl_range = mean of (high - low) / open per session
    # TODO: Compute up_ratio = fraction of bars where close > open per session
    # TODO: Compute close_rel_range = (close[49] - session_low) / (session_high - session_low)
    # TODO: Return DataFrame indexed by session
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Tier 2 — additional OHLC features (hypothesized, test in Phase 2)
# ---------------------------------------------------------------------------

def _ohlc_features_tier2(bars: pd.DataFrame) -> pd.DataFrame:
    """Tier-2 OHLC features — add incrementally, keep only if they improve CV Sharpe.

    Returns DataFrame indexed by ``session`` with columns:
      - max_drawdown       max peak-to-trough decline in close prices
      - vol_trend          std(log_ret bars 25-49) / std(log_ret bars 0-24)
      - price_slope        linear regression slope of close prices vs bar_ix
      - range_contraction  mean(hl_range last 10 bars) / mean(hl_range first 10 bars)
      - gap_mean           mean of (open[i] - close[i-1]) gaps between consecutive bars
      - ret_x_vol          interaction: first_half_return * volatility
    """
    # TODO: max_drawdown — for each session, compute running max of close,
    #       then drawdown = close / running_max - 1, feature = min(drawdown)
    # TODO: vol_trend — split bars at midpoint, ratio of late vol to early vol
    # TODO: price_slope — scipy.stats.linregress(bar_ix, close).slope
    # TODO: range_contraction — mean hl_range of last 10 bars / first 10 bars
    # TODO: gap_mean — mean(open[i+1] - close[i]) for consecutive bars
    # TODO: ret_x_vol — multiply first_half_return by volatility
    # TODO: Return DataFrame indexed by session
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Headline features
# ---------------------------------------------------------------------------

# Keyword lists for event-type classification
REVENUE_KEYWORDS = ["revenue", "quarterly", "earnings", "estimates"]
POSITIVE_KEYWORDS = ["breakthrough", "surge", "record", "strong demand", "raises outlook"]
NEGATIVE_KEYWORDS = ["misses", "decline", "withdraws", "warns", "disruptions",
                      "downgrades", "recall", "loses"]
CONTRACT_KEYWORDS = ["secures", "contract"]
EXPANSION_KEYWORDS = ["expands operations", "opens new", "new office"]


def _headline_features(headlines: pd.DataFrame) -> pd.DataFrame:
    """Compute per-session headline features from seen headlines only.

    Returns DataFrame indexed by ``session`` with columns:
      - revenue_event_count    (r=-0.082, p=0.010) — STRONGEST single feature
      - positive_event_count   (r=+0.063, p=0.048)
      - negative_event_count
      - contract_event_count
      - expansion_event_count
      - headline_count
      - net_event_sentiment    (positive + expansion + contract) - (negative + revenue)

    NOTE: Simple keyword sentiment (seen-only) has r=0.018, p=0.57 — DEAD.
    Only event-type counts carry marginal signal.
    """
    # TODO: Lowercase all headline text
    # TODO: For each headline, flag whether it matches each keyword list
    # TODO: Aggregate counts per session using groupby("session").sum()
    # TODO: Compute net_event_sentiment as combination score
    # TODO: Return DataFrame indexed by session
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Public API — called by run_pipeline.py, evaluate.py, submit.py
# ---------------------------------------------------------------------------

def build_features(
    bars: pd.DataFrame,
    headlines: pd.DataFrame | None = None,
    *,
    include_tier2: bool = False,
) -> pd.DataFrame:
    """Build session-level feature matrix from raw data.

    Parameters
    ----------
    bars : DataFrame
        OHLC bars with columns: session, bar_ix, open, high, low, close.
        May contain only the seen half (bar_ix 0-49).
    headlines : DataFrame or None
        Headlines with columns: session, bar_ix, headline.
        Pass None to skip headline features.
    include_tier2 : bool
        If True, include Tier-2 OHLC features (experimental, Phase 2).

    Returns
    -------
    DataFrame indexed by session, one row per session, columns = feature names.

    Called by
    --------
    - scripts/run_pipeline.py  (train + test)
    - src/evaluate.py          (inside CV loop, on train folds)
    - src/submit.py            (on test data)
    - notebooks/01_eda.ipynb   (for correlation analysis)
    """
    # TODO: Call _ohlc_features(bars)
    # TODO: If include_tier2, call _ohlc_features_tier2(bars) and join
    # TODO: If headlines provided, call _headline_features(headlines) and join
    # TODO: fillna(0.0), sort by session index
    # TODO: Return combined DataFrame
    raise NotImplementedError


def get_target(bars_seen: pd.DataFrame, bars_unseen: pd.DataFrame) -> pd.Series:
    """Compute target variable: second-half return per session.

    target = close[bar 99] / close[bar 49] - 1

    Only usable on training data (unseen bars required).
    Training distribution: mean=+0.35%, std=2.0%, 57% positive.

    Called by
    --------
    - scripts/run_pipeline.py  (to get training labels)
    - src/evaluate.py          (inside CV to split X/y)
    """
    # TODO: halfway_close = bars_seen grouped by session, last close (bar 49)
    # TODO: end_close = bars_unseen grouped by session, last close (bar 99)
    # TODO: return end_close / halfway_close - 1
    raise NotImplementedError


def get_halfway_close(bars_seen: pd.DataFrame) -> pd.Series:
    """Return the close price at bar 49 for each session.

    Needed by evaluate.compute_sharpe() to convert positions to PnL.

    Called by
    --------
    - src/evaluate.py  (Sharpe calculation)
    - src/submit.py    (sanity checks)
    """
    # TODO: Group by session, take last close value
    raise NotImplementedError
