"""Per-session feature engineering for the Zurich Datathon 2026 challenge.

This module is the single source of truth for the feature matrix consumed by
`src/model.py`. It is intentionally minimal at the scaffolding stage — a single
`_const` column keeps the end-to-end pipeline runnable while the researcher
identifies signals worth turning into features.

Public surface:
    - make_features(bars_seen, headlines_seen, sentiment_cache, training_stats=None)
      -> (X, feature_names, fitted_stats)
    - compute_realized_vol(bars_seen) -> pd.Series  (used for position sizing only)
    - validate_no_leakage(X) -> None

Target construction lives in `src.data.compute_targets()` — deliberately *not*
returned from `make_features` so there is a single source of truth and so the
function can be called identically for train and test (test has no target).

# --------------------------------------------------------------------------
# NaN-policy template (fill in as features are added)
# --------------------------------------------------------------------------
# PRICE features         : default fill 0.0 unless noted below
#   - close_pos_in_range : fill 0.5 (midpoint) when high == low
#   - wick_asymmetry     : fill 1.0 (neutral) when denominator is zero
#   - rsi_at_halftime    : fill 50.0 (neutral) when insufficient data
# CROSS-SESSION features : default fill 0.0 unless noted below
#   - vol_percentile     : fill 0.5 (median) for edge cases
# NLP features           : fill 0.0 when sentiment cache has no coverage
# INTERACTION features   : fill 0.0 (composed from already-filled inputs)
#
# Invariant: X.isna().sum().sum() == 0 before return.
# --------------------------------------------------------------------------
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Substrings that would indicate a column leaks unseen/second-half information.
_LEAK_SUBSTRINGS: tuple[str, ...] = (
    "unseen",
    "future",
    "second_half",
    # Anything referencing bars 50-99: bar_5?, bar_6?, bar_7?, bar_8?, bar_9?
    "bar_5",
    "bar_6",
    "bar_7",
    "bar_8",
    "bar_9",
)


def make_features(
    bars_seen: pd.DataFrame,
    headlines_seen: pd.DataFrame,
    sentiment_cache: pd.DataFrame,
    training_stats: dict | None = None,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Build a per-session feature matrix from seen bars, headlines, and sentiment.

    Fit mode (``training_stats=None``): compute any cross-session statistics
    needed (decile bin edges, vol distribution, bucket means, etc.) and return
    them as ``fitted_stats`` so they can be reused at inference.

    Apply mode (``training_stats=<dict>``): use the provided stats verbatim;
    never re-fit anything on test data.

    The current baseline is an empty scaffold: X has a single ``_const`` column
    of zeros so the model pipeline runs end-to-end. Real features land here as
    the researcher confirms signal in `src/signals.md`.

    Args:
        bars_seen: First-half OHLC bars with columns
            [session, bar_ix, open, high, low, close]. bar_ix must be in 0..49.
        headlines_seen: Headlines for seen bars with columns
            [session, bar_ix, headline].
        sentiment_cache: FinBERT-scored headlines with columns
            [headline, pos, neg, neu, sent_score].
        training_stats: None during training; the dict returned from a previous
            training call during inference.

    Returns:
        X: DataFrame indexed by session, columns = feature_names.
        feature_names: list[str] equal to ``list(X.columns)``.
        fitted_stats: dict of fitted cross-session statistics. Empty during the
            scaffolding stage; pass-through of ``training_stats`` when provided.
    """
    sessions = np.sort(bars_seen["session"].unique())
    session_index = pd.Index(sessions, name="session")

    # ------------------------------------------------------------------
    # PRICE FEATURES (per-session, computed from seen bars only)
    # ------------------------------------------------------------------
    # TODO: add price-derived features here (returns, vol, range, slope, rsi, ...)

    # ------------------------------------------------------------------
    # CROSS-SESSION FEATURES (require training_stats)
    # ------------------------------------------------------------------
    # TODO: add features that depend on the training distribution (deciles,
    #       percentiles, historical bucket means). Fit in fit mode, apply in
    #       apply mode. NEVER fit on test data.

    # ------------------------------------------------------------------
    # NLP FEATURES (merged from headlines + sentiment_cache per session)
    # ------------------------------------------------------------------
    # TODO: add sentiment aggregates (mean, std, count, extremes) and any
    #       cleaning/deduplication.

    # ------------------------------------------------------------------
    # INTERACTION FEATURES (compositions of the above)
    # ------------------------------------------------------------------
    # TODO: add interactions such as sentiment_x_invvol, sentiment_price_agree.

    X = pd.DataFrame({"_const": 0.0}, index=session_index)
    feature_names: list[str] = list(X.columns)

    # Pass-through contract: apply mode returns the stats it was given so callers
    # can thread a single dict through train/predict without branching.
    if training_stats is None:
        fitted_stats: dict = {}
    else:
        fitted_stats = training_stats

    # Final invariant: no NaN/inf allowed to leak into the model.
    assert X.isna().sum().sum() == 0, "make_features produced NaN values"
    return X, feature_names, fitted_stats


def compute_realized_vol(bars_seen: pd.DataFrame) -> pd.Series:
    """Per-session std of log-returns over seen bars, indexed by session.

    Used ONLY for position sizing — not a model feature. A single
    ``sort_values(["session", "bar_ix"])`` is done up front so the diff and
    groupby operations share the same ordering without repeating the sort.
    Sessions with missing/insufficient data get filled with 0.01 (a neutral
    non-zero vol that avoids division-by-zero downstream in position sizing).

    Args:
        bars_seen: First-half OHLC bars with at least [session, bar_ix, close].

    Returns:
        pd.Series indexed by session (all unique sessions from ``bars_seen``),
        values = std of per-bar log-returns. No NaN, strictly finite.
    """
    sessions = np.sort(bars_seen["session"].unique())
    session_index = pd.Index(sessions, name="session")

    ordered = bars_seen.sort_values(["session", "bar_ix"])
    log_close = np.log(ordered["close"].clip(lower=1e-12))
    log_ret = log_close.groupby(ordered["session"]).diff()
    vol = log_ret.groupby(ordered["session"]).std()

    return vol.reindex(session_index).fillna(0.01)


def validate_no_leakage(X: pd.DataFrame) -> None:
    """Raise ValueError if X leaks unseen-bar info or contains NaN/inf.

    Checks column names for substrings that would indicate the feature touched
    bars 50-99 or any 'future'/'second_half'/'unseen' data, and verifies the
    matrix is finite and NaN-free. This is a cheap last line of defense — the
    real leakage prevention lives inside the feature builders themselves.

    Raises:
        ValueError: with a descriptive message naming the offending column(s)
            or the kind of invalid values found.
    """
    bad_cols = [
        col
        for col in X.columns
        if any(substr in str(col).lower() for substr in _LEAK_SUBSTRINGS)
    ]
    if bad_cols:
        raise ValueError(
            "validate_no_leakage: columns reference unseen/second-half data: "
            f"{bad_cols}. Feature names must not mention bars 50-99, 'unseen', "
            "'future', or 'second_half'."
        )

    nan_count = int(X.isna().sum().sum())
    if nan_count > 0:
        nan_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"validate_no_leakage: X contains {nan_count} NaN value(s) in "
            f"columns {nan_cols}. make_features must impute every feature."
        )

    numeric = X.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        inf_cols = [
            col
            for col in numeric.columns
            if not np.isfinite(numeric[col].to_numpy()).all()
        ]
        raise ValueError(
            "validate_no_leakage: X contains non-finite (inf) values in "
            f"columns {inf_cols}."
        )
