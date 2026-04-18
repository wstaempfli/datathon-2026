"""Per-session feature engineering for the Zurich Datathon 2026 challenge.

This module is the single source of truth for the feature matrix consumed by
`src/model.py`. It is intentionally minimal at the scaffolding stage — a single
`_const` column keeps the end-to-end pipeline runnable while the researcher
identifies signals worth turning into features.

Public surface:
    - make_features(bars_seen, headlines_seen, sentiment_cache, training_stats=None)
      -> (X, feature_names, fitted_stats)
    - validate_no_leakage(X) -> None

Target construction lives in `src.data.compute_targets()` — deliberately *not*
returned from `make_features` so there is a single source of truth and so the
function can be called identically for train and test (test has no target).

# --------------------------------------------------------------------------
# NaN-policy template (fill in as features are added)
# --------------------------------------------------------------------------
# PRICE features         : default fill 0.0 unless noted below
#   - fh_return          : fill 0.0 when open[0] or close[49] is missing or zero
#   - yz_vol             : fill 0.0 when variance components are NaN (degenerate / constant-price sessions)
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


def _yang_zhang_vol(bars_seen: pd.DataFrame) -> pd.Series:
    """Yang-Zhang volatility over all seen bars per session.

    Drift-independent OHLC volatility estimator. For 50 bars:
      V_o  = var( ln(O_i / C_{i-1}) )       # bar-to-bar gap, skips bar 0
      V_c  = var( ln(C_i / O_i) )           # intraday open-to-close
      V_rs = mean( u(u-c) + d(d-c) )        # Rogers-Satchell, u=ln(H/O), d=ln(L/O)
      k    = 0.34 / (1.34 + (n+1)/(n-1))    # n = gap returns = 49
      yz   = sqrt(V_o + k*V_c + (1-k)*V_rs)

    Reference: Yang & Zhang (2000), Journal of Business 73:477.
    """
    b = bars_seen.sort_values(["session", "bar_ix"]).copy()
    b["close_prev"] = b.groupby("session")["close"].shift(1)
    b["o"] = np.log(b["open"]  / b["close_prev"])
    b["c"] = np.log(b["close"] / b["open"])
    b["u"] = np.log(b["high"]  / b["open"])
    b["d"] = np.log(b["low"]   / b["open"])

    g = b.groupby("session")
    v_o  = g["o"].var(ddof=1)
    v_c  = g["c"].var(ddof=1)
    v_rs = g.apply(lambda df: (df["u"] * (df["u"] - df["c"])
                               + df["d"] * (df["d"] - df["c"])).mean())

    n = 49
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return np.sqrt(v_o + k * v_c + (1 - k) * v_rs).rename("yz_vol")


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
    # fh_return: first-half return from seen bars only. Uses bar 0's open and
    # bar 49's close. Mean-reverting sign against target (r ~= -0.07, p ~= 0.02).
    opens = bars_seen.loc[bars_seen["bar_ix"] == 0].set_index("session")["open"]
    closes = bars_seen.loc[bars_seen["bar_ix"] == 49].set_index("session")["close"]
    fh_return = (closes / opens.replace(0.0, np.nan) - 1.0).reindex(session_index)
    fh_return = fh_return.fillna(0.0)

    # yz_vol: Yang-Zhang (2000) drift-independent OHLC volatility over bars 0-49.
    # Gate 1 |r|=0.079, MI=0.0092 vs target; Gate 2 CV Sharpe lift +0.12 (4/5 folds);
    # Gate 3 Wasserstein = 2% of train std (very stable across splits).
    yz_vol = _yang_zhang_vol(bars_seen).reindex(session_index).fillna(0.0)

    # recent_return_3: close[49]/close[46] - 1. Late-session 3-bar momentum
    # (continuation signal, opposite sign to fh_return's reversion).
    close_pivot = (
        bars_seen.pivot(index="session", columns="bar_ix", values="close")
        .reindex(session_index)
    )
    c46 = close_pivot[46].replace(0.0, np.nan)
    c49 = close_pivot[49]
    recent_return_3 = (c49 / c46 - 1.0).fillna(0.0)

    # upper_wick_49: upper-wick ratio at bar 49. (high - max(open, close)) / (high - low).
    # Candlestick rejection / late-session buying pressure proxy. r ~= +0.065 vs target.
    bar49 = bars_seen.loc[bars_seen["bar_ix"] == 49].set_index("session")
    hi = bar49["high"].reindex(session_index)
    lo = bar49["low"].reindex(session_index)
    op = bar49["open"].reindex(session_index)
    cl = bar49["close"].reindex(session_index)
    body_top = np.maximum(op.to_numpy(), cl.to_numpy())
    range_ = (hi - lo).replace(0.0, np.nan)
    upper_wick_49 = ((hi - body_top) / range_).fillna(0.0)

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

    X = pd.DataFrame(
        {
            "_const": 0.0,
            "fh_return": fh_return.to_numpy(),
            "yz_vol": yz_vol.to_numpy(),
            "recent_return_3": recent_return_3.to_numpy(),
            "upper_wick_49": upper_wick_49.to_numpy(),
        },
        index=session_index,
    )
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
