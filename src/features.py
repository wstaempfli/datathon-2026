"""Price/OHLC features computed from bars_seen only (bar_ix 0-49).

Outputs two frames:
  - per_bar: rolling features keyed by (session, bar_ix) — used by the viz.
  - per_session: aggregated features at bar 49 — consumed by the model.

Leakage contract: this module imports nothing about bars_unseen. All features are
derived from the seen window. Sessions are treated independently.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-bar rolling features
# ---------------------------------------------------------------------------

def _rsi_14_by_session(close: pd.Series, group: pd.Series) -> pd.Series:
    """Wilder 14-period RSI computed within each session (vectorized, no Python loop)."""
    diff = close.groupby(group).diff()
    up = diff.clip(lower=0.0)
    down = (-diff).clip(lower=0.0)
    # Wilder smoothing ≈ EMA with alpha = 1/14 (span = 27 equivalence: use com=13).
    avg_up = up.groupby(group).transform(
        lambda s: s.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    )
    avg_down = down.groupby(group).transform(
        lambda s: s.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    )
    rs = avg_up / avg_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When avg_down == 0 and avg_up > 0 → rsi = 100; when both 0 → NaN (we leave it).
    rsi = rsi.where(~((avg_down == 0) & (avg_up > 0)), 100.0)
    return rsi


def _build_per_bar(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling features (ma/ema/vol/rsi/ret_1) per (session, bar_ix)."""
    df = bars.sort_values(["session", "bar_ix"]).reset_index(drop=True).copy()
    g = df.groupby("session", sort=False)["close"]

    df["ma_5"] = g.transform(lambda s: s.rolling(5, min_periods=1).mean())
    df["ma_10"] = g.transform(lambda s: s.rolling(10, min_periods=1).mean())
    df["ma_20"] = g.transform(lambda s: s.rolling(20, min_periods=1).mean())
    df["ema_10"] = g.transform(lambda s: s.ewm(span=10, adjust=False).mean())

    log_close = np.log(df["close"].clip(lower=1e-12))
    log_ret = log_close.groupby(df["session"]).diff()
    df["vol_10"] = log_ret.groupby(df["session"]).transform(
        lambda s: s.rolling(10, min_periods=3).std()
    )

    df["rsi_14"] = _rsi_14_by_session(df["close"], df["session"])
    df["ret_1"] = g.pct_change()

    keep = [
        "session",
        "bar_ix",
        "close",
        "ma_5",
        "ma_10",
        "ma_20",
        "ema_10",
        "vol_10",
        "rsi_14",
        "ret_1",
    ]
    return df[keep]


# ---------------------------------------------------------------------------
# Per-session aggregate features (at bar 49)
# ---------------------------------------------------------------------------

def _bar_slope(closes: np.ndarray) -> float:
    """OLS slope of close ~ bar_ix using closed-form cov/var. Assumes 1D array."""
    n = closes.shape[0]
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = closes.mean()
    var_x = ((x - x_mean) ** 2).sum()
    if var_x == 0.0:
        return 0.0
    cov_xy = ((x - x_mean) * (closes - y_mean)).sum()
    return float(cov_xy / var_x)


def _build_per_session(bars: pd.DataFrame, per_bar: pd.DataFrame) -> pd.DataFrame:
    """Aggregate features per session, evaluated at the final seen bar."""
    bars = bars.sort_values(["session", "bar_ix"]).reset_index(drop=True)
    per_bar = per_bar.sort_values(["session", "bar_ix"]).reset_index(drop=True)

    g = bars.groupby("session", sort=True)

    first = g.first()
    last = g.last()

    out = pd.DataFrame(index=first.index.rename("session"))
    out["close_0"] = first["close"]
    out["close_49"] = last["close"]
    out["fh_return"] = out["close_49"] / out["close_0"] - 1.0

    # log-return based vol over full seen window
    log_close = np.log(bars["close"].clip(lower=1e-12))
    log_ret = log_close.groupby(bars["session"]).diff()
    out["realized_vol"] = log_ret.groupby(bars["session"]).std()

    # Per-bar feature snapshots at last bar
    pb_last = per_bar.groupby("session").last()
    out["realized_vol_10"] = pb_last["vol_10"]
    out["ma_5_at_49"] = pb_last["ma_5"]
    out["ma_10_at_49"] = pb_last["ma_10"]
    out["ma_20_at_49"] = pb_last["ma_20"]
    out["ema_10_at_49"] = pb_last["ema_10"]
    out["rsi_at_49"] = pb_last["rsi_14"]

    # Ratios to moving averages
    out["close_over_ma10"] = out["close_49"] / out["ma_10_at_49"]
    out["close_over_ma20"] = out["close_49"] / out["ma_20_at_49"]

    # Trailing returns (last k bars)
    def _tail_return(k: int) -> pd.Series:
        def _f(s: pd.Series) -> float:
            if len(s) <= k:
                return s.iloc[-1] / s.iloc[0] - 1.0
            return s.iloc[-1] / s.iloc[-(k + 1)] - 1.0
        return bars.groupby("session")["close"].apply(_f)

    out["ret_last_5"] = _tail_return(5)
    out["ret_last_10"] = _tail_return(10)
    out["ret_last_20"] = _tail_return(20)

    # Drawdown/runup over seen window
    def _dd_ru(s: pd.Series) -> tuple[float, float]:
        arr = s.to_numpy()
        cummax = np.maximum.accumulate(arr)
        cummin = np.minimum.accumulate(arr)
        dd = float((arr / cummax - 1.0).min())
        ru = float((arr / cummin - 1.0).max())
        return dd, ru

    dd_ru = bars.groupby("session")["close"].apply(lambda s: pd.Series(_dd_ru(s), index=["max_drawdown", "max_runup"]))
    # dd_ru may come back as a DataFrame (unstacked) or Series depending on pandas version.
    if isinstance(dd_ru, pd.Series):
        dd_ru = dd_ru.unstack()
    out["max_drawdown"] = dd_ru["max_drawdown"]
    out["max_runup"] = dd_ru["max_runup"]

    # Bar slope (OLS)
    slopes = bars.groupby("session")["close"].apply(lambda s: _bar_slope(s.to_numpy()))
    out["bar_slope"] = slopes

    # High/low range features
    hl_range = (bars["high"] - bars["low"]) / bars["close"].replace(0.0, np.nan)
    out["hl_range_mean"] = hl_range.groupby(bars["session"]).mean()

    hi = g["high"].max()
    lo = g["low"].min()
    rng = (hi - lo).replace(0.0, np.nan)
    out["close_pos_in_range"] = (out["close_49"] - lo) / rng

    # Up-bar ratio
    up_bar = (bars["close"] > bars["open"]).astype(float)
    out["up_bar_ratio"] = up_bar.groupby(bars["session"]).mean()

    # Wick features at last bar
    last_body_hi = np.maximum(last["open"], last["close"])
    last_body_lo = np.minimum(last["open"], last["close"])
    out["wick_upper_last"] = (last["high"] - last_body_hi) / last["close"].replace(0.0, np.nan)
    out["wick_lower_last"] = (last_body_lo - last["low"]) / last["close"].replace(0.0, np.nan)

    out = out.fillna(0.0)
    # replace any residual inf (e.g., division edge cases) with 0.0
    out = out.replace([np.inf, -np.inf], 0.0)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_price_features(
    bars_seen: pd.DataFrame,
    rolling_out: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per_session: DataFrame indexed by session, per_bar: DataFrame with session,bar_ix cols).

    If rolling_out is given, writes per_bar to that path (parquet) for the viz.
    """
    per_bar = _build_per_bar(bars_seen)
    per_session = _build_per_session(bars_seen, per_bar)
    if rolling_out is not None:
        rolling_out = Path(rolling_out)
        rolling_out.parent.mkdir(parents=True, exist_ok=True)
        per_bar.to_parquet(rolling_out)
    return per_session, per_bar
