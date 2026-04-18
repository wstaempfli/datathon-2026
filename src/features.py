"""Per-session feature engineering for the Zurich Datathon 2026 challenge.

Builds the 14-column feature matrix consumed by `src/model.py` from seen bars
(0..49), headlines, and the FinBERT sentiment cache. No parquet feature store
is required at runtime — every feature is computed inline.

Public surface:
    make_features(bars, headlines, sentiment_cache=None) -> (X, feature_names)
    validate_no_leakage(X) -> None
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_SENTIMENT_CACHE_PATH = _ROOT / "features" / "sentiment_cache.parquet"
_N_BARS = 50

# Frozen output column order.
FEATURE_NAMES: tuple[str, ...] = (
    "_const",
    "fh_return",
    "yz_vol",
    "recent_return_3",
    "upper_wick_49",
    "max_drawdown_fh",
    "garman_klass_vol",
    "rogers_satchell_vol",
    "parkinson_vol",
    "realized_skewness",
    "rsi_14",
    "sent_sma10",
    "sent_ema20",
    "sent_wt_decay20",
    "bmb",
)

# Curated keyword patterns for the bull-minus-bear (BMB) event score. These
# capture specific event verbs that FinBERT polarity alone underweights.
_BULL_PATTERNS: tuple[str, ...] = (
    r"reports record quarterly revenue",
    r"raises outlook",
    r"reports strong demand",
    r"reports \d+% increase in customer acquisition",
    r"sees \d+% margin improvement",
    r"announces \$[\d.]+b share buyback",
    r"launches next-generation",
    r"completes strategic acquisition",
    r"announces breakthrough",
    r"expands operations into",
    r"secures \$\d+m contract",
    r"wins industry award",
    r"files for regulatory approval",
    r"announces significant capital expenditure",
)

_BEAR_PATTERNS: tuple[str, ...] = (
    r"warns of supply chain disruptions",
    r"delays product launch",
    r"misses quarterly revenue estimates",
    r"sees \d+% drop in new customer orders",
    r"reports \d+% decline in operating income",
    r"reports unexpected decline",
    r"faces regulatory review",
    r"faces class action",
    r"explores strategic alternatives",
    r"loses key contract",
    r"withdraws from .* market citing unfavorable",
    r"recalls products",
    r"reports rising costs pressuring margins",
    r"steps down unexpectedly",
    r"sees mixed results",
    r"addresses investor concerns in open letter",
    r"revises long-term strategy",
)

_BULL_RE = re.compile("|".join(_BULL_PATTERNS), re.IGNORECASE)
_BEAR_RE = re.compile("|".join(_BEAR_PATTERNS), re.IGNORECASE)

# Substrings that would indicate a column leaks unseen/second-half information.
_LEAK_SUBSTRINGS: tuple[str, ...] = (
    "unseen", "future", "second_half",
    "bar_5", "bar_6", "bar_7", "bar_8", "bar_9",
)


# ---- price helpers --------------------------------------------------------

def _pivot_close(bars: pd.DataFrame, sessions: pd.Index) -> pd.DataFrame:
    """Rows = session, cols = bar_ix 0..49, values = close."""
    return (
        bars.pivot(index="session", columns="bar_ix", values="close")
        .reindex(index=sessions, columns=range(_N_BARS))
    )


def _yz_vol(bars: pd.DataFrame, sessions: pd.Index) -> pd.Series:
    """Yang-Zhang (2000) drift-independent OHLC volatility over bars 0..49.

    yz = sqrt(V_o + k*V_c + (1-k)*V_rs), where
        V_o  = var(ln(O_i / C_{i-1}))           # gap, skips bar 0
        V_c  = var(ln(C_i / O_i))               # intraday open-to-close
        V_rs = mean(u(u-c) + d(d-c))            # u=ln(H/O), d=ln(L/O), c=ln(C/O)
        k    = 0.34 / (1.34 + (n+1)/(n-1)),  n = 49
    """
    b = bars.sort_values(["session", "bar_ix"]).copy()
    b["close_prev"] = b.groupby("session")["close"].shift(1)
    b["o"] = np.log(b["open"] / b["close_prev"])
    b["c"] = np.log(b["close"] / b["open"])
    b["u"] = np.log(b["high"] / b["open"])
    b["d"] = np.log(b["low"] / b["open"])
    g = b.groupby("session")
    v_o = g["o"].var(ddof=1)
    v_c = g["c"].var(ddof=1)
    v_rs = g.apply(lambda df: (df["u"] * (df["u"] - df["c"])
                               + df["d"] * (df["d"] - df["c"])).mean())
    n = 49
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return np.sqrt(v_o + k * v_c + (1 - k) * v_rs).reindex(sessions).fillna(0.0)


def _parkinson_vol(bars: pd.DataFrame, sessions: pd.Index) -> pd.Series:
    """sqrt( mean(log(H/L)^2) / (4 ln 2) )."""
    hl = np.log(np.where(bars["low"] > 0, bars["high"] / bars["low"], 1.0))
    mean_hl2 = bars[["session"]].assign(hl2=hl ** 2).groupby("session")["hl2"].mean()
    return np.sqrt(mean_hl2 / (4.0 * np.log(2.0))).reindex(sessions).fillna(0.0)


def _garman_klass_vol(bars: pd.DataFrame, sessions: pd.Index) -> pd.Series:
    """sqrt( mean( 0.5*log(H/L)^2 - (2 ln 2 - 1)*log(C/O)^2 ) )."""
    hl = np.log(np.where(bars["low"] > 0, bars["high"] / bars["low"], 1.0))
    co = np.log(np.where(bars["open"] > 0, bars["close"] / bars["open"], 1.0))
    term = 0.5 * hl ** 2 - (2 * np.log(2.0) - 1.0) * co ** 2
    mean_term = bars[["session"]].assign(t=term).groupby("session")["t"].mean().clip(lower=0.0)
    return np.sqrt(mean_term).reindex(sessions).fillna(0.0)


def _rogers_satchell_vol(bars: pd.DataFrame, sessions: pd.Index) -> pd.Series:
    """sqrt( mean( log(H/C)*log(H/O) + log(L/C)*log(L/O) ) )."""
    safe_o = np.where(bars["open"] > 0, bars["open"], 1.0)
    safe_c = np.where(bars["close"] > 0, bars["close"], 1.0)
    hc = np.log(bars["high"] / safe_c)
    ho = np.log(bars["high"] / safe_o)
    lc = np.log(bars["low"] / safe_c)
    lo = np.log(bars["low"] / safe_o)
    term = hc * ho + lc * lo
    mean_term = bars[["session"]].assign(t=term).groupby("session")["t"].mean().clip(lower=0.0)
    return np.sqrt(mean_term).reindex(sessions).fillna(0.0)


def _max_drawdown_fh(close_pivot: pd.DataFrame) -> pd.Series:
    """min over t of close[t]/cummax(close[0..t]) - 1 (most-negative drawdown)."""
    arr = close_pivot.to_numpy()
    cummax = np.maximum.accumulate(arr, axis=1)
    dd = np.where(cummax > 0, arr / cummax - 1.0, 0.0)
    return pd.Series(dd.min(axis=1), index=close_pivot.index).fillna(0.0)


def _realized_skewness(close_pivot: pd.DataFrame) -> pd.Series:
    """Scipy-free skew of log-returns over bars 0..48.

    ((x - mean)^3).mean() / std^3, with bias-corrected (ddof=1) std.
    """
    arr = close_pivot.to_numpy()
    safe = np.where(arr > 0, arr, np.nan)
    lr = np.log(safe[:, 1:] / safe[:, :-1])
    lr = np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)
    m = lr.mean(axis=1, keepdims=True)
    std = lr.std(axis=1, ddof=1)
    centered_cubed = ((lr - m) ** 3).mean(axis=1)
    skew = np.divide(
        centered_cubed, std ** 3,
        out=np.zeros_like(centered_cubed),
        where=std > 1e-18,
    )
    skew = np.nan_to_num(skew, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(skew, index=close_pivot.index)


def _rsi_14(close_pivot: pd.DataFrame) -> pd.Series:
    """Wilder's 14-bar RSI at bar 49.

    Seed avg_gain/avg_loss with the simple mean of the first 14 diffs, then
    apply Wilder smoothing (alpha = 1/14). 50.0 default on flat sessions.
    """
    arr = close_pivot.to_numpy()
    diff = np.diff(arr, axis=1)
    up = np.where(diff > 0, diff, 0.0)
    down = np.where(diff < 0, -diff, 0.0)
    period = 14
    avg_up = up[:, :period].mean(axis=1)
    avg_down = down[:, :period].mean(axis=1)
    alpha = 1.0 / period
    for i in range(period, diff.shape[1]):
        avg_up = avg_up * (1 - alpha) + up[:, i] * alpha
        avg_down = avg_down * (1 - alpha) + down[:, i] * alpha
    rs = np.divide(avg_up, avg_down, out=np.zeros_like(avg_up),
                   where=avg_down > 1e-18)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    flat = (avg_up == 0) & (avg_down == 0)
    rsi = np.where(flat, 50.0, rsi)
    rsi = np.nan_to_num(rsi, nan=50.0, posinf=100.0, neginf=0.0)
    return pd.Series(rsi, index=close_pivot.index)


# ---- sentiment helpers ----------------------------------------------------

def _bar_sentiment_matrix(
    headlines: pd.DataFrame,
    sentiment_cache: pd.DataFrame,
    sessions: pd.Index,
) -> np.ndarray:
    """Shape (n_sessions, 50) of per-bar mean sent_score (0 if no headline)."""
    cache = sentiment_cache[["headline", "sent_score"]]
    merged = headlines.merge(cache, on="headline", how="left")
    merged["sent_score"] = merged["sent_score"].fillna(0.0)
    grp = (
        merged.groupby(["session", "bar_ix"], sort=False)["sent_score"]
        .mean()
        .reset_index()
    )
    sess_to_idx = {s: i for i, s in enumerate(sessions.tolist())}
    S = np.zeros((len(sessions), _N_BARS), dtype=np.float64)
    grp = grp[(grp["bar_ix"] >= 0) & (grp["bar_ix"] < _N_BARS)
              & grp["session"].isin(sess_to_idx)]
    if len(grp):
        rows = grp["session"].map(sess_to_idx).to_numpy()
        cols = grp["bar_ix"].to_numpy()
        S[rows, cols] = grp["sent_score"].to_numpy(dtype=np.float64)
    return S


def _sent_ema20(S: np.ndarray) -> np.ndarray:
    """Last-bar value of pd.Series(S).ewm(span=20, adjust=False).mean()."""
    alpha = 2.0 / 21.0
    out = S[:, 0].astype(np.float64).copy()
    for t in range(1, S.shape[1]):
        out = alpha * S[:, t] + (1.0 - alpha) * out
    return out


def _sent_wt_decay20(S: np.ndarray) -> np.ndarray:
    """sum_b s[b] * exp(-(49 - b)/20)."""
    bars = np.arange(_N_BARS)
    w = np.exp(-(_N_BARS - 1 - bars) / 20.0)
    return S @ w


def _compute_bmb(headlines: pd.DataFrame, sessions: pd.Index) -> pd.Series:
    """Per-session count of bull-keyword hits minus bear-keyword hits."""
    text = headlines["headline"].astype(str).str.lower()
    is_bull = text.str.contains(_BULL_RE, regex=True, na=False).astype(int)
    is_bear = text.str.contains(_BEAR_RE, regex=True, na=False).astype(int)
    tally = (
        pd.DataFrame(
            {
                "session": headlines["session"].to_numpy(),
                "bull": is_bull.to_numpy(),
                "bear": is_bear.to_numpy(),
            }
        )
        .groupby("session")[["bull", "bear"]]
        .sum()
    )
    return (tally["bull"] - tally["bear"]).reindex(sessions, fill_value=0).astype(float)


# ---- public API -----------------------------------------------------------

def make_features(
    bars: pd.DataFrame,
    headlines: pd.DataFrame,
    sentiment_cache: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the 14-column per-session feature matrix.

    Args:
        bars: Seen OHLC bars [session, bar_ix, open, high, low, close], bar_ix 0..49.
        headlines: [session, bar_ix, headline] for the seen bars.
        sentiment_cache: FinBERT-scored cache [headline, ..., sent_score]. If None,
            loads from features/sentiment_cache.parquet.

    Returns:
        X indexed by session with columns matching FEATURE_NAMES, and the column list.
    """
    if sentiment_cache is None:
        sentiment_cache = pd.read_parquet(_SENTIMENT_CACHE_PATH)

    sessions = pd.Index(np.sort(bars["session"].unique()), name="session")
    close_pivot = _pivot_close(bars, sessions)

    # fh_return = close[49]/close[0] - 1
    c0 = close_pivot[0].replace(0.0, np.nan)
    c49 = close_pivot[49]
    fh_return = (c49 / c0 - 1.0).fillna(0.0)

    # recent_return_3 = close[49]/close[46] - 1
    c46 = close_pivot[46].replace(0.0, np.nan)
    recent_return_3 = (c49 / c46 - 1.0).fillna(0.0)

    # upper_wick_49 = (high[49] - max(open[49], close[49])) / (high[49] - low[49])
    bar49 = bars.loc[bars["bar_ix"] == 49].set_index("session").reindex(sessions)
    body_top = np.maximum(bar49["open"].to_numpy(), bar49["close"].to_numpy())
    range_ = (bar49["high"] - bar49["low"]).replace(0.0, np.nan)
    upper_wick_49 = ((bar49["high"] - body_top) / range_).fillna(0.0)

    S = _bar_sentiment_matrix(headlines, sentiment_cache, sessions)

    X = pd.DataFrame(
        {
            "_const": np.ones(len(sessions), dtype=float),
            "fh_return": fh_return.to_numpy(),
            "yz_vol": _yz_vol(bars, sessions).to_numpy(),
            "recent_return_3": recent_return_3.to_numpy(),
            "upper_wick_49": upper_wick_49.to_numpy(),
            "max_drawdown_fh": _max_drawdown_fh(close_pivot).to_numpy(),
            "garman_klass_vol": _garman_klass_vol(bars, sessions).to_numpy(),
            "rogers_satchell_vol": _rogers_satchell_vol(bars, sessions).to_numpy(),
            "parkinson_vol": _parkinson_vol(bars, sessions).to_numpy(),
            "realized_skewness": _realized_skewness(close_pivot).to_numpy(),
            "rsi_14": _rsi_14(close_pivot).to_numpy(),
            "sent_sma10": S[:, -10:].mean(axis=1),
            "sent_ema20": _sent_ema20(S),
            "sent_wt_decay20": _sent_wt_decay20(S),
            "bmb": _compute_bmb(headlines, sessions).to_numpy(),
        },
        index=sessions,
    )[list(FEATURE_NAMES)]

    # Defensive: replace any inf and NaN with 0.0 before returning.
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return X, list(X.columns)


def validate_no_leakage(X: pd.DataFrame) -> None:
    """Raise ValueError if X leaks unseen-bar info or contains NaN/inf."""
    bad_cols = [
        col for col in X.columns
        if any(substr in str(col).lower() for substr in _LEAK_SUBSTRINGS)
    ]
    if bad_cols:
        raise ValueError(
            f"validate_no_leakage: columns reference unseen/second-half data: {bad_cols}."
        )

    nan_count = int(X.isna().sum().sum())
    if nan_count > 0:
        nan_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"validate_no_leakage: X contains {nan_count} NaN value(s) in columns {nan_cols}."
        )

    numeric = X.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        inf_cols = [
            col for col in numeric.columns
            if not np.isfinite(numeric[col].to_numpy()).all()
        ]
        raise ValueError(
            f"validate_no_leakage: X contains non-finite (inf) values in columns {inf_cols}."
        )
