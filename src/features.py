"""Feature engineering for learned variants (V2-V6).

Produces 11 features per session on the seen half (bars 0-49). Pure function;
no I/O. Reuses the pivot idiom from src/pipeline.py._fh_return.

Features:
    fh_0_5, fh_0_10, fh_0_25, fh_0_49   - cumulative returns from bar 0
    fh_25_49, fh_40_49, fh_46_49        - late-window momentum
    max_dd                              - max drawdown on bars 0-49
    rv                                  - std of diff(log(close)) on bars 0-49
    parkinson                           - sqrt((1/(4 ln2 · 49)) · Σ ln(h/l)^2)
    bmb_recent                          - bull/bear regex score from pipeline
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline import _bmb_recent

FEATURE_NAMES: tuple[str, ...] = (
    "fh_0_5",
    "fh_0_10",
    "fh_0_25",
    "fh_0_49",
    "fh_25_49",
    "fh_40_49",
    "fh_46_49",
    "max_dd",
    "rv",
    "parkinson",
    "bmb_recent",
)


def _pivot(bars: pd.DataFrame, col: str) -> tuple[pd.Index, np.ndarray]:
    sessions = pd.Index(np.sort(bars["session"].unique()), name="session")
    pv = (
        bars.pivot(index="session", columns="bar_ix", values=col)
        .reindex(index=sessions, columns=range(50))
    )
    return sessions, pv.to_numpy(dtype=float)


def _safe_ret(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = np.where(b == 0.0, np.nan, b)
    return np.nan_to_num(a / b - 1.0, nan=0.0, posinf=0.0, neginf=0.0)


def realized_vol(close: np.ndarray) -> np.ndarray:
    """Std of diff(log(close)) per row. close shape (n_sessions, 50)."""
    c = np.where(close <= 0.0, np.nan, close)
    log_ret = np.diff(np.log(c), axis=1)
    return np.nan_to_num(np.nanstd(log_ret, axis=1, ddof=1), nan=0.0)


def parkinson_vol(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """sqrt((1/(4 ln2 · 49)) · Σ ln(h/l)^2) per row (uses bars 1..49 like rv)."""
    h = np.where(high <= 0.0, np.nan, high)
    lo = np.where(low <= 0.0, np.nan, low)
    r = np.log(h[:, 1:] / lo[:, 1:]) ** 2
    n = r.shape[1]
    coeff = 1.0 / (4.0 * np.log(2.0) * n)
    return np.nan_to_num(np.sqrt(coeff * np.nansum(r, axis=1)), nan=0.0)


def max_drawdown(close: np.ndarray) -> np.ndarray:
    c = np.where(close <= 0.0, np.nan, close)
    cummax = np.fmax.accumulate(c, axis=1)
    dd = c / cummax - 1.0
    return np.nan_to_num(np.nanmin(dd, axis=1), nan=0.0)


def build_features(bars: pd.DataFrame, heads: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute the 11-feature matrix indexed by session.

    If `heads` is None, bmb_recent is filled with zeros. Callers that want the
    full feature set should pass headlines for the same split as `bars`.
    """
    sessions, close = _pivot(bars, "close")
    _, high = _pivot(bars, "high")
    _, low = _pivot(bars, "low")

    c0 = close[:, 0]
    fh_0_5 = _safe_ret(close[:, 5], c0)
    fh_0_10 = _safe_ret(close[:, 10], c0)
    fh_0_25 = _safe_ret(close[:, 25], c0)
    fh_0_49 = _safe_ret(close[:, 49], c0)
    fh_25_49 = _safe_ret(close[:, 49], close[:, 25])
    fh_40_49 = _safe_ret(close[:, 49], close[:, 40])
    fh_46_49 = _safe_ret(close[:, 49], close[:, 46])

    max_dd = max_drawdown(close)
    rv = realized_vol(close)
    pk = parkinson_vol(high, low)

    if heads is None or len(heads) == 0:
        bmb = np.zeros(len(sessions), dtype=float)
    else:
        bmb = _bmb_recent(heads, sessions, tau=20.0).to_numpy()

    data = np.column_stack(
        [fh_0_5, fh_0_10, fh_0_25, fh_0_49, fh_25_49, fh_40_49, fh_46_49, max_dd, rv, pk, bmb]
    )
    return pd.DataFrame(data, index=sessions, columns=list(FEATURE_NAMES))
