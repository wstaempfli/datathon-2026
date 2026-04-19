"""V1b: drift-preserving risk-parity position sizing.

    pos = clip(1 + scaler * (-K * fh + W * bmb_recent), -2, 2)
    scaler = clip(sigma_ref / max(sigma, floor * sigma_ref), 0.5, 2.0)
    sigma  = std(diff(log(close[0:50])))
    bmb_recent = sum_h sign(h) * exp(-(49 - bar_ix_h) / tau)

Scaling only the directional component preserves the +1 positive-drift prior
while risk-parity-sizing the fade+sentiment bet. CV 5-fold: mean=3.128,
min=2.182 (beats baseline mean=3.119, min=2.021 on both).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

K = 24.0
W = 0.375
TAU = 20.0
LO, HI = -2.0, 2.0
SCALER_CLIP = (0.5, 2.0)
FLOOR_FRAC = 0.25

_BULL_PATTERNS: tuple[str, ...] = (
    r"raises outlook",
    r"reports strong demand",
    r"reports \d+% increase in customer acquisition",
    r"sees \d+% margin improvement",
    r"announces \$[\d.]+b share buyback",
    r"launches next-generation",
    r"completes strategic acquisition",
    r"announces breakthrough",
    r"expands operations into",
    r"opens new office in",
    r"completes planned facility upgrade",
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

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_bars(split: str) -> pd.DataFrame:
    return pd.read_parquet(_DATA_DIR / f"bars_seen_{split}.parquet")


def load_headlines(split: str) -> pd.DataFrame:
    return pd.read_parquet(_DATA_DIR / f"headlines_seen_{split}.parquet")


def _fh_return(bars: pd.DataFrame) -> pd.Series:
    sessions = pd.Index(np.sort(bars["session"].unique()), name="session")
    close_pivot = (
        bars.pivot(index="session", columns="bar_ix", values="close")
        .reindex(index=sessions, columns=range(50))
    )
    c0 = close_pivot[0].replace(0.0, np.nan)
    c49 = close_pivot[49]
    return (c49 / c0 - 1.0).fillna(0.0)


def _bmb_recent(heads: pd.DataFrame, sessions: pd.Index, tau: float = TAU) -> pd.Series:
    text = heads["headline"].astype(str)
    is_bull = text.str.contains(_BULL_RE, regex=True, na=False)
    is_bear = text.str.contains(_BEAR_RE, regex=True, na=False)
    pol = (is_bull.astype(int) - (is_bear & ~is_bull).astype(int)).astype(float)
    bar_ix = heads["bar_ix"].to_numpy().astype(float)
    decay = np.exp(-(49.0 - bar_ix) / float(tau))
    contrib = pol.to_numpy() * decay
    agg = (
        pd.DataFrame({"session": heads["session"].to_numpy(), "c": contrib})
        .groupby("session")["c"]
        .sum()
    )
    return agg.reindex(sessions, fill_value=0.0).astype(float)


def realized_vol(bars: pd.DataFrame) -> pd.Series:
    """Per-session std(diff(log(close))) on bars 0-49."""
    sessions = pd.Index(np.sort(bars["session"].unique()), name="session")
    pv = (
        bars.pivot(index="session", columns="bar_ix", values="close")
        .reindex(index=sessions, columns=range(50))
    )
    close = np.where(pv.to_numpy(dtype=float) <= 0.0, np.nan, pv.to_numpy(dtype=float))
    log_ret = np.diff(np.log(close), axis=1)
    sigma = np.nanstd(log_ret, axis=1, ddof=1)
    return pd.Series(np.nan_to_num(sigma, nan=0.0), index=sessions, name="rv")


def vol_scale(
    sigma: np.ndarray,
    sigma_ref: float,
    floor_frac: float = FLOOR_FRAC,
    scaler_clip: tuple[float, float] = SCALER_CLIP,
) -> np.ndarray:
    """Risk-parity scaler: sigma_ref / max(sigma, floor·sigma_ref), clipped."""
    floor = floor_frac * sigma_ref
    denom = np.maximum(np.asarray(sigma, dtype=float), floor)
    denom = np.where(denom <= 0.0, floor, denom)
    return np.clip(sigma_ref / denom, scaler_clip[0], scaler_clip[1])


def predict(bars: pd.DataFrame, heads: pd.DataFrame, sigma_ref: float) -> pd.Series:
    """V1b: drift-preserving vol-scaled rule. Caller supplies sigma_ref from train."""
    fh = _fh_return(bars)
    bmb = _bmb_recent(heads, fh.index).to_numpy()
    sigma = realized_vol(bars).reindex(fh.index).to_numpy()
    scaler = vol_scale(sigma, sigma_ref)
    directional = -K * fh.to_numpy() + W * bmb
    pos = np.clip(1.0 + scaler * directional, LO, HI)
    return pd.Series(pos, index=fh.index, name="target_position")
