"""Minimal self-contained pipeline reproducing rule_bmb_recent byte-for-byte.

Formula: clip(1 - 35 * fh_return + 0.25 * bmb_recent, -2.0, 2.0) with tau=40.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

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


def _bmb_recent(heads: pd.DataFrame, sessions: pd.Index, tau: float = 40.0) -> pd.Series:
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


def predict(bars: pd.DataFrame, heads: pd.DataFrame) -> pd.Series:
    k = 24.0
    w = 0.375
    lo = -2.0
    hi = 2.0
    tau = 20.0

    fh = _fh_return(bars)
    bmb = _bmb_recent(heads, fh.index, tau=tau)
    raw = 1.0 - k * fh.to_numpy() + w * bmb.to_numpy()
    return pd.Series(np.clip(raw, lo, hi), index=fh.index, name="target_position")
