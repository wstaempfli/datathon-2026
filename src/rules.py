"""Rule-based position sizing variants for cross-validation submissions.

Pure functions that take (bars, headlines, sent=None) and return a Series of
target_position values clipped to [0.2, 2.0], indexed by session.

Category-net regex tuples are inlined (copied from `src/alphas.py`) to keep
this module split-agnostic and avoid `build_alpha_candidates("train")`'s
hardcoded split.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.features import make_features

# ---- constants ------------------------------------------------------------

CLIP_LO, CLIP_HI = 0.2, 2.0
K_FH = 25.0
W_BMB = 0.15


# ---- inlined phrase-category regex taxonomy (mirrors src/alphas.py) -------

_GUIDANCE_POS = (
    r"raises outlook",
    r"raises guidance",
    r"reports strong demand",
    r"reiterates full-year guidance",
)
_GUIDANCE_NEG = (
    r"cuts guidance",
    r"lowers outlook",
    r"withdraws guidance",
    r"revises long-term strategy",
)

_MA_POS = (
    r"completes strategic acquisition",
    r"enters joint venture",
    r"signs multi-year partnership",
    r"announces merger",
)
_MA_NEG = (
    r"explores strategic alternatives",
    r"terminates acquisition",
    r"calls off merger",
    r"divests non-core",
)

_CONTRACT_POS = (
    r"secures \$\d+m contract",
    r"secures \$\d+b contract",
    r"wins industry award",
    r"wins .* award",
    r"announces significant capital expenditure",
)
_CONTRACT_NEG = (
    r"loses key contract",
    r"contract terminated",
    r"fails to renew",
)

_REGULATORY_POS = (
    r"files for regulatory approval",
    r"receives regulatory approval",
    r"cleared by regulators",
    r"files routine patent applications",
)
_REGULATORY_NEG = (
    r"faces regulatory review",
    r"faces class action",
    r"under regulatory investigation",
    r"addresses investor concerns in open letter",
)

_EARNINGS_POS = (
    r"reports record .*revenue",
    r"reports record quarterly revenue",
    r"reports \d+% increase in customer acquisition",
    r"sees \d+% margin improvement",
    r"announces \$[\d.]+b share buyback",
)
_EARNINGS_NEG = (
    r"misses .*estimates",
    r"misses quarterly revenue estimates",
    r"reports \d+% decline in operating income",
    r"reports unexpected decline",
    r"reports rising costs pressuring margins",
    r"sees \d+% drop in new customer orders",
)

_OPS_POS = (
    r"launches next-generation",
    r"expands operations into",
    r"announces breakthrough",
    r"opens new office",
    r"completes planned facility upgrade",
)
_OPS_NEG = (
    r"delays product launch",
    r"recalls products",
    r"warns of supply chain disruptions",
    r"withdraws from .* market citing unfavorable",
    r"steps down unexpectedly",
)


_CATEGORIES: dict[str, tuple[re.Pattern, re.Pattern]] = {
    "guidance_net": (
        re.compile("|".join(_GUIDANCE_POS), re.IGNORECASE),
        re.compile("|".join(_GUIDANCE_NEG), re.IGNORECASE),
    ),
    "ma_net": (
        re.compile("|".join(_MA_POS), re.IGNORECASE),
        re.compile("|".join(_MA_NEG), re.IGNORECASE),
    ),
    "contract_net": (
        re.compile("|".join(_CONTRACT_POS), re.IGNORECASE),
        re.compile("|".join(_CONTRACT_NEG), re.IGNORECASE),
    ),
    "regulatory_net": (
        re.compile("|".join(_REGULATORY_POS), re.IGNORECASE),
        re.compile("|".join(_REGULATORY_NEG), re.IGNORECASE),
    ),
    "earnings_net": (
        re.compile("|".join(_EARNINGS_POS), re.IGNORECASE),
        re.compile("|".join(_EARNINGS_NEG), re.IGNORECASE),
    ),
    "ops_net": (
        re.compile("|".join(_OPS_POS), re.IGNORECASE),
        re.compile("|".join(_OPS_NEG), re.IGNORECASE),
    ),
}


# ---- helpers --------------------------------------------------------------

def _pattern_counts(
    headlines: pd.DataFrame, sessions: pd.Index, pattern: re.Pattern
) -> pd.Series:
    """Count per-session headlines matching `pattern` (IGNORECASE substring)."""
    text = headlines["headline"].astype(str)
    hits = text.str.contains(pattern, regex=True, na=False).astype(int)
    tally = (
        pd.DataFrame(
            {"session": headlines["session"].to_numpy(), "hit": hits.to_numpy()}
        )
        .groupby("session")["hit"]
        .sum()
    )
    return tally.reindex(sessions, fill_value=0).astype(float)


def _category_nets(headlines: pd.DataFrame, sessions: pd.Index) -> pd.DataFrame:
    """Six per-session category-net columns: #pos hits - #neg hits."""
    cols: dict[str, np.ndarray] = {}
    for name, (pos_re, neg_re) in _CATEGORIES.items():
        pos_cnt = _pattern_counts(headlines, sessions, pos_re)
        neg_cnt = _pattern_counts(headlines, sessions, neg_re)
        cols[name] = (pos_cnt - neg_cnt).to_numpy()
    return pd.DataFrame(cols, index=sessions)


# ---- rule variants --------------------------------------------------------

def _clip(values: np.ndarray) -> np.ndarray:
    return np.clip(values, CLIP_LO, CLIP_HI)


def rule_bmbv2(bars: pd.DataFrame, heads: pd.DataFrame, sent=None) -> pd.Series:
    """Base formula. BMB regex v2 is already in features.py."""
    X, _ = make_features(bars, heads, sentiment_cache=sent)
    raw = 1.0 - K_FH * X["fh_return"].to_numpy() + W_BMB * X["bmb"].to_numpy()
    return pd.Series(_clip(raw), index=X.index, name="target_position")


def rule_dd(bars: pd.DataFrame, heads: pd.DataFrame, sent=None) -> pd.Series:
    """Base + -20*max_drawdown_fh (mdd is negative -> positive contribution)."""
    X, _ = make_features(bars, heads, sentiment_cache=sent)
    raw = (
        1.0
        - K_FH * X["fh_return"].to_numpy()
        + W_BMB * X["bmb"].to_numpy()
        - 20.0 * X["max_drawdown_fh"].to_numpy()
    )
    return pd.Series(_clip(raw), index=X.index, name="target_position")


def rule_bmbv2_dd(bars: pd.DataFrame, heads: pd.DataFrame, sent=None) -> pd.Series:
    """Same as rule_dd; BMB v2 already in features.py. Explicit name for log."""
    return rule_dd(bars, heads, sent=sent)


def rule_cat(bars: pd.DataFrame, heads: pd.DataFrame, sent=None) -> pd.Series:
    """Base with bmb replaced by category split."""
    X, _ = make_features(bars, heads, sentiment_cache=sent)
    cats = _category_nets(heads, X.index)
    raw = (
        1.0
        - K_FH * X["fh_return"].to_numpy()
        + 0.10 * cats["guidance_net"].to_numpy()
        + 0.10 * cats["contract_net"].to_numpy()
        + 0.15 * cats["regulatory_net"].to_numpy()
        + 0.10 * cats["earnings_net"].to_numpy()
    )
    return pd.Series(_clip(raw), index=X.index, name="target_position")


def rule_sink(bars: pd.DataFrame, heads: pd.DataFrame, sent=None) -> pd.Series:
    """Kitchen sink: BMBv2 + 0.15*bmb + -15*mdd + 0.05*(guidance+contract)."""
    X, _ = make_features(bars, heads, sentiment_cache=sent)
    cats = _category_nets(heads, X.index)
    raw = (
        1.0
        - K_FH * X["fh_return"].to_numpy()
        + 0.15 * X["bmb"].to_numpy()
        - 15.0 * X["max_drawdown_fh"].to_numpy()
        + 0.05 * cats["guidance_net"].to_numpy()
        + 0.05 * cats["contract_net"].to_numpy()
    )
    return pd.Series(_clip(raw), index=X.index, name="target_position")
