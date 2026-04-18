"""Diagnostic-only alpha-signal module.

Computes candidate alpha signals and headline-template hit counts for
cross-sectional analysis in the viz tool and notebooks. This file is a
*diagnostic* surface: it is NOT imported by `src/features.py`, `src/model.py`,
or `src/pipeline.py`, and it does not alter the production feature matrix.

Public surface:
    extract_skeleton(headline) -> str
    build_alpha_candidates(split="train") -> pd.DataFrame
    build_template_hits(split="train", top_k=30) -> pd.DataFrame
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_bars, load_headlines
from src.features import (
    _BEAR_RE,
    _BULL_RE,
    _bar_sentiment_matrix,
    _max_drawdown_fh,
    _pivot_close,
    _realized_skewness,
    _rsi_14,
    _sent_wt_decay20,
    _yz_vol,
)

_ROOT = Path(__file__).resolve().parent.parent
_SENTIMENT_CACHE_PATH = _ROOT / "features" / "sentiment_cache.parquet"
_TEMPLATE_DICT_PATH = _ROOT / "features" / "template_dictionary.tsv"
_N_BARS = 50


# ---- skeleton extractor ---------------------------------------------------

_MONEY = re.compile(r"\$[\d.,]+\s*(?:[mMbBkK])?")
_PCT = re.compile(r"\b\d+(?:\.\d+)?\s*%")
_NUM = re.compile(r"\b\d+(?:\.\d+)?\b")
# Runs of Title/UPPER case words — order matters: collapse entities BEFORE
# residual bare numbers. Must run AFTER [MONEY]/[PCT] substitutions so their
# bracket tokens survive without getting rewritten.
_CAPS = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")


def extract_skeleton(h: str) -> str:
    """Normalize a headline to a template by masking numbers and entities.

    Replaces currency ($500M), percentages (18%), and runs of Title-Case or
    UPPERCASE words (company names, regions) with placeholder tokens, then
    lowercases. Deterministic and pure-string.
    """
    s = _MONEY.sub("[MONEY]", h)
    s = _PCT.sub("[PCT]", s)
    s = _CAPS.sub("[ENT]", s)  # order matters: ENT before NUM
    s = _NUM.sub("[NUM]", s)
    return " ".join(s.split()).lower()


# ---- phrase-category regex taxonomy ---------------------------------------
# Each category is `#positive patterns − #negative patterns`. Patterns are
# 3–6 per side, disjoint-first-design. All compiled case-insensitive.

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


# Frozen output column order for build_alpha_candidates.
_ALPHA_COLUMNS: tuple[str, ...] = (
    # Price alphas
    "fh_return",
    "recent_return_3",
    "rsi_centered",
    "max_drawdown_fh",
    "upper_wick_49",
    "yz_vol",
    "realized_skewness",
    # Headline aggregates
    "n_headlines",
    "bull_count",
    "bear_count",
    "bmb",
    "sent_mean_all",
    "sent_wt_decay20",
    "sent_dispersion",
    # Phrase-category alphas
    "guidance_net",
    "ma_net",
    "contract_net",
    "regulatory_net",
    "earnings_net",
    "ops_net",
)


# ---- helpers --------------------------------------------------------------

def _pattern_counts(
    headlines: pd.DataFrame, sessions: pd.Index, pattern: re.Pattern
) -> pd.Series:
    """Count per-session headlines matching `pattern` (IGNORECASE substring)."""
    text = headlines["headline"].astype(str)
    hits = text.str.contains(pattern, regex=True, na=False).astype(int)
    tally = (
        pd.DataFrame({"session": headlines["session"].to_numpy(), "hit": hits.to_numpy()})
        .groupby("session")["hit"]
        .sum()
    )
    return tally.reindex(sessions, fill_value=0).astype(float)


def _upper_wick_49(bars: pd.DataFrame, sessions: pd.Index) -> pd.Series:
    """(high[49] - max(o, c)) / (h - l) at bar 49. 0.0 when h == l."""
    bar49 = bars.loc[bars["bar_ix"] == 49].set_index("session").reindex(sessions)
    body_top = np.maximum(bar49["open"].to_numpy(), bar49["close"].to_numpy())
    range_ = (bar49["high"] - bar49["low"]).replace(0.0, np.nan)
    wick = ((bar49["high"] - body_top) / range_).fillna(0.0)
    return wick.astype(float)


def _price_alphas(bars: pd.DataFrame, sessions: pd.Index) -> pd.DataFrame:
    """Seven price alphas indexed by session."""
    close_pivot = _pivot_close(bars, sessions)

    c0 = close_pivot[0].replace(0.0, np.nan)
    c46 = close_pivot[46].replace(0.0, np.nan)
    c49 = close_pivot[49]
    fh_return = (c49 / c0 - 1.0).fillna(0.0)
    recent_return_3 = (c49 / c46 - 1.0).fillna(0.0)

    rsi = _rsi_14(close_pivot)
    rsi_centered = (rsi - 50.0) / 50.0

    out = pd.DataFrame(
        {
            "fh_return": fh_return.to_numpy(),
            "recent_return_3": recent_return_3.to_numpy(),
            "rsi_centered": rsi_centered.to_numpy(),
            "max_drawdown_fh": _max_drawdown_fh(close_pivot).to_numpy(),
            "upper_wick_49": _upper_wick_49(bars, sessions).to_numpy(),
            "yz_vol": _yz_vol(bars, sessions).to_numpy(),
            "realized_skewness": _realized_skewness(close_pivot).to_numpy(),
        },
        index=sessions,
    )
    return out


def _headline_aggregates(
    headlines: pd.DataFrame,
    sentiment_cache: pd.DataFrame,
    sessions: pd.Index,
) -> pd.DataFrame:
    """Seven per-session headline aggregates indexed by session."""
    n_headlines = (
        headlines.groupby("session").size().reindex(sessions, fill_value=0).astype(float)
    )

    bull_count = _pattern_counts(headlines, sessions, _BULL_RE)
    bear_count = _pattern_counts(headlines, sessions, _BEAR_RE)
    bmb = (bull_count - bear_count).astype(float)

    # Sentiment aggregates joined from cache.
    merged = headlines.merge(
        sentiment_cache[["headline", "sent_score"]], on="headline", how="left"
    )
    merged["sent_score"] = merged["sent_score"].fillna(0.0)
    g = merged.groupby("session")["sent_score"]
    sent_mean_all = g.mean().reindex(sessions, fill_value=0.0).astype(float)
    sent_dispersion = (
        g.std(ddof=0).reindex(sessions, fill_value=0.0).fillna(0.0).astype(float)
    )

    S = _bar_sentiment_matrix(headlines, sentiment_cache, sessions)
    sent_wt_decay20 = pd.Series(_sent_wt_decay20(S), index=sessions).astype(float)

    return pd.DataFrame(
        {
            "n_headlines": n_headlines.to_numpy(),
            "bull_count": bull_count.to_numpy(),
            "bear_count": bear_count.to_numpy(),
            "bmb": bmb.to_numpy(),
            "sent_mean_all": sent_mean_all.to_numpy(),
            "sent_wt_decay20": sent_wt_decay20.to_numpy(),
            "sent_dispersion": sent_dispersion.to_numpy(),
        },
        index=sessions,
    )


def _phrase_category_nets(
    headlines: pd.DataFrame, sessions: pd.Index
) -> pd.DataFrame:
    """Six phrase-category net columns: #pos hits - #neg hits per category."""
    cols: dict[str, np.ndarray] = {}
    for name, (pos_re, neg_re) in _CATEGORIES.items():
        pos_cnt = _pattern_counts(headlines, sessions, pos_re)
        neg_cnt = _pattern_counts(headlines, sessions, neg_re)
        cols[name] = (pos_cnt - neg_cnt).to_numpy()
    return pd.DataFrame(cols, index=sessions)


# ---- public API -----------------------------------------------------------

def build_alpha_candidates(split: str = "train") -> pd.DataFrame:
    """Build the per-session alpha-candidates DataFrame.

    Returns a DataFrame indexed by `session` with exactly the columns in
    `_ALPHA_COLUMNS` (20 columns, dtype float64, zero NaN).
    """
    bars = load_bars(split, seen=True)
    headlines = load_headlines(split, seen=True)
    sentiment_cache = pd.read_parquet(_SENTIMENT_CACHE_PATH)

    sessions = pd.Index(np.sort(bars["session"].unique()), name="session")

    price = _price_alphas(bars, sessions)
    heads = _headline_aggregates(headlines, sentiment_cache, sessions)
    cats = _phrase_category_nets(headlines, sessions)

    out = pd.concat([price, heads, cats], axis=1)[list(_ALPHA_COLUMNS)]
    out = out.astype(np.float64)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    assert list(out.columns) == list(_ALPHA_COLUMNS), (
        f"alpha column drift: got {list(out.columns)}, expected {list(_ALPHA_COLUMNS)}"
    )
    assert int(out.isna().sum().sum()) == 0, "build_alpha_candidates produced NaN"
    return out


def build_template_hits(split: str = "train", top_k: int = 30) -> pd.DataFrame:
    """Build a per-session wide DataFrame of top-K template hit counts.

    1. Apply `extract_skeleton` to every headline in the split.
    2. If unique-skeleton count is outside [200, 1000], raise with the count.
    3. Rank skeletons by total frequency (tie-break alphabetically).
    4. Write sidecar `features/template_dictionary.tsv`.
    5. Return DataFrame indexed by session with columns `tpl_1..tpl_K`.
    """
    headlines = load_headlines(split, seen=True)
    sessions = pd.Index(
        np.sort(headlines["session"].unique()), name="session"
    )

    skel = headlines["headline"].astype(str).map(extract_skeleton)
    n_unique = int(skel.nunique())
    if n_unique < 200 or n_unique > 1000:
        raise RuntimeError(
            f"build_template_hits: skeleton count {n_unique} outside [200, 1000]; "
            "halting per contract — adjust regex before proceeding."
        )

    # Per-skeleton total frequency, with alphabetic tie-break.
    vc = (
        pd.DataFrame({"skel": skel.to_numpy()})
        .groupby("skel")
        .size()
        .rename("total_count")
        .reset_index()
        .sort_values(["total_count", "skel"], ascending=[False, True])
        .reset_index(drop=True)
    )
    top = vc.head(top_k).copy()
    top["rank"] = np.arange(1, len(top) + 1)

    # Per-session counts for each top skeleton.
    work = pd.DataFrame(
        {"session": headlines["session"].to_numpy(), "skel": skel.to_numpy()}
    )
    work = work[work["skel"].isin(top["skel"])]
    counts = (
        work.groupby(["session", "skel"]).size().unstack(fill_value=0)
    )

    # Rename columns to tpl_<rank> in the frozen rank order.
    rank_by_skel = dict(zip(top["skel"], top["rank"].astype(int)))
    rename_map = {skel_val: f"tpl_{rank_by_skel[skel_val]}" for skel_val in counts.columns}
    counts = counts.rename(columns=rename_map)

    col_order = [f"tpl_{r}" for r in range(1, len(top) + 1)]
    # Some top skeletons may have zero matches in `counts` if no session was
    # captured (unlikely, but defensive reindexing).
    counts = counts.reindex(columns=col_order, fill_value=0)
    counts = counts.reindex(index=sessions, fill_value=0).astype(np.float64)

    # Write dictionary sidecar.
    dict_df = top[["rank", "skel", "total_count"]].rename(columns={"skel": "skeleton"})
    _TEMPLATE_DICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dict_df.to_csv(_TEMPLATE_DICT_PATH, sep="\t", index=False)

    assert int(counts.isna().sum().sum()) == 0, "build_template_hits produced NaN"
    return counts
