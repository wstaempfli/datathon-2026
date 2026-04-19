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

from pathlib import Path

from src.features import _BEAR_RE, _BULL_RE, make_features

_ROOT = Path(__file__).resolve().parent.parent
_SENTIMENT_CACHE_PATH = _ROOT / "features" / "sentiment_cache.parquet"

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


# ---- experimental helpers -------------------------------------------------

def _load_sentiment_cache(sent: pd.DataFrame | None) -> pd.DataFrame:
    if sent is not None:
        return sent
    return pd.read_parquet(_SENTIMENT_CACHE_PATH)


def _headline_polarity(heads: pd.DataFrame) -> pd.Series:
    """Signed polarity: +1 if bull regex matches, -1 if bear, 0 otherwise.

    Bull takes precedence over bear when both match (matches rule_bmbv2 which
    counts as bull if _BULL_RE matched, regardless of _BEAR_RE).
    """
    text = heads["headline"].astype(str)
    is_bull = text.str.contains(_BULL_RE, regex=True, na=False)
    is_bear = text.str.contains(_BEAR_RE, regex=True, na=False)
    pol = is_bull.astype(int) - (is_bear & ~is_bull).astype(int)
    return pol.astype(float)


def _bmb_sent_weighted(
    heads: pd.DataFrame, sessions: pd.Index, sent: pd.DataFrame
) -> pd.Series:
    """Per-session Σ sign_h · |sent_score_h| over matched bull/bear headlines."""
    pol = _headline_polarity(heads)
    if not pol.any():
        return pd.Series(0.0, index=sessions, name="bmb_sent")
    merged = heads[["session", "headline"]].assign(pol=pol.to_numpy())
    merged = merged.merge(
        sent[["headline", "sent_score"]], on="headline", how="left"
    )
    merged["sent_score"] = merged["sent_score"].fillna(0.0)
    merged["contrib"] = merged["pol"] * merged["sent_score"].abs()
    agg = merged.groupby("session")["contrib"].sum()
    return agg.reindex(sessions, fill_value=0.0).astype(float).rename("bmb_sent")


def _bmb_recency_weighted(
    heads: pd.DataFrame, sessions: pd.Index, tau: float = 20.0
) -> pd.Series:
    """Per-session Σ sign_h · exp(-(49 - bar_ix_h)/tau)."""
    pol = _headline_polarity(heads)
    bar_ix = heads["bar_ix"].to_numpy().astype(float)
    decay = np.exp(-(49.0 - bar_ix) / float(tau))
    contrib = pol.to_numpy() * decay
    agg = (
        pd.DataFrame({"session": heads["session"].to_numpy(), "c": contrib})
        .groupby("session")["c"]
        .sum()
    )
    return agg.reindex(sessions, fill_value=0.0).astype(float).rename("bmb_recent")


def _bmb_late_window(
    heads: pd.DataFrame, sessions: pd.Index, min_bar: int = 35
) -> pd.Series:
    """Per-session Σ sign_h · 1[bar_ix_h ≥ min_bar] (hard late-session cutoff)."""
    pol = _headline_polarity(heads)
    keep = (heads["bar_ix"].to_numpy() >= int(min_bar)).astype(float)
    contrib = pol.to_numpy() * keep
    agg = (
        pd.DataFrame({"session": heads["session"].to_numpy(), "c": contrib})
        .groupby("session")["c"]
        .sum()
    )
    return agg.reindex(sessions, fill_value=0.0).astype(float).rename("bmb_late")


def _bmb_recent_sent_weighted(
    heads: pd.DataFrame, sessions: pd.Index, sent: pd.DataFrame, tau: float = 40.0
) -> pd.Series:
    """Per-session Σ sign_h · |sent_score_h| · exp(-(49 - bar_ix_h)/tau).

    Combines recency decay (rewards late-session news) and FinBERT confidence
    (rewards high-conviction headlines) on the same bull/bear-matched set.
    """
    pol = _headline_polarity(heads)
    if not pol.any():
        return pd.Series(0.0, index=sessions, name="bmb_recent_sent")
    decay = np.exp(-(49.0 - heads["bar_ix"].to_numpy().astype(float)) / float(tau))
    merged = heads[["session", "headline"]].assign(
        pol=pol.to_numpy(), decay=decay
    )
    merged = merged.merge(
        sent[["headline", "sent_score"]], on="headline", how="left"
    )
    merged["sent_score"] = merged["sent_score"].fillna(0.0)
    merged["contrib"] = merged["pol"] * merged["sent_score"].abs() * merged["decay"]
    agg = merged.groupby("session")["contrib"].sum()
    return (
        agg.reindex(sessions, fill_value=0.0)
        .astype(float)
        .rename("bmb_recent_sent")
    )


# ---- experimental rule variants ------------------------------------------

def rule_bmb_sent(bars: pd.DataFrame, heads: pd.DataFrame, sent=None) -> pd.Series:
    """clip(1 - 25·fh_return + 0.15·bmb_sent).

    bmb_sent = Σ_h sign(h) · |sent_score_h| over matched bull/bear headlines.
    """
    X, _ = make_features(bars, heads, sentiment_cache=sent)
    sent_df = _load_sentiment_cache(sent)
    bmb_sent = _bmb_sent_weighted(heads, X.index, sent_df)
    raw = 1.0 - K_FH * X["fh_return"].to_numpy() + 0.15 * bmb_sent.to_numpy()
    return pd.Series(_clip(raw), index=X.index, name="target_position")


def rule_bmb_recent(bars: pd.DataFrame, heads: pd.DataFrame, sent=None) -> pd.Series:
    """clip(1 - 35·fh_return + 0.25·bmb_recent, -2.0, 2.0) with tau=40.

    Hard-coded to Phase 0 sweep_clip variant (lo=-2.0, hi=2.0, K=35, W=0.25, tau=40)
    selected by LB diagnostic on submissions/sweep_clip_lon2p0_hi2p0_k35.csv.
    CV: mean=3.044, min=1.982, std_fold=1.469.
    """
    X, _ = make_features(bars, heads, sentiment_cache=sent)
    bmb_recent = _bmb_recency_weighted(heads, X.index, tau=40.0)
    raw = 1.0 - 35.0 * X["fh_return"].to_numpy() + 0.25 * bmb_recent.to_numpy()
    return pd.Series(np.clip(raw, -2.0, 2.0), index=X.index, name="target_position")


def rule_bmb_late(bars: pd.DataFrame, heads: pd.DataFrame, sent=None) -> pd.Series:
    """clip(1 - 25·fh_return + 0.20·bmb_late) with min_bar=35 (last 15 bars only).

    Holdout sweep peaks at min_bar=35 — only bull/bear headlines from the second
    half-half of the seen window count. Holdout=2.789 vs 2.744 for tau=40.
    """
    X, _ = make_features(bars, heads, sentiment_cache=sent)
    bmb_late = _bmb_late_window(heads, X.index, min_bar=35)
    raw = 1.0 - K_FH * X["fh_return"].to_numpy() + 0.20 * bmb_late.to_numpy()
    return pd.Series(_clip(raw), index=X.index, name="target_position")


def rule_bmb_recent_sent(bars: pd.DataFrame, heads: pd.DataFrame, sent=None) -> pd.Series:
    """clip(1 - 25·fh_return + 0.20·bmb_recent_sent) with tau=40.

    Multiplies recency decay by FinBERT |sent_score|: late-session, high-conviction
    headlines weigh most. Same coefficient and tau as rule_bmb_recent so direct compare.
    """
    X, _ = make_features(bars, heads, sentiment_cache=sent)
    sent_df = _load_sentiment_cache(sent)
    bmb_rs = _bmb_recent_sent_weighted(heads, X.index, sent_df, tau=40.0)
    raw = 1.0 - K_FH * X["fh_return"].to_numpy() + 0.20 * bmb_rs.to_numpy()
    return pd.Series(_clip(raw), index=X.index, name="target_position")


def rule_tanh(bars: pd.DataFrame, heads: pd.DataFrame, sent=None) -> pd.Series:
    """clip(1 - 25·tanh(fh_return/sigma) + 0.15·bmb) with sigma=0.02.

    Caps the mean-reversion tilt for extreme fh_return values.
    """
    X, _ = make_features(bars, heads, sentiment_cache=sent)
    sigma = 0.02
    fh = X["fh_return"].to_numpy()
    raw = 1.0 - K_FH * np.tanh(fh / sigma) * sigma + W_BMB * X["bmb"].to_numpy()
    # Note: K·tanh(fh/sigma)·sigma preserves small-signal slope K (since
    # tanh(x)≈x near 0, so K·tanh(fh/sigma)·sigma ≈ K·fh). For large |fh| the
    # contribution saturates at ±K·sigma = ±0.5.
    return pd.Series(_clip(raw), index=X.index, name="target_position")


def rule_volnorm(bars: pd.DataFrame, heads: pd.DataFrame, sent=None) -> pd.Series:
    """clip(1 - K_eff·(fh_return/yz_vol) + 0.15·bmb).

    Scales reversal strength inversely by session volatility. K_eff=0.35 chosen
    so that for typical yz_vol≈0.014 the effective slope matches 25 (0.35/0.014 ≈ 25).
    """
    X, _ = make_features(bars, heads, sentiment_cache=sent)
    fh = X["fh_return"].to_numpy()
    vol = X["yz_vol"].to_numpy()
    vol_safe = np.where(vol > 1e-6, vol, 1e-6)
    raw = 1.0 - 0.35 * (fh / vol_safe) + W_BMB * X["bmb"].to_numpy()
    return pd.Series(_clip(raw), index=X.index, name="target_position")


def rule_rsi(bars: pd.DataFrame, heads: pd.DataFrame, sent=None) -> pd.Series:
    """clip(1 - 25·fh_return + 0.15·bmb - 0.2·(rsi_14 - 50)/50).

    RSI > 50 (overbought tilt) adds short pressure; RSI < 50 adds long pressure.
    """
    X, _ = make_features(bars, heads, sentiment_cache=sent)
    rsi_centered = (X["rsi_14"].to_numpy() - 50.0) / 50.0
    raw = (
        1.0
        - K_FH * X["fh_return"].to_numpy()
        + W_BMB * X["bmb"].to_numpy()
        - 0.2 * rsi_centered
    )
    return pd.Series(_clip(raw), index=X.index, name="target_position")


def rule_tplridge(bars: pd.DataFrame, heads: pd.DataFrame, sent=None) -> pd.Series:
    """clip(1 - 25·fh_return + weighted_template_score).

    Uses per-template weights fitted on train targets via ridge regression
    (see src/learners.py). Falls back to rule_bmbv2 if weights unavailable.
    """
    from src.learners import compute_template_features, load_template_weights

    X, _ = make_features(bars, heads, sentiment_cache=sent)
    try:
        weights, skeletons = load_template_weights()
    except FileNotFoundError:
        return rule_bmbv2(bars, heads, sent=sent)

    tpl_hits = compute_template_features(heads, X.index, skeletons)
    # Weighted sum: Σ_t β_t · count(tpl_t, s).
    score = tpl_hits.to_numpy() @ weights
    raw = 1.0 - K_FH * X["fh_return"].to_numpy() + score
    return pd.Series(_clip(raw), index=X.index, name="target_position")
