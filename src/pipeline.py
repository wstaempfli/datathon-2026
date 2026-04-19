"""Prediction pipeline: baseline rule (V0), vol-scaled rule (V1), learned variants.

V0 `predict(bars, heads)` is preserved byte-for-byte so existing tests and the
shipped submission reproduce identically. New variants live in dedicated
functions and a `LearnedPredictor` dataclass.

Formulas:
    V0  pos = clip(1 - 24·fh + 0.375·bmb_recent,            -2, 2)
    V1  pos = clip((1 - 24·fh + 0.375·bmb_recent) · s,      -2, 2)
    V2  Ridge / V3 Lasso / V4 Huber on 11 feats:
        pos = clip(pred_centered · k + 1,                   -2, 2)
    V5  Huber + vol scaling:
        pos = clip(pred_centered · k · s + 1,               -2, 2)
    V6  0.5·V1 + 0.5·V5
where s = clip(sigma_ref / max(sigma, 0.25·sigma_ref), 0.5, 2.0).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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


def load_unseen_bars(split: str = "train") -> pd.DataFrame:
    return pd.read_parquet(_DATA_DIR / f"bars_unseen_{split}.parquet")


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
    """V0: shipped rule. Preserved byte-for-byte."""
    k = 24.0
    w = 0.375
    lo = -2.0
    hi = 2.0
    tau = 20.0

    fh = _fh_return(bars)
    bmb = _bmb_recent(heads, fh.index, tau=tau)
    raw = 1.0 - k * fh.to_numpy() + w * bmb.to_numpy()
    return pd.Series(np.clip(raw, lo, hi), index=fh.index, name="target_position")


def _rule_raw(
    bars: pd.DataFrame,
    heads: pd.DataFrame,
    k: float = 24.0,
    w: float = 0.375,
    tau: float = 20.0,
) -> pd.Series:
    fh = _fh_return(bars)
    bmb = _bmb_recent(heads, fh.index, tau=tau)
    return pd.Series(1.0 - k * fh.to_numpy() + w * bmb.to_numpy(), index=fh.index)


def realized_vol_series(bars: pd.DataFrame) -> pd.Series:
    """Per-session std(diff(log(close))) on bars 0-49. Index = session."""
    sessions = pd.Index(np.sort(bars["session"].unique()), name="session")
    pv = (
        bars.pivot(index="session", columns="bar_ix", values="close")
        .reindex(index=sessions, columns=range(50))
    )
    close = pv.to_numpy(dtype=float)
    close = np.where(close <= 0.0, np.nan, close)
    log_ret = np.diff(np.log(close), axis=1)
    sigma = np.nanstd(log_ret, axis=1, ddof=1)
    return pd.Series(np.nan_to_num(sigma, nan=0.0), index=sessions, name="rv")


def predict_v1(
    bars: pd.DataFrame,
    heads: pd.DataFrame,
    sigma_ref: float,
    floor_frac: float = 0.25,
    scaler_clip: tuple[float, float] = (0.5, 2.0),
    lo: float = -2.0,
    hi: float = 2.0,
) -> pd.Series:
    """V1: shipped rule × risk-parity vol scaler (whole signal including drift)."""
    from src.positions import to_position, vol_scale

    raw = _rule_raw(bars, heads)
    rv = realized_vol_series(bars).reindex(raw.index).to_numpy()
    scaler = vol_scale(rv, sigma_ref, floor_frac, scaler_clip)
    pos = to_position(raw.to_numpy(), scaler, intercept=0.0, clip_lo=lo, clip_hi=hi)
    return pd.Series(pos, index=raw.index, name="target_position")


def predict_v1b(
    bars: pd.DataFrame,
    heads: pd.DataFrame,
    sigma_ref: float,
    drift: float = 1.0,
    floor_frac: float = 0.25,
    scaler_clip: tuple[float, float] = (0.5, 2.0),
    k: float = 24.0,
    w: float = 0.375,
    tau: float = 20.0,
    lo: float = -2.0,
    hi: float = 2.0,
) -> pd.Series:
    """V1b: vol-scale ONLY the directional (fade + sentiment) component; keep drift unscaled.

    pos = clip(drift + scaler·(−k·fh + w·bmb), lo, hi).

    Rationale: the rule's k=24 already encodes an inverse-vol fade strength
    implicitly (high-vol sessions have larger |fh|, already yielding larger
    positions before clip). Multiplying the whole signal including the +1
    drift intercept double-dips. V1b decouples them.
    """
    from src.positions import vol_scale

    fh = _fh_return(bars)
    bmb = _bmb_recent(heads, fh.index, tau=tau).reindex(fh.index).to_numpy()
    rv = realized_vol_series(bars).reindex(fh.index).to_numpy()
    scaler = vol_scale(rv, sigma_ref, floor_frac, scaler_clip)
    directional = -k * fh.to_numpy() + w * bmb
    pos = np.clip(drift + scaler * directional, lo, hi)
    return pd.Series(pos, index=fh.index, name="target_position")


def predict_v1c(
    bars: pd.DataFrame,
    heads: pd.DataFrame,
    sigma_ref: float,
    drift: float = 1.0,
    floor_frac: float = 0.5,
    scaler_clip: tuple[float, float] = (0.75, 1.5),
    k: float = 24.0,
    w: float = 0.375,
    tau: float = 20.0,
    lo: float = -2.0,
    hi: float = 2.0,
) -> pd.Series:
    """V1c: V1b with tighter scaler bounds (0.75–1.5, floor 0.5). Conservative scaling."""
    return predict_v1b(
        bars, heads, sigma_ref=sigma_ref, drift=drift, floor_frac=floor_frac,
        scaler_clip=scaler_clip, k=k, w=w, tau=tau, lo=lo, hi=hi,
    )


@dataclass
class LearnedPredictor:
    """Captures a fitted (scaler, estimator) + tuned k + sigma_ref.

    Instantiate via `fit_learned(...)`. Call `.predict(bars_test, heads_test)`
    to produce positions for a held-out split.

    `vol_mode`:
        "none"         no vol scaling (const scaler=1)
        "full"         scaler multiplies (pred*k + intercept) — legacy V5
        "directional"  scaler multiplies only pred*k; intercept unscaled (V5b)
    """

    kind: Literal["ridge", "lasso", "huber"]
    pipe: object
    y_mean: float
    k: float
    sigma_ref: float
    vol_mode: Literal["none", "full", "directional"] = "none"
    intercept: float = 1.0
    lo: float = -2.0
    hi: float = 2.0

    def predict(self, bars: pd.DataFrame, heads: pd.DataFrame) -> pd.Series:
        from src.features import build_features
        from src.positions import vol_scale

        feats = build_features(bars, heads)
        pred = self.pipe.predict(feats.to_numpy()) - self.y_mean
        if self.vol_mode == "none":
            scaler = np.ones(len(feats), dtype=float)
        else:
            rv = realized_vol_series(bars).reindex(feats.index).to_numpy()
            scaler = vol_scale(rv, self.sigma_ref)

        if self.vol_mode == "directional":
            raw = self.intercept + scaler * (pred * self.k)
        else:
            raw = scaler * (pred * self.k + self.intercept)
        pos = np.clip(raw, self.lo, self.hi)
        return pd.Series(pos, index=feats.index, name="target_position")


def fit_learned(
    bars_train: pd.DataFrame,
    heads_train: pd.DataFrame,
    target_return_train: pd.Series,
    kind: Literal["ridge", "lasso", "huber"],
    vol_mode: Literal["none", "full", "directional"] = "none",
    k_grid: tuple[int, ...] = (10, 15, 20, 25, 30, 35, 40, 45, 50),
    intercept: float = 1.0,
) -> LearnedPredictor:
    """Fit regressor on train, tune k in-sample on train, return predictor."""
    from src.features import build_features
    from src.models import _make
    from src.positions import vol_scale

    X = build_features(bars_train, heads_train)
    y = target_return_train.reindex(X.index).to_numpy(dtype=float)
    pipe = _make(kind).fit(X.to_numpy(), y)

    y_mean = float(y.mean())
    rv_train = realized_vol_series(bars_train).reindex(X.index).to_numpy()
    sigma_ref = float(np.median(rv_train))

    pred_tr = pipe.predict(X.to_numpy()) - y_mean
    if vol_mode == "none":
        scaler_tr = np.ones(len(X), dtype=float)
    else:
        scaler_tr = vol_scale(rv_train, sigma_ref)

    # In-sample k line search, honoring vol_mode
    best_k, best_s = k_grid[0], -np.inf
    for kg in k_grid:
        if vol_mode == "directional":
            raw = intercept + scaler_tr * (pred_tr * kg)
        else:
            raw = scaler_tr * (pred_tr * kg + intercept)
        pos = np.clip(raw, -2.0, 2.0)
        pnl = pos * y
        sd = float(np.std(pnl))
        sharpe = (float(np.mean(pnl)) / sd * 16.0) if sd > 0 else -np.inf
        if sharpe > best_s:
            best_s, best_k = sharpe, kg

    return LearnedPredictor(
        kind=kind, pipe=pipe, y_mean=y_mean, k=float(best_k),
        sigma_ref=sigma_ref, vol_mode=vol_mode, intercept=intercept,
    )


def predict_v6_blend(
    bars: pd.DataFrame,
    heads: pd.DataFrame,
    sigma_ref: float,
    learned: LearnedPredictor,
    w_rule: float = 0.5,
    w_learned: float = 0.5,
    lo: float = -2.0,
    hi: float = 2.0,
) -> pd.Series:
    """V6: weighted average of V1b (drift-preserving rule+vol) and a learned predictor."""
    pos_v1 = predict_v1b(bars, heads, sigma_ref=sigma_ref, lo=lo, hi=hi)
    pos_learned = learned.predict(bars, heads).reindex(pos_v1.index)
    blended = w_rule * pos_v1.to_numpy() + w_learned * pos_learned.to_numpy()
    return pd.Series(np.clip(blended, lo, hi), index=pos_v1.index, name="target_position")
