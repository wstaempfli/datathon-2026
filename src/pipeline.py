"""End-to-end pipeline. Add feature builders to FEATURE_BUILDERS, then run scripts/train_predict.py.

Each feature builder is a callable `builder(bars_seen: pd.DataFrame, headlines_seen: pd.DataFrame,
sentiment_cache: pd.DataFrame) -> pd.DataFrame` that returns a per-session DataFrame indexed by
session, where each column is a feature. Missing sessions get filled with 0.0 automatically.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.data import load_bars, load_headlines, compute_targets
from src.sentiment import load_cache
from src.model import train_cv, fit_final, predict
from src.position import size_positions

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "features"
MODELS_DIR = ROOT / "models"
SUBMISSIONS_DIR = ROOT / "submissions"

FeatureBuilder = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame]

# Add feature-builder callables here — each returns a DataFrame indexed by session.
FEATURE_BUILDERS: list[FeatureBuilder] = []


def _build_X_vol(split: str, sent_cache: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    bars = load_bars(split, seen=True)
    heads = load_headlines(split, seen=True)
    sessions = bars["session"].drop_duplicates().sort_values().to_numpy()
    if FEATURE_BUILDERS:
        frames = [b(bars, heads, sent_cache) for b in FEATURE_BUILDERS]
        X = pd.concat(frames, axis=1).reindex(sessions).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    else:
        # No features registered yet — model trains on a single constant column.
        X = pd.DataFrame({"_const": np.zeros(len(sessions))}, index=pd.Index(sessions, name="session"))
    # Realized vol from seen bars, used only for position sizing (not a model feature).
    log_close = np.log(bars.sort_values(["session","bar_ix"])["close"].clip(lower=1e-12))
    log_ret = log_close.groupby(bars.sort_values(["session","bar_ix"])["session"]).diff()
    vol = log_ret.groupby(bars.sort_values(["session","bar_ix"])["session"]).std().reindex(X.index).fillna(0.01)
    return X, vol


def run_pipeline() -> None:
    FEATURES_DIR.mkdir(exist_ok=True); MODELS_DIR.mkdir(exist_ok=True); SUBMISSIONS_DIR.mkdir(exist_ok=True)
    sent_cache = load_cache(FEATURES_DIR / "sentiment_cache.parquet")
    targets = compute_targets().set_index("session")["target_return"]

    X_train, vol_train = _build_X_vol("train", sent_cache)
    y_train = targets.loc[X_train.index]
    print(f"X_train shape: {X_train.shape}  columns: {list(X_train.columns)}")
    if not FEATURE_BUILDERS:
        print("WARNING: no FEATURE_BUILDERS registered — model is training on a constant.")

    boosters, oof, metrics = train_cv(X_train, y_train, vol_train)
    print({k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))})

    pd.DataFrame({"session": X_train.index, "pred_return": oof, "target_return": y_train.values}) \
      .to_parquet(FEATURES_DIR / "model_oof.parquet", index=False)

    booster = fit_final(X_train, y_train)

    parts = []
    for split in ("public_test", "private_test"):
        X_t, vol_t = _build_X_vol(split, sent_cache)
        preds = predict(booster, X_t)
        pos = size_positions(preds, vol_t.values)
        sub = pd.DataFrame({"session": X_t.index.astype(int), "target_position": pos})
        parts.append(sub)
        print(f"split={split}  rows={len(sub)}  mean={pos.mean():.3f} std={pos.std():.3f}")

    combined = pd.concat(parts, ignore_index=True)
    out = SUBMISSIONS_DIR / "submission.csv"
    combined.to_csv(out, index=False)
    print(f"wrote {out}  rows={len(combined)}")

    fi = metrics.get("feature_importance_gain", {})
    if fi:
        pd.Series(fi).sort_values(ascending=False).to_csv(MODELS_DIR / "feature_importance.csv")
