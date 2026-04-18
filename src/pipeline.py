from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.data import load_bars, load_headlines, compute_targets
from src.features import build_price_features
from src.sentiment import load_cache, build_sentiment_features
from src.model import train_cv, fit_final, predict, sharpe
from src.position import size_positions

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "features"
MODELS_DIR = ROOT / "models"
SUBMISSIONS_DIR = ROOT / "submissions"


def _build_X(split: str, sent_cache: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    bars = load_bars(split, seen=True)
    heads = load_headlines(split, seen=True)
    # Still build price features — realized_vol is needed for size_positions().
    per_sess_price, _ = build_price_features(bars)
    per_sess_sent, _ = build_sentiment_features(heads, sent_cache)

    # SENTIMENT-ONLY feature matrix. Align to the full price session index so
    # sessions with zero headlines are present with zero-valued sentiment rows.
    sent_cols = [
        "sent_mean",
        "sent_max",
        "sent_min",
        "sent_last",
        "sent_count",
        "sent_abs_mean",
        "pos_share",
        "neg_share",
        "sent_sum",
    ]
    sessions = per_sess_price.sort_index().index
    if per_sess_sent is None or len(per_sess_sent) == 0:
        X = pd.DataFrame(0.0, index=sessions, columns=sent_cols)
    else:
        X = (
            per_sess_sent.reindex(sessions)
            .reindex(columns=sent_cols)
            .astype(float)
        )
    X.index.name = "session"
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.sort_index()
    assert not X.isna().any().any(), "sentiment feature matrix contains NaN"

    vol_series = per_sess_price.loc[X.index, "realized_vol"]
    vol_series = vol_series.fillna(per_sess_price["realized_vol"].median())
    return X, vol_series


def run_pipeline() -> None:
    FEATURES_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    sent_cache = load_cache(FEATURES_DIR / "sentiment_cache.parquet")
    targets = compute_targets().set_index("session")["target_return"]

    X_train, vol_train = _build_X("train", sent_cache)
    y_train = targets.loc[X_train.index]

    print(f"X columns ({len(X_train.columns)}): {list(X_train.columns)}")

    boosters, oof, metrics = train_cv(X_train, y_train, vol_train)
    print(
        "CV metrics:",
        {
            k: round(v, 4) if isinstance(v, (int, float)) else v
            for k, v in metrics.items()
            if not isinstance(v, dict)
        },
    )

    # Save OOF parquet for viz diagnostics.
    pd.DataFrame(
        {
            "session": X_train.index,
            "pred_return": oof,
            "target_return": y_train.values,
        }
    ).to_parquet(FEATURES_DIR / "model_oof.parquet", index=False)

    booster = fit_final(X_train, y_train)

    split_subs = []
    for split in ("public_test", "private_test"):
        X_t, vol_t = _build_X(split, sent_cache)
        preds = predict(booster, X_t)
        pos = size_positions(preds, vol_t.values)
        sub = pd.DataFrame(
            {"session": X_t.index.astype(int), "target_position": pos}
        )
        split_subs.append(sub)
        print(
            f"split={split}  rows={len(sub)}  pos stats: "
            f"mean={pos.mean():.3f} std={pos.std():.3f} "
            f"min={pos.min():.3f} max={pos.max():.3f}"
        )

        # Remove legacy per-split files if present.
        legacy_path = SUBMISSIONS_DIR / f"submission_v1_{split}.csv"
        if legacy_path.exists():
            legacy_path.unlink()

    combined = pd.concat(split_subs, ignore_index=True)
    combined_path = SUBMISSIONS_DIR / "submission_v1.csv"
    combined.to_csv(combined_path, index=False)
    print(
        f"wrote {combined_path}  rows={len(combined)}  pos stats: "
        f"mean={combined['target_position'].mean():.3f} "
        f"std={combined['target_position'].std():.3f} "
        f"min={combined['target_position'].min():.3f} "
        f"max={combined['target_position'].max():.3f}"
    )

    # Also persist feature importances.
    fi = metrics.get("feature_importance_gain", {})
    pd.Series(fi).sort_values(ascending=False).to_csv(
        MODELS_DIR / "feature_importance.csv"
    )
