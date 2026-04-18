"""End-to-end pipeline. Imports `make_features` from `src.features` — the feature-engineer agent owns that file; this module only consumes it."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_bars, load_headlines, compute_targets
from src.sentiment import load_cache
from src.features import make_features, validate_no_leakage
from src.model import train_cv, fit_final, predict

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "features"
MODELS_DIR = ROOT / "models"
SUBMISSIONS_DIR = ROOT / "submissions"


def _build_X(
    split: str,
    sent_cache: pd.DataFrame,
    training_stats: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build (X, fitted_stats) for a given split.

    Fit mode: pass ``training_stats=None`` — ``make_features`` computes any
    cross-session statistics and returns them in ``fitted_stats``.
    Apply mode: pass the stats returned from the training call so nothing is
    re-fit on test data.
    """
    bars = load_bars(split, seen=True)
    heads = load_headlines(split, seen=True)

    X, _feature_names, fitted_stats = make_features(
        bars, heads, sent_cache, training_stats=training_stats
    )

    # Defensive reindex + NaN/inf guard. `make_features` is contracted to return
    # a finite, NaN-free frame already; this catches regressions without relying
    # on the callee alone.
    sessions = bars["session"].drop_duplicates().sort_values().to_numpy()
    X = (
        X.reindex(sessions)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )

    return X, fitted_stats


def run_pipeline() -> None:
    FEATURES_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    sent_cache = load_cache(FEATURES_DIR / "sentiment_cache.parquet")
    targets = compute_targets().set_index("session")["target_return"]

    # --- Train: fit mode ---------------------------------------------------
    X_train, fitted_stats = _build_X("train", sent_cache, training_stats=None)
    y_train = targets.loc[X_train.index]

    # Fail fast on leakage / NaN / inf before burning time on training.
    validate_no_leakage(X_train)

    print(
        f"X_train shape: {X_train.shape}  "
        f"n_features={X_train.shape[1]}  "
        f"columns: {list(X_train.columns)}"
    )
    if list(X_train.columns) == ["_const"]:
        print("WARNING: only _const feature — model has no real signal yet.")

    boosters, oof, metrics = train_cv(X_train, y_train)
    print({k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))})

    pd.DataFrame(
        {
            "session": X_train.index,
            "pred_return": oof,
            "target_return": y_train.values,
        }
    ).to_parquet(FEATURES_DIR / "model_oof.parquet", index=False)

    booster = fit_final(X_train, y_train)

    # --- Test: apply mode using fitted_stats from training -----------------
    # target_position is the raw model prediction for now. Position sizing
    # (vol-scaling, clipping, rescaling) will be added back later.
    parts = []
    for split in ("public_test", "private_test"):
        X_t, _ = _build_X(split, sent_cache, training_stats=fitted_stats)
        assert set(X_t.columns) == set(X_train.columns), (
            f"Column mismatch on {split}: "
            f"train_only={set(X_train.columns) - set(X_t.columns)}, "
            f"test_only={set(X_t.columns) - set(X_train.columns)}"
        )
        # Enforce column *order* as well, since XGBoost DMatrix is positional.
        X_t = X_t[list(X_train.columns)]

        preds = predict(booster, X_t)
        sub = pd.DataFrame(
            {"session": X_t.index.astype(int), "target_position": preds}
        )
        parts.append(sub)
        print(
            f"split={split}  rows={len(sub)}  "
            f"mean={preds.mean():.5f} std={preds.std():.5f}"
        )

    combined = pd.concat(parts, ignore_index=True)
    out = SUBMISSIONS_DIR / "submission.csv"
    combined.to_csv(out, index=False)
    print(f"wrote {out}  rows={len(combined)}")

    fi = metrics.get("feature_importance_gain", {})
    if fi:
        pd.Series(fi).sort_values(ascending=False).to_csv(
            MODELS_DIR / "feature_importance.csv"
        )
