"""End-to-end training and submission pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.data import compute_targets, load_bars, load_headlines  # noqa: E402
from src.features import make_features, validate_no_leakage  # noqa: E402
from src.model import apply_sizing, fit_final, predict, train_cv, tune_k  # noqa: E402

MODELS_DIR = ROOT / "models"
SUBMISSIONS_DIR = ROOT / "submissions"


def _build_X(split: str) -> pd.DataFrame:
    bars = load_bars(split, seen=True)
    heads = load_headlines(split, seen=True)
    X, _ = make_features(bars, heads)
    return X


def run_pipeline() -> None:
    """Train on the train split and write submissions/submission.csv."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = _build_X("train")
    y_train = compute_targets().set_index("session")["target_return"].loc[X_train.index]

    validate_no_leakage(X_train)
    print(f"X_train shape: {X_train.shape}  columns: {list(X_train.columns)}")

    _models, oof, metrics = train_cv(X_train, y_train)
    print({k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))})

    best_k, k_diag = tune_k(oof, y_train.values.astype(float))
    print("k sweep:")
    for k_val, sh in sorted(k_diag.items()):
        marker = "  <-- best" if k_val == best_k else ""
        print(f"  k={k_val:>7.1f}  sharpe={sh:+.4f}{marker}")
    print(f"selected k = {best_k}")

    booster = fit_final(X_train, y_train)

    parts: list[pd.DataFrame] = []
    for split in ("public_test", "private_test"):
        X_t = _build_X(split)
        assert list(X_t.columns) == list(X_train.columns), (
            f"Column mismatch on {split}: train={list(X_train.columns)} "
            f"vs test={list(X_t.columns)}"
        )
        raw_preds = predict(booster, X_t)
        positions = apply_sizing(raw_preds, best_k)
        parts.append(pd.DataFrame(
            {"session": X_t.index.astype(int), "target_position": positions}
        ))
        print(
            f"split={split}  rows={len(positions)}  "
            f"raw_mean={raw_preds.mean():+.5f} raw_std={raw_preds.std():.5f}  "
            f"pos_mean={positions.mean():.4f} pos_std={positions.std():.4f}  "
            f"pos_min={positions.min():.3f} pos_max={positions.max():.3f}"
        )

    out = SUBMISSIONS_DIR / "submission.csv"
    pd.concat(parts, ignore_index=True).to_csv(out, index=False)
    print(f"wrote {out}  rows={sum(len(p) for p in parts)}")


if __name__ == "__main__":
    run_pipeline()
