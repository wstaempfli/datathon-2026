#!/usr/bin/env python3
"""
End-to-end pipeline: load data → build features → cross-validate → train → submit.

Run from project root:
  python scripts/run_pipeline.py --name v01_ohlc
  python scripts/run_pipeline.py --name v02_full --headlines --alpha 0.5

Steps:
  1. Parse CLI args (--name, --headlines, --alpha)
  2. Load all parquet data
  3. Build features (OHLC-only by default; add headlines if --headlines)
  4. Compute target return
  5. Cross-validate (20 seeds, 5 folds)
  6. Train final Ridge model on all training data
  7. Print feature importance
  8. Generate one combined submission (public + private test)
  9. Print summary
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.features import build_features, get_target
from src.evaluate import cross_validate
from src.models import train_ridge, feature_importance
from src.submit import generate_submission

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# All parquet files that follow the naming schema:
#   {bars,headlines}_{seen,unseen}_{train,public_test,private_test}.parquet
# Unseen splits only exist for train (we don't have future bars/headlines for test).
_PARQUET_FILES = [
    "bars_seen_train",
    "bars_unseen_train",
    "bars_seen_public_test",
    "bars_seen_private_test",
    "headlines_seen_train",
    "headlines_unseen_train",
    "headlines_seen_public_test",
    "headlines_seen_private_test",
]


def load_data() -> dict[str, pd.DataFrame]:
    """Load all parquet files from data/.

    Returns
    -------
    dict with keys:
        bars_seen_train, bars_unseen_train,
        bars_seen_public_test, bars_seen_private_test,
        headlines_seen_train, headlines_unseen_train,
        headlines_seen_public_test, headlines_seen_private_test
    """
    data = {}
    for name in _PARQUET_FILES:
        path = DATA_DIR / f"{name}.parquet"
        data[name] = pd.read_parquet(path)
        print(f"  Loaded {name}: {data[name].shape}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Run the datathon pipeline end-to-end."
    )
    parser.add_argument(
        "--name", default="submission",
        help="Base name for submission file (default: submission)"
    )
    parser.add_argument(
        "--headlines", action="store_true",
        help="Include headline-derived features (default: OHLC-only)"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Ridge regression alpha (default: 1.0)"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=== LOADING DATA ===")
    data = load_data()
    bars_seen = data["bars_seen_train"]
    bars_unseen = data["bars_unseen_train"]
    headlines_seen = data["headlines_seen_train"] if args.headlines else None

    # ------------------------------------------------------------------
    # 2. Build features and target
    # ------------------------------------------------------------------
    print("\n=== BUILDING FEATURES ===")
    features = build_features(bars_seen, headlines=headlines_seen, include_tier2=False)
    X = features.values
    feature_names = features.columns.tolist()
    print(f"Features: {features.shape} — {feature_names}")

    target = get_target(bars_seen, bars_unseen)
    y = target.values

    # ------------------------------------------------------------------
    # 3. Cross-validate
    # ------------------------------------------------------------------
    print(f"\n=== CROSS-VALIDATION (alpha={args.alpha}) ===")
    train_fn = lambda x_tr, y_tr: train_ridge(x_tr, y_tr, alpha=args.alpha)
    cv_results = cross_validate(X, y, train_fn, n_splits=5, n_seeds=20)
    print(f"Sharpe: {cv_results['sharpe_mean']:.4f} ± {cv_results['sharpe_std']:.4f}")
    print(f"Direction accuracy: {cv_results['accuracy_mean']:.4f}")

    # ------------------------------------------------------------------
    # 4. Train final model on all data
    # ------------------------------------------------------------------
    print("\n=== TRAINING FINAL MODEL ===")
    model = train_ridge(X, y, alpha=args.alpha)

    # ------------------------------------------------------------------
    # 5. Feature importance
    # ------------------------------------------------------------------
    imp = feature_importance(model, feature_names)
    print("\n=== FEATURE IMPORTANCE ===")
    print(imp.to_string())

    # ------------------------------------------------------------------
    # 6. Generate submission
    # ------------------------------------------------------------------
    print("\n=== GENERATING SUBMISSION ===")
    bars_pub = data["bars_seen_public_test"]
    hl_pub = data["headlines_seen_public_test"] if args.headlines else None
    bars_priv = data["bars_seen_private_test"]
    hl_priv = data["headlines_seen_private_test"] if args.headlines else None

    sub_pub = generate_submission(model, bars_pub, hl_pub)
    sub_priv = generate_submission(model, bars_priv, hl_priv)
    sub = pd.concat([sub_pub, sub_priv], ignore_index=True)

    sub_path = f"submissions/{args.name}.csv"
    Path(sub_path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(sub_path, index=False)

    positions = sub["target_position"]
    print(f"Saved {sub_path}: {len(sub)} rows, "
          f"positions in [{positions.min():.2f}, {positions.max():.2f}], "
          f"mean={positions.mean():.3f}, "
          f"{(positions > 0).mean() * 100:.1f}% long, "
          f"{(positions < 0).mean() * 100:.1f}% short")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print("\n=== SUMMARY ===")
    print(f"Headlines: {'yes' if args.headlines else 'no'}")
    print(f"Ridge alpha: {args.alpha}")
    print(f"CV Sharpe: {cv_results['sharpe_mean']:.4f} ± {cv_results['sharpe_std']:.4f}")
    print(f"Submission: {sub_path} ({len(sub)} sessions)")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
