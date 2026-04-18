#!/usr/bin/env python3
"""
End-to-end pipeline: load data → build features → train → evaluate → submit.

This is the main entry point that ties all src/ modules together.
Run from project root:  python scripts/run_pipeline.py

Expected output:
  1. Print baseline Sharpe scores
  2. Train primary model (Ridge) with Tier-1 features
  3. Print CV results (Sharpe mean ± std, direction accuracy)
  4. Print feature importance
  5. Generate submissions for public_test and private_test
  6. Generate always-long fallback submissions
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

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
        help="Base name for submission files (default: submission)"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    data = load_data()
    bars_seen = data["bars_seen_train"]
    bars_unseen = data["bars_unseen_train"]
    headlines_seen = data["headlines_seen_train"]

    # ------------------------------------------------------------------
    # 2. Build features and target
    # ------------------------------------------------------------------
    from src.features import build_features, get_target, get_halfway_close
    from src.evaluate import run_baselines, cross_validate, compare_models
    from src.models import train_ridge, feature_importance
    from src.submit import generate_submission, generate_always_long_submission

    target = get_target(bars_seen, bars_unseen)
    y = target.values

    # ------------------------------------------------------------------
    # 3. Run baselines
    # ------------------------------------------------------------------
    baselines = run_baselines(y)
    print("\n=== BASELINES ===")
    print(baselines.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Compare: OHLC-only vs OHLC+Headlines
    # ------------------------------------------------------------------
    all_results = {}

    # --- OHLC-only (6 features) ---
    features_ohlc = build_features(bars_seen, headlines=None, include_tier2=False)
    X_ohlc = features_ohlc.values
    fn_ohlc = features_ohlc.columns.tolist()
    print(f"\nOHLC features: {features_ohlc.shape} — {fn_ohlc}")

    ridge_ohlc_fn = lambda X_tr, y_tr: train_ridge(X_tr, y_tr, alpha=1.0)
    ridge_ohlc = cross_validate(X_ohlc, y, ridge_ohlc_fn, n_splits=5, n_seeds=20)
    all_results["Ridge OHLC-only"] = ridge_ohlc
    print(f"\n=== RIDGE OHLC-ONLY (alpha=1.0) ===")
    print(f"Sharpe: {ridge_ohlc['sharpe_mean']:.4f} ± {ridge_ohlc['sharpe_std']:.4f}")
    print(f"Direction accuracy: {ridge_ohlc['accuracy_mean']:.4f}")

    # --- OHLC + Headlines (13 features) ---
    features_full = build_features(bars_seen, headlines_seen, include_tier2=False)
    X_full = features_full.values
    fn_full = features_full.columns.tolist()
    print(f"\nFull features: {features_full.shape} — {fn_full}")

    ridge_full_fn = lambda X_tr, y_tr: train_ridge(X_tr, y_tr, alpha=1.0)
    ridge_full = cross_validate(X_full, y, ridge_full_fn, n_splits=5, n_seeds=20)
    all_results["Ridge OHLC+Headlines"] = ridge_full
    print(f"\n=== RIDGE OHLC+HEADLINES (alpha=1.0) ===")
    print(f"Sharpe: {ridge_full['sharpe_mean']:.4f} ± {ridge_full['sharpe_std']:.4f}")
    print(f"Direction accuracy: {ridge_full['accuracy_mean']:.4f}")

    # ------------------------------------------------------------------
    # 5. Select best model and show feature importance
    # ------------------------------------------------------------------
    # Pick whichever feature set gave higher mean Sharpe
    if ridge_ohlc["sharpe_mean"] >= ridge_full["sharpe_mean"]:
        best_name = "Ridge OHLC-only"
        best_X, best_fn = X_ohlc, fn_ohlc
        use_headlines = False
    else:
        best_name = "Ridge OHLC+Headlines"
        best_X, best_fn = X_full, fn_full
        use_headlines = True

    model = train_ridge(best_X, y, alpha=1.0)
    imp = feature_importance(model, best_fn)
    print(f"\n=== FEATURE IMPORTANCE ({best_name}) ===")
    print(imp.to_string())

    # ------------------------------------------------------------------
    # 6. Generate submissions
    # ------------------------------------------------------------------
    print("\n=== GENERATING SUBMISSIONS ===")
    bars_pub = data["bars_seen_public_test"]
    hl_pub = data["headlines_seen_public_test"] if use_headlines else None
    bars_priv = data["bars_seen_private_test"]
    hl_priv = data["headlines_seen_private_test"] if use_headlines else None

    # Model submission: concat public + private into one file
    sub_pub = generate_submission(model, bars_pub, hl_pub)
    sub_priv = generate_submission(model, bars_priv, hl_priv)
    sub_model = pd.concat([sub_pub, sub_priv], ignore_index=True)
    model_path = f"submissions/{args.name}.csv"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    sub_model.to_csv(model_path, index=False)
    positions = sub_model["target_position"]
    print(f"Saved {model_path}: {len(sub_model)} rows, "
          f"positions in [{positions.min():.2f}, {positions.max():.2f}], "
          f"mean={positions.mean():.3f}, "
          f"{(positions > 0).mean() * 100:.1f}% long, "
          f"{(positions < 0).mean() * 100:.1f}% short")

    # Always-long fallback: concat public + private into one file
    al_pub = generate_always_long_submission(bars_pub)
    al_priv = generate_always_long_submission(bars_priv)
    sub_always_long = pd.concat([al_pub, al_priv], ignore_index=True)
    always_long_path = "submissions/v00_always_long.csv"
    Path(always_long_path).parent.mkdir(parents=True, exist_ok=True)
    sub_always_long.to_csv(always_long_path, index=False)
    print(f"Saved always-long submission to {always_long_path}: "
          f"{len(sub_always_long)} sessions, all positions = +1.0")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    summary = compare_models(all_results)
    print("\n=== MODEL COMPARISON ===")
    print(summary.to_string(index=False))
    print(f"\nAlways-long baseline Sharpe: {baselines[baselines['strategy'] == 'always_long']['sharpe'].values[0]:.4f}")
    print(f"\nBest model: {best_name}")
    print(f"Headlines used: {use_headlines}")
    print("\nSubmissions generated:")
    print(f"  - {model_path}  (primary)")
    print(f"  - {always_long_path}  (fallback)")
    print("\nPipeline complete.")
    print(f"\nUsage: python scripts/run_pipeline.py --name <submission_name>")


if __name__ == "__main__":
    main()
