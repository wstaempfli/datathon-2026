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

import sys
from pathlib import Path

# Add project root to path so we can import src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def load_data():
    """Load all parquet files from data/.

    Returns
    -------
    dict with keys:
        bars_seen_train, bars_unseen_train,
        bars_seen_public_test, bars_seen_private_test,
        headlines_seen_train, headlines_unseen_train,
        headlines_seen_public_test, headlines_seen_private_test
    """
    # TODO: Load all 8 parquet files from data/
    # TODO: Return as dict
    raise NotImplementedError


def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    # TODO: data = load_data()
    # TODO: bars_seen = data["bars_seen_train"]
    # TODO: bars_unseen = data["bars_unseen_train"]
    # TODO: headlines_seen = data["headlines_seen_train"]

    # ------------------------------------------------------------------
    # 2. Build features and target
    # ------------------------------------------------------------------
    # TODO: from src.features import build_features, get_target, get_halfway_close
    # TODO: features = build_features(bars_seen, headlines_seen, include_tier2=False)
    # TODO: target = get_target(bars_seen, bars_unseen)
    # TODO: X = features.values
    # TODO: y = target.values
    # TODO: feature_names = features.columns.tolist()
    # TODO: print(f"Features: {features.shape}, Target: {y.shape}")

    # ------------------------------------------------------------------
    # 3. Run baselines
    # ------------------------------------------------------------------
    # TODO: from src.evaluate import run_baselines, cross_validate, compare_models
    # TODO: baselines = run_baselines(y)
    # TODO: print("\n=== BASELINES ===")
    # TODO: print(baselines.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Train and evaluate primary model (Ridge)
    # ------------------------------------------------------------------
    # TODO: from src.models import train_ridge, apply_position_sizing, feature_importance
    # TODO: ridge_fn = lambda X, y: train_ridge(X, y, alpha=1.0)
    # TODO: ridge_results = cross_validate(X, y, ridge_fn, n_splits=5, n_seeds=20)
    # TODO: print(f"\n=== RIDGE (alpha=1.0) ===")
    # TODO: print(f"Sharpe: {ridge_results['sharpe_mean']:.4f} ± {ridge_results['sharpe_std']:.4f}")
    # TODO: print(f"Direction accuracy: {ridge_results['accuracy_mean']:.4f}")

    # ------------------------------------------------------------------
    # 5. Feature importance
    # ------------------------------------------------------------------
    # TODO: model = train_ridge(X, y, alpha=1.0)
    # TODO: imp = feature_importance(model, feature_names)
    # TODO: print(f"\n=== FEATURE IMPORTANCE ===")
    # TODO: print(imp)

    # ------------------------------------------------------------------
    # 6. Generate submissions
    # ------------------------------------------------------------------
    # TODO: from src.submit import generate_submission, generate_always_long_submission
    # TODO: For public_test:
    #   TODO: bars_pub = data["bars_seen_public_test"]
    #   TODO: hl_pub = data["headlines_seen_public_test"]
    #   TODO: generate_submission(model, bars_pub, hl_pub,
    #                             "submissions/v01_ridge_public.csv")
    # TODO: For private_test:
    #   TODO: bars_priv = data["bars_seen_private_test"]
    #   TODO: hl_priv = data["headlines_seen_private_test"]
    #   TODO: generate_submission(model, bars_priv, hl_priv,
    #                             "submissions/v01_ridge_private.csv")
    # TODO: Always-long fallbacks:
    #   TODO: generate_always_long_submission(bars_pub, "submissions/v00_always_long_public.csv")
    #   TODO: generate_always_long_submission(bars_priv, "submissions/v00_always_long_private.csv")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    # TODO: Print final summary table comparing all models
    # TODO: Remind which submission to upload

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
