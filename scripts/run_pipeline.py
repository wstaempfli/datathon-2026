#!/usr/bin/env python3
"""End-to-end pipeline: load data, build features, cross-validate, train, submit."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.features import build_features, get_target
from src.evaluate import cross_validate
from src.models import train_xgboost, feature_importance
from src.submit import generate_submission

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_PARQUET_FILES = [
    "bars_seen_train",
    "bars_unseen_train",
    "bars_seen_public_test",
    "bars_seen_private_test",
]


def load_data() -> dict[str, pd.DataFrame]:
    """Load all parquet files from data/."""
    return {name: pd.read_parquet(DATA_DIR / f"{name}.parquet") for name in _PARQUET_FILES}


def main():
    parser = argparse.ArgumentParser(description="Run the datathon pipeline end-to-end.")
    parser.add_argument("--name", default="submission", help="Submission file base name")
    args = parser.parse_args()

    data = load_data()
    bars_seen = data["bars_seen_train"]
    bars_unseen = data["bars_unseen_train"]

    features = build_features(bars_seen)
    X = features.values
    feature_names = features.columns.tolist()
    y = get_target(bars_seen, bars_unseen).values

    cv_results = cross_validate(X, y, train_xgboost, n_splits=5, n_seeds=20)
    print(
        f"CV Sharpe: {cv_results['sharpe_mean']:.4f} \u00b1 {cv_results['sharpe_std']:.4f} "
        f"(acc {cv_results['accuracy_mean']:.4f})"
    )

    model = train_xgboost(X, y)
    imp = feature_importance(model, feature_names)
    print("Top features:")
    print(imp.head(5).to_string())

    sub_pub = generate_submission(model, data["bars_seen_public_test"])
    sub_priv = generate_submission(model, data["bars_seen_private_test"])
    sub = pd.concat([sub_pub, sub_priv], ignore_index=True)

    sub_path = f"submissions/{args.name}.csv"
    Path(sub_path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(sub_path, index=False)
    print(f"Saved {sub_path}: {len(sub)} rows")


if __name__ == "__main__":
    main()
