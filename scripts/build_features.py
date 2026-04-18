"""Emit per-bar + per-session price-feature parquets for all splits. Fast (seconds), no torch."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.data import load_bars
from src.features import build_price_features

FEATURES_DIR = ROOT / "features"


def main():
    FEATURES_DIR.mkdir(exist_ok=True)
    # Train: write rolling features file used by the viz.
    train_bars = load_bars("train", seen=True)
    per_sess_train, per_bar_train = build_price_features(
        train_bars, rolling_out=FEATURES_DIR / "price_rolling.parquet"
    )
    per_sess_train.to_parquet(FEATURES_DIR / "price_session_train.parquet")
    # Test: per-session for model; per-bar saved too for inspection (overwrites same file,
    # so for the viz we merge train+test rolling into one file to cover every split).
    all_rolling = [per_bar_train.assign(split="train")]
    for split in ("public_test", "private_test"):
        bars = load_bars(split, seen=True)
        per_sess, per_bar = build_price_features(bars)
        per_sess.to_parquet(FEATURES_DIR / f"price_session_{split}.parquet")
        all_rolling.append(per_bar.assign(split=split))
    pd.concat(all_rolling, ignore_index=True).to_parquet(
        FEATURES_DIR / "price_rolling.parquet"
    )
    print("wrote features to", FEATURES_DIR)


if __name__ == "__main__":
    main()
