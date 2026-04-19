"""Generate (and optionally submit) the rule_bmb_recent CSV to Kaggle.

Usage:
    python scripts/submit_rules.py --dry-run    # write CSV + diagnostics, no submit
    python scripts/submit_rules.py              # write CSV + submit once

Emits exactly one submission: rule_bmb_recent (recency-weighted BMB, tau=40, W=0.25) — CV combined=2.758.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import compute_targets, load_bars, load_headlines  # noqa: E402
from src.rules import rule_bmb_recent  # noqa: E402

SUBMISSIONS_DIR = ROOT / "submissions"
COMPETITION = "hrt-eth-zurich-datathon-2026"

RULES = [
    ("rule_bmb_recent", "rule_bmb_recent recency-decay tau=40 W=0.25 (CV combined=2.758)"),
]

RULE_FNS = {
    "rule_bmb_recent": rule_bmb_recent,
}


def _train_sharpe(rule_name: str) -> tuple[float, pd.Series]:
    """Compute (positions * y).mean()/std * 16 on train, return (sharpe, positions)."""
    bars = load_bars("train", seen=True)
    heads = load_headlines("train", seen=True)
    fn = RULE_FNS[rule_name]
    pos = fn(bars, heads).astype(float)
    y = (
        compute_targets()
        .set_index("session")["target_return"]
        .reindex(pos.index)
        .astype(float)
    )
    pnl = pos.values * y.values
    sharpe = float(pnl.mean() / pnl.std() * 16.0) if pnl.std() > 0 else 0.0
    return sharpe, pos


def _predict_for_split(rule_name: str, split: str) -> pd.DataFrame:
    bars = load_bars(split, seen=True)
    heads = load_headlines(split, seen=True)
    fn = RULE_FNS[rule_name]
    positions = fn(bars, heads).astype(float)
    return pd.DataFrame(
        {
            "session": positions.index.astype(int),
            "target_position": positions.values,
        }
    )


def _validate(df: pd.DataFrame, rule_name: str) -> None:
    assert len(df) == 20000, f"{rule_name}: expected 20000 rows, got {len(df)}"
    assert df["target_position"].isna().sum() == 0, f"{rule_name}: NaN in positions"
    lo = float(df["target_position"].min())
    hi = float(df["target_position"].max())
    assert lo >= 0.2 - 1e-9, f"{rule_name}: pos_min={lo} < 0.2"
    assert hi <= 2.0 + 1e-9, f"{rule_name}: pos_max={hi} > 2.0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate CSVs + diagnostics; do not call kaggle.")
    args = parser.parse_args()

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    for rule_name, msg in RULES:
        # Train-Sharpe diagnostic.
        train_sharpe, train_pos = _train_sharpe(rule_name)

        # Public + private predictions.
        pub_df = _predict_for_split(rule_name, "public_test")
        priv_df = _predict_for_split(rule_name, "private_test")
        full = pd.concat([pub_df, priv_df], ignore_index=True)

        out = SUBMISSIONS_DIR / f"{rule_name}.csv"
        full.to_csv(out, index=False)

        _validate(full, rule_name)

        pos = full["target_position"]
        print(
            f"{rule_name}: train_sharpe={train_sharpe:.4f}  "
            f"pos_mean={pos.mean():.3f}  pos_std={pos.std():.3f}  "
            f"pos_min={pos.min():.3f}  pos_max={pos.max():.3f}  "
            f"rows={len(full)}"
        )

        if not args.dry_run:
            try:
                subprocess.run(
                    [
                        "kaggle", "competitions", "submit",
                        "-c", COMPETITION,
                        "-f", str(out),
                        "-m", msg,
                    ],
                    check=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                print(f"  ! kaggle submit failed for {rule_name}: {exc}")


if __name__ == "__main__":
    main()
