"""Produce + submit V1b (drift-preserving vol-scaled rule) to Kaggle.

V1b formula:
    pos = clip(1 + scaler * (-24*fh + 0.375*bmb_recent), -2, 2)
    scaler = clip(sigma_ref / max(sigma, 0.25*sigma_ref), 0.5, 2.0)

sigma_ref = median(realized_vol) on train. CV 5-fold: mean=3.128, min=2.182.

Usage:
    PYTHONPATH=. python3 scripts/submit.py --dry-run
    PYTHONPATH=. python3 scripts/submit.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import load_bars, load_headlines, predict, realized_vol  # noqa: E402

KAGGLE_COMP = "hrt-eth-zurich-datathon-2026"
FILENAME = "v1b_rule_volscaled_directional_drift1_k24_w0.375_tau20.csv"
MSG = (
    "V1b rule+volscale(directional) | "
    "pos=clip(1+scaler*(-24*fh+0.375*bmb),+-2) "
    "scaler=clip(sigma_ref/sigma,0.5,2.0) floor=0.25*sigma_ref | "
    "CV mean=3.128 min=2.182"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sigma_ref = float(realized_vol(load_bars("train")).median())
    print(f"sigma_ref (train median) = {sigma_ref:.6f}")

    parts: list[pd.DataFrame] = []
    for split in ("public_test", "private_test"):
        pos = predict(load_bars(split), load_headlines(split), sigma_ref=sigma_ref)
        parts.append(pd.DataFrame({
            "session": pos.index.astype(int).to_numpy(),
            "target_position": pos.to_numpy(),
        }))

    df = pd.concat(parts, ignore_index=True)
    assert len(df) == 20000, f"expected 20000 rows, got {len(df)}"
    assert df["target_position"].notna().all()
    assert (df["target_position"] >= -2.0 - 1e-9).all()
    assert (df["target_position"] <= 2.0 + 1e-9).all()

    out = ROOT / "submissions" / FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  rows={len(df)}")

    if args.dry_run:
        print(f"dry-run: skipping kaggle submit")
        print(f"  msg: {MSG}")
        return

    print(f"submitting to kaggle: {MSG}")
    subprocess.run(
        ["kaggle", "competitions", "submit", "-c", KAGGLE_COMP,
         "-f", str(out), "-m", MSG],
        check=True,
    )


if __name__ == "__main__":
    main()
