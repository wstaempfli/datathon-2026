"""Compute and submit the rule_bmb_recent CSV to Kaggle.

The rule has no learned parameters — predict is deterministic. Use --dry-run to
write the CSV without submitting.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import load_bars, load_headlines, predict  # noqa: E402

MSG = (
    "temp "
    "...."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    parts: list[pd.DataFrame] = []
    for split in ("public_test", "private_test"):
        positions = predict(load_bars(split), load_headlines(split))
        parts.append(
            pd.DataFrame(
                {"session": positions.index.astype(int), "target_position": positions.to_numpy()}
            )
        )

    out_df = pd.concat(parts, ignore_index=True)
    assert len(out_df) == 20000, f"expected 20000 rows, got {len(out_df)}"

    out = ROOT / "submissions" / "rule_bmb_recent.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"wrote {out}  rows={len(out_df)}")

    if args.dry_run:
        print("dry-run: skipping kaggle submit")
        return

    print(f"submitting to kaggle: {MSG}")
    subprocess.run(
        ["kaggle", "competitions", "submit", "-c", "hrt-eth-zurich-datathon-2026",
         "-f", str(out), "-m", MSG],
        check=True,
    )


if __name__ == "__main__":
    main()
