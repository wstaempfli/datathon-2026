"""Build and persist diagnostic alpha-candidate parquets.

Writes:
    features/alpha_candidates.parquet   (per-session alpha signals)
    features/template_hits.parquet      (per-session top-K template counts)
    features/template_dictionary.tsv    (rank -> skeleton -> total_count)

Usage (from repo root):
    source .venv/bin/activate
    python3 scripts/build_alpha_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.alphas import (  # noqa: E402
    build_alpha_candidates,
    build_template_hits,
    extract_skeleton,
)
from src.data import load_headlines  # noqa: E402

FEATURES_DIR = ROOT / "features"
ALPHA_PATH = FEATURES_DIR / "alpha_candidates.parquet"
TEMPLATE_PATH = FEATURES_DIR / "template_hits.parquet"


def _print_schema(name: str, df: pd.DataFrame) -> None:
    print(f"\n[{name}]")
    print(f"  shape       : {df.shape}")
    print(f"  columns     : {list(df.columns)}")
    print(f"  dtypes.nunq : {df.dtypes.nunique()}  (sample: {df.dtypes.iloc[0]})")
    print(f"  nan_total   : {int(df.isna().sum().sum())}")


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Skeleton-count telemetry (independent of build_template_hits guard).
    headlines = load_headlines("train", seen=True)
    n_unique = int(headlines["headline"].astype(str).map(extract_skeleton).nunique())
    print(f"unique skeletons (train): {n_unique}")

    # Build + persist alpha candidates.
    alpha = build_alpha_candidates("train")
    assert int(alpha.isna().sum().sum()) == 0, "alpha_candidates contains NaN — aborting."
    alpha_out = alpha.reset_index()  # viz server needs session as a column
    alpha_out.to_parquet(ALPHA_PATH, index=False)

    # Build + persist template hits.
    templates = build_template_hits("train", top_k=30)
    assert int(templates.isna().sum().sum()) == 0, "template_hits contains NaN — aborting."
    tpl_out = templates.reset_index()
    tpl_out.to_parquet(TEMPLATE_PATH, index=False)

    _print_schema("alpha_candidates.parquet", alpha_out)
    _print_schema("template_hits.parquet", tpl_out)

    print(f"\nwrote {ALPHA_PATH.relative_to(ROOT)}")
    print(f"wrote {TEMPLATE_PATH.relative_to(ROOT)}")
    print(f"wrote {(FEATURES_DIR / 'template_dictionary.tsv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
