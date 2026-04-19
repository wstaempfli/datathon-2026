"""Diagnostic harness for experimental rule variants.

Loads train bars/headlines + targets once, evaluates each rule on a deterministic
80/20 session-id split, ranks by min(train_sh, holdout_sh), and (optionally)
materializes test-split CSVs for the top-N rules.

Usage:
    PYTHONPATH=. python3 scripts/experiments.py               # diagnostics only
    PYTHONPATH=. python3 scripts/experiments.py --write-top 4 # + write CSVs

Does NOT submit to Kaggle. Does NOT modify src/rules.py, src/learners.py, or
src/features.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import compute_targets, load_bars, load_headlines  # noqa: E402
from src.model import sharpe  # noqa: E402
from src.rules import (  # noqa: E402
    rule_bmb_recent,
    rule_bmb_sent,
    rule_bmbv2,
    rule_rsi,
    rule_tanh,
    rule_tplridge,
    rule_volnorm,
)

SUBMISSIONS_DIR = ROOT / "submissions"

RULES: list[str] = [
    "rule_bmbv2",      # baseline
    "rule_bmb_sent",
    "rule_bmb_recent",
    "rule_tanh",
    "rule_volnorm",
    "rule_rsi",
    "rule_tplridge",
]

RULE_FNS = {
    "rule_bmbv2": rule_bmbv2,
    "rule_bmb_sent": rule_bmb_sent,
    "rule_bmb_recent": rule_bmb_recent,
    "rule_tanh": rule_tanh,
    "rule_volnorm": rule_volnorm,
    "rule_rsi": rule_rsi,
    "rule_tplridge": rule_tplridge,
}


def _holdout_sessions(sessions: pd.Index) -> pd.Index:
    """Deterministic holdout: last ~20% of sorted unique sessions."""
    ordered = pd.Series(sorted(pd.unique(sessions)))
    n = len(ordered)
    cutoff_idx = int(0.8 * n)
    # Holdout = sessions at index >= cutoff_idx in the sorted list.
    holdout_values = ordered.iloc[cutoff_idx:].to_numpy()
    return pd.Index(holdout_values)


def _pos_stats(pos: pd.Series) -> tuple[float, float, float]:
    values = pos.to_numpy(dtype=float)
    saturated = np.isclose(values, 0.2) | np.isclose(values, 2.0)
    return (
        float(values.mean()),
        float(values.std()),
        float(saturated.mean()),
    )


def _run_diagnostics(
    bars: pd.DataFrame,
    heads: pd.DataFrame,
    targets: pd.Series,
    holdout_idx: pd.Index,
) -> pd.DataFrame:
    rows: list[dict] = []
    train_idx = targets.index.difference(holdout_idx)

    for rule_name in RULES:
        fn = RULE_FNS[rule_name]
        pos = fn(bars, heads).astype(float)
        pos = pos.reindex(targets.index)

        assert pos.notna().all(), f"{rule_name}: NaN in positions after reindex"

        pnl = (pos * targets).astype(float)
        train_pnl = pnl.loc[train_idx].to_numpy()
        holdout_pnl = pnl.loc[holdout_idx].to_numpy()

        train_sh = sharpe(train_pnl)
        holdout_sh = sharpe(holdout_pnl)
        combined = min(train_sh, holdout_sh)

        pos_mean, pos_std, pos_sat = _pos_stats(pos)

        rows.append(
            {
                "rule": rule_name,
                "train_sh": train_sh,
                "holdout_sh": holdout_sh,
                "combined": combined,
                "pos_mean": pos_mean,
                "pos_std": pos_std,
                "pos_sat": pos_sat,
            }
        )

    df = pd.DataFrame(rows).sort_values("combined", ascending=False).reset_index(drop=True)
    return df


def _print_table(df: pd.DataFrame) -> None:
    cols = ["rule", "train_sh", "holdout_sh", "combined", "pos_mean", "pos_std", "pos_sat"]
    widths = {
        "rule": max(max(len(r) for r in df["rule"]), len("rule")),
        "train_sh": max(len("train_sh"), 8),
        "holdout_sh": max(len("holdout_sh"), 10),
        "combined": max(len("combined"), 8),
        "pos_mean": max(len("pos_mean"), 8),
        "pos_std": max(len("pos_std"), 7),
        "pos_sat": max(len("pos_sat"), 7),
    }
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        line = (
            row["rule"].ljust(widths["rule"])
            + "  " + f"{row['train_sh']:.3f}".ljust(widths["train_sh"])
            + "  " + f"{row['holdout_sh']:.3f}".ljust(widths["holdout_sh"])
            + "  " + f"{row['combined']:.3f}".ljust(widths["combined"])
            + "  " + f"{row['pos_mean']:.3f}".ljust(widths["pos_mean"])
            + "  " + f"{row['pos_std']:.3f}".ljust(widths["pos_std"])
            + "  " + f"{row['pos_sat']:.3f}".ljust(widths["pos_sat"])
        )
        print(line)


def _validate_submission(full: pd.DataFrame, rule_name: str) -> None:
    assert len(full) == 20000, f"{rule_name}: expected 20000 rows, got {len(full)}"
    assert full["target_position"].isna().sum() == 0, f"{rule_name}: NaN in positions"
    lo = float(full["target_position"].min())
    hi = float(full["target_position"].max())
    assert lo >= 0.2 - 1e-9, f"{rule_name}: pos_min={lo} < 0.2"
    assert hi <= 2.0 + 1e-9, f"{rule_name}: pos_max={hi} > 2.0"


def _write_top_csvs(top_rules: list[str]) -> None:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("Loading public_test bars/headlines...")
    pub_bars = load_bars("public_test", seen=True)
    pub_heads = load_headlines("public_test", seen=True)

    print("Loading private_test bars/headlines...")
    priv_bars = load_bars("private_test", seen=True)
    priv_heads = load_headlines("private_test", seen=True)

    for rule_name in top_rules:
        fn = RULE_FNS[rule_name]
        pos_pub = fn(pub_bars, pub_heads).astype(float)
        pos_priv = fn(priv_bars, priv_heads).astype(float)

        full = pd.concat(
            [
                pd.DataFrame(
                    {
                        "session": pos_pub.index.astype(int),
                        "target_position": pos_pub.values,
                    }
                ),
                pd.DataFrame(
                    {
                        "session": pos_priv.index.astype(int),
                        "target_position": pos_priv.values,
                    }
                ),
            ],
            ignore_index=True,
        )

        _validate_submission(full, rule_name)

        out = SUBMISSIONS_DIR / f"exp_{rule_name}.csv"
        full.to_csv(out, index=False)
        print(
            f"wrote {out}  rows={len(full)}  "
            f"pos_mean={full['target_position'].mean():.3f}  "
            f"pos_min={full['target_position'].min():.3f}  "
            f"pos_max={full['target_position'].max():.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-top",
        type=int,
        default=0,
        help="If >0, write submissions/exp_<rule>.csv for the top-N rules.",
    )
    args = parser.parse_args()

    print("Loading train bars/headlines + targets...")
    bars = load_bars("train", seen=True)
    heads = load_headlines("train", seen=True)
    targets = (
        compute_targets()
        .set_index("session")["target_return"]
        .astype(float)
        .sort_index()
    )

    holdout_idx = _holdout_sessions(targets.index)
    n_total = len(targets)
    n_holdout = len(holdout_idx)
    print(
        f"Sessions: total={n_total}  train={n_total - n_holdout}  "
        f"holdout={n_holdout}  holdout_min_session={int(holdout_idx.min())}"
    )
    print()

    df = _run_diagnostics(bars, heads, targets, holdout_idx)
    _print_table(df)

    if args.write_top and args.write_top > 0:
        n = min(args.write_top, len(df))
        top_rules = df["rule"].iloc[:n].tolist()
        print()
        print(f"Top-{n} rules by combined score: {top_rules}")
        _write_top_csvs(top_rules)


if __name__ == "__main__":
    main()
