"""Grid-search (K_FH, W_BMB) for the BMBv2 clipped rule.

Formula:
    target_position = clip(1 - K_FH * fh_return + W_BMB * bmb, 0.2, 2.0)

Usage:
    PYTHONPATH=. python3 scripts/sweep_bmbv2.py --dry-run
    PYTHONPATH=. python3 scripts/sweep_bmbv2.py --write-top 5
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
from src.features import make_features  # noqa: E402

SUBMISSIONS_DIR = ROOT / "submissions"
CLIP_LO, CLIP_HI = 0.2, 2.0

K_GRID = [15, 20, 25, 30, 35, 40, 50]
W_GRID = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


def _positions(fh_return: np.ndarray, bmb: np.ndarray, K: float, W: float) -> np.ndarray:
    raw = 1.0 - K * fh_return + W * bmb
    return np.clip(raw, CLIP_LO, CLIP_HI)


def _sharpe(pos: np.ndarray, y: np.ndarray) -> float:
    pnl = pos * y
    sd = pnl.std()
    return float(pnl.mean() / sd * 16.0) if sd > 0 else 0.0


def _fname(K: float, W: float) -> str:
    return f"sweep_bmbv2_k{int(K)}_w{int(round(W * 100)):02d}.csv"


def _validate(df: pd.DataFrame, tag: str) -> None:
    assert len(df) == 20000, f"{tag}: expected 20000 rows, got {len(df)}"
    assert df["target_position"].isna().sum() == 0, f"{tag}: NaN in positions"
    lo = float(df["target_position"].min())
    hi = float(df["target_position"].max())
    assert lo >= CLIP_LO - 1e-9, f"{tag}: pos_min={lo} < {CLIP_LO}"
    assert hi <= CLIP_HI + 1e-9, f"{tag}: pos_max={hi} > {CLIP_HI}"


def _run_sweep_train() -> list[dict]:
    """Load train once, call make_features once, evaluate all 49 combos."""
    bars = load_bars("train", seen=True)
    heads = load_headlines("train", seen=True)
    X, _ = make_features(bars, heads)

    y_df = compute_targets().set_index("session")
    y = y_df["target_return"].reindex(X.index).astype(float).to_numpy()

    fh = X["fh_return"].to_numpy()
    bmb = X["bmb"].to_numpy()

    results: list[dict] = []
    for K in K_GRID:
        for W in W_GRID:
            pos = _positions(fh, bmb, K, W)
            sh = _sharpe(pos, y)
            pos_sat = float(
                ((pos <= CLIP_LO + 1e-9) | (pos >= CLIP_HI - 1e-9)).mean()
            )
            results.append(
                {
                    "K_FH": K,
                    "W_BMB": W,
                    "sharpe": sh,
                    "pos_mean": float(pos.mean()),
                    "pos_std": float(pos.std()),
                    "pos_saturation": pos_sat,
                }
            )
    return results


def _print_grid(results: list[dict]) -> None:
    df = pd.DataFrame(results)
    grid = df.pivot(index="K_FH", columns="W_BMB", values="sharpe")
    grid = grid.reindex(index=K_GRID, columns=W_GRID)

    header = "K_FH\\W_BMB | " + "  ".join(f"{w:>6.2f}" for w in W_GRID)
    print(header)
    print("-" * len(header))
    for K in K_GRID:
        cells = "  ".join(f"{grid.loc[K, w]:>6.3f}" for w in W_GRID)
        print(f"{K:>9d} | {cells}")


def _print_top(results: list[dict], n: int = 10) -> list[dict]:
    ranked = sorted(results, key=lambda r: r["sharpe"], reverse=True)
    print(f"\nTop-{n} combos (by train Sharpe):")
    for r in ranked[:n]:
        print(
            f"K={int(r['K_FH']):>2d} W={r['W_BMB']:.2f}  "
            f"sharpe={r['sharpe']:.3f}  "
            f"pos_mean={r['pos_mean']:.2f}  "
            f"pos_sat={r['pos_saturation']:.2f}"
        )
    return ranked


def _predict_split(split: str, K: float, W: float) -> pd.DataFrame:
    bars = load_bars(split, seen=True)
    heads = load_headlines(split, seen=True)
    X, _ = make_features(bars, heads)
    pos = _positions(X["fh_return"].to_numpy(), X["bmb"].to_numpy(), K, W)
    return pd.DataFrame({"session": X.index.astype(int), "target_position": pos})


def _write_top(ranked: list[dict], top_n: int) -> list[Path]:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Cache public/private features once.
    print(f"\nLoading public_test + private_test features (once each)...")
    pub_bars = load_bars("public_test", seen=True)
    pub_heads = load_headlines("public_test", seen=True)
    X_pub, _ = make_features(pub_bars, pub_heads)

    priv_bars = load_bars("private_test", seen=True)
    priv_heads = load_headlines("private_test", seen=True)
    X_priv, _ = make_features(priv_bars, priv_heads)

    fh_pub, bmb_pub = X_pub["fh_return"].to_numpy(), X_pub["bmb"].to_numpy()
    fh_priv, bmb_priv = X_priv["fh_return"].to_numpy(), X_priv["bmb"].to_numpy()

    for r in ranked[:top_n]:
        K, W = r["K_FH"], r["W_BMB"]
        pub_pos = _positions(fh_pub, bmb_pub, K, W)
        priv_pos = _positions(fh_priv, bmb_priv, K, W)
        pub_df = pd.DataFrame({"session": X_pub.index.astype(int), "target_position": pub_pos})
        priv_df = pd.DataFrame({"session": X_priv.index.astype(int), "target_position": priv_pos})
        full = pd.concat([pub_df, priv_df], ignore_index=True)

        out = SUBMISSIONS_DIR / _fname(K, W)
        full.to_csv(out, index=False)
        _validate(full, out.name)
        print(
            f"  wrote {out.name}  "
            f"train_sharpe={r['sharpe']:.3f}  "
            f"pos_mean={full['target_position'].mean():.3f}  "
            f"pos_min={full['target_position'].min():.3f}  "
            f"pos_max={full['target_position'].max():.3f}  "
            f"rows={len(full)}"
        )
        written.append(out)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Train-only sweep; no CSV writes.")
    parser.add_argument("--write-top", type=int, default=0, metavar="N",
                        help="After ranking, write the top-N combos to submissions/.")
    args = parser.parse_args()

    print(f"Running sweep over {len(K_GRID)}x{len(W_GRID)} = "
          f"{len(K_GRID)*len(W_GRID)} combos on train split...")
    results = _run_sweep_train()

    print("\nTrain-Sharpe grid (rows=K_FH, cols=W_BMB):")
    _print_grid(results)
    ranked = _print_top(results, n=10)

    if args.write_top and not args.dry_run:
        _write_top(ranked, args.write_top)
    elif args.dry_run:
        print("\n--dry-run: no CSVs written.")


if __name__ == "__main__":
    main()
