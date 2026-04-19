"""Phase 0: sweep clip bounds + K_FH for the bmb_recent formula.

The base formula is `clip(1 − K_FH·fh_return + 0.25·bmb_recent, lo, hi)` with
τ=40. Current shipped rule uses (lo, hi) = (0.2, 2.0) — which forbids shorts.
This sweep explores widened bounds (including negative lower bound) and a
re-tuned K_FH.

Usage:
    PYTHONPATH=. python3 scripts/sweep_clip.py
    PYTHONPATH=. python3 scripts/sweep_clip.py --write-csv
    PYTHONPATH=. python3 scripts/sweep_clip.py --submit          # kaggle submit top-3

Ranking: `min` fold Sharpe, then `mean`. No LB-driven selection.
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
from src.features import make_features  # noqa: E402
from src.rules import _bmb_recency_weighted  # noqa: E402

SUBMISSIONS_DIR = ROOT / "submissions"
COMPETITION = "hrt-eth-zurich-datathon-2026"

# Grid.
CLIPS: list[tuple[float, float]] = [
    (0.2, 2.0),      # baseline
    (0.0, 2.0),
    (-0.5, 2.0),
    (-1.0, 2.0),
    (-1.0, 3.0),
    (-2.0, 2.0),
]
KS: list[float] = [20.0, 25.0, 30.0, 35.0]
W_BMB: float = 0.25
TAU: float = 40.0
N_FOLDS = 5
FOLD_SIZE = 200


def _sharpe(pos: np.ndarray, y: np.ndarray) -> float:
    pnl = pos * y
    s = pnl.std()
    return float(pnl.mean() / s * 16.0) if s > 0 else 0.0


def _positions(fh: np.ndarray, bmb: np.ndarray, k: float, lo: float, hi: float) -> np.ndarray:
    raw = 1.0 - k * fh + W_BMB * bmb
    return np.clip(raw, lo, hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--submit", action="store_true",
                        help="kaggle submit the top-3 CV-best variants.")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.submit:
        args.write_csv = True

    print(">> loading train features...")
    bars_tr = load_bars("train", seen=True)
    heads_tr = load_headlines("train", seen=True)
    X, _ = make_features(bars_tr, heads_tr)
    fh = X["fh_return"].to_numpy().astype(float)
    bmb = _bmb_recency_weighted(heads_tr, X.index, tau=TAU).to_numpy().astype(float)
    sessions = X.index.to_numpy()
    y_full = (
        compute_targets()
        .set_index("session")["target_return"]
        .reindex(X.index)
        .astype(float)
    )
    y = y_full.to_numpy()

    rows = []
    for (lo, hi) in CLIPS:
        for k in KS:
            pos = _positions(fh, bmb, k, lo, hi)
            folds = []
            sat = 0.0
            for fi in range(N_FOLDS):
                m = (sessions >= fi * FOLD_SIZE) & (sessions < (fi + 1) * FOLD_SIZE)
                folds.append(_sharpe(pos[m], y[m]))
                sat += float(((pos[m] == lo) | (pos[m] == hi)).mean()) / N_FOLDS
            f = np.asarray(folds)
            rows.append({
                "lo": lo, "hi": hi, "k": k,
                "mean": float(f.mean()), "min": float(f.min()),
                "std_fold": float(f.std(ddof=1)),
                "sat": sat,
                "pos_mean": float(pos.mean()),
                "pos_std": float(pos.std()),
                "folds": np.round(f, 4).tolist(),
            })

    cv = pd.DataFrame(rows).sort_values(["min", "mean"], ascending=False).reset_index(drop=True)
    pd.options.display.float_format = "{:.4f}".format
    print("\n-- Phase 0: clip × K_FH sweep (ranked by min) --")
    print(cv.to_string(index=False))

    baseline = cv[(cv["lo"] == 0.2) & (cv["hi"] == 2.0) & (cv["k"] == 25.0)].iloc[0]
    print(f"\nbaseline (0.2, 2.0, K=25): mean={baseline['mean']:.4f}  "
          f"min={baseline['min']:.4f}  std_fold={baseline['std_fold']:.4f}")

    best = cv.iloc[0]
    std_guard = float(baseline["std_fold"])
    wins = (best["mean"] > baseline["mean"] + std_guard) and (best["min"] > baseline["min"] + std_guard)
    print(f"best variant: lo={best['lo']} hi={best['hi']} k={best['k']:.0f}  "
          f"mean={best['mean']:.4f} min={best['min']:.4f}")
    print(f"verdict: {'SHIP (beats by ≥1σ on both)' if wins else 'KEEP BASELINE'}")

    if not args.write_csv:
        return

    print(f"\n>> writing top-{args.top_k} CSVs + (optionally) submitting to kaggle...")
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print(">> loading public + private test...")
    bars_pub = load_bars("public_test", seen=True)
    heads_pub = load_headlines("public_test", seen=True)
    bars_priv = load_bars("private_test", seen=True)
    heads_priv = load_headlines("private_test", seen=True)

    X_pub, _ = make_features(bars_pub, heads_pub)
    X_priv, _ = make_features(bars_priv, heads_priv)
    fh_pub = X_pub["fh_return"].to_numpy()
    fh_priv = X_priv["fh_return"].to_numpy()
    bmb_pub = _bmb_recency_weighted(heads_pub, X_pub.index, tau=TAU).to_numpy()
    bmb_priv = _bmb_recency_weighted(heads_priv, X_priv.index, tau=TAU).to_numpy()

    for i in range(min(args.top_k, len(cv))):
        row = cv.iloc[i]
        lo, hi, k = float(row["lo"]), float(row["hi"]), float(row["k"])
        pos_pub = _positions(fh_pub, bmb_pub, k, lo, hi)
        pos_priv = _positions(fh_priv, bmb_priv, k, lo, hi)
        full = pd.concat([
            pd.DataFrame({"session": X_pub.index.astype(int), "target_position": pos_pub}),
            pd.DataFrame({"session": X_priv.index.astype(int), "target_position": pos_priv}),
        ], ignore_index=True)
        assert len(full) == 20000
        assert full["target_position"].isna().sum() == 0

        tag = f"clip_lo{_lo_str(lo)}_hi{_hi_str(hi)}_k{int(k):02d}"
        out = SUBMISSIONS_DIR / f"sweep_{tag}.csv"
        full.to_csv(out, index=False)

        print(f"  wrote {out.name}  pos_mean={full['target_position'].mean():.3f}  "
              f"cv_mean={row['mean']:.4f} cv_min={row['min']:.4f}")

        if args.submit:
            msg = (
                f"phase0_clip lo={lo} hi={hi} K={int(k)} "
                f"CV(mean={row['mean']:.4f},min={row['min']:.4f}) "
                f"— diagnostic, not selection"
            )
            try:
                subprocess.run(
                    ["kaggle", "competitions", "submit",
                     "-c", COMPETITION, "-f", str(out), "-m", msg],
                    check=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                print(f"  ! kaggle submit failed: {exc}")


def _lo_str(x: float) -> str:
    s = f"{x:+.1f}".replace(".", "p").replace("+", "p").replace("-", "n")
    return s


def _hi_str(x: float) -> str:
    s = f"{x:.1f}".replace(".", "p")
    return s


if __name__ == "__main__":
    main()
