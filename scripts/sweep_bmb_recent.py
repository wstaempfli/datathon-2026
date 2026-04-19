"""Grid-search (tau, W_recent) for rule_bmb_recent on train + holdout.

The base formula is `clip(1 - 25·fh_return + W_recent·bmb_recent, 0.2, 2.0)`
where `bmb_recent_s = Σ_h sign(h) · exp(-(49 - bar_ix_h)/tau)` over matched
bull/bear headlines in session s.

Holdout = sessions 800..999 (deterministic 80/20 cut). Combined ranking metric
is `min(train_sharpe, holdout_sharpe)` — pessimistic ordering.

Usage:
    PYTHONPATH=. python3 scripts/sweep_bmb_recent.py
    PYTHONPATH=. python3 scripts/sweep_bmb_recent.py --write-top 3
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
from src.rules import K_FH, _bmb_recency_weighted, _clip  # noqa: E402

SUBMISSIONS_DIR = ROOT / "submissions"

TAUS = [35.0, 40.0, 45.0, 50.0, 60.0, 80.0, 120.0]
W_RECENTS = [0.15, 0.18, 0.20, 0.22, 0.25, 0.30]
HOLDOUT_START = 800


def _positions(bars: pd.DataFrame, heads: pd.DataFrame, tau: float, w: float) -> pd.Series:
    X, _ = make_features(bars, heads)
    bmb_recent = _bmb_recency_weighted(heads, X.index, tau=tau)
    raw = 1.0 - K_FH * X["fh_return"].to_numpy() + w * bmb_recent.to_numpy()
    return pd.Series(_clip(raw), index=X.index, name="target_position")


def _sharpe(pos: pd.Series, y: pd.Series) -> float:
    pnl = pos.values * y.values
    s = pnl.std()
    return float(pnl.mean() / s * 16.0) if s > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-top", type=int, default=0,
                        help="Materialize submission CSVs for top-N by combined score.")
    args = parser.parse_args()

    bars_tr = load_bars("train", seen=True)
    heads_tr = load_headlines("train", seen=True)
    y_tr_full = (
        compute_targets()
        .set_index("session")["target_return"]
        .astype(float)
    )

    rows = []
    cache: dict[tuple[float, float], pd.Series] = {}
    for tau in TAUS:
        for w in W_RECENTS:
            pos = _positions(bars_tr, heads_tr, tau, w)
            y = y_tr_full.reindex(pos.index)
            mask_tr = pos.index < HOLDOUT_START
            mask_hd = pos.index >= HOLDOUT_START
            s_tr = _sharpe(pos[mask_tr], y[mask_tr])
            s_hd = _sharpe(pos[mask_hd], y[mask_hd])
            s_all = _sharpe(pos, y)
            sat = float(((pos == 0.2) | (pos == 2.0)).mean())
            combined = min(s_tr, s_hd)
            rows.append({
                "tau": tau, "w": w,
                "sharpe_train": s_tr, "sharpe_holdout": s_hd,
                "sharpe_all": s_all, "sat": sat,
                "combined": combined,
                "pos_mean": float(pos.mean()),
                "pos_std": float(pos.std()),
            })
            cache[(tau, w)] = pos

    df = pd.DataFrame(rows).sort_values("combined", ascending=False).reset_index(drop=True)
    pd.options.display.float_format = "{:.4f}".format
    print(df.to_string(index=False))

    if args.write_top > 0:
        SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
        bars_pub = load_bars("public_test", seen=True)
        heads_pub = load_headlines("public_test", seen=True)
        bars_priv = load_bars("private_test", seen=True)
        heads_priv = load_headlines("private_test", seen=True)
        for i in range(min(args.write_top, len(df))):
            tau = float(df.loc[i, "tau"])
            w = float(df.loc[i, "w"])
            pub = _positions(bars_pub, heads_pub, tau, w)
            priv = _positions(bars_priv, heads_priv, tau, w)
            pub_df = pd.DataFrame({
                "session": pub.index.astype(int),
                "target_position": pub.values,
            })
            priv_df = pd.DataFrame({
                "session": priv.index.astype(int),
                "target_position": priv.values,
            })
            full = pd.concat([pub_df, priv_df], ignore_index=True)
            tau_str = f"tau{int(tau):02d}"
            w_str = f"w{int(round(w * 100)):02d}"
            out = SUBMISSIONS_DIR / f"sweep_recent_{tau_str}_{w_str}.csv"
            full.to_csv(out, index=False)
            assert len(full) == 20000
            assert full["target_position"].isna().sum() == 0
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
