"""Export JSON data files consumed by the Üetlibytes React pitch deck.

Writes 5 JSON files into presentation/src/data/:
    models.json          — LB Sharpe per modelling approach we tried
    scatter.json         — (fh_return, target_return) downsample + regression + corr
    quintile.json        — mean target_return binned by fh_return quintile
    cv.json              — per-fold CV Sharpe (V1b vs pre-V1b baseline)
    session_example.json — one session close 0-99 for the "challenge" slide

Usage:
    PYTHONPATH=. python3 scripts/export_plot_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import load_bars  # noqa: E402

OUT_DIR = ROOT / "presentation" / "src" / "data"


# --- models.json: LB Sharpe per approach (sourced from git log of prior submissions) ---

MODELS: list[dict] = [
    {"name": "LightGBM",    "features": "price5+gate1top10", "lb": 1.820, "isFinal": False},
    {"name": "SklGBR",      "features": "price5+gate1top10", "lb": 1.981, "isFinal": False},
    {"name": "Huber",       "features": "price5+gate1top10", "lb": 2.564, "isFinal": False},
    {"name": "Ridge",       "features": "price5+gate1top10", "lb": 2.672, "isFinal": False},
    {"name": "Quantile",    "features": "price5+gate1top10", "lb": 2.676, "isFinal": False},
    {"name": "ElasticNet",  "features": "price5+gate1top10", "lb": 2.784, "isFinal": False},
    {"name": "Lasso",       "features": "price5+all12",      "lb": 2.784, "isFinal": False},
    {"name": "Huber",       "features": "price5",            "lb": 2.799, "isFinal": False},
    {"name": "XGBoost",     "features": "price5+gate1top10", "lb": 2.930, "isFinal": False},
    {"name": "V1b (rule)",  "features": "fh+bmb+volscale",   "lb": 2.811, "isFinal": True},
]


# --- CV per-fold (known from prior `scripts/cv.py` runs) ---

CV = {
    "folds": [1, 2, 3, 4, 5],
    "v1b":      [2.293, 3.033, 2.182, 5.476, 2.654],
    "baseline": [2.273, 3.128, 2.021, 5.501, 2.670],
    "v1b_mean": 3.128,
    "v1b_min":  2.182,
    "baseline_mean": 3.119,
    "baseline_min":  2.021,
}


def _target_return_train() -> pd.DataFrame:
    """Per-session (fh_return, target_return) from train bars only."""
    seen = load_bars("train")
    unseen = pd.read_parquet(ROOT / "data" / "bars_unseen_train.parquet")
    sessions = pd.Index(np.sort(seen["session"].unique()), name="session")

    pv = (
        seen.pivot(index="session", columns="bar_ix", values="close")
        .reindex(index=sessions, columns=range(50))
    )
    c0 = pv[0].replace(0.0, np.nan)
    c49 = pv[49]
    fh = (c49 / c0 - 1.0).rename("fh_return")

    c49_u = seen.groupby("session")["close"].last().reindex(sessions)
    c99 = unseen.groupby("session")["close"].last().reindex(sessions)
    target = (c99 / c49_u - 1.0).rename("target_return")

    df = pd.concat([fh, target], axis=1).dropna()
    return df


def _scatter_payload(df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(42)
    n = min(300, len(df))
    idx = rng.choice(len(df), size=n, replace=False)
    sample = df.iloc[np.sort(idx)].copy()

    fh = df["fh_return"].to_numpy()
    tg = df["target_return"].to_numpy()
    corr = float(np.corrcoef(fh, tg)[0, 1])
    # y = a + b*x  via closed form on full 1k
    b = float(np.cov(fh, tg, ddof=1)[0, 1] / np.var(fh, ddof=1))
    a = float(tg.mean() - b * fh.mean())
    xmin = float(fh.min())
    xmax = float(fh.max())

    return {
        "points": [
            {"fh": float(r.fh_return), "target": float(r.target_return)}
            for r in sample.itertuples()
        ],
        "corr": corr,
        "regression": {
            "a": a,
            "b": b,
            "x_min": xmin,
            "x_max": xmax,
            "y_at_xmin": a + b * xmin,
            "y_at_xmax": a + b * xmax,
        },
        "n_total": int(len(df)),
        "n_sampled": int(n),
    }


def _quintile_payload(df: pd.DataFrame) -> dict:
    q = pd.qcut(df["fh_return"], 5, labels=False)
    rows: list[dict] = []
    for bin_ix in range(5):
        mask = q == bin_ix
        rows.append({
            "quintile": f"Q{bin_ix + 1}",
            "fh_mean": float(df.loc[mask, "fh_return"].mean()),
            "target_mean": float(df.loc[mask, "target_return"].mean()),
            "n": int(mask.sum()),
        })
    return {"bins": rows, "overall_target_mean": float(df["target_return"].mean())}


def _session_example_payload() -> dict:
    seen = load_bars("train")
    unseen = pd.read_parquet(ROOT / "data" / "bars_unseen_train.parquet")

    c49 = seen.groupby("session")["close"].last()
    c99 = unseen.groupby("session")["close"].last()
    target = (c99 / c49 - 1.0).dropna()
    # Pick a visually dramatic session: one of the largest |target_return|
    # that also has a clean up-then-down (or down-then-up) shape.
    cand = target.abs().sort_values(ascending=False).head(50).index.tolist()

    # pick first candidate where fh_return has opposite sign to target_return
    # (classic mean-reversion example) — falls back to the most extreme otherwise
    pivot_seen = seen.pivot(index="session", columns="bar_ix", values="close")
    chosen = None
    for s in cand:
        if s not in pivot_seen.index:
            continue
        c0 = pivot_seen.loc[s, 0]
        c49v = pivot_seen.loc[s, 49]
        if c0 == 0 or np.isnan(c0) or np.isnan(c49v):
            continue
        fh = c49v / c0 - 1.0
        if np.sign(fh) != np.sign(target.loc[s]) and abs(fh) > 0.01:
            chosen = int(s)
            break
    if chosen is None:
        chosen = int(cand[0])

    seen_row = pivot_seen.loc[chosen].reindex(range(50)).to_numpy()
    unseen_pivot = unseen.pivot(index="session", columns="bar_ix", values="close")
    unseen_row = unseen_pivot.loc[chosen].reindex(range(50, 100)).to_numpy()
    close = np.concatenate([seen_row, unseen_row]).tolist()

    return {
        "session": chosen,
        "close": [float(x) for x in close],
        "fh_return": float(seen_row[-1] / seen_row[0] - 1.0),
        "target_return": float(target.loc[chosen]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {OUT_DIR}")

    df = _target_return_train()
    print(f"train rows: {len(df)}")

    payloads = {
        "models.json":          {"models": MODELS},
        "cv.json":               CV,
        "scatter.json":          _scatter_payload(df),
        "quintile.json":         _quintile_payload(df),
        "session_example.json":  _session_example_payload(),
    }

    for name, data in payloads.items():
        path = OUT_DIR / name
        with path.open("w") as f:
            json.dump(data, f, indent=2)
        print(f"wrote {path.relative_to(ROOT)}")

    # sanity prints
    scatter = payloads["scatter.json"]
    q = payloads["quintile.json"]
    print(
        f"\ncorr(fh, target) = {scatter['corr']:+.4f}  "
        f"| regression b = {scatter['regression']['b']:+.4f}"
    )
    print("quintile target_mean:")
    for r in q["bins"]:
        print(f"  {r['quintile']}  fh={r['fh_mean']:+.4f}  target={r['target_mean']:+.4f}  n={r['n']}")


if __name__ == "__main__":
    main()
