"""5-fold contiguous CV harness for variants V0-V6.

Splits sessions into 5 bins of 200 by sorted session id. For each fold:
fit on the other 4 bins (for learned variants), predict on the held-out bin,
compute OOF Sharpe = mean(pnl)/std(pnl)*16 where
pnl = position * (close[99]/close[49] - 1).

Prints a table `variant | f0 f1 f2 f3 f4 | mean | min` and writes
cv_report.json to the repo root (gitignored).

Usage:
    PYTHONPATH=. python3 scripts/cv.py
    PYTHONPATH=. python3 scripts/cv.py --variants v0,v1,v4,v5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import (  # noqa: E402
    fit_learned,
    load_bars,
    load_headlines,
    load_unseen_bars,
    predict,
    predict_v1,
    predict_v1b,
    predict_v1c,
    predict_v6_blend,
    realized_vol_series,
)

ALL_VARIANTS = ("v0", "v1", "v1b", "v1c", "v2", "v3", "v4", "v5", "v5b", "v6")


def _target_return(bars_unseen: pd.DataFrame, bars_seen: pd.DataFrame) -> pd.Series:
    c49 = bars_seen.groupby("session")["close"].last()
    c99 = bars_unseen.groupby("session")["close"].last()
    return (c99 / c49 - 1.0).rename("target_return")


def _fold_split(sessions: np.ndarray, n_folds: int = 5) -> list[np.ndarray]:
    sessions_sorted = np.sort(np.asarray(sessions))
    return np.array_split(sessions_sorted, n_folds)


def _sharpe(pos: np.ndarray, ret: np.ndarray) -> float:
    pnl = pos * ret
    sd = float(np.std(pnl))
    if sd <= 0.0:
        return float("-inf")
    return float(np.mean(pnl) / sd * 16.0)


def _select(bars: pd.DataFrame, sessions: np.ndarray) -> pd.DataFrame:
    return bars[bars["session"].isin(sessions)].reset_index(drop=True)


def _predict_variant(
    variant: str,
    bars_tr: pd.DataFrame,
    heads_tr: pd.DataFrame,
    target_tr: pd.Series,
    bars_te: pd.DataFrame,
    heads_te: pd.DataFrame,
    sigma_ref: float,
) -> pd.Series:
    if variant == "v0":
        return predict(bars_te, heads_te)
    if variant == "v1":
        return predict_v1(bars_te, heads_te, sigma_ref=sigma_ref)
    if variant == "v1b":
        return predict_v1b(bars_te, heads_te, sigma_ref=sigma_ref)
    if variant == "v1c":
        return predict_v1c(bars_te, heads_te, sigma_ref=sigma_ref)
    if variant == "v2":
        lp = fit_learned(bars_tr, heads_tr, target_tr, kind="ridge", vol_mode="none")
        return lp.predict(bars_te, heads_te)
    if variant == "v3":
        lp = fit_learned(bars_tr, heads_tr, target_tr, kind="lasso", vol_mode="none")
        return lp.predict(bars_te, heads_te)
    if variant == "v4":
        lp = fit_learned(bars_tr, heads_tr, target_tr, kind="huber", vol_mode="none")
        return lp.predict(bars_te, heads_te)
    if variant == "v5":
        lp = fit_learned(bars_tr, heads_tr, target_tr, kind="huber", vol_mode="full")
        return lp.predict(bars_te, heads_te)
    if variant == "v5b":
        lp = fit_learned(bars_tr, heads_tr, target_tr, kind="huber", vol_mode="directional")
        return lp.predict(bars_te, heads_te)
    if variant == "v6":
        lp = fit_learned(bars_tr, heads_tr, target_tr, kind="huber", vol_mode="directional")
        return predict_v6_blend(bars_te, heads_te, sigma_ref=sigma_ref, learned=lp)
    raise ValueError(f"unknown variant: {variant}")


def run_cv(variants: list[str], n_folds: int = 5) -> dict:
    print(f"loading train data ...")
    bars_all = load_bars("train")
    heads_all = load_headlines("train")
    bars_unseen = load_unseen_bars("train")
    target = _target_return(bars_unseen, bars_all)

    sessions = np.sort(bars_all["session"].unique())
    folds = _fold_split(sessions, n_folds)

    rv_all = realized_vol_series(bars_all)

    results: dict[str, dict] = {}
    for variant in variants:
        per_fold: list[float] = []
        ks: list[float] = []
        for fold_ix, test_sessions in enumerate(folds):
            train_sessions = np.setdiff1d(sessions, test_sessions)
            bars_tr = _select(bars_all, train_sessions)
            heads_tr = heads_all[heads_all["session"].isin(train_sessions)]
            bars_te = _select(bars_all, test_sessions)
            heads_te = heads_all[heads_all["session"].isin(test_sessions)]
            target_tr = target.reindex(train_sessions)
            sigma_ref_tr = float(rv_all.reindex(train_sessions).median())

            pos = _predict_variant(
                variant, bars_tr, heads_tr, target_tr, bars_te, heads_te, sigma_ref_tr
            )
            pos = pos.reindex(test_sessions)
            ret = target.reindex(test_sessions).to_numpy()
            s = _sharpe(pos.to_numpy(), ret)
            per_fold.append(s)
            ks.append(np.nan)

        mean = float(np.mean(per_fold))
        mn = float(np.min(per_fold))
        score = 0.5 * mean + 0.5 * mn
        results[variant] = {
            "per_fold": per_fold, "mean": mean, "min": mn, "score": score,
        }
        fold_str = " ".join(f"{s:5.3f}" for s in per_fold)
        print(f"{variant:4s} | {fold_str} | mean={mean:6.3f} min={mn:6.3f} score={score:6.3f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=str, default=",".join(ALL_VARIANTS))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in ALL_VARIANTS]
    if bad:
        raise SystemExit(f"unknown variants: {bad}")

    print(f"CV variants={variants} folds={args.folds}")
    print("-" * 72)
    results = run_cv(variants, n_folds=args.folds)
    print("-" * 72)
    ranked = sorted(results.items(), key=lambda kv: kv[1]["score"], reverse=True)
    print("ranked by 0.5*mean + 0.5*min:")
    for v, r in ranked:
        print(f"  {v}: score={r['score']:.3f} mean={r['mean']:.3f} min={r['min']:.3f}")

    out = ROOT / "cv_report.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
