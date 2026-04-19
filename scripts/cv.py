"""5-fold contiguous CV for V1b. Prints per-fold + mean + min Sharpe.

Uses only train bars/headlines/unseen. Each fold holds out 200 contiguous
sessions; sigma_ref is computed on the train-fold only (no leakage).

Usage:
    PYTHONPATH=. python3 scripts/cv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import load_bars, load_headlines, predict, realized_vol  # noqa: E402


def _target_return(bars_seen: pd.DataFrame, bars_unseen: pd.DataFrame) -> pd.Series:
    c49 = bars_seen.groupby("session")["close"].last()
    c99 = bars_unseen.groupby("session")["close"].last()
    return (c99 / c49 - 1.0).rename("target_return")


def main() -> None:
    bars = load_bars("train")
    heads = load_headlines("train")
    bars_unseen = pd.read_parquet(ROOT / "data" / "bars_unseen_train.parquet")
    target = _target_return(bars, bars_unseen)

    sessions = np.sort(bars["session"].unique())
    folds = np.array_split(sessions, 5)
    rv_all = realized_vol(bars)

    per_fold: list[float] = []
    for fold_ix, test_sessions in enumerate(folds):
        train_sessions = np.setdiff1d(sessions, test_sessions)
        bars_te = bars[bars["session"].isin(test_sessions)]
        heads_te = heads[heads["session"].isin(test_sessions)]
        sigma_ref = float(rv_all.reindex(train_sessions).median())

        pos = predict(bars_te, heads_te, sigma_ref=sigma_ref).reindex(test_sessions)
        ret = target.reindex(test_sessions).to_numpy()
        pnl = pos.to_numpy() * ret
        sd = float(np.std(pnl))
        sharpe = float(np.mean(pnl) / sd * 16.0) if sd > 0 else float("-inf")
        per_fold.append(sharpe)
        print(f"fold{fold_ix} sessions=[{test_sessions[0]:4d},{test_sessions[-1]:4d}] "
              f"sigma_ref={sigma_ref:.6f} sharpe={sharpe:.3f}")

    mean = float(np.mean(per_fold))
    mn = float(np.min(per_fold))
    score = 0.5 * mean + 0.5 * mn
    print("-" * 60)
    print(f"V1b  | {' '.join(f'{s:5.3f}' for s in per_fold)} | "
          f"mean={mean:6.3f} min={mn:6.3f} score={score:6.3f}")


if __name__ == "__main__":
    main()
