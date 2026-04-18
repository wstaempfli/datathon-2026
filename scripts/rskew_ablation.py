"""5-fold CV ablation for the realized-skewness (`rskew`) feature. Research-only."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import compute_targets, load_bars, load_headlines  # noqa: E402
from src.features import make_features  # noqa: E402
from src.model import train_cv  # noqa: E402

JSON_PATH = ROOT / "features" / "realized_skewness_research.json"
RSKEW_CSV = ROOT / "features" / "realized_skewness.csv"


def main() -> None:
    bars = load_bars("train", seen=True)
    heads = load_headlines("train", seen=True)
    X_base, _, _ = make_features(bars, heads, sentiment_cache=pd.DataFrame())

    rskew = pd.read_csv(RSKEW_CSV).set_index("session")["rskew"]
    rskew_aligned = rskew.reindex(X_base.index)
    assert rskew_aligned.notna().all(), "rskew has sessions missing from X_base"
    X_with = X_base.assign(rskew=rskew_aligned.to_numpy())

    y = compute_targets().set_index("session")["target_return"].loc[X_base.index]

    print("=== CV without rskew ===")
    _, _, m_wo = train_cv(X_base, y, seed=42)
    print("=== CV with rskew ===")
    _, _, m_wi = train_cv(X_with, y, seed=42)

    pf_wo = [float(v) for v in m_wo["per_fold_sharpe_raw"]]
    pf_wi = [float(v) for v in m_wi["per_fold_sharpe_raw"]]
    lift = [wi - wo for wi, wo in zip(pf_wi, pf_wo)]
    lift_mean = float(m_wi["sharpe_raw_mean"] - m_wo["sharpe_raw_mean"])
    win_rate = float(sum(1 for x in lift if x > 0) / len(lift))
    gate2 = bool((lift_mean >= 0.05) and (win_rate >= 0.6))

    with open(JSON_PATH) as f:
        payload = json.load(f)
    payload["ablation"] = {
        "seed": 42,
        "n_splits": 5,
        "per_fold_sharpe_without": pf_wo,
        "per_fold_sharpe_with": pf_wi,
        "per_fold_lift": lift,
        "mean_without": float(m_wo["sharpe_raw_mean"]),
        "std_without": float(m_wo["sharpe_raw_std"]),
        "mean_with": float(m_wi["sharpe_raw_mean"]),
        "std_with": float(m_wi["sharpe_raw_std"]),
        "lift_mean": lift_mean,
        "win_rate": win_rate,
        "gate2": gate2,
    }
    payload.setdefault("gates", {})["gate2"] = gate2

    tmp = JSON_PATH.with_suffix(JSON_PATH.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, JSON_PATH)

    wins = int(round(win_rate * 5))
    print(
        f"FINDING: rskew ablation lift={lift_mean:+.3f} "
        f"(with={m_wi['sharpe_raw_mean']:.3f}\u00b1{m_wi['sharpe_raw_std']:.3f} "
        f"vs without={m_wo['sharpe_raw_mean']:.3f}\u00b1{m_wo['sharpe_raw_std']:.3f}) "
        f"win_rate={wins}/5 gate2={gate2}"
    )


if __name__ == "__main__":
    main()
