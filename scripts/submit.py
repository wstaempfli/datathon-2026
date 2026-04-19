"""Produce + submit CSVs for all pipeline variants.

Variant → filename + descriptive Kaggle message (approach + parameters).
CV-best variant (by 0.5*mean + 0.5*min) is marked with tag `ship`; others
are `explore`. Selection is CV-only — we do not re-rank on LB.

Usage:
    PYTHONPATH=. python3 scripts/submit.py --all --dry-run
    PYTHONPATH=. python3 scripts/submit.py --variant v1b
    PYTHONPATH=. python3 scripts/submit.py --all
"""

from __future__ import annotations

import argparse
import json
import subprocess
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

KAGGLE_COMP = "hrt-eth-zurich-datathon-2026"

# CV scores from scripts/cv.py (fill in after each re-run; never use LB to select).
CV_SCORES: dict[str, tuple[float, float]] = {
    "v0":  (3.119, 2.021),
    "v1":  (3.004, 1.940),
    "v1b": (3.128, 2.182),
    "v1c": (3.125, 2.179),
    "v2":  (2.849, 1.362),
    "v3":  (2.855, 1.383),
    "v4":  (2.873, 1.524),
    "v5":  (2.778, 1.320),
    "v5b": (2.876, 1.596),
    "v6":  (3.068, 1.939),
}

# Filename + human description of the approach+params. CV numbers are
# interpolated at submit time from CV_SCORES.
VARIANT_META: dict[str, dict[str, str]] = {
    "v0": {
        "filename": "v0_rule_k24_w0.375_tau20.csv",
        "msg": "V0 baseline | clip(1-24*fh+0.375*bmb_recent,+-2) tau=20 | tripwire",
    },
    "v1": {
        "filename": "v1_rule_volscaled_full_k24_w0.375_tau20_rp0.5-2.0.csv",
        "msg": "V1 rule+volscale(full) | pos=clip(raw*clip(sigma_ref/sigma,0.5,2),+-2) raw=1-24*fh+0.375*bmb sigma=std(dlog close)",
    },
    "v1b": {
        "filename": "v1b_rule_volscaled_directional_drift1_k24_w0.375_tau20.csv",
        "msg": "V1b rule+volscale(directional) | pos=clip(1+scaler*(-24*fh+0.375*bmb),+-2) scaler=clip(sigma_ref/sigma,0.5,2.0) floor=0.25*sigma_ref",
    },
    "v1c": {
        "filename": "v1c_rule_volscaled_directional_tight_clip0.75-1.5_floor0.5.csv",
        "msg": "V1c V1b with tighter scaler | scaler clip [0.75,1.5] floor=0.5*sigma_ref | conservative vol scaling",
    },
    "v2": {
        "filename": "v2_ridge_a1.0_feat11_constvol.csv",
        "msg": "V2 Ridge alpha=1.0 | 11 feats [fh_{0-5,0-10,0-25,0-49,25-49,40-49,46-49},max_dd,rv,parkinson,bmb] | pos=clip(pred*k+1,+-2) k in-sample",
    },
    "v3": {
        "filename": "v3_lasso_a0.0005_feat11_constvol.csv",
        "msg": "V3 Lasso alpha=5e-4 | 11 feats | pos=clip(pred*k+1,+-2) k in-sample",
    },
    "v4": {
        "filename": "v4_huber_eps1.35_a0.001_feat11_constvol.csv",
        "msg": "V4 Huber eps=1.35 alpha=1e-3 | 11 feats | pos=clip(pred*k+1,+-2) k in-sample",
    },
    "v5": {
        "filename": "v5_huber_volscaled_full_eps1.35_a0.001.csv",
        "msg": "V5 Huber+volscale(full) eps=1.35 alpha=1e-3 | pos=clip(scaler*(pred*k+1),+-2) scaler=clip(sigma_ref/sigma,0.5,2)",
    },
    "v5b": {
        "filename": "v5b_huber_volscaled_directional_eps1.35_a0.001.csv",
        "msg": "V5b Huber+volscale(directional) eps=1.35 alpha=1e-3 | pos=clip(1+scaler*pred*k,+-2) drift-preserving vol scaling",
    },
    "v6": {
        "filename": "v6_blend_0.5v1b_0.5v5b_huber.csv",
        "msg": "V6 blend | 0.5*V1b(rule+vol directional) + 0.5*V5b(Huber+vol directional) | pos=clip(avg,+-2)",
    },
}

ALL_VARIANTS = tuple(VARIANT_META.keys())


def _msg_with_cv(variant: str, tag: str) -> str:
    mean, mn = CV_SCORES.get(variant, (float("nan"), float("nan")))
    return f"{VARIANT_META[variant]['msg']} | CV mean={mean:.3f} min={mn:.3f} | {tag}"


def _pick_ship() -> str:
    best = max(CV_SCORES.items(), key=lambda kv: 0.5 * kv[1][0] + 0.5 * kv[1][1])
    return best[0]


def _target_return(bars_unseen: pd.DataFrame, bars_seen: pd.DataFrame) -> pd.Series:
    c49 = bars_seen.groupby("session")["close"].last()
    c99 = bars_unseen.groupby("session")["close"].last()
    return (c99 / c49 - 1.0).rename("target_return")


def _predict_variant(
    variant: str,
    bars_te: pd.DataFrame,
    heads_te: pd.DataFrame,
    bars_tr: pd.DataFrame,
    heads_tr: pd.DataFrame,
    target_tr: pd.Series,
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


def produce_csv(variant: str, bars_tr, heads_tr, target_tr, sigma_ref) -> Path:
    parts: list[pd.DataFrame] = []
    for split in ("public_test", "private_test"):
        bars_te = load_bars(split)
        heads_te = load_headlines(split)
        pos = _predict_variant(
            variant, bars_te, heads_te, bars_tr, heads_tr, target_tr, sigma_ref
        )
        parts.append(pd.DataFrame({
            "session": pos.index.astype(int).to_numpy(),
            "target_position": pos.to_numpy(),
        }))
    df = pd.concat(parts, ignore_index=True)
    assert len(df) == 20000, f"expected 20000 rows, got {len(df)}"
    assert df["target_position"].notna().all(), "NaN in positions"
    assert (df["target_position"] >= -2.0 - 1e-9).all()
    assert (df["target_position"] <= 2.0 + 1e-9).all()

    out = ROOT / "submissions" / VARIANT_META[variant]["filename"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def submit(path: Path, msg: str) -> None:
    print(f"submitting {path.name} | {msg}")
    subprocess.run(
        ["kaggle", "competitions", "submit", "-c", KAGGLE_COMP, "-f", str(path), "-m", msg],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=ALL_VARIANTS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.all and not args.variant:
        parser.error("specify --variant vX or --all")

    variants = list(ALL_VARIANTS) if args.all else [args.variant]
    ship = _pick_ship()
    print(f"CV-best ship = {ship} (score={0.5*sum(CV_SCORES[ship]):.3f})")

    # Train-side artifacts are shared across variants/splits.
    bars_tr = load_bars("train")
    heads_tr = load_headlines("train")
    bars_unseen = load_unseen_bars("train")
    target_tr = _target_return(bars_unseen, bars_tr)
    sigma_ref = float(realized_vol_series(bars_tr).median())

    for variant in variants:
        path = produce_csv(variant, bars_tr, heads_tr, target_tr, sigma_ref)
        tag = "ship" if variant == ship else "explore"
        msg = _msg_with_cv(variant, tag)
        print(f"wrote {path.name}  tag={tag}")
        if args.dry_run:
            print(f"  (dry-run) msg: {msg}")
            continue
        submit(path, msg)


if __name__ == "__main__":
    main()
