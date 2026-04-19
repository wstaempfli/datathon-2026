"""Phase 5: stack MiniLM emb-relevance (Phase 2) with widened clip (Phase 0).

Formula: clip(1 − K_FH·fh + 0.25·bmb_recent + W·ws_emb, lo, hi) where
  ws_emb_s = Σ_h pol_regex(h) · relevance_h · exp(−(49 − bar_ix_h)/40)

Grid: clip ∈ {(0.2, 2.0), (-1.0, 2.0), (-2.0, 2.0)}, K ∈ {25, 30, 35},
W_emb ∈ {0.25, 0.50, 0.75}. 5-fold contiguous CV with per-fold centroid
refit. Ship if CV mean AND min both beat baseline by ≥1×std_fold.

Usage:
    PYTHONPATH=. python3 scripts/sweep_stack.py
    PYTHONPATH=. python3 scripts/sweep_stack.py --submit
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
from src.rules import _bmb_recency_weighted, _headline_polarity  # noqa: E402

SUBMISSIONS_DIR = ROOT / "submissions"
_EMB_CACHE = ROOT / "features" / "headline_embeddings.parquet"
COMPETITION = "hrt-eth-zurich-datathon-2026"

N_FOLDS = 5
FOLD_SIZE = 200
TAU = 40.0
W_BMB = 0.25
IMPACT_SIGMA = 1.5

CLIPS: list[tuple[float, float]] = [(0.2, 2.0), (-1.0, 2.0), (-2.0, 2.0)]
KS: list[float] = [25.0, 30.0, 35.0]
W_EMBS: list[float] = [0.25, 0.50, 0.75]


def _sharpe(pos: np.ndarray, y: np.ndarray) -> float:
    pnl = pos * y
    s = pnl.std()
    return float(pnl.mean() / s * 16.0) if s > 0 else 0.0


def _ws_per_session(
    heads: pd.DataFrame, sessions: pd.Index,
    pol: np.ndarray, rel: np.ndarray, tau: float = TAU,
) -> np.ndarray:
    bar_ix = heads["bar_ix"].to_numpy().astype(float)
    decay = np.exp(-(49.0 - bar_ix) / float(tau))
    contrib = pol * rel * decay
    agg = (
        pd.DataFrame({"session": heads["session"].to_numpy(), "c": contrib})
        .groupby("session")["c"].sum()
        .reindex(sessions, fill_value=0.0)
    )
    return agg.to_numpy().astype(float)


def _lookup_embeddings(heads: pd.DataFrame, emb_df: pd.DataFrame) -> np.ndarray:
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    idx = heads[["headline"]].merge(emb_df[["headline"] + emb_cols], on="headline", how="left")
    return idx[emb_cols].to_numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.submit:
        args.write_csv = True

    if not _EMB_CACHE.exists():
        print(f"missing {_EMB_CACHE}; run scripts/sweep_emb.py first")
        sys.exit(1)
    emb_df = pd.read_parquet(_EMB_CACHE)

    print(">> loading train...")
    bars_tr = load_bars("train", seen=True)
    heads_tr = load_headlines("train", seen=True)
    X, _ = make_features(bars_tr, heads_tr)
    sessions = X.index.to_numpy()
    fh = X["fh_return"].to_numpy().astype(float)
    bmb = _bmb_recency_weighted(heads_tr, X.index, tau=TAU).to_numpy().astype(float)
    y_full = (
        compute_targets()
        .set_index("session")["target_return"]
        .reindex(X.index).astype(float)
    )
    y = y_full.to_numpy()
    pol_tr = _headline_polarity(heads_tr).to_numpy()
    emb_tr = _lookup_embeddings(heads_tr, emb_df)

    # ---- Per-fold relevance (same as Phase 2) ---------------------------
    rel_tr = np.zeros(len(heads_tr), dtype=np.float32)
    for fi in range(N_FOLDS):
        lo, hi = fi * FOLD_SIZE, (fi + 1) * FOLD_SIZE
        test_mask_sess = (sessions >= lo) & (sessions < hi)
        train_sess = sessions[~test_mask_sess]
        y_train = y_full.reindex(pd.Index(train_sess, name="session")).to_numpy()
        thresh = IMPACT_SIGMA * np.nanstd(y_train)
        imp_sess = train_sess[np.abs(y_train) > thresh]
        imp_mask = heads_tr["session"].isin(imp_sess).to_numpy()
        centroid = emb_tr[imp_mask].mean(axis=0) if imp_mask.sum() else emb_tr.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        test_mask_h = heads_tr["session"].isin(sessions[test_mask_sess]).to_numpy()
        cos = np.clip(emb_tr[test_mask_h] @ centroid, 0.0, 1.0)
        rel_tr[test_mask_h] = cos
    ws_tr = _ws_per_session(heads_tr, X.index, pol_tr, rel_tr, TAU)

    # ---- Baseline -------------------------------------------------------
    base_pos = np.clip(1.0 - 25.0 * fh + W_BMB * bmb, 0.2, 2.0)
    base_folds = []
    for fi in range(N_FOLDS):
        m = (sessions >= fi * FOLD_SIZE) & (sessions < (fi + 1) * FOLD_SIZE)
        base_folds.append(_sharpe(base_pos[m], y[m]))
    base_folds = np.asarray(base_folds)
    print(f"\n-- baseline rule_bmb_recent: folds={np.round(base_folds, 4).tolist()}  "
          f"mean={base_folds.mean():.4f} min={base_folds.min():.4f}  "
          f"std={base_folds.std(ddof=1):.4f}")

    # ---- Stack sweep ----------------------------------------------------
    rows = []
    for (lo_c, hi_c) in CLIPS:
        for k in KS:
            for w_emb in W_EMBS:
                raw = 1.0 - k * fh + W_BMB * bmb + w_emb * ws_tr
                pos = np.clip(raw, lo_c, hi_c)
                folds = []
                for fi in range(N_FOLDS):
                    m = (sessions >= fi * FOLD_SIZE) & (sessions < (fi + 1) * FOLD_SIZE)
                    folds.append(_sharpe(pos[m], y[m]))
                f = np.asarray(folds)
                rows.append({
                    "lo": lo_c, "hi": hi_c, "k": k, "w_emb": w_emb,
                    "mean": float(f.mean()), "min": float(f.min()),
                    "std_fold": float(f.std(ddof=1)),
                    "sat": float(((pos == lo_c) | (pos == hi_c)).mean()),
                    "folds": np.round(f, 4).tolist(),
                })
    cv = pd.DataFrame(rows).sort_values(["min", "mean"], ascending=False).reset_index(drop=True)
    pd.options.display.float_format = "{:.4f}".format
    print("\n-- Phase 5: stack emb-relevance × widened clip × K_FH (ranked by min) --")
    print(cv.to_string(index=False))

    std_guard = float(base_folds.std(ddof=1))
    best = cv.iloc[0]
    wins = (best["mean"] > base_folds.mean() + std_guard) and (best["min"] > base_folds.min() + std_guard)
    print(f"\nverdict: {'SHIP' if wins else 'KEEP BASELINE'}")

    # also print the Pareto frontier (best min per mean tier).
    print("\nPareto tableau (mean ≥ baseline_mean − 0.03):")
    par = cv[cv["mean"] >= base_folds.mean() - 0.03].copy()
    par = par.sort_values("min", ascending=False).head(10)
    print(par.to_string(index=False))

    if not args.write_csv:
        return

    # ---- Fit centroid on ALL train for test inference -------------------
    print("\n>> computing ALL-train centroid + test relevance...")
    thresh_all = IMPACT_SIGMA * np.nanstd(y)
    imp_sess_all = sessions[np.abs(y) > thresh_all]
    imp_mask_all = heads_tr["session"].isin(imp_sess_all).to_numpy()
    centroid_all = emb_tr[imp_mask_all].mean(axis=0)
    centroid_all = centroid_all / (np.linalg.norm(centroid_all) + 1e-12)

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
    pol_pub = _headline_polarity(heads_pub).to_numpy()
    pol_priv = _headline_polarity(heads_priv).to_numpy()
    emb_pub = _lookup_embeddings(heads_pub, emb_df)
    emb_priv = _lookup_embeddings(heads_priv, emb_df)
    rel_pub = np.clip(emb_pub @ centroid_all, 0.0, 1.0)
    rel_priv = np.clip(emb_priv @ centroid_all, 0.0, 1.0)
    ws_pub = _ws_per_session(heads_pub, X_pub.index, pol_pub, rel_pub, TAU)
    ws_priv = _ws_per_session(heads_priv, X_priv.index, pol_priv, rel_priv, TAU)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(min(args.top_k, len(cv))):
        row = cv.iloc[i]
        lo_c, hi_c, k, w_emb = float(row["lo"]), float(row["hi"]), float(row["k"]), float(row["w_emb"])
        pos_pub = np.clip(1.0 - k * fh_pub + W_BMB * bmb_pub + w_emb * ws_pub, lo_c, hi_c)
        pos_priv = np.clip(1.0 - k * fh_priv + W_BMB * bmb_priv + w_emb * ws_priv, lo_c, hi_c)
        full = pd.concat([
            pd.DataFrame({"session": X_pub.index.astype(int), "target_position": pos_pub}),
            pd.DataFrame({"session": X_priv.index.astype(int), "target_position": pos_priv}),
        ], ignore_index=True)
        assert len(full) == 20000

        def _fmt(x):
            return f"{x:+.1f}".replace(".", "p").replace("+", "p").replace("-", "n")

        tag = f"lo{_fmt(lo_c)}_hi{_fmt(hi_c)}_k{int(k):02d}_we{int(round(w_emb*100)):03d}"
        out = SUBMISSIONS_DIR / f"sweep_stack_{tag}.csv"
        full.to_csv(out, index=False)
        print(f"  wrote {out.name}  cv_mean={row['mean']:.4f} cv_min={row['min']:.4f}")

        if args.submit:
            msg = (f"phase5_stack lo={lo_c} hi={hi_c} K={int(k)} w_emb={w_emb} "
                   f"CV(mean={row['mean']:.4f},min={row['min']:.4f}) — diagnostic")
            try:
                subprocess.run(
                    ["kaggle", "competitions", "submit",
                     "-c", COMPETITION, "-f", str(out), "-m", msg],
                    check=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                print(f"  ! kaggle submit failed: {exc}")


if __name__ == "__main__":
    main()
