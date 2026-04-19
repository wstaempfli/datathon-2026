"""Phase 2: MiniLM embedding-based headline relevance.

1. Embed each unique headline (all splits) with all-MiniLM-L6-v2.
   Cached to `features/headline_embeddings.parquet`.
2. Per CV fold, on train-fold sessions only: identify "high-impact" sessions
   (|target_return| above 1.5σ of train-fold), compute the mean embedding of
   their headlines → centroid.
3. Per headline: relevance = cosine_similarity(emb_h, centroid) ∈ [-1, +1].
   Clamped to [0, 1] to serve as a non-negative weight.
4. Per-session weighted-BMB feature (with recency decay):
   ws(s) = Σ_h  pol_regex(h) · relevance_h · exp(-(49-bar_ix_h)/40)
5. Rule: clip(1 - 25·fh + W·ws). Sweep W ∈ {0.25, 0.5, 0.75, 1.0}.

Usage:
    PYTHONPATH=. python3 scripts/sweep_emb.py
    PYTHONPATH=. python3 scripts/sweep_emb.py --submit
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
from src.rules import _bmb_recency_weighted, _clip, _headline_polarity  # noqa: E402

SUBMISSIONS_DIR = ROOT / "submissions"
_EMB_CACHE = ROOT / "features" / "headline_embeddings.parquet"
COMPETITION = "hrt-eth-zurich-datathon-2026"

N_FOLDS = 5
FOLD_SIZE = 200
K_FH = 25.0
TAU = 40.0
W_SWEEP: list[float] = [0.15, 0.25, 0.50, 0.75, 1.00]
IMPACT_SIGMA = 1.5  # sessions with |target_return| > IMPACT_SIGMA · σ are "impactful".


def _sharpe(pos: np.ndarray, y: np.ndarray) -> float:
    pnl = pos * y
    s = pnl.std()
    return float(pnl.mean() / s * 16.0) if s > 0 else 0.0


def _ws_per_session(
    heads: pd.DataFrame, sessions: pd.Index, pol_regex: np.ndarray,
    rel: np.ndarray, tau: float = TAU,
) -> np.ndarray:
    bar_ix = heads["bar_ix"].to_numpy().astype(float)
    decay = np.exp(-(49.0 - bar_ix) / float(tau))
    contrib = pol_regex * rel * decay
    agg = (
        pd.DataFrame({"session": heads["session"].to_numpy(), "c": contrib})
        .groupby("session")["c"].sum()
        .reindex(sessions, fill_value=0.0)
    )
    return agg.to_numpy().astype(float)


def _ensure_embeddings() -> pd.DataFrame:
    """Return DataFrame [headline, emb_0..emb_{d-1}] (one row per unique headline)."""
    if _EMB_CACHE.exists():
        print(f">> loading cached embeddings from {_EMB_CACHE.name}")
        return pd.read_parquet(_EMB_CACHE)

    print(">> computing embeddings (cache miss)...")
    from sentence_transformers import SentenceTransformer

    parts = []
    for split in ["train", "public_test", "private_test"]:
        h = load_headlines(split, seen=True)
        parts.append(h["headline"].astype(str))
    all_heads = pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)
    print(f"   {len(all_heads):,} unique headlines")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(
        all_heads.tolist(), batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,  # unit vectors — cos-sim = dot
    )
    df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    df.insert(0, "headline", all_heads.values)
    _EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_EMB_CACHE, index=False)
    print(f"   wrote {_EMB_CACHE} ({emb.shape})")
    return df


def _lookup_embeddings(heads: pd.DataFrame, emb_df: pd.DataFrame) -> np.ndarray:
    """Per-headline-row embedding matrix (n_rows, d)."""
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    idx = heads[["headline"]].merge(
        emb_df[["headline"] + emb_cols], on="headline", how="left"
    )
    return idx[emb_cols].to_numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.submit:
        args.write_csv = True

    emb_df = _ensure_embeddings()

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
    emb_tr = _lookup_embeddings(heads_tr, emb_df)  # (n_head, d)

    # ---- Per-fold centroid + relevance ----------------------------------
    rel_tr = np.zeros(len(heads_tr), dtype=np.float32)
    for fi in range(N_FOLDS):
        lo, hi = fi * FOLD_SIZE, (fi + 1) * FOLD_SIZE
        test_mask_sess = (sessions >= lo) & (sessions < hi)
        train_sess = sessions[~test_mask_sess]

        y_train = y_full.reindex(pd.Index(train_sess, name="session")).to_numpy()
        thresh = IMPACT_SIGMA * np.nanstd(y_train)
        impactful_sess = train_sess[np.abs(y_train) > thresh]

        # Headlines in impactful train sessions.
        imp_head_mask = heads_tr["session"].isin(impactful_sess).to_numpy()
        if imp_head_mask.sum() == 0:
            centroid = emb_tr.mean(axis=0)
        else:
            centroid = emb_tr[imp_head_mask].mean(axis=0)
        # normalize for cosine-sim (embeddings are already unit-norm).
        cn = np.linalg.norm(centroid) + 1e-12
        centroid = centroid / cn

        # Cosine similarity of every test-fold headline with centroid.
        test_head_mask = heads_tr["session"].isin(sessions[test_mask_sess]).to_numpy()
        cos = emb_tr[test_head_mask] @ centroid  # (n_test_head,) since both unit-norm
        # Clamp to [0, 1] to keep weight non-negative.
        cos = np.clip(cos, 0.0, 1.0)
        rel_tr[test_head_mask] = cos

    ws_tr = _ws_per_session(heads_tr, X.index, pol_tr, rel_tr, TAU)

    # ---- Baseline --------------------------------------------------------
    base_pos = _clip(1.0 - K_FH * fh + 0.25 * bmb)
    base_folds = []
    for fi in range(N_FOLDS):
        m = (sessions >= fi * FOLD_SIZE) & (sessions < (fi + 1) * FOLD_SIZE)
        base_folds.append(_sharpe(base_pos[m], y[m]))
    base_folds = np.asarray(base_folds)
    print(f"\n-- baseline rule_bmb_recent: folds={np.round(base_folds, 4).tolist()}  "
          f"mean={base_folds.mean():.4f} min={base_folds.min():.4f}  "
          f"std={base_folds.std(ddof=1):.4f}")

    # ---- W sweep: 'replace' vs 'add' -------------------------------------
    rows = []
    for mode in ["replace_bmb", "add_bmb"]:
        for w in W_SWEEP:
            if mode == "replace_bmb":
                pos = _clip(1.0 - K_FH * fh + w * ws_tr)
            else:
                pos = _clip(1.0 - K_FH * fh + 0.25 * bmb + w * ws_tr)
            folds = []
            for fi in range(N_FOLDS):
                m = (sessions >= fi * FOLD_SIZE) & (sessions < (fi + 1) * FOLD_SIZE)
                folds.append(_sharpe(pos[m], y[m]))
            f = np.asarray(folds)
            rows.append({
                "w": w, "mode": mode,
                "mean": float(f.mean()), "min": float(f.min()),
                "std_fold": float(f.std(ddof=1)),
                "sat": float(((pos == 0.2) | (pos == 2.0)).mean()),
                "folds": np.round(f, 4).tolist(),
            })
    cv = pd.DataFrame(rows).sort_values(["min", "mean"], ascending=False).reset_index(drop=True)
    pd.options.display.float_format = "{:.4f}".format
    print("\n-- Phase 2: emb-relevance sweep (ranked by min) --")
    print(cv.to_string(index=False))

    std_guard = float(base_folds.std(ddof=1))
    best = cv.iloc[0]
    wins = (best["mean"] > base_folds.mean() + std_guard) and (best["min"] > base_folds.min() + std_guard)
    print(f"\nverdict: {'SHIP' if wins else 'KEEP BASELINE'}")

    if not args.write_csv:
        return

    # ---- Fit on ALL train for test inference -----------------------------
    print("\n>> building centroid from ALL train + computing test relevance...")
    thresh_all = IMPACT_SIGMA * np.nanstd(y)
    imp_sess_all = sessions[np.abs(y) > thresh_all]
    imp_head_mask_all = heads_tr["session"].isin(imp_sess_all).to_numpy()
    centroid_all = emb_tr[imp_head_mask_all].mean(axis=0)
    centroid_all = centroid_all / (np.linalg.norm(centroid_all) + 1e-12)

    print(">> loading test splits...")
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
        w = float(row["w"])
        mode = row["mode"]
        if mode == "replace_bmb":
            pos_pub = _clip(1.0 - K_FH * fh_pub + w * ws_pub)
            pos_priv = _clip(1.0 - K_FH * fh_priv + w * ws_priv)
        else:
            pos_pub = _clip(1.0 - K_FH * fh_pub + 0.25 * bmb_pub + w * ws_pub)
            pos_priv = _clip(1.0 - K_FH * fh_priv + 0.25 * bmb_priv + w * ws_priv)
        full = pd.concat([
            pd.DataFrame({"session": X_pub.index.astype(int), "target_position": pos_pub}),
            pd.DataFrame({"session": X_priv.index.astype(int), "target_position": pos_priv}),
        ], ignore_index=True)
        assert len(full) == 20000
        out = SUBMISSIONS_DIR / f"sweep_emb_{mode}_w{int(round(w*100)):03d}.csv"
        full.to_csv(out, index=False)
        print(f"  wrote {out.name}  cv_mean={row['mean']:.4f} cv_min={row['min']:.4f}")
        if args.submit:
            msg = (f"phase2_emb {mode} w={w} "
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
