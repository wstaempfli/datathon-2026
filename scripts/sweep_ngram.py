"""Phase 1: data-driven n-gram loadings on top of bmb_recent.

For each CV fold, on train-fold sessions only:
  1. CountVectorizer(ngram_range=(1,3), min_df=20, max_features=2000) over
     headlines from those sessions.
  2. For each n-gram: correlate per-session "contains-ngram-at-least-once"
     with session-level `target_return`.
  3. Keep top-TOPK positive + top-TOPK negative n-grams by correlation.
  4. Each retained ngram t gets β_t = its correlation (Pearson).

Then per fold-test session: `ngram_score = Σ_h Σ_{t ∈ headline_h} β_t`.
Rule: `clip(1 − 25·fh + 0.25·bmb_recent + W·ngram_score, 0.2, 2.0)`.

Usage:
    PYTHONPATH=. python3 scripts/sweep_ngram.py
    PYTHONPATH=. python3 scripts/sweep_ngram.py --submit
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import compute_targets, load_bars, load_headlines  # noqa: E402
from src.features import make_features  # noqa: E402
from src.rules import _bmb_recency_weighted, _clip  # noqa: E402

SUBMISSIONS_DIR = ROOT / "submissions"
COMPETITION = "hrt-eth-zurich-datathon-2026"

N_FOLDS = 5
FOLD_SIZE = 200
K_FH = 25.0
W_BMB = 0.25
TAU = 40.0
TOPK_PER_SIDE = 50
W_NGRAMS: list[float] = [0.05, 0.10, 0.15, 0.20, 0.30]
NGRAM_RANGE = (1, 3)
MIN_DF = 20
MAX_FEATURES = 2000


def _sharpe(pos: np.ndarray, y: np.ndarray) -> float:
    pnl = pos * y
    s = pnl.std()
    return float(pnl.mean() / s * 16.0) if s > 0 else 0.0


def _session_presence_matrix(
    heads: pd.DataFrame, sessions: pd.Index, M_headlines, vocab_size: int
) -> np.ndarray:
    """(n_sessions, n_ngrams) where cell = 1 if any headline in that session contains the ngram.

    `M_headlines` rows correspond to `heads.iterrows()` order (the vectorizer
    input was that order).
    """
    sess = heads["session"].to_numpy()
    sess_to_idx = {s: i for i, s in enumerate(sessions.tolist())}
    out = np.zeros((len(sessions), vocab_size), dtype=np.float32)
    # M_headlines is sparse: iterate rows once.
    M_csr = M_headlines.tocsr()
    for row_idx, s in enumerate(sess):
        if s not in sess_to_idx:
            continue
        start, end = M_csr.indptr[row_idx], M_csr.indptr[row_idx + 1]
        cols = M_csr.indices[start:end]
        out[sess_to_idx[s], cols] = 1.0  # presence (any headline with ngram)
    return out


def _ngram_score_per_session(
    heads: pd.DataFrame, sessions: pd.Index, vectorizer: CountVectorizer, betas: np.ndarray
) -> np.ndarray:
    """Per-session Σ_h Σ_{t ∈ headline_h} β_t. Headlines transform via the
    *already-fitted* vectorizer.
    """
    M = vectorizer.transform(heads["headline"].astype(str).tolist())  # (n_head, vocab)
    # Per-headline score = row · β. Dense vector of length n_head.
    hs = M @ betas  # (n_head,)
    df = pd.DataFrame({"session": heads["session"].to_numpy(), "s": hs})
    agg = df.groupby("session")["s"].sum().reindex(sessions, fill_value=0.0)
    return agg.to_numpy().astype(float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.submit:
        args.write_csv = True

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
        .reindex(X.index)
        .astype(float)
    )
    y = y_full.to_numpy()

    # ---- per-fold n-gram fit ---------------------------------------------
    # For each fold, fit vectorizer+betas on train-fold sessions, compute
    # ngram_score on test-fold sessions.
    ngram_scores = np.zeros(len(sessions), dtype=float)
    for fi in range(N_FOLDS):
        lo, hi = fi * FOLD_SIZE, (fi + 1) * FOLD_SIZE
        test_mask = (sessions >= lo) & (sessions < hi)
        train_sess = sessions[~test_mask]
        heads_fold = heads_tr[heads_tr["session"].isin(train_sess)]

        print(f">> fold {fi}: fitting CountVectorizer on {len(heads_fold):,} headlines...")
        vec = CountVectorizer(
            ngram_range=NGRAM_RANGE, min_df=MIN_DF, max_features=MAX_FEATURES,
            lowercase=True,
        )
        M = vec.fit_transform(heads_fold["headline"].astype(str).tolist())

        presence = _session_presence_matrix(
            heads_fold, pd.Index(train_sess, name="session"), M, vocab_size=M.shape[1]
        )
        y_train = y_full.reindex(pd.Index(train_sess, name="session")).to_numpy()

        # Per-ngram correlation with target_return across train-fold sessions.
        # Compute Pearson: corr = cov(X, y) / (std_x * std_y).
        y_c = y_train - np.nanmean(y_train)
        std_y = np.nanstd(y_train) + 1e-12
        X_c = presence - presence.mean(axis=0, keepdims=True)
        std_x = presence.std(axis=0) + 1e-12
        cov = (X_c * y_c[:, None]).mean(axis=0)
        corr = cov / (std_x * std_y)

        # Keep top-K positive + top-K negative by correlation.
        order = np.argsort(corr)  # asc
        bear_idx = order[:TOPK_PER_SIDE]
        bull_idx = order[-TOPK_PER_SIDE:]
        keep = np.zeros_like(corr, dtype=bool)
        keep[bear_idx] = True
        keep[bull_idx] = True
        betas = np.where(keep, corr, 0.0).astype(np.float32)

        # Apply to test-fold: transform test-fold headlines through the SAME fitted vectorizer.
        test_sess = sessions[test_mask]
        heads_test = heads_tr[heads_tr["session"].isin(test_sess)]
        ns_test = _ngram_score_per_session(
            heads_test, pd.Index(test_sess, name="session"), vec, betas
        )
        # Map to global positions.
        idx = np.where(test_mask)[0]
        ngram_scores[idx] = ns_test

        if fi == 0:
            feat = np.array(vec.get_feature_names_out())
            print("  top-10 bullish ngrams (fold 0):")
            for j in bull_idx[::-1][:10]:
                print(f"    corr={corr[j]:+.4f}  ngram={feat[j]!r}")
            print("  top-10 bearish ngrams (fold 0):")
            for j in bear_idx[:10]:
                print(f"    corr={corr[j]:+.4f}  ngram={feat[j]!r}")

    # ---- baseline row ----------------------------------------------------
    baseline_pos = _clip(1.0 - K_FH * fh + W_BMB * bmb)
    base_folds = []
    for fi in range(N_FOLDS):
        m = (sessions >= fi * FOLD_SIZE) & (sessions < (fi + 1) * FOLD_SIZE)
        base_folds.append(_sharpe(baseline_pos[m], y[m]))
    base_folds = np.asarray(base_folds)
    print(f"\n-- baseline rule_bmb_recent: folds={np.round(base_folds, 4).tolist()}  "
          f"mean={base_folds.mean():.4f} min={base_folds.min():.4f}  "
          f"std={base_folds.std(ddof=1):.4f}")

    # ---- W sweep ---------------------------------------------------------
    rows = []
    for w in W_NGRAMS:
        pos = _clip(1.0 - K_FH * fh + W_BMB * bmb + w * ngram_scores)
        folds = []
        for fi in range(N_FOLDS):
            m = (sessions >= fi * FOLD_SIZE) & (sessions < (fi + 1) * FOLD_SIZE)
            folds.append(_sharpe(pos[m], y[m]))
        f = np.asarray(folds)
        rows.append({
            "w_ngram": w,
            "mean": float(f.mean()),
            "min": float(f.min()),
            "std_fold": float(f.std(ddof=1)),
            "sat": float(((pos == 0.2) | (pos == 2.0)).mean()),
            "folds": np.round(f, 4).tolist(),
        })
    cv = pd.DataFrame(rows).sort_values(["min", "mean"], ascending=False).reset_index(drop=True)
    pd.options.display.float_format = "{:.4f}".format
    print("\n-- Phase 1: ngram-loadings sweep (ranked by min) --")
    print(cv.to_string(index=False))

    std_guard = float(base_folds.std(ddof=1))
    best = cv.iloc[0]
    wins = (best["mean"] > base_folds.mean() + std_guard) and (best["min"] > base_folds.min() + std_guard)
    print(f"\nverdict: {'SHIP' if wins else 'KEEP BASELINE'}")

    if not args.write_csv:
        return

    # ---- Fit ALL-train ngram loadings for test inference ----------------
    print("\n>> fitting ngram loadings on ALL train for test inference...")
    vec_all = CountVectorizer(
        ngram_range=NGRAM_RANGE, min_df=MIN_DF, max_features=MAX_FEATURES,
        lowercase=True,
    )
    M_all = vec_all.fit_transform(heads_tr["headline"].astype(str).tolist())
    presence_all = _session_presence_matrix(
        heads_tr, X.index, M_all, vocab_size=M_all.shape[1]
    )
    y_c = y - np.nanmean(y)
    std_y = np.nanstd(y) + 1e-12
    X_c = presence_all - presence_all.mean(axis=0, keepdims=True)
    std_x = presence_all.std(axis=0) + 1e-12
    cov = (X_c * y_c[:, None]).mean(axis=0)
    corr_all = cov / (std_x * std_y)
    order = np.argsort(corr_all)
    bear_idx = order[:TOPK_PER_SIDE]
    bull_idx = order[-TOPK_PER_SIDE:]
    keep = np.zeros_like(corr_all, dtype=bool)
    keep[bear_idx] = True
    keep[bull_idx] = True
    betas_all = np.where(keep, corr_all, 0.0).astype(np.float32)

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
    ns_pub = _ngram_score_per_session(heads_pub, X_pub.index, vec_all, betas_all)
    ns_priv = _ngram_score_per_session(heads_priv, X_priv.index, vec_all, betas_all)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(min(args.top_k, len(cv))):
        row = cv.iloc[i]
        w = float(row["w_ngram"])
        pos_pub = _clip(1.0 - K_FH * fh_pub + W_BMB * bmb_pub + w * ns_pub)
        pos_priv = _clip(1.0 - K_FH * fh_priv + W_BMB * bmb_priv + w * ns_priv)
        full = pd.concat([
            pd.DataFrame({"session": X_pub.index.astype(int), "target_position": pos_pub}),
            pd.DataFrame({"session": X_priv.index.astype(int), "target_position": pos_priv}),
        ], ignore_index=True)
        assert len(full) == 20000
        out = SUBMISSIONS_DIR / f"sweep_ngram_w{int(round(w*100)):03d}.csv"
        full.to_csv(out, index=False)
        print(f"  wrote {out.name}  cv_mean={row['mean']:.4f} cv_min={row['min']:.4f}")

        if args.submit:
            msg = (f"phase1_ngram w={w} topK={TOPK_PER_SIDE} "
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
