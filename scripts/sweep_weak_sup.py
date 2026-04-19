"""Phase 3: weak-supervised headline-relevance classifier.

Per-headline noisy label (train-fold only):
    relevant_h = 1 if sign(fh_return_of_session_h) == sign(finbert_sent_score_h)
                 else 0

Train logistic regression on TF-IDF of the headline text → relevance label.
At inference: `p_rel(h) ∈ [0, 1]`.

Per-session weighted-BMB feature (with recency decay):
    ws(s) = Σ_h  pol_regex(h) · p_rel(h) · exp(-(49 - bar_ix_h) / 40)

where `pol_regex ∈ {-1, 0, +1}` from the existing bull/bear regex.

Rule: `clip(1 - 25·fh + W·ws, 0.2, 2.0)`. Sweep W.

Leakage guard: LR is refit per-fold on train-fold headlines only; p_rel for
test-fold headlines comes from that fold's classifier.

Usage:
    PYTHONPATH=. python3 scripts/sweep_weak_sup.py
    PYTHONPATH=. python3 scripts/sweep_weak_sup.py --submit
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import compute_targets, load_bars, load_headlines  # noqa: E402
from src.features import make_features  # noqa: E402
from src.rules import _bmb_recency_weighted, _clip, _headline_polarity  # noqa: E402

SUBMISSIONS_DIR = ROOT / "submissions"
_SENTIMENT_CACHE_PATH = ROOT / "features" / "sentiment_cache.parquet"
COMPETITION = "hrt-eth-zurich-datathon-2026"

N_FOLDS = 5
FOLD_SIZE = 200
K_FH = 25.0
TAU = 40.0
W_SWEEP: list[float] = [0.15, 0.25, 0.35, 0.50]


def _sharpe(pos: np.ndarray, y: np.ndarray) -> float:
    pnl = pos * y
    s = pnl.std()
    return float(pnl.mean() / s * 16.0) if s > 0 else 0.0


def _session_fh_return(X: pd.DataFrame) -> pd.Series:
    return X["fh_return"].astype(float)


def _per_headline_df(heads: pd.DataFrame, sent_df: pd.DataFrame, fh_by_sess: pd.Series) -> pd.DataFrame:
    """Per-headline row with fh_return, finbert sent_score, regex polarity."""
    merged = heads[["session", "bar_ix", "headline"]].merge(
        sent_df[["headline", "sent_score"]], on="headline", how="left"
    )
    merged["sent_score"] = merged["sent_score"].fillna(0.0).astype(float)
    merged["fh_return"] = merged["session"].map(fh_by_sess).astype(float).fillna(0.0)
    merged["pol_regex"] = _headline_polarity(heads).to_numpy()
    return merged


def _ws_per_session(
    heads: pd.DataFrame, sessions: pd.Index, pol_regex: np.ndarray,
    p_rel: np.ndarray, tau: float = TAU
) -> np.ndarray:
    bar_ix = heads["bar_ix"].to_numpy().astype(float)
    decay = np.exp(-(49.0 - bar_ix) / float(tau))
    contrib = pol_regex * p_rel * decay
    agg = (
        pd.DataFrame({"session": heads["session"].to_numpy(), "c": contrib})
        .groupby("session")["c"].sum()
        .reindex(sessions, fill_value=0.0)
    )
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
    sent_df = pd.read_parquet(_SENTIMENT_CACHE_PATH)
    fh_by_sess = pd.Series(fh, index=X.index)
    hdf = _per_headline_df(heads_tr, sent_df, fh_by_sess)

    # ---- Per-fold weak-supervision training -----------------------------
    ws_scores = np.zeros(len(sessions), dtype=float)
    for fi in range(N_FOLDS):
        lo, hi = fi * FOLD_SIZE, (fi + 1) * FOLD_SIZE
        test_mask = (sessions >= lo) & (sessions < hi)
        train_sess = sessions[~test_mask]
        test_sess = sessions[test_mask]

        # Training rows = headlines in train-fold sessions.
        tr_mask = hdf["session"].isin(train_sess)
        te_mask = hdf["session"].isin(test_sess)
        hdf_tr = hdf[tr_mask]
        # Label: does fh_return sign agree with sent_score sign?
        # Drop rows with zero sentiment (no signal to agree/disagree with).
        nonzero = hdf_tr["sent_score"].abs() > 1e-6
        agree = (np.sign(hdf_tr["sent_score"].to_numpy()) ==
                 np.sign(hdf_tr["fh_return"].to_numpy())).astype(int)
        mask_lbl = nonzero.to_numpy()
        X_train_text = hdf_tr.loc[mask_lbl, "headline"].astype(str).tolist()
        y_train = agree[mask_lbl]

        print(f">> fold {fi}: training LR on {len(X_train_text):,} headlines, "
              f"pos_rate={y_train.mean():.3f}")
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=10000,
                              lowercase=True, sublinear_tf=True)
        Xtr = vec.fit_transform(X_train_text)
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
        clf.fit(Xtr, y_train)

        # Predict on ALL test-fold headlines (not filtering by nonzero — we want
        # a relevance score for every headline).
        hdf_te = hdf[te_mask]
        Xte = vec.transform(hdf_te["headline"].astype(str).tolist())
        p_rel = clf.predict_proba(Xte)[:, 1].astype(float)

        # Compute ws for test-fold sessions.
        ws_fold = _ws_per_session(
            heads=heads_tr[heads_tr["session"].isin(test_sess)],
            sessions=pd.Index(test_sess, name="session"),
            pol_regex=hdf_te["pol_regex"].to_numpy(),
            p_rel=p_rel,
            tau=TAU,
        )
        # Map to global.
        idx = np.where(test_mask)[0]
        ws_scores[idx] = ws_fold

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

    # ---- W sweep (weak_sup REPLACES bmb_recent) --------------------------
    rows = []
    for w in W_SWEEP:
        pos = _clip(1.0 - K_FH * fh + w * ws_scores)
        folds = []
        for fi in range(N_FOLDS):
            m = (sessions >= fi * FOLD_SIZE) & (sessions < (fi + 1) * FOLD_SIZE)
            folds.append(_sharpe(pos[m], y[m]))
        f = np.asarray(folds)
        rows.append({
            "w": w, "mode": "replace_bmb",
            "mean": float(f.mean()), "min": float(f.min()),
            "std_fold": float(f.std(ddof=1)),
            "sat": float(((pos == 0.2) | (pos == 2.0)).mean()),
            "folds": np.round(f, 4).tolist(),
        })

    # Also test additive mode (bmb_recent + weak_sup together).
    for w in W_SWEEP:
        pos = _clip(1.0 - K_FH * fh + 0.25 * bmb + w * ws_scores)
        folds = []
        for fi in range(N_FOLDS):
            m = (sessions >= fi * FOLD_SIZE) & (sessions < (fi + 1) * FOLD_SIZE)
            folds.append(_sharpe(pos[m], y[m]))
        f = np.asarray(folds)
        rows.append({
            "w": w, "mode": "add_bmb",
            "mean": float(f.mean()), "min": float(f.min()),
            "std_fold": float(f.std(ddof=1)),
            "sat": float(((pos == 0.2) | (pos == 2.0)).mean()),
            "folds": np.round(f, 4).tolist(),
        })

    cv = pd.DataFrame(rows).sort_values(["min", "mean"], ascending=False).reset_index(drop=True)
    pd.options.display.float_format = "{:.4f}".format
    print("\n-- Phase 3: weak-sup sweep (ranked by min) --")
    print(cv.to_string(index=False))

    std_guard = float(base_folds.std(ddof=1))
    best = cv.iloc[0]
    wins = (best["mean"] > base_folds.mean() + std_guard) and (best["min"] > base_folds.min() + std_guard)
    print(f"\nverdict: {'SHIP' if wins else 'KEEP BASELINE'}")

    if not args.write_csv:
        return

    # ---- Fit on ALL train, write test CSVs ------------------------------
    print("\n>> fitting LR on ALL train for test inference...")
    nonzero_all = hdf["sent_score"].abs() > 1e-6
    agree_all = (np.sign(hdf["sent_score"].to_numpy()) ==
                 np.sign(hdf["fh_return"].to_numpy())).astype(int)
    vec_all = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=10000,
                              lowercase=True, sublinear_tf=True)
    Xtr_all = vec_all.fit_transform(hdf.loc[nonzero_all, "headline"].astype(str).tolist())
    y_all = agree_all[nonzero_all.to_numpy()]
    clf_all = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
    clf_all.fit(Xtr_all, y_all)

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
    p_rel_pub = clf_all.predict_proba(vec_all.transform(heads_pub["headline"].astype(str).tolist()))[:, 1]
    p_rel_priv = clf_all.predict_proba(vec_all.transform(heads_priv["headline"].astype(str).tolist()))[:, 1]
    ws_pub = _ws_per_session(heads_pub, X_pub.index, pol_pub, p_rel_pub, TAU)
    ws_priv = _ws_per_session(heads_priv, X_priv.index, pol_priv, p_rel_priv, TAU)

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
        out = SUBMISSIONS_DIR / f"sweep_weaksup_{mode}_w{int(round(w*100)):03d}.csv"
        full.to_csv(out, index=False)
        print(f"  wrote {out.name}  cv_mean={row['mean']:.4f} cv_min={row['min']:.4f}")
        if args.submit:
            msg = (f"phase3_weaksup {mode} w={w} "
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
