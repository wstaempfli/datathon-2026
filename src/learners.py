"""Data-driven per-template weights via ridge regression.

Fits Ridge on (per-session top-30 template counts) → target_return on the
train split. Weights are persisted to `models/template_weights.parquet` and
consumed at inference time by `src/rules.py::rule_tplridge`.

Public surface:
    fit_template_weights(alpha=10.0) -> (weights: np.ndarray, skeletons: list[str])
    load_template_weights() -> (weights, skeletons)
    compute_template_features(heads, sessions, skeletons) -> pd.DataFrame
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.alphas import extract_skeleton
from src.data import compute_targets

_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATE_HITS_PATH = _ROOT / "features" / "template_hits.parquet"
_TEMPLATE_DICT_PATH = _ROOT / "features" / "template_dictionary.tsv"
_MODELS_DIR = _ROOT / "models"
_WEIGHTS_PATH = _MODELS_DIR / "template_weights.parquet"

# Clip individual β magnitudes so a single noisy template cannot dominate.
BETA_CLIP = 0.5


def fit_template_weights(
    alpha: float = 10.0, clip: float = BETA_CLIP
) -> tuple[np.ndarray, list[str]]:
    """Fit Ridge(target_return ~ template counts) on the train split.

    Returns (weights_array, skeletons_list) in the fixed rank order from
    `features/template_dictionary.tsv`.
    """
    hits = pd.read_parquet(_TEMPLATE_HITS_PATH).set_index("session")
    td = pd.read_csv(_TEMPLATE_DICT_PATH, sep="\t")

    # Align columns to dictionary rank order (tpl_1, tpl_2, ...).
    col_order = [f"tpl_{r}" for r in td["rank"].to_numpy()]
    X = hits.reindex(columns=col_order, fill_value=0).astype(float)
    skeletons = td["skeleton"].astype(str).tolist()

    y = (
        compute_targets()
        .set_index("session")["target_return"]
        .reindex(X.index)
        .astype(float)
    )

    mask = y.notna().to_numpy()
    X_fit = X.to_numpy()[mask]
    y_fit = y.to_numpy()[mask]

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_fit, y_fit)
    beta = np.clip(model.coef_, -clip, clip)

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"skeleton": skeletons, "beta": beta, "rank": td["rank"].to_numpy()}
    ).to_parquet(_WEIGHTS_PATH, index=False)
    return beta.astype(float), skeletons


def load_template_weights() -> tuple[np.ndarray, list[str]]:
    """Load persisted (weights, skeletons). Raises FileNotFoundError if unfit."""
    if not _WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {_WEIGHTS_PATH}. Run fit_template_weights() first."
        )
    df = pd.read_parquet(_WEIGHTS_PATH).sort_values("rank")
    return df["beta"].to_numpy().astype(float), df["skeleton"].astype(str).tolist()


def compute_template_features(
    heads: pd.DataFrame, sessions: pd.Index, skeletons: list[str]
) -> pd.DataFrame:
    """Per-session × per-skeleton hit counts (same skeleton set as train).

    Applies `extract_skeleton` to every headline in `heads` and tallies matches
    against the provided `skeletons` list. Skeletons absent in a split
    contribute zero columns (explicit zeros via reindex).
    """
    skel = heads["headline"].astype(str).map(extract_skeleton)
    work = pd.DataFrame(
        {"session": heads["session"].to_numpy(), "skel": skel.to_numpy()}
    )
    skel_set = set(skeletons)
    work = work[work["skel"].isin(skel_set)]
    if len(work) == 0:
        return pd.DataFrame(
            np.zeros((len(sessions), len(skeletons)), dtype=float),
            index=sessions,
            columns=skeletons,
        )
    counts = work.groupby(["session", "skel"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=skeletons, fill_value=0)
    counts = counts.reindex(index=sessions, fill_value=0)
    return counts.astype(float)


if __name__ == "__main__":
    beta, skels = fit_template_weights()
    order = np.argsort(-np.abs(beta))
    print(f"Fit {len(skels)} template weights. Top-10 by |β|:")
    for i in order[:10]:
        print(f"  β={beta[i]:+.5f}  rank={i + 1:>2}  skeleton={skels[i]}")
    print(f"wrote {_WEIGHTS_PATH}")
