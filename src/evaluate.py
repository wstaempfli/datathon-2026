"""
Evaluation framework: Sharpe computation, cross-validation, and baselines.

All evaluation uses the official Sharpe formula:
    pnl_i = target_position_i * (close_end_i / close_halfway_i - 1)
    sharpe = mean(pnl) / std(pnl) * 16

Usage:
    from src.evaluate import compute_sharpe, cross_validate, run_baselines

    # Single evaluation
    sharpe = compute_sharpe(positions, halfway_close, end_close)

    # Full CV with stability test
    results = cross_validate(X, y, train_ridge, n_splits=5, n_seeds=20)

    # Baselines to beat
    baselines = run_baselines(y)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.models import apply_position_sizing


# ---------------------------------------------------------------------------
# Core Sharpe computation
# ---------------------------------------------------------------------------

def compute_sharpe(
    positions: np.ndarray,
    halfway_close: np.ndarray,
    end_close: np.ndarray,
) -> float:
    """Compute Sharpe ratio using the official competition formula.

    pnl_i = positions_i * (end_close_i / halfway_close_i - 1)
    sharpe = mean(pnl) / std(pnl) * 16

    The factor 16 = sqrt(256) is fixed regardless of session count.

    Called by
    --------
    - cross_validate()   (per-fold evaluation)
    - run_baselines()    (baseline computation)
    - notebooks          (ad-hoc analysis)
    """
    positions = np.asarray(positions, dtype=float)
    halfway_close = np.asarray(halfway_close, dtype=float)
    end_close = np.asarray(end_close, dtype=float)

    pnl = positions * (end_close / halfway_close - 1)
    if np.std(pnl) == 0:
        return 0.0
    return float(np.mean(pnl) / np.std(pnl) * 16)


def compute_direction_accuracy(predictions: np.ndarray, actual_returns: np.ndarray) -> float:
    """Fraction of sessions where sign(prediction) == sign(actual_return).

    Always-long baseline accuracy = 57.0% (the positive drift rate).
    We need > 57% to beat it.
    """
    pred_sign = np.sign(np.asarray(predictions, dtype=float))
    actual_sign = np.sign(np.asarray(actual_returns, dtype=float))
    return float(np.mean(pred_sign == actual_sign))


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: callable,
    *,
    n_splits: int = 5,
    n_seeds: int = 20,
    sizing_strategy: str = "sign",
    sizing_kwargs: dict | None = None,
) -> dict:
    """K-fold CV with multi-seed stability test.

    For each of n_seeds random KFold splits:
      1. Split into train/test folds
      2. Train model with model_fn(X_train, y_train)
      3. Predict on test fold
      4. Apply position sizing
      5. Compute Sharpe and direction accuracy

    Parameters
    ----------
    X : array (n_sessions, n_features)
    y : array (n_sessions,) — target returns
    model_fn : callable(X, y) -> fitted model with .predict(X)
    n_splits : int — number of CV folds (default 5)
    n_seeds : int — number of random seeds for stability (default 20)
    sizing_strategy : str — passed to apply_position_sizing
    sizing_kwargs : dict — extra kwargs for position sizing

    Returns
    -------
    dict with keys:
        sharpe_mean, sharpe_std : float — across all seeds
        sharpe_per_seed : list[float] — one Sharpe per seed (averaged over folds)
        accuracy_mean : float — direction accuracy averaged across all folds/seeds
        per_fold_details : list[dict] — detailed per-fold metrics

    Called by
    --------
    - scripts/run_pipeline.py      (main evaluation)
    - scripts/leaderboard.py       (model comparison)
    - notebooks/01_eda.ipynb        (feature ablation)

    NOTE: Sessions are independent synthetic stocks — random KFold is
    appropriate (no temporal leakage concern).
    """
    if sizing_kwargs is None:
        sizing_kwargs = {}

    X = np.asarray(X)
    y = np.asarray(y)

    sharpe_per_seed = []
    all_accuracies = []
    per_fold_details = []

    for seed in range(n_seeds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_sharpes = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = model_fn(X_train, y_train)
            raw_pred = model.predict(X_test)
            positions = apply_position_sizing(raw_pred, sizing_strategy, **sizing_kwargs)

            # Simplified Sharpe for CV: pnl = positions * y_test
            pnl = positions * y_test
            if np.std(pnl) == 0:
                fold_sharpe = 0.0
            else:
                fold_sharpe = float(np.mean(pnl) / np.std(pnl) * 16)

            fold_accuracy = compute_direction_accuracy(raw_pred, y_test)

            fold_sharpes.append(fold_sharpe)
            all_accuracies.append(fold_accuracy)
            per_fold_details.append({
                "seed": seed,
                "fold": fold_idx,
                "sharpe": fold_sharpe,
                "accuracy": fold_accuracy,
                "n_test": len(test_idx),
            })

        seed_sharpe = float(np.mean(fold_sharpes))
        sharpe_per_seed.append(seed_sharpe)

    return {
        "sharpe_mean": float(np.mean(sharpe_per_seed)),
        "sharpe_std": float(np.std(sharpe_per_seed)),
        "sharpe_per_seed": sharpe_per_seed,
        "accuracy_mean": float(np.mean(all_accuracies)),
        "per_fold_details": per_fold_details,
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def run_baselines(y: np.ndarray) -> pd.DataFrame:
    """Compute Sharpe for all baseline strategies on training data.

    Baselines (from BATTLE_PLAN.md):
      1. Always-long:      position = +1 for all sessions
      2. Momentum:         position = sign(first_half_return)
      3. Mean-reversion:   position = -sign(first_half_return)
      4. Random:           position = random ±1

    NOTE: Momentum/mean-reversion need first_half_return, which is a feature.
    For simplicity, pass y (target returns) and we compute Sharpe directly.
    Always-long Sharpe should be ~2.766.

    Returns DataFrame with columns: strategy, sharpe, direction_accuracy.

    Called by
    --------
    - scripts/run_pipeline.py  (print baselines at start)
    - notebooks/01_eda.ipynb   (comparison table)
    """
    y = np.asarray(y, dtype=float)
    rows = []

    # Always-long: positions = +1
    pnl_long = y.copy()
    if np.std(pnl_long) == 0:
        sharpe_long = 0.0
    else:
        sharpe_long = float(np.mean(pnl_long) / np.std(pnl_long) * 16)
    accuracy_long = float(np.mean(y > 0))
    rows.append({"strategy": "always_long", "sharpe": sharpe_long, "direction_accuracy": accuracy_long})

    # Random: average over 100 random seeds
    random_sharpes = []
    for seed in range(100):
        rng = np.random.RandomState(seed)
        positions = rng.choice([-1.0, 1.0], size=len(y))
        pnl = positions * y
        if np.std(pnl) == 0:
            random_sharpes.append(0.0)
        else:
            random_sharpes.append(float(np.mean(pnl) / np.std(pnl) * 16))
    sharpe_random = float(np.mean(random_sharpes))
    accuracy_random = 0.5  # random guessing
    rows.append({"strategy": "random", "sharpe": sharpe_random, "direction_accuracy": accuracy_random})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------

def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Format multiple cross_validate() results into a comparison table.

    Parameters
    ----------
    results : dict mapping model_name -> cross_validate() return dict

    Returns
    -------
    DataFrame with columns: model, sharpe_mean, sharpe_std, accuracy,
    sorted by sharpe_mean descending.

    Called by
    --------
    - scripts/leaderboard.py   (display leaderboard)
    - scripts/run_pipeline.py  (summary output)
    """
    rows = []
    for model_name, res in results.items():
        rows.append({
            "model": model_name,
            "sharpe_mean": res["sharpe_mean"],
            "sharpe_std": res["sharpe_std"],
            "accuracy": res["accuracy_mean"],
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("sharpe_mean", ascending=False).reset_index(drop=True)
    return df
