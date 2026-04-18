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
    returns = end_close / halfway_close - 1
    pnl = positions * returns

    std = pnl.std()
    if std == 0:
        return 0.0
    
    return pnl.mean() / std * 16
    

def compute_direction_accuracy(predictions: np.ndarray, actual_returns: np.ndarray) -> float:
    """Fraction of sessions where sign(prediction) == sign(actual_return).

    Always-long baseline accuracy = 57.0% (the positive drift rate).
    We need > 57% to beat it.
    """
    return (np.sign(predictions) == np.sign(actual_returns)).mean()


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
    sizing_kwargs = sizing_kwargs or {}

    sharpe_per_seed = []
    accuracy_per_seed = []
    per_fold_details = []

    for seed in range(n_seeds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_sharpes = []
        fold_accuracies = []

        for fold_ix, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = model_fn(X_train, y_train)
            raw_pred = model.predict(X_test)

            positions = apply_position_sizing(raw_pred, strategy=sizing_strategy, **sizing_kwargs)

            pnl = positions * y_test
            if pnl.std() > 0:
                fold_sharpe = pnl.mean() / pnl.std() * 16
            else:
                fold_sharpe = 0.0
            
            fold_accuracy = compute_direction_accuracy(raw_pred, y_test)

            fold_sharpes.append(fold_sharpe)
            fold_accuracies.append(fold_accuracy)
            per_fold_details.append({
                "seed": seed,
                "fold": fold_ix,
                "sharpe": fold_sharpe,
                "accuracy": fold_accuracy,
                "n_test": len(test_idx),
            })
        
        sharpe_per_seed.append(np.mean(fold_sharpes))
        accuracy_per_seed.append(np.mean(fold_accuracies))

    sharpe_per_seed = np.array(sharpe_per_seed)

    return {
        "sharpe_mean": float(sharpe_per_seed.mean()),
        "sharpe_std": float(sharpe_per_seed.std()),
        "sharpe_per_seed": sharpe_per_seed.tolist(),
        "accuracy_mean": float(np.mean(accuracy_per_seed)),
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
    # Always long
    pnl_long = 1.0 * y
    sharpe_long = pnl_long.mean() / pnl_long.std() * 16
    acc_long = (y > 0).mean()

    # Always short
    pnl_short = -1.0 * y
    sharpe_short = pnl_short.mean() / pnl_short.std() * 16
    acc_short = (y < 0).mean()

    # Random
    rng = np.random.default_rng(0)
    random_sharpes = []
    for _ in range(50):
        positions = rng.choice([-1, 1], size=len(y))
        pnl = positions * y
        if pnl.std() > 0:
            random_sharpes.append(pnl.mean() / pnl.std() * 16)
    sharpe_random = np.mean(random_sharpes)
    acc_random = 0.5

    return pd.DataFrame({
        "strategy": ["always_long", "always_short", "random"],
        "sharpe": [sharpe_long, sharpe_short, sharpe_random],
        "direction_accuracy": [acc_long, acc_short, acc_random],
    })



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
    for name, res in results.items():
        rows.append({
            "model": name,
            "sharpe_mean": res["sharpe_mean"],
            "sharpe_std": res["sharpe_std"],
            "accuracy_mean": res["accuracy_mean"],
        })
    return pd.DataFrame(rows).sort_values("sharpe_mean", ascending=False).reset_index(drop=True)
