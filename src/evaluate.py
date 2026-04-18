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
    # TODO: Compute per-session PnL
    # TODO: Return mean(pnl) / std(pnl) * 16, handle std==0 edge case
    raise NotImplementedError


def compute_direction_accuracy(predictions: np.ndarray, actual_returns: np.ndarray) -> float:
    """Fraction of sessions where sign(prediction) == sign(actual_return).

    Always-long baseline accuracy = 57.0% (the positive drift rate).
    We need > 57% to beat it.
    """
    # TODO: Compare signs, return fraction correct
    raise NotImplementedError


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
    # TODO: Loop over n_seeds
    #   TODO: Create KFold(n_splits, shuffle=True, random_state=seed)
    #   TODO: For each fold:
    #     TODO: Split X, y into train/test
    #     TODO: Fit model = model_fn(X_train, y_train)
    #     TODO: raw_pred = model.predict(X_test)
    #     TODO: positions = apply_position_sizing(raw_pred, sizing_strategy, **sizing_kwargs)
    #     TODO: fold_sharpe = compute_sharpe(positions, ..., y_test)
    #          NOTE: For CV, we can simplify since halfway_close cancels out
    #          when positions use sign-only sizing.  PnL = positions * y_test.
    #          sharpe = mean(pnl) / std(pnl) * 16
    #     TODO: fold_accuracy = compute_direction_accuracy(raw_pred, y_test)
    #   TODO: Average fold Sharpes for this seed
    # TODO: Aggregate across seeds: mean, std of per-seed Sharpes
    # TODO: Return results dict
    raise NotImplementedError


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
    # TODO: Always-long: pnl = 1.0 * y, compute Sharpe
    # TODO: Random: pnl = random_sign * y, compute Sharpe (avg over many seeds)
    # TODO: Return DataFrame with strategy name, Sharpe, accuracy
    # NOTE: Momentum and mean-reversion need first_half_return from features.
    #       Either accept it as a parameter or compute from bars here.
    raise NotImplementedError


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
    # TODO: Extract sharpe_mean, sharpe_std, accuracy_mean from each result
    # TODO: Return sorted DataFrame
    raise NotImplementedError
