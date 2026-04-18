"""Evaluation: Sharpe computation and cross-validation."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.models import apply_position_sizing


def compute_sharpe(
    positions: np.ndarray,
    halfway_close: np.ndarray,
    end_close: np.ndarray,
) -> float:
    """Sharpe ratio: mean(pnl) / std(pnl) * 16, with pnl = positions * (end/halfway - 1)."""
    positions = np.asarray(positions, dtype=float)
    halfway_close = np.asarray(halfway_close, dtype=float)
    end_close = np.asarray(end_close, dtype=float)

    pnl = positions * (end_close / halfway_close - 1)
    if np.std(pnl) == 0:
        return 0.0
    return float(np.mean(pnl) / np.std(pnl) * 16)


def compute_direction_accuracy(predictions: np.ndarray, actual_returns: np.ndarray) -> float:
    """Fraction of sessions where sign(prediction) == sign(actual_return)."""
    pred_sign = np.sign(np.asarray(predictions, dtype=float))
    actual_sign = np.sign(np.asarray(actual_returns, dtype=float))
    return float(np.mean(pred_sign == actual_sign))


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: callable,
    *,
    n_splits: int = 5,
    n_seeds: int = 20,
) -> dict:
    """K-fold CV with multi-seed stability test."""
    X = np.asarray(X)
    y = np.asarray(y)

    sharpe_per_seed = []
    all_accuracies = []

    for seed in range(n_seeds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_sharpes = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = model_fn(X_train, y_train)
            raw_pred = model.predict(X_test)
            positions = apply_position_sizing(raw_pred)

            pnl = positions * y_test
            if np.std(pnl) == 0:
                fold_sharpe = 0.0
            else:
                fold_sharpe = float(np.mean(pnl) / np.std(pnl) * 16)

            fold_sharpes.append(fold_sharpe)
            all_accuracies.append(compute_direction_accuracy(raw_pred, y_test))

        sharpe_per_seed.append(float(np.mean(fold_sharpes)))

    return {
        "sharpe_mean": float(np.mean(sharpe_per_seed)),
        "sharpe_std": float(np.std(sharpe_per_seed)),
        "sharpe_per_seed": sharpe_per_seed,
        "accuracy_mean": float(np.mean(all_accuracies)),
    }
