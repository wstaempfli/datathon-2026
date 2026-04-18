"""XGBoost model training, position sizing, and feature importance."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def sharpe_objective(y_true: np.ndarray, y_pred: np.ndarray):
    """Custom XGBoost objective: minimise negative Sharpe of pnl = pred * y_true.

    Returns (grad, hess) per sample. Gradient is rescaled so split-gain clears
    XGBoost's thresholds; constant Hessian (true Hessian is singular due to
    Sharpe's scale-invariance — reg_lambda pins output scale).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_pred)
    pnl = y_pred * y_true
    sigma = pnl.std()
    if sigma < 1e-8:
        return -y_true, np.ones_like(y_pred)
    mu = pnl.mean()
    grad = -(y_true / sigma) + (mu * y_true * (pnl - mu)) / (sigma**3)
    hess = np.ones_like(y_pred)
    return grad, hess


def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    params: dict | None = None,
    loss: str = "mse",
):
    """Train XGBoost regressor. loss={'mse','sharpe'}; 'sharpe' uses a custom objective."""
    defaults = dict(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        verbosity=0,
    )
    if loss == "sharpe":
        defaults["objective"] = sharpe_objective
        defaults["base_score"] = 1e-3
    elif loss != "mse":
        raise ValueError(f"unknown loss: {loss}")
    if params is not None:
        defaults.update(params)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBRegressor(**defaults)),
    ]).fit(X, y)


def apply_position_sizing(predictions):
    """Pass raw predictions through as scalar positions (confidence-weighted)."""
    return np.asarray(predictions, dtype=float)


def feature_importance(model, feature_names: list[str]) -> pd.Series:
    """Return XGBoost feature importances as a sorted Series."""
    importance = model[-1].feature_importances_
    return pd.Series(importance, index=feature_names).sort_values(ascending=False)
