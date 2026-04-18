"""XGBoost model training, position sizing, and feature importance."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def train_xgboost(X: np.ndarray, y: np.ndarray, params: dict | None = None):
    """Train XGBoost regressor with conservative defaults inside a StandardScaler pipeline."""
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
