"""Linear / robust-linear regressors for learned variants (V2-V5)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.linear_model import HuberRegressor, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ModelKind = Literal["ridge", "lasso", "huber"]


def _make(kind: ModelKind) -> Pipeline:
    if kind == "ridge":
        est = Ridge(alpha=1.0)
    elif kind == "lasso":
        est = Lasso(alpha=0.0005, max_iter=20000)
    elif kind == "huber":
        est = HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=500)
    else:
        raise ValueError(f"unknown kind: {kind}")
    return Pipeline([("scaler", StandardScaler()), ("est", est)])


def fit_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, kind: ModelKind
) -> np.ndarray:
    """Fit (scaler+regressor) on train, return predictions on test."""
    pipe = _make(kind)
    pipe.fit(X_train, y_train)
    return pipe.predict(X_test)


def fit_predict_with_train(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, kind: ModelKind
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_preds, test_preds). Useful for in-sample k tuning."""
    pipe = _make(kind)
    pipe.fit(X_train, y_train)
    return pipe.predict(X_train), pipe.predict(X_test)
