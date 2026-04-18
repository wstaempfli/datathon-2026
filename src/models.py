"""
Model training and prediction.

Uniform interface: every train_* function returns a fitted object with
a .predict(X) method.  All models use StandardScaler internally via
sklearn Pipeline.

Usage:
    from src.models import train_ridge, apply_position_sizing, feature_importance

    model = train_ridge(X_train, y_train, alpha=1.0)
    raw_pred = model.predict(X_test)
    positions = apply_position_sizing(raw_pred, strategy="sign")
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression


# ---------------------------------------------------------------------------
# Primary: Ridge regression  (CV Sharpe 2.79 +/- 0.04 — most stable)
# ---------------------------------------------------------------------------

def train_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    """Train Ridge regression with StandardScaler in a Pipeline.

    Returns sklearn Pipeline with .predict(X) method.

    Called by
    --------
    - scripts/run_pipeline.py  (main training)
    - src/evaluate.py          (inside CV loop)
    """
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))]).fit(X, y)


# ---------------------------------------------------------------------------
# Secondary: Logistic regression (direction prediction, native probabilities)
# ---------------------------------------------------------------------------

def train_logistic(X: np.ndarray, y: np.ndarray, C: float = 1.0, penalty: str = "l2"):
    """Train logistic regression on sign(y) for direction prediction.

    The model predicts P(return > 0).  For soft sizing use:
        positions = 2 * model.predict_proba(X)[:, 1] - 1

    Called by
    --------
    - scripts/run_pipeline.py
    - src/evaluate.py
    """
    y_binary = (y > 0).astype(int)
    solver = "saga" if penalty in ("l1", "elasticnet") else "lbfgs"
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=5000)),
    ]).fit(X, y_binary)


# ---------------------------------------------------------------------------
# Phase 2 alternatives: Lasso, ElasticNet
# ---------------------------------------------------------------------------

def train_lasso(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    """Train Lasso (L1) — useful for automatic feature selection."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=alpha, max_iter=5000)),
    ]).fit(X, y)


def train_elasticnet(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, l1_ratio: float = 0.5):
    """Train ElasticNet (blended L1+L2)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("elasticnet", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)),
    ]).fit(X, y)


# ---------------------------------------------------------------------------
# Tree models (CAUTION: overfit badly with 1K samples)
# LightGBM CV Sharpe: 2.43 +/- 0.79 — only use in ensembles
# ---------------------------------------------------------------------------

def train_lightgbm(X: np.ndarray, y: np.ndarray, params: dict | None = None):
    """Train LightGBM regressor with conservative defaults.

    Default params: n_estimators=50, max_depth=3, min_child_samples=30,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0,
    reg_lambda=1.0.

    WARNING: Overfits with 1K sessions. Only use if heavily regularized AND
    it still beats Ridge in 20-seed stability test.
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        raise ImportError("lightgbm is required for train_lightgbm. Install with: pip install lightgbm")

    defaults = dict(
        n_estimators=50,
        max_depth=3,
        min_child_samples=30,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        verbose=-1,
    )
    if params is not None:
        defaults.update(params)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("lgbm", LGBMRegressor(**defaults)),
    ]).fit(X, y)


def train_xgboost(X: np.ndarray, y: np.ndarray, params: dict | None = None):
    """Train XGBoost regressor. Same cautions as LightGBM."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("xgboost is required for train_xgboost. Install with: pip install xgboost")

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


# ---------------------------------------------------------------------------
# Ensemble (Phase 3)
# ---------------------------------------------------------------------------

class EnsembleModel:
    """Weighted average of multiple fitted models' sign predictions.

    predict(X) returns values in [-1, +1] — the weighted average of
    np.sign(model_i.predict(X)).

    Called by
    --------
    - train_ensemble() below
    - src/evaluate.py (treated like any other model with .predict)
    """

    def __init__(self, models: list, weights: list[float] | None = None):
        self.models = models
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]
        self.weights = np.array(weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        sign_preds = np.array([np.sign(model.predict(X)) for model in self.models])
        return np.dot(self.weights, sign_preds)


def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    model_fns: list | None = None,
    weights: list[float] | None = None,
) -> EnsembleModel:
    """Train multiple models and wrap in EnsembleModel.

    Default model_fns: [train_ridge, train_logistic].
    """
    if model_fns is None:
        model_fns = [train_ridge, train_logistic]
    fitted_models = [fn(X, y) for fn in model_fns]
    return EnsembleModel(fitted_models, weights)


# ---------------------------------------------------------------------------
# Position sizing — converts raw predictions to target_position values
# ---------------------------------------------------------------------------

def apply_position_sizing(
    predictions: np.ndarray,
    strategy: str = "sign",
    **kwargs,
) -> np.ndarray:
    """Convert raw model predictions into target positions.

    Strategies (test each with 20-seed stability in Phase 2):
      "sign"       : ±1 always.  Default and safest.
      "asymmetric" : +1 if pred>0, else -short_scale (default 0.5).
                     Respects positive drift (57% up).
      "logistic"   : 2*P(up) - 1.  Requires probabilities kwarg.
      "clipped"    : pred * scale, clipped to [-max_pos, +max_pos].
      "quantile"   : Map pred rank to {-1, -0.5, 0, +0.5, +1}.

    Called by
    --------
    - scripts/run_pipeline.py  (final position generation)
    - src/evaluate.py          (inside CV to compute Sharpe)
    - src/submit.py            (test predictions → positions)
    """
    predictions = np.asarray(predictions, dtype=float)

    if strategy == "sign":
        return np.sign(predictions)

    elif strategy == "asymmetric":
        short_scale = kwargs.get("short_scale", 0.5)
        return np.where(predictions > 0, 1.0, -short_scale)

    elif strategy == "clipped":
        scale = kwargs.get("scale", 1.0)
        max_pos = kwargs.get("max_pos", 1.0)
        return np.clip(predictions * scale, -max_pos, max_pos)

    elif strategy == "quantile":
        bins = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        n = len(predictions)
        if n == 0:
            return predictions.copy()
        ranks = np.argsort(np.argsort(predictions)).astype(float)
        # Map ranks to 0-4 bin indices (5 equal bins)
        bin_indices = np.clip((ranks / n * 5).astype(int), 0, 4)
        return bins[bin_indices]

    else:
        # Unknown strategy — default to sign
        return np.sign(predictions)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def feature_importance(model, feature_names: list[str]) -> pd.Series:
    """Extract feature importance from a fitted Pipeline.

    Works for Ridge/Lasso/ElasticNet (|coef|), Logistic (|coef|),
    and LightGBM/XGBoost (feature_importances_).

    Called by
    --------
    - scripts/run_pipeline.py  (print after training)
    - notebooks/01_eda.ipynb   (visualization)
    """
    # Access the last step (the estimator) inside the Pipeline
    if hasattr(model, 'named_steps'):
        estimator = model[len(model) - 1]
    else:
        estimator = model

    if hasattr(estimator, 'coef_'):
        importance = np.abs(estimator.coef_).ravel()
    elif hasattr(estimator, 'feature_importances_'):
        importance = estimator.feature_importances_
    else:
        raise ValueError(f"Cannot extract importance from {type(estimator).__name__}")

    return pd.Series(importance, index=feature_names).sort_values(ascending=False)
