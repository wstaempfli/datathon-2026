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
    # TODO: Create Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    # TODO: Fit and return
    raise NotImplementedError


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
    # TODO: Convert y to binary: y_binary = (y > 0).astype(int)
    # TODO: Create Pipeline with StandardScaler + LogisticRegression
    # TODO: Handle solver choice: saga for elasticnet, lbfgs otherwise
    # TODO: Fit and return
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Phase 2 alternatives: Lasso, ElasticNet
# ---------------------------------------------------------------------------

def train_lasso(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    """Train Lasso (L1) — useful for automatic feature selection."""
    # TODO: Pipeline with StandardScaler + Lasso(alpha, max_iter=5000)
    raise NotImplementedError


def train_elasticnet(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, l1_ratio: float = 0.5):
    """Train ElasticNet (blended L1+L2)."""
    # TODO: Pipeline with StandardScaler + ElasticNet
    raise NotImplementedError


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
    # TODO: Import lightgbm (guard with try/except ImportError)
    # TODO: Merge user params over conservative defaults
    # TODO: Pipeline with StandardScaler + LGBMRegressor
    raise NotImplementedError


def train_xgboost(X: np.ndarray, y: np.ndarray, params: dict | None = None):
    """Train XGBoost regressor. Same cautions as LightGBM."""
    # TODO: Import xgboost (guard with try/except ImportError)
    # TODO: Merge user params over conservative defaults
    # TODO: Pipeline with StandardScaler + XGBRegressor
    raise NotImplementedError


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
        # TODO: Store models and normalize weights (default = equal)
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: For each model, get sign(model.predict(X))
        # TODO: Return weighted average across models
        raise NotImplementedError


def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    model_fns: list | None = None,
    weights: list[float] | None = None,
) -> EnsembleModel:
    """Train multiple models and wrap in EnsembleModel.

    Default model_fns: [train_ridge, train_logistic].
    """
    # TODO: Default to [train_ridge, train_logistic] if model_fns is None
    # TODO: Fit each model_fn(X, y)
    # TODO: Return EnsembleModel(fitted_models, weights)
    raise NotImplementedError


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
    # TODO: Implement each strategy branch
    # TODO: "sign" → np.sign(predictions)
    # TODO: "asymmetric" → +1 if pred>0 else -short_scale
    # TODO: "logistic" → 2 * probabilities - 1
    # TODO: "clipped" → np.clip(predictions * scale, -max_pos, max_pos)
    # TODO: "quantile" → rank predictions, bin into 5 buckets
    raise NotImplementedError


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
    # TODO: Access the estimator inside the Pipeline via named_steps
    # TODO: For linear models → np.abs(estimator.coef_)
    # TODO: For tree models → estimator.feature_importances_
    # TODO: Return pd.Series sorted descending
    raise NotImplementedError
