from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# QuantileRegressor at the median targets absolute deviation, which aligns with
# Sharpe's penalty on PnL dispersion better than squared error.
DEFAULT_PARAMS = {
    "quantile": 0.5,
    "alpha": 0.01,
    "solver": "highs",
}

N_SPLITS = 5


def sharpe(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, dtype=float)
    s = float(np.std(pnl))
    if s == 0 or not np.isfinite(s):
        return 0.0
    return float(np.mean(pnl) / s * 16.0)


def _make_pipeline(params: dict) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("est", QuantileRegressor(**params)),
        ]
    )


def train_cv(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    n_splits: int = N_SPLITS,
    seed: int = 42,
) -> tuple[list[Pipeline], np.ndarray, dict]:
    """Cross-validated QuantileRegressor (median) training.

    Returns (models, oof_preds, metrics) with per-fold RMSE and Sharpe on raw
    predictions used as positions, plus CV aggregates.
    """
    if params is None:
        params = dict(DEFAULT_PARAMS)
    else:
        merged = dict(DEFAULT_PARAMS)
        merged.update(params)
        params = merged

    assert X.index.equals(y.index), "X and y must share the same index"

    X_vals = X.values
    y_vals = y.values.astype(float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=float)

    models: list[Pipeline] = []
    per_fold_rmse: list[float] = []
    per_fold_sharpe_raw: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
        m = _make_pipeline(params)
        m.fit(X_vals[tr_idx], y_vals[tr_idx])
        preds = m.predict(X_vals[va_idx])
        oof[va_idx] = preds
        models.append(m)

        rmse = float(np.sqrt(mean_squared_error(y_vals[va_idx], preds)))
        s_raw = sharpe(preds * y_vals[va_idx])
        per_fold_rmse.append(rmse)
        per_fold_sharpe_raw.append(s_raw)

        print(f"[fold {fold + 1}/{n_splits}] rmse={rmse:.5f}  sharpe_raw={s_raw:.3f}")

    cv_rmse = float(np.sqrt(mean_squared_error(y_vals, oof)))

    def _mean_std(xs: list[float]) -> tuple[float, float]:
        arr = np.array(xs, dtype=float)
        return float(arr.mean()), float(arr.std())

    rmse_mean, rmse_std = _mean_std(per_fold_rmse)
    sraw_mean, sraw_std = _mean_std(per_fold_sharpe_raw)

    metrics: dict = {
        "per_fold_rmse": per_fold_rmse,
        "per_fold_sharpe_raw": per_fold_sharpe_raw,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "sharpe_raw_mean": sraw_mean,
        "sharpe_raw_std": sraw_std,
        "cv_rmse": cv_rmse,
        "cv_sharpe_raw": sharpe(oof * y_vals),
    }

    print(
        f"CV: rmse={rmse_mean:.5f}±{rmse_std:.5f}  "
        f"sharpe_raw={sraw_mean:.3f}±{sraw_std:.3f}"
    )

    if abs(rmse_mean) > 0 and rmse_std > 0.3 * abs(rmse_mean):
        print(
            f"WARNING: high variance — model may be unstable "
            f"(rmse: mean={rmse_mean:.4f}, std={rmse_std:.4f})"
        )

    return models, oof, metrics


def fit_final(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
) -> Pipeline:
    """Fit on all train; no validation. Saves to models/quantile_final.pkl."""
    if params is None:
        params = dict(DEFAULT_PARAMS)
    else:
        merged = dict(DEFAULT_PARAMS)
        merged.update(params)
        params = merged

    m = _make_pipeline(params)
    m.fit(X.values, y.values.astype(float))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "quantile_final.pkl", "wb") as f:
        pickle.dump(m, f)
    return m


def predict(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X.values)
