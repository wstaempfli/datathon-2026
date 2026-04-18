from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

DEFAULT_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "colsample_bylevel": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
    "tree_method": "hist",
    "seed": 42,
    "verbosity": 0,
}

NUM_BOOST_ROUND = 800
EARLY_STOPPING_ROUNDS = 50
N_SPLITS = 5


def sharpe(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, dtype=float)
    s = float(np.std(pnl))
    if s == 0 or not np.isfinite(s):
        return 0.0
    return float(np.mean(pnl) / s * 16.0)


def _accumulate_importance(boosters: list[xgb.Booster], feature_names: list[str]) -> dict:
    agg = np.zeros(len(feature_names), dtype=float)
    idx = {n: i for i, n in enumerate(feature_names)}
    for b in boosters:
        # get_score returns {feature_name: gain}; features with zero splits are absent.
        score = b.get_score(importance_type="gain")
        for n, v in score.items():
            if n in idx:
                agg[idx[n]] += float(v)
    if len(boosters) > 0:
        agg /= len(boosters)
    return {n: float(v) for n, v in zip(feature_names, agg)}


def train_cv(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    n_splits: int = N_SPLITS,
    num_boost_round: int = NUM_BOOST_ROUND,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    seed: int = 42,
) -> tuple[list[xgb.Booster], np.ndarray, dict]:
    """Cross-validated XGBoost training.

    Returns (boosters, oof_preds, metrics) with per-fold RMSE, per-fold Sharpe on
    raw predictions used as positions, plus CV aggregates and feature importance.
    """
    if params is None:
        params = dict(DEFAULT_PARAMS)
    else:
        merged = dict(DEFAULT_PARAMS)
        merged.update(params)
        params = merged

    assert X.index.equals(y.index), "X and y must share the same index"

    feature_names = list(X.columns)
    X_vals = X.values
    y_vals = y.values.astype(float)

    # sessions are independent synthetic stocks → KFold not TimeSeriesSplit
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=float)

    boosters: list[xgb.Booster] = []
    per_fold_rmse: list[float] = []
    per_fold_sharpe_raw: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
        dtrain = xgb.DMatrix(
            X_vals[tr_idx], label=y_vals[tr_idx], feature_names=feature_names
        )
        dval = xgb.DMatrix(
            X_vals[va_idx], label=y_vals[va_idx], feature_names=feature_names
        )

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        boosters.append(booster)

        best_iter = booster.best_iteration or 0
        preds = booster.predict(dval, iteration_range=(0, best_iter + 1))
        oof[va_idx] = preds

        rmse = float(np.sqrt(mean_squared_error(y_vals[va_idx], preds)))
        # Sharpe(raw): pred used directly as position
        s_raw = sharpe(preds * y_vals[va_idx])

        per_fold_rmse.append(rmse)
        per_fold_sharpe_raw.append(s_raw)

        print(
            f"[fold {fold + 1}/{n_splits}] rmse={rmse:.5f}  "
            f"sharpe_raw={s_raw:.3f}  best_iter={best_iter}"
        )

    cv_rmse = float(np.sqrt(mean_squared_error(y_vals, oof)))

    def _mean_std(xs: list[float]) -> tuple[float, float]:
        arr = np.array(xs, dtype=float)
        return float(arr.mean()), float(arr.std())

    rmse_mean, rmse_std = _mean_std(per_fold_rmse)
    sraw_mean, sraw_std = _mean_std(per_fold_sharpe_raw)

    fi = _accumulate_importance(boosters, feature_names)

    metrics: dict = {
        "per_fold_rmse": per_fold_rmse,
        "per_fold_sharpe_raw": per_fold_sharpe_raw,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "sharpe_raw_mean": sraw_mean,
        "sharpe_raw_std": sraw_std,
        "cv_rmse": cv_rmse,
        "cv_sharpe_raw": sharpe(oof * y_vals),
        "feature_importance_gain": fi,
    }

    print(
        f"CV: rmse={rmse_mean:.5f}±{rmse_std:.5f}  "
        f"sharpe_raw={sraw_mean:.3f}±{sraw_std:.3f}"
    )

    # Instability warning: flag when std exceeds 0.3 * |mean| for any headline metric.
    if abs(rmse_mean) > 0 and rmse_std > 0.3 * abs(rmse_mean):
        print(
            f"WARNING: high variance — model may be unstable "
            f"(rmse: mean={rmse_mean:.4f}, std={rmse_std:.4f})"
        )

    return boosters, oof, metrics


def fit_final(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    num_boost_round: int = NUM_BOOST_ROUND,
) -> xgb.Booster:
    """Fit on all train; no validation. Saves to models/xgb_final.json."""
    if params is None:
        params = dict(DEFAULT_PARAMS)
    else:
        merged = dict(DEFAULT_PARAMS)
        merged.update(params)
        params = merged

    feature_names = list(X.columns)
    dtrain = xgb.DMatrix(
        X.values, label=y.values.astype(float), feature_names=feature_names
    )
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=False,
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODELS_DIR / "xgb_final.json"))
    return booster


def predict(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Predict with all boosted rounds.

    Intentionally omits `iteration_range` so boosters from `fit_final` (which
    trains without a validation set, so `best_iteration == 0`) use every tree.
    The CV fold-level predict inside `train_cv` keeps its own `iteration_range`
    because it *does* have early stopping.
    """
    feature_names = list(X.columns)
    dmat = xgb.DMatrix(X.values, feature_names=feature_names)
    return booster.predict(dmat)
