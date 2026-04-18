from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from src.position import size_positions

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
}


def build_feature_matrix(
    price_session: pd.DataFrame,
    sent_session: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge per-session price + sentiment features on session index.

    Returns a DataFrame indexed by session, NaN-filled with 0.0 (sentiment
    missing for sessions with no headlines). Target columns are not included.
    """
    price = price_session.copy()
    price.index.name = "session"

    # Drop any target-like columns defensively.
    drop_cols = [c for c in ("target_return", "target_position") if c in price.columns]
    if drop_cols:
        price = price.drop(columns=drop_cols)

    if sent_session is None or len(sent_session) == 0:
        X = price
    else:
        sent = sent_session.copy()
        sent.index.name = "session"
        # Left-join: keep every session present in price features.
        X = price.join(sent, how="left")

    X = X.sort_index()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def sharpe(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, dtype=float)
    s = float(np.std(pnl))
    if s == 0 or not np.isfinite(s):
        return 0.0
    return float(np.mean(pnl) / s * 16.0)


def _accumulate_importance(boosters: list[lgb.Booster], feature_names: list[str]) -> dict:
    agg = np.zeros(len(feature_names), dtype=float)
    for b in boosters:
        imp = b.feature_importance(importance_type="gain")
        # Defensive: LightGBM returns importances in booster feature order.
        names = b.feature_name()
        idx = {n: i for i, n in enumerate(feature_names)}
        for n, v in zip(names, imp):
            if n in idx:
                agg[idx[n]] += float(v)
    if len(boosters) > 0:
        agg /= len(boosters)
    return {n: float(v) for n, v in zip(feature_names, agg)}


def train_cv(
    X: pd.DataFrame,
    y: pd.Series,
    vol: pd.Series,
    params: dict | None = None,
    n_splits: int = 5,
    num_boost_round: int = 1500,
    early_stopping: int = 100,
    seed: int = 42,
) -> tuple[list[lgb.Booster], np.ndarray, dict]:
    """Cross-validated LightGBM training.

    Returns (boosters, oof_preds, metrics) where metrics include per-fold RMSE,
    per-fold Sharpe using raw predictions as positions, and per-fold Sharpe using
    vol-scaled predictions, plus CV aggregates and a feature_importance_gain dict.
    """
    if params is None:
        params = dict(DEFAULT_PARAMS)
    else:
        merged = dict(DEFAULT_PARAMS)
        merged.update(params)
        params = merged

    assert X.index.equals(y.index), "X and y must share the same index"
    assert X.index.equals(vol.index), "X and vol must share the same index"

    feature_names = list(X.columns)
    X_vals = X.values
    y_vals = y.values.astype(float)
    vol_vals = vol.values.astype(float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=float)

    boosters: list[lgb.Booster] = []
    per_fold_rmse: list[float] = []
    per_fold_sharpe_raw: list[float] = []
    per_fold_sharpe_vol: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
        dtr = lgb.Dataset(X_vals[tr_idx], label=y_vals[tr_idx], feature_name=feature_names)
        dva = lgb.Dataset(X_vals[va_idx], label=y_vals[va_idx], feature_name=feature_names,
                          reference=dtr)

        booster = lgb.train(
            params,
            dtr,
            num_boost_round=num_boost_round,
            valid_sets=[dva],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(early_stopping, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        boosters.append(booster)

        preds = booster.predict(X_vals[va_idx], num_iteration=booster.best_iteration or None)
        oof[va_idx] = preds

        rmse = float(np.sqrt(mean_squared_error(y_vals[va_idx], preds)))
        # Sharpe(raw): pred used directly as position
        pnl_raw = preds * y_vals[va_idx]
        s_raw = sharpe(pnl_raw)
        # Sharpe(vol-scaled): pos = pred / max(vol, floor), NO clipping here
        pos_vol = preds / np.maximum(vol_vals[va_idx], 1e-4)
        pnl_vol = pos_vol * y_vals[va_idx]
        s_vol = sharpe(pnl_vol)

        per_fold_rmse.append(rmse)
        per_fold_sharpe_raw.append(s_raw)
        per_fold_sharpe_vol.append(s_vol)

        print(
            f"[fold {fold + 1}/{n_splits}] rmse={rmse:.5f}  "
            f"sharpe_raw={s_raw:.3f}  sharpe_vol={s_vol:.3f}  "
            f"best_iter={booster.best_iteration}"
        )

    # CV Sharpe(vol-scaled) via full OOF through size_positions
    pos_full = size_positions(oof, vol_vals)
    cv_sharpe_vol_full = sharpe(pos_full * y_vals)
    cv_rmse = float(np.sqrt(mean_squared_error(y_vals, oof)))

    def _mean_std(xs: list[float]) -> tuple[float, float]:
        arr = np.array(xs, dtype=float)
        return float(arr.mean()), float(arr.std())

    rmse_mean, rmse_std = _mean_std(per_fold_rmse)
    sraw_mean, sraw_std = _mean_std(per_fold_sharpe_raw)
    svol_mean, svol_std = _mean_std(per_fold_sharpe_vol)

    fi = _accumulate_importance(boosters, feature_names)

    metrics: dict = {
        "per_fold_rmse": per_fold_rmse,
        "per_fold_sharpe_raw": per_fold_sharpe_raw,
        "per_fold_sharpe_vol": per_fold_sharpe_vol,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "sharpe_raw_mean": sraw_mean,
        "sharpe_raw_std": sraw_std,
        "sharpe_vol_mean": svol_mean,
        "sharpe_vol_std": svol_std,
        "cv_rmse": cv_rmse,
        "cv_sharpe_raw": sharpe(oof * y_vals),
        "cv_sharpe_vol": cv_sharpe_vol_full,
        "feature_importance_gain": fi,
    }

    print(
        f"CV: rmse={rmse_mean:.5f}±{rmse_std:.5f}  "
        f"sharpe_raw={sraw_mean:.3f}±{sraw_std:.3f}  "
        f"sharpe_vol={svol_mean:.3f}±{svol_std:.3f}  "
        f"cv_sharpe_vol_full={cv_sharpe_vol_full:.3f}"
    )

    # Instability warning: flag when std exceeds 0.3 * |mean| for any headline metric.
    def _warn(label: str, mean: float, std: float) -> None:
        if abs(mean) > 0 and std > 0.3 * abs(mean):
            print(f"WARNING: high variance — model may be unstable ({label}: "
                  f"mean={mean:.4f}, std={std:.4f})")

    _warn("rmse", rmse_mean, rmse_std)
    _warn("sharpe_vol", svol_mean, svol_std)

    return boosters, oof, metrics


def fit_final(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    num_boost_round: int = 1000,
) -> lgb.Booster:
    """Fit on all train; no validation. Saves to models/lgbm_final.txt."""
    if params is None:
        params = dict(DEFAULT_PARAMS)
    else:
        merged = dict(DEFAULT_PARAMS)
        merged.update(params)
        params = merged

    feature_names = list(X.columns)
    dtr = lgb.Dataset(X.values, label=y.values.astype(float), feature_name=feature_names)
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=num_boost_round,
        callbacks=[lgb.log_evaluation(0)],
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODELS_DIR / "lgbm_final.txt"))
    return booster


def predict(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return booster.predict(X.values, num_iteration=booster.best_iteration or None)
