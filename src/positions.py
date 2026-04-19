"""Position sizing utilities: risk-parity vol scaling + final clip."""

from __future__ import annotations

import numpy as np


def vol_scale(
    sigma: np.ndarray,
    sigma_ref: float,
    floor_frac: float = 0.25,
    scaler_clip: tuple[float, float] = (0.5, 2.0),
) -> np.ndarray:
    """Risk-parity scaler: sigma_ref / max(sigma, floor * sigma_ref), clipped.

    Halving sigma doubles the scaler (within clip); a tiny-sigma session cannot
    exceed scaler_clip[1].
    """
    sigma = np.asarray(sigma, dtype=float)
    floor = floor_frac * sigma_ref
    denom = np.maximum(sigma, floor)
    raw = sigma_ref / np.where(denom <= 0.0, floor, denom)
    return np.clip(raw, scaler_clip[0], scaler_clip[1])


def to_position(
    raw: np.ndarray,
    scaler: np.ndarray | float = 1.0,
    intercept: float = 0.0,
    clip_lo: float = -2.0,
    clip_hi: float = 2.0,
) -> np.ndarray:
    """Apply scaler + intercept, then clip. Final step of every variant."""
    out = np.asarray(raw, dtype=float) * np.asarray(scaler, dtype=float) + intercept
    return np.clip(out, clip_lo, clip_hi)


def fit_k_insample(
    pred_centered: np.ndarray,
    scaler: np.ndarray | float,
    target_return: np.ndarray,
    intercept: float = 1.0,
    k_grid: tuple[int, ...] = (10, 15, 20, 25, 30, 35, 40, 45, 50),
    clip_lo: float = -2.0,
    clip_hi: float = 2.0,
) -> tuple[float, float]:
    """Pick k that maximizes in-sample Sharpe on (pred, scaler, target).

    Returns (k_best, sharpe_at_k_best). Train-fold only; never call with
    held-out target_return.
    """
    best_k, best_s = k_grid[0], -np.inf
    for k in k_grid:
        pos = to_position(pred_centered * k, scaler, intercept, clip_lo, clip_hi)
        pnl = pos * target_return
        mu = float(np.mean(pnl))
        sd = float(np.std(pnl))
        sharpe = (mu / sd * 16.0) if sd > 0 else -np.inf
        if sharpe > best_s:
            best_s, best_k = sharpe, k
    return float(best_k), float(best_s)
