from __future__ import annotations
import numpy as np


def size_positions(
    pred_return: np.ndarray,
    realized_vol: np.ndarray,
    clip_quantile: float = 0.99,
    target_std: float = 1.0,
    vol_floor: float = 1e-4,
) -> np.ndarray:
    """Vol-scaled + clipped + globally rescaled position size.

    - raw = pred / max(vol, vol_floor)
    - clip to [-q, +q] where q = quantile(|raw|, clip_quantile)
    - scale so raw.std() == target_std (Sharpe-invariant; cosmetic).
    """
    pred_return = np.asarray(pred_return, dtype=float)
    realized_vol = np.asarray(realized_vol, dtype=float)
    raw = pred_return / np.maximum(realized_vol, vol_floor)
    q = float(np.quantile(np.abs(raw), clip_quantile))
    if q > 0:
        raw = np.clip(raw, -q, q)
    std = raw.std()
    if std and np.isfinite(std):
        raw = raw * (target_std / std)
    return raw
