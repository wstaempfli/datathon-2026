"""Smoke tests for build_features."""

import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES, build_features, max_drawdown, parkinson_vol, realized_vol


def _make_bars(n_sessions: int = 3, n_bars: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        prices = np.exp(np.cumsum(rng.normal(0, 0.003, n_bars))) * 1.0
        for ix, c in enumerate(prices):
            rows.append({
                "session": s, "bar_ix": ix,
                "open": float(c), "high": float(c * 1.001),
                "low": float(c * 0.999), "close": float(c),
            })
    return pd.DataFrame(rows)


def _heads() -> pd.DataFrame:
    return pd.DataFrame([
        {"session": 0, "bar_ix": 48, "headline": "Company raises outlook"},
        {"session": 1, "bar_ix": 49, "headline": "Company misses quarterly revenue estimates"},
    ])


def test_build_features_shape():
    bars = _make_bars(n_sessions=3)
    X = build_features(bars, _heads())
    assert list(X.columns) == list(FEATURE_NAMES)
    assert len(X) == 3
    assert X.index.name == "session"
    assert X.isna().sum().sum() == 0


def test_build_features_11_columns():
    X = build_features(_make_bars(2), None)
    assert X.shape[1] == 11


def test_realized_vol_nonneg():
    bars = _make_bars(5)
    close = bars.pivot(index="session", columns="bar_ix", values="close").to_numpy()
    rv = realized_vol(close)
    assert (rv >= 0).all()


def test_parkinson_vs_constant():
    bars = _make_bars(1)
    close = bars.pivot(index="session", columns="bar_ix", values="close").to_numpy()
    high = bars.pivot(index="session", columns="bar_ix", values="high").to_numpy()
    low = bars.pivot(index="session", columns="bar_ix", values="low").to_numpy()
    pk = parkinson_vol(high, low)
    assert pk.shape == (1,)
    assert pk[0] > 0


def test_max_drawdown_negative_or_zero():
    bars = _make_bars(3)
    close = bars.pivot(index="session", columns="bar_ix", values="close").to_numpy()
    dd = max_drawdown(close)
    assert (dd <= 0).all()
