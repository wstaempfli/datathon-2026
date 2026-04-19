"""Tests for V1b pipeline."""

import numpy as np
import pandas as pd

from src.pipeline import (
    _bmb_recent,
    _fh_return,
    predict,
    realized_vol,
    vol_scale,
)


def _make_bars(trajectories: dict[int, list[float]]) -> pd.DataFrame:
    rows = []
    for session, closes in trajectories.items():
        for bar_ix, c in enumerate(closes):
            rows.append({
                "session": session, "bar_ix": bar_ix,
                "open": c, "high": c, "low": c, "close": c,
            })
    return pd.DataFrame(rows)


def _trivial_bars() -> pd.DataFrame:
    return _make_bars({
        0: [1.0] * 49 + [1.10],
        1: [1.0] * 49 + [0.95],
        2: [1.0] * 50,
    })


def _trivial_heads() -> pd.DataFrame:
    return pd.DataFrame([
        {"session": 0, "bar_ix": 49, "headline": "Company raises outlook"},
        {"session": 1, "bar_ix": 49, "headline": "Company misses quarterly revenue estimates"},
        {"session": 2, "bar_ix": 49, "headline": "Company holds investor day"},
    ])


def test_fh_return_value():
    fh = _fh_return(_trivial_bars())
    assert abs(fh.loc[0] - 0.10) < 1e-9
    assert abs(fh.loc[1] - (-0.05)) < 1e-9
    assert abs(fh.loc[2] - 0.0) < 1e-9


def test_bmb_polarity():
    heads = pd.DataFrame([
        {"session": 0, "bar_ix": 49, "headline": "Company raises outlook"},
        {"session": 1, "bar_ix": 49, "headline": "Company misses quarterly revenue estimates"},
        {"session": 2, "bar_ix": 49, "headline": "Company raises outlook and misses quarterly revenue estimates"},
        {"session": 3, "bar_ix": 49, "headline": "Nothing happens here"},
    ])
    sessions = pd.Index([0, 1, 2, 3], name="session")
    bmb = _bmb_recent(heads, sessions, tau=40.0)
    assert abs(bmb.loc[0] - 1.0) < 1e-9
    assert abs(bmb.loc[1] - (-1.0)) < 1e-9
    assert abs(bmb.loc[2] - 1.0) < 1e-9  # bull takes precedence
    assert abs(bmb.loc[3] - 0.0) < 1e-9


def test_realized_vol_nonneg():
    rv = realized_vol(_trivial_bars())
    assert (rv >= 0).all()


def test_vol_scale_halving_doubles():
    sigma_ref = 0.003
    assert abs(vol_scale(np.array([0.003]), sigma_ref)[0] - 1.0) < 1e-9
    assert abs(vol_scale(np.array([0.0015]), sigma_ref)[0] - 2.0) < 1e-9


def test_vol_scale_clip_bounds():
    sigma_ref = 0.003
    assert vol_scale(np.array([1e-9]), sigma_ref)[0] <= 2.0 + 1e-9
    assert vol_scale(np.array([1.0]), sigma_ref)[0] >= 0.5 - 1e-9


def test_predict_shape_and_clip():
    out = predict(_trivial_bars(), _trivial_heads(), sigma_ref=0.003)
    assert isinstance(out, pd.Series)
    assert len(out) == 3
    assert out.index.name == "session"
    assert (out >= -2.0).all()
    assert (out <= 2.0).all()


def test_predict_preserves_drift_when_flat():
    # Flat prices + no matching headline => fh=0, bmb=0 => pos = 1.0 regardless of scaler.
    bars = _make_bars({0: [1.0] * 50})
    heads = pd.DataFrame([{"session": 0, "bar_ix": 49, "headline": "Nothing happens"}])
    out = predict(bars, heads, sigma_ref=0.003)
    assert abs(out.loc[0] - 1.0) < 1e-9
