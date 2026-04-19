"""Tests for the minimal pipeline."""

import numpy as np
import pandas as pd

from src.pipeline import _bmb_recent, _fh_return, predict


def _make_bars(trajectories: dict[int, list[float]]) -> pd.DataFrame:
    rows = []
    for session, closes in trajectories.items():
        for bar_ix, c in enumerate(closes):
            rows.append({"session": session, "bar_ix": bar_ix, "close": c})
    return pd.DataFrame(rows)


def _trivial_bars() -> pd.DataFrame:
    # 3 sessions, close[0]=1.0, close[49]=1.10 / 0.95 / 1.00
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


def test_predict_shape():
    out = predict(_trivial_bars(), _trivial_heads())
    assert isinstance(out, pd.Series)
    assert len(out) == 3
    assert out.index.name == "session"


def test_predict_clip_range():
    out = predict(_trivial_bars(), _trivial_heads())
    assert (out >= -2.0).all()
    assert (out <= 2.0).all()


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
    # At bar_ix=49: decay = exp(0) = 1.0
    assert abs(bmb.loc[0] - 1.0) < 1e-9
    assert abs(bmb.loc[1] - (-1.0)) < 1e-9
    # bull-takes-precedence when both match
    assert abs(bmb.loc[2] - 1.0) < 1e-9
    assert abs(bmb.loc[3] - 0.0) < 1e-9
