"""Tests for position sizing utilities."""

import numpy as np

from src.positions import to_position, vol_scale


def test_vol_scale_halving_doubles():
    sigma_ref = 0.003
    s = vol_scale(np.array([0.003]), sigma_ref)
    s_half = vol_scale(np.array([0.0015]), sigma_ref)
    assert abs(s[0] - 1.0) < 1e-9
    assert abs(s_half[0] - 2.0) < 1e-9  # doubled, hits scaler_clip cap


def test_vol_scale_clip_upper():
    sigma_ref = 0.003
    s = vol_scale(np.array([1e-9]), sigma_ref)
    assert s[0] <= 2.0 + 1e-9


def test_vol_scale_clip_lower():
    sigma_ref = 0.001
    s = vol_scale(np.array([0.1]), sigma_ref)
    assert s[0] >= 0.5 - 1e-9


def test_to_position_clips():
    raw = np.array([10.0, -10.0, 0.0])
    pos = to_position(raw, 1.0, clip_lo=-2.0, clip_hi=2.0)
    assert pos[0] == 2.0
    assert pos[1] == -2.0
    assert pos[2] == 0.0


def test_to_position_intercept():
    raw = np.array([0.0, 1.0])
    pos = to_position(raw, 1.0, intercept=1.0)
    assert pos[0] == 1.0
    assert pos[1] == 2.0
