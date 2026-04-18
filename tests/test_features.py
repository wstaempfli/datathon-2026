"""Tests for src/features.py.

Uses a seeded inline synthetic fixture — no external fixture files. Keep these
tests cheap so they can run on every change to the feature pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import make_features, validate_no_leakage


def _synth(
    n_sessions: int = 3,
    n_bars: int = 50,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a tiny synthetic (bars_seen, headlines_seen, sentiment_cache).

    Bars start at 1.0 and walk via small normal returns — realistic enough to
    exercise log-returns/vol without hitting zeros or negatives. Headlines are
    sparse (one per session) and the sentiment cache covers them all.
    """
    rng = np.random.default_rng(seed)

    # ----- bars_seen -----------------------------------------------------
    rows = []
    for s in range(n_sessions):
        price = 1.0
        for b in range(n_bars):
            ret = rng.normal(0.0, 0.01)
            prev = price
            price = max(prev * (1.0 + ret), 1e-6)
            open_ = prev
            close = price
            high = max(open_, close) * (1.0 + abs(rng.normal(0.0, 0.002)))
            low = min(open_, close) * (1.0 - abs(rng.normal(0.0, 0.002)))
            rows.append(
                {
                    "session": s,
                    "bar_ix": b,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                }
            )
    bars_seen = pd.DataFrame(rows)

    # ----- headlines_seen ------------------------------------------------
    headlines = pd.DataFrame(
        [
            {"session": s, "bar_ix": 0, "headline": f"Synthetic headline {s}"}
            for s in range(n_sessions)
        ]
    )

    # ----- sentiment_cache ----------------------------------------------
    sent_rows = []
    for s in range(n_sessions):
        pos = float(rng.uniform(0.0, 1.0))
        neg = float(rng.uniform(0.0, 1.0 - pos))
        neu = max(0.0, 1.0 - pos - neg)
        sent_rows.append(
            {
                "headline": f"Synthetic headline {s}",
                "pos": pos,
                "neg": neg,
                "neu": neu,
                "sent_score": pos - neg,
            }
        )
    sentiment_cache = pd.DataFrame(sent_rows)

    return bars_seen, headlines, sentiment_cache


# --------------------------------------------------------------------------
# make_features
# --------------------------------------------------------------------------


def test_make_features_returns_correct_types() -> None:
    bars_seen, headlines_seen, sent = _synth()
    X, feature_names, fitted_stats = make_features(bars_seen, headlines_seen, sent)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(feature_names, list)
    assert all(isinstance(n, str) for n in feature_names)
    assert isinstance(fitted_stats, dict)
    assert X.index.name == "session"


def test_no_nan_or_inf() -> None:
    bars_seen, headlines_seen, sent = _synth()
    X, _, _ = make_features(bars_seen, headlines_seen, sent)

    assert X.isna().sum().sum() == 0
    numeric = X.select_dtypes(include=[np.number]).to_numpy()
    assert np.isfinite(numeric).all()


def test_feature_names_match_columns() -> None:
    bars_seen, headlines_seen, sent = _synth()
    X, feature_names, _ = make_features(bars_seen, headlines_seen, sent)

    assert list(X.columns) == feature_names


def test_validate_no_leakage_passes() -> None:
    bars_seen, headlines_seen, sent = _synth()
    X, _, _ = make_features(bars_seen, headlines_seen, sent)

    # Should not raise on a valid feature matrix.
    validate_no_leakage(X)


def test_validate_no_leakage_rejects_leaky_column() -> None:
    bars_seen, headlines_seen, sent = _synth()
    X, _, _ = make_features(bars_seen, headlines_seen, sent)

    leaky = X.copy()
    leaky["close_bar_55"] = 0.0
    with pytest.raises(ValueError):
        validate_no_leakage(leaky)

    # NaN rejection path.
    nan_X = X.copy()
    nan_X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_no_leakage(nan_X)


def test_training_stats_roundtrip() -> None:
    bars_seen, headlines_seen, sent = _synth()

    # Fit mode: stats come back from the function.
    _, _, fitted = make_features(bars_seen, headlines_seen, sent, training_stats=None)
    assert isinstance(fitted, dict)

    # Apply mode: contract is pass-through at the baseline.
    sentinel = {"_sentinel": 1.0}
    _, _, passed_through = make_features(
        bars_seen, headlines_seen, sent, training_stats=sentinel
    )
    assert passed_through == sentinel


def test_fh_return_present_and_named() -> None:
    bars_seen, headlines_seen, sent = _synth()
    _, feature_names, _ = make_features(bars_seen, headlines_seen, sent)
    assert "fh_return" in feature_names


def test_fh_return_domain_sane() -> None:
    bars_seen, headlines_seen, sent = _synth()
    X, _, _ = make_features(bars_seen, headlines_seen, sent)
    assert (X["fh_return"].abs() < 0.5).all()


def test_yz_vol_present_and_named() -> None:
    bars_seen, headlines_seen, sent = _synth()
    _, feature_names, _ = make_features(bars_seen, headlines_seen, sent)
    assert "yz_vol" in feature_names


def test_yz_vol_nonneg() -> None:
    bars_seen, headlines_seen, sent = _synth()
    X, _, _ = make_features(bars_seen, headlines_seen, sent)
    assert (X["yz_vol"] >= 0).all()
