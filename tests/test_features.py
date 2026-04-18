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
    """Generate a tiny synthetic (bars, headlines, sentiment_cache)."""
    rng = np.random.default_rng(seed)

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
    bars = pd.DataFrame(rows)

    headlines = pd.DataFrame(
        [
            {"session": s, "bar_ix": 0, "headline": f"Synthetic headline {s}"}
            for s in range(n_sessions)
        ]
    )

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

    return bars, headlines, sentiment_cache


# --------------------------------------------------------------------------
# make_features
# --------------------------------------------------------------------------


def test_make_features_returns_correct_types() -> None:
    bars, headlines, sent = _synth()
    X, feature_names = make_features(bars, headlines, sentiment_cache=sent)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(feature_names, list)
    assert all(isinstance(n, str) for n in feature_names)
    assert X.index.name == "session"


def test_no_nan_or_inf() -> None:
    bars, headlines, sent = _synth()
    X, _ = make_features(bars, headlines, sentiment_cache=sent)

    assert X.isna().sum().sum() == 0
    numeric = X.select_dtypes(include=[np.number]).to_numpy()
    assert np.isfinite(numeric).all()


def test_feature_names_match_columns() -> None:
    bars, headlines, sent = _synth()
    X, feature_names = make_features(bars, headlines, sentiment_cache=sent)

    assert list(X.columns) == feature_names


def test_validate_no_leakage_passes() -> None:
    bars, headlines, sent = _synth()
    X, _ = make_features(bars, headlines, sentiment_cache=sent)

    validate_no_leakage(X)


def test_validate_no_leakage_rejects_leaky_column() -> None:
    bars, headlines, sent = _synth()
    X, _ = make_features(bars, headlines, sentiment_cache=sent)

    leaky = X.copy()
    leaky["close_bar_55"] = 0.0
    with pytest.raises(ValueError):
        validate_no_leakage(leaky)

    # NaN rejection path.
    nan_X = X.copy()
    nan_X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_no_leakage(nan_X)


def test_fh_return_present_and_named() -> None:
    bars, headlines, sent = _synth()
    _, feature_names = make_features(bars, headlines, sentiment_cache=sent)
    assert "fh_return" in feature_names


def test_fh_return_domain_sane() -> None:
    bars, headlines, sent = _synth()
    X, _ = make_features(bars, headlines, sentiment_cache=sent)
    assert (X["fh_return"].abs() < 0.5).all()


def test_yz_vol_present_and_named() -> None:
    bars, headlines, sent = _synth()
    _, feature_names = make_features(bars, headlines, sentiment_cache=sent)
    assert "yz_vol" in feature_names


def test_yz_vol_nonneg() -> None:
    bars, headlines, sent = _synth()
    X, _ = make_features(bars, headlines, sentiment_cache=sent)
    assert (X["yz_vol"] >= 0).all()


def test_bmb_counts_match_patterns() -> None:
    bars, _headlines_unused, sent = _synth(n_sessions=3)

    custom_headlines = pd.DataFrame(
        [
            # Session 0: balanced (1 bull + 1 bear + 1 neutral) -> bmb = 0.
            {"session": 0, "bar_ix": 0, "headline": "Acme raises outlook for next quarter"},
            {"session": 0, "bar_ix": 1, "headline": "Acme faces regulatory review"},
            {"session": 0, "bar_ix": 2, "headline": "Generic commentary about markets"},
            # Session 1: two bulls + zero bears -> bmb = 2.
            {"session": 1, "bar_ix": 0, "headline": "BetaCo wins industry award"},
            {"session": 1, "bar_ix": 3, "headline": "BetaCo launches next-generation platform"},
            # Session 2: zero bull + zero bear -> bmb = 0.
            {"session": 2, "bar_ix": 5, "headline": "Routine filing submitted"},
        ]
    )

    X, _ = make_features(bars, custom_headlines, sentiment_cache=sent)
    assert "bmb" in X.columns
    assert X.loc[0, "bmb"] == 0.0
    assert X.loc[1, "bmb"] == 2.0
    assert X.loc[2, "bmb"] == 0.0
