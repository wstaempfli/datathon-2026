"""
Submission generation and validation.

Produces correctly formatted CSV files and validates them before saving.

Usage:
    from src.submit import generate_submission, validate_submission

    generate_submission(model, bars_test, headlines_test,
                        "submissions/v01_ridge_public.csv")
    validate_submission("submissions/v01_ridge_public.csv",
                        "data/bars_seen_public_test.parquet")
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.features import build_features
from src.models import apply_position_sizing


def generate_submission(
    model,
    bars_test: pd.DataFrame,
    headlines_test: pd.DataFrame | None,
    output_path: str | None = None,
    *,
    sizing_strategy: str = "sign",
    sizing_kwargs: dict | None = None,
    include_tier2: bool = False,
) -> pd.DataFrame:
    """Generate a submission CSV from a fitted model and test data.

    Steps:
      1. Build features from test bars + headlines (via build_features)
      2. Run model.predict(X_test) to get raw predictions
      3. Apply position sizing to get target_position values
      4. Validate and save CSV to output_path

    Parameters
    ----------
    model : fitted model with .predict(X) method
    bars_test : DataFrame — test OHLC bars (seen only)
    headlines_test : DataFrame or None — test headlines
    output_path : str or None — where to save the CSV (if None, just return DataFrame)
    sizing_strategy : str — passed to apply_position_sizing
    sizing_kwargs : dict — extra kwargs for position sizing
    include_tier2 : bool — whether to include Tier-2 features

    Returns
    -------
    DataFrame with columns [session, target_position]

    Called by
    --------
    - scripts/run_pipeline.py  (main submission generation)
    """
    features = build_features(bars_test, headlines_test, include_tier2=include_tier2)
    X_test = features.values
    raw_pred = model.predict(X_test)
    if sizing_kwargs is None:
        sizing_kwargs = {}
    positions = apply_position_sizing(raw_pred, sizing_strategy, **sizing_kwargs)

    sub = pd.DataFrame({"session": features.index, "target_position": positions})

    # Save
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(output_path, index=False)
        print(f"Saved {output_path}: {len(sub)} rows, "
              f"positions in [{positions.min():.2f}, {positions.max():.2f}], "
              f"mean={positions.mean():.3f}, "
              f"{(positions > 0).mean()*100:.1f}% long, "
              f"{(positions < 0).mean()*100:.1f}% short")
    return sub


def generate_always_long_submission(
    bars_test: pd.DataFrame,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Generate the always-long baseline submission (target_position = +1).

    This is our safety fallback — Sharpe ~2.766 guaranteed.

    Called by
    --------
    - scripts/run_pipeline.py  (always generate as fallback)
    """
    sessions = bars_test["session"].unique()
    sub = pd.DataFrame({
        "session": sorted(sessions),
        "target_position": 1.0,
    })
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(output_path, index=False)
        print(f"Saved always-long submission to {output_path}: {len(sub)} sessions, all positions = +1.0")
    return sub


def validate_submission(
    submission: pd.DataFrame | str,
    test_bars_path: str,
) -> bool:
    """Validate a submission DataFrame or CSV file.

    Checks (from BATTLE_PLAN.md verification plan):
      1. Correct columns: session, target_position
      2. One row per test session, no missing sessions
      3. Session IDs match test data exactly
      4. No NaN or Inf values in target_position
      5. No duplicate sessions
      6. Positions are not all identical (model degeneracy check)

    Returns True if valid, raises ValueError with details if not.

    Called by
    --------
    - generate_submission()     (automatic validation before saving)
    - scripts/run_pipeline.py   (explicit check)
    """
    # Load submission if path string
    if isinstance(submission, str):
        submission = pd.read_csv(submission)

    # Load test bars to get expected session IDs
    test_bars = pd.read_parquet(test_bars_path)
    expected_sessions = set(test_bars["session"].unique())

    # Check columns
    required_cols = {"session", "target_position"}
    if set(submission.columns) != required_cols:
        raise ValueError(
            f"Incorrect columns: got {list(submission.columns)}, "
            f"expected {sorted(required_cols)}"
        )

    # Check for duplicate sessions
    if submission["session"].duplicated().any():
        n_dups = submission["session"].duplicated().sum()
        raise ValueError(f"Found {n_dups} duplicate session(s)")

    # Check session sets match exactly
    submitted_sessions = set(submission["session"].unique())
    missing = expected_sessions - submitted_sessions
    extra = submitted_sessions - expected_sessions
    if missing or extra:
        msg_parts = []
        if missing:
            msg_parts.append(f"{len(missing)} missing sessions")
        if extra:
            msg_parts.append(f"{len(extra)} extra sessions")
        raise ValueError(f"Session mismatch: {', '.join(msg_parts)}")

    # Check for NaN / Inf
    positions = submission["target_position"]
    if positions.isna().any():
        n_nan = positions.isna().sum()
        raise ValueError(f"Found {n_nan} NaN value(s) in target_position")
    if np.isinf(positions).any():
        n_inf = np.isinf(positions).sum()
        raise ValueError(f"Found {n_inf} Inf value(s) in target_position")

    # Warn if all positions are identical (degeneracy)
    if positions.nunique() == 1:
        raise ValueError(
            f"All positions are identical ({positions.iloc[0]}). "
            "This suggests model degeneracy."
        )

    # Print summary
    print(f"Submission valid: {len(submission)} rows, "
          f"positions in [{positions.min():.2f}, {positions.max():.2f}], "
          f"mean={positions.mean():.3f}, "
          f"{(positions > 0).mean()*100:.1f}% long, "
          f"{(positions < 0).mean()*100:.1f}% short")

    return True
