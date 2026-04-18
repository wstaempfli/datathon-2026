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


def generate_submission(
    model,
    bars_test: pd.DataFrame,
    headlines_test: pd.DataFrame | None,
    output_path: str,
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
    output_path : str — where to save the CSV
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
    # TODO: Import build_features from src.features
    # TODO: features = build_features(bars_test, headlines_test, include_tier2=include_tier2)
    # TODO: X_test = features.values
    # TODO: raw_pred = model.predict(X_test)
    # TODO: positions = apply_position_sizing(raw_pred, sizing_strategy, **sizing_kwargs)
    # TODO: Build DataFrame with session and target_position columns
    # TODO: Call validate_submission() on the result
    # TODO: Save to output_path
    # TODO: Print confirmation with position stats (min, max, mean, %long, %short)
    raise NotImplementedError


def generate_always_long_submission(
    bars_test: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """Generate the always-long baseline submission (target_position = +1).

    This is our safety fallback — Sharpe ~2.766 guaranteed.

    Called by
    --------
    - scripts/run_pipeline.py  (always generate as fallback)
    """
    # TODO: Get unique session IDs from bars_test
    # TODO: Create DataFrame with session and target_position=1.0
    # TODO: Save to output_path
    raise NotImplementedError


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
    # TODO: Load submission if path string, otherwise use DataFrame
    # TODO: Load test bars to get expected session IDs
    # TODO: Check columns == ["session", "target_position"]
    # TODO: Check session sets match exactly
    # TODO: Check for NaN / Inf
    # TODO: Check for duplicates
    # TODO: Warn if all positions are identical (degeneracy)
    # TODO: Print summary: n_rows, position range, %long/%short
    raise NotImplementedError
