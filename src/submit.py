"""Submission CSV generation."""

import pandas as pd
from pathlib import Path

from src.features import build_features
from src.models import apply_position_sizing


def generate_submission(
    model,
    bars_test: pd.DataFrame,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Build features, predict, apply sizing, and optionally save submission CSV."""
    features = build_features(bars_test)
    raw_pred = model.predict(features.values)
    positions = apply_position_sizing(raw_pred)

    sub = pd.DataFrame({"session": features.index, "target_position": positions})

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(output_path, index=False)
    return sub
