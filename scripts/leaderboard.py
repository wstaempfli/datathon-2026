#!/usr/bin/env python3
"""
Model leaderboard: track, compare, and display all model runs.

Stores results in a JSON file (submissions/leaderboard.json) so they
persist across runs.  Each entry records: model name, CV Sharpe (mean/std),
direction accuracy, feature set, sizing strategy, timestamp.

Usage:
    python scripts/leaderboard.py                    # display current leaderboard
    python scripts/leaderboard.py --add "ridge_v1"   # add latest run results

Programmatic usage (from notebooks or run_pipeline.py):
    from scripts.leaderboard import add_entry, display

    add_entry("ridge_v1", cv_results, feature_names, "sign")
    display()
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from datetime import datetime

LEADERBOARD_PATH = Path("submissions/leaderboard.json")


def load_leaderboard() -> list[dict]:
    """Load existing leaderboard from JSON file.

    Returns empty list if file doesn't exist.
    """
    # TODO: Check if LEADERBOARD_PATH exists
    # TODO: Load and return list of dicts, or empty list
    raise NotImplementedError


def save_leaderboard(entries: list[dict]) -> None:
    """Save leaderboard to JSON file."""
    # TODO: Write entries to LEADERBOARD_PATH with indent=2
    raise NotImplementedError


def add_entry(
    model_name: str,
    cv_results: dict,
    feature_names: list[str],
    sizing_strategy: str,
    notes: str = "",
) -> None:
    """Add a new entry to the leaderboard.

    Parameters
    ----------
    model_name : str — descriptive name (e.g., "ridge_alpha1_tier1")
    cv_results : dict — output of evaluate.cross_validate()
    feature_names : list — which features were used
    sizing_strategy : str — which position sizing was used
    notes : str — any additional notes

    Called by
    --------
    - scripts/run_pipeline.py  (after each model evaluation)
    - notebooks                (after experiments)
    """
    # TODO: Build entry dict with:
    #   model_name, sharpe_mean, sharpe_std, accuracy,
    #   n_features, feature_names, sizing_strategy,
    #   timestamp (ISO format), notes
    # TODO: Load existing leaderboard
    # TODO: Append new entry
    # TODO: Save
    # TODO: Print confirmation
    raise NotImplementedError


def display() -> None:
    """Print the leaderboard sorted by sharpe_mean descending.

    Output format:
    Rank | Model                | Sharpe (mean±std)  | Accuracy | Features | Sizing
    ---- | -------------------- | ------------------ | -------- | -------- | ------
    1    | ridge_alpha1_tier1   | 2.788 ± 0.043      | 57.3%    | 6        | sign
    2    | always_long          | 2.766 ± 0.000      | 57.0%    | 0        | fixed

    Called by
    --------
    - __main__ (when run as script)
    - scripts/run_pipeline.py  (end of pipeline)
    """
    # TODO: Load leaderboard
    # TODO: Sort by sharpe_mean descending
    # TODO: Print formatted table
    # TODO: Highlight if any model beats always-long (Sharpe > 2.766)
    raise NotImplementedError


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model leaderboard")
    parser.add_argument("--add", type=str, help="Add a named entry (requires running pipeline first)")
    args = parser.parse_args()

    if args.add:
        print(f"TODO: Integrate with run_pipeline to add '{args.add}' results")
    else:
        display()
