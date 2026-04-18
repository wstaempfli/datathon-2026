"""End-to-end: build feature matrix from prior artifacts, CV-train LGBM, emit submissions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline()
