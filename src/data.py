import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_bars(split: str, seen: bool = True) -> pd.DataFrame:
    """split in {train, public_test, private_test}. seen=True loads bars 0-49."""
    suffix = "seen" if seen else "unseen"
    return pd.read_parquet(DATA_DIR / f"bars_{suffix}_{split}.parquet")


def load_headlines(split: str, seen: bool = True) -> pd.DataFrame:
    suffix = "seen" if seen else "unseen"
    return pd.read_parquet(DATA_DIR / f"headlines_{suffix}_{split}.parquet")


def compute_targets() -> pd.DataFrame:
    """Returns DataFrame with columns [session, target_return] for train."""
    seen = load_bars("train", seen=True)
    unseen = load_bars("train", seen=False)
    half = seen.sort_values(["session", "bar_ix"]).groupby("session").last()["close"]
    end = unseen.sort_values(["session", "bar_ix"]).groupby("session").last()["close"]
    r = (end / half - 1.0).rename("target_return").reset_index()
    return r


def all_unique_headlines() -> list[str]:
    """Deduplicated headlines across every available split (seen + unseen-train)."""
    frames = []
    for split in ("train", "public_test", "private_test"):
        frames.append(load_headlines(split, seen=True))
    frames.append(load_headlines("train", seen=False))
    all_h = (
        pd.concat(frames, ignore_index=True)["headline"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    return all_h
