"""Score all unique headlines with FinBERT. Idempotent — reuses features/sentiment_cache.parquet."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import all_unique_headlines, load_headlines
from src.sentiment import update_cache, build_sentiment_features, load_cache

FEATURES_DIR = ROOT / "features"


def main():
    FEATURES_DIR.mkdir(exist_ok=True)
    cache_path = FEATURES_DIR / "sentiment_cache.parquet"
    texts = all_unique_headlines()
    print(f"{len(texts)} unique headlines")
    cache = update_cache(cache_path, texts)
    # Emit per-bar viz file using train headlines (so it overlays on train session candles).
    h_train_seen = load_headlines("train", seen=True)
    _, per_bar = build_sentiment_features(
        h_train_seen, cache, bar_out=FEATURES_DIR / "sentiment_bar.parquet"
    )
    print("wrote", FEATURES_DIR / "sentiment_bar.parquet", len(per_bar), "rows")


if __name__ == "__main__":
    main()
