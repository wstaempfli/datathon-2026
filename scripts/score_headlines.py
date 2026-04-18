"""Score all unique headlines with FinBERT. Idempotent — reuses features/sentiment_cache.parquet."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import all_unique_headlines
from src.sentiment import update_cache

FEATURES_DIR = ROOT / "features"


def main():
    FEATURES_DIR.mkdir(exist_ok=True)
    cache_path = FEATURES_DIR / "sentiment_cache.parquet"
    texts = all_unique_headlines()
    print(f"{len(texts)} unique headlines")
    cache = update_cache(cache_path, texts)
    print(f"cache rows: {len(cache)}  path: {cache_path}")


if __name__ == "__main__":
    main()
