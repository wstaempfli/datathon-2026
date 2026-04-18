"""FinBERT sentiment scoring with on-disk caching.

`transformers` + `torch` are imported lazily inside `score_headlines` so that the
rest of the pipeline (price features, data loading) still works even if they
aren't installed yet. A fallback path returns zero-valued sentiment in that case.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_FINBERT_WARNED = False


def _device(preference: str | None):
    """Resolve torch device with mps > cuda > cpu fallback."""
    import torch  # local import — only needed when scoring

    if preference:
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _zero_scores(texts: list[str]) -> pd.DataFrame:
    """Return a zero-filled sentiment DataFrame. Used as graceful fallback."""
    uniq = list(dict.fromkeys(texts))
    return pd.DataFrame(
        {
            "headline": uniq,
            "pos": 0.0,
            "neg": 0.0,
            "neu": 0.0,
            "sent_score": 0.0,
        }
    )


def score_headlines(
    texts: list[str],
    batch_size: int = 32,
    model_name: str = "ProsusAI/finbert",
    device: str | None = None,
) -> pd.DataFrame:
    """Score unique headlines with FinBERT.

    Returns DataFrame[headline, pos, neg, neu, sent_score] where
    sent_score = pos - neg. Deduplicates input before calling the model.
    If transformers/torch are unavailable, returns zeros and warns once.
    """
    global _FINBERT_WARNED

    texts = [str(t) for t in texts if t is not None and str(t).strip()]
    if not texts:
        return pd.DataFrame(columns=["headline", "pos", "neg", "neu", "sent_score"])

    # Dedup while preserving order.
    uniq = list(dict.fromkeys(texts))

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:  # pragma: no cover - optional dep
        if not _FINBERT_WARNED:
            warnings.warn(
                f"transformers/torch unavailable ({e!r}); returning zero sentiment scores."
            )
            _FINBERT_WARNED = True
        return _zero_scores(uniq)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    dev = _device(device)
    model.to(dev)

    # FinBERT (ProsusAI) is typically {0: positive, 1: negative, 2: neutral}, but we
    # resolve the mapping explicitly from config rather than assuming order.
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    label_to_idx = {v: k for k, v in id2label.items()}
    # guard against unexpected label set
    for needed in ("positive", "negative", "neutral"):
        if needed not in label_to_idx:
            raise RuntimeError(
                f"FinBERT config missing expected label '{needed}'; got {id2label}"
            )
    pos_i = label_to_idx["positive"]
    neg_i = label_to_idx["negative"]
    neu_i = label_to_idx["neutral"]

    rows = []
    with torch.inference_mode():
        for start in range(0, len(uniq), batch_size):
            batch = uniq[start : start + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=96,
                return_tensors="pt",
            ).to(dev)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            for text, p in zip(batch, probs):
                pos = float(p[pos_i])
                neg = float(p[neg_i])
                neu = float(p[neu_i])
                rows.append(
                    {
                        "headline": text,
                        "pos": pos,
                        "neg": neg,
                        "neu": neu,
                        "sent_score": pos - neg,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_CACHE_COLS = ["headline", "pos", "neg", "neu", "sent_score"]


def load_cache(path: Path) -> pd.DataFrame:
    """Returns cached sentiment scores or an empty frame with the right schema."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=_CACHE_COLS)
    return pd.read_parquet(path)


def update_cache(path: Path, texts: list[str], batch_size: int = 32) -> pd.DataFrame:
    """Score only texts missing from cache, append, persist, return the full cache."""
    path = Path(path)
    cache = load_cache(path)
    have = set(cache["headline"].astype(str).tolist()) if len(cache) else set()
    missing = [t for t in dict.fromkeys(str(x) for x in texts) if t and t not in have]
    if missing:
        scored = score_headlines(missing, batch_size=batch_size)
        if len(scored):
            cache = pd.concat([cache, scored], ignore_index=True)
            cache = cache.drop_duplicates(subset=["headline"], keep="last").reset_index(drop=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            cache.to_parquet(path)
    return cache


# ---------------------------------------------------------------------------
# Session + bar aggregation
# ---------------------------------------------------------------------------

def _empty_per_session(sessions: pd.Index) -> pd.DataFrame:
    cols = [
        "sent_mean",
        "sent_max",
        "sent_min",
        "sent_last",
        "sent_count",
        "sent_abs_mean",
        "pos_share",
        "neg_share",
        "sent_sum",
    ]
    return pd.DataFrame(0.0, index=sessions, columns=cols)


def build_sentiment_features(
    headlines: pd.DataFrame,
    cache: pd.DataFrame,
    bar_out: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate cached sentiment into per-session and per-bar frames.

    per_session columns: sent_mean, sent_max, sent_min, sent_last, sent_count,
                        sent_abs_mean, pos_share, neg_share, sent_sum.
    per_bar columns: session, bar_ix, sent_pos, sent_neg, sent_score
                    (mean-aggregated when multiple headlines share a bar).
    Sessions with no headlines get zero rows in per_session.
    """
    if headlines is None or len(headlines) == 0:
        per_session = pd.DataFrame(
            columns=[
                "sent_mean",
                "sent_max",
                "sent_min",
                "sent_last",
                "sent_count",
                "sent_abs_mean",
                "pos_share",
                "neg_share",
                "sent_sum",
            ]
        )
        per_session.index.name = "session"
        per_bar = pd.DataFrame(
            columns=["session", "bar_ix", "sent_pos", "sent_neg", "sent_score"]
        )
        if bar_out is not None:
            Path(bar_out).parent.mkdir(parents=True, exist_ok=True)
            per_bar.to_parquet(bar_out)
        return per_session, per_bar

    h = headlines.copy()
    h["headline"] = h["headline"].astype(str)
    if len(cache):
        lookup = cache.drop_duplicates(subset=["headline"], keep="last").set_index("headline")
    else:
        lookup = pd.DataFrame(columns=["pos", "neg", "neu", "sent_score"])
    merged = h.merge(
        lookup[["pos", "neg", "sent_score"]],
        left_on="headline",
        right_index=True,
        how="left",
    )
    merged[["pos", "neg", "sent_score"]] = merged[["pos", "neg", "sent_score"]].fillna(0.0)

    # ----- per-bar aggregation -----
    per_bar = (
        merged.groupby(["session", "bar_ix"], as_index=False)
        .agg(sent_pos=("pos", "mean"), sent_neg=("neg", "mean"), sent_score=("sent_score", "mean"))
    )
    if bar_out is not None:
        Path(bar_out).parent.mkdir(parents=True, exist_ok=True)
        per_bar.to_parquet(bar_out)

    # ----- per-session aggregation -----
    merged_sorted = merged.sort_values(["session", "bar_ix"])
    grp = merged_sorted.groupby("session")

    per_session = pd.DataFrame(index=grp.size().index.rename("session"))
    per_session["sent_mean"] = grp["sent_score"].mean()
    per_session["sent_max"] = grp["sent_score"].max()
    per_session["sent_min"] = grp["sent_score"].min()
    per_session["sent_last"] = grp["sent_score"].last()
    per_session["sent_count"] = grp["sent_score"].size().astype(float)
    per_session["sent_abs_mean"] = grp["sent_score"].apply(lambda s: s.abs().mean())
    per_session["pos_share"] = grp["sent_score"].apply(lambda s: float((s > 0).mean()))
    per_session["neg_share"] = grp["sent_score"].apply(lambda s: float((s < 0).mean()))
    per_session["sent_sum"] = grp["sent_score"].sum()

    per_session = per_session.fillna(0.0)
    return per_session, per_bar
