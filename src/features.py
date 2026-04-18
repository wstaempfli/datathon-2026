"""
Feature engineering pipeline.

Builds a session-level feature matrix from raw OHLC bars and headlines.
Every function must work identically on train and test data (no target leakage).

Usage (from run_pipeline.py or notebooks):
    from src.features import build_features, get_target, get_halfway_close

    features = build_features(bars_seen, headlines_seen, include_tier2=True)
    target   = get_target(bars_seen, bars_unseen)          # train only
    halfway  = get_halfway_close(bars_seen)                # for Sharpe calc
"""

import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Tier 1 — OHLC features (verified signal, build in Phase 1)
# ---------------------------------------------------------------------------

def _ohlc_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute per-session OHLC features from seen bars (bar_ix 0-49).

    Returns DataFrame indexed by ``session`` with columns:
      - first_half_return  (r=-0.069, p=0.028)  close[49]/close[0] - 1
      - ret_from_30        (r=-0.081, p=0.010)  close[49]/close[30] - 1
      - volatility         (r=+0.072, p=0.023)  std of log(close/open)
      - avg_hl_range       (r=+0.078, p=0.013)  mean of (high-low)/open
      - up_ratio           (r=-0.059, p=0.061)  fraction of bars where close > open
      - close_rel_range    (r=-0.055, p=0.083)  (close[49] - session_low) / (session_high - session_low)
    """
    # TODO: Compute first_half_return = close at bar 49 / close at bar 0 - 1
    # TODO: Compute ret_from_30 = close at bar 49 / close at bar 30 - 1
    # TODO: Compute volatility = std of log(close/open) per session
    # TODO: Compute avg_hl_range = mean of (high - low) / open per session
    # TODO: Compute up_ratio = fraction of bars where close > open per session
    # TODO: Compute close_rel_range = (close[49] - session_low) / (session_high - session_low)
    # TODO: Return DataFrame indexed by session
    bars = bars.copy()
    g = bars.groupby("session")
    first_close = g["close"].first()
    last_close = g["close"].last()

    first_half_return = last_close / first_close - 1

    close_30 = bars[bars["bar_ix"] == 30].set_index("session")["close"]
    ret_from_30 = last_close / close_30 - 1


    bars["log_ret"] = np.log(bars["close"] / bars["open"])
    volatility = g["log_ret"].std()

    bars["hl_range"] = (bars["high"] - bars["low"]) / bars["open"]
    avg_hl_range = g["hl_range"].mean()

    bars["up"] = (bars["close"] > bars["open"]).astype(int)
    up_ratio = g["up"].mean()

    session_low = g["low"].min()
    session_high = g["high"].max()
    close_rel_range = (last_close - session_low) / (session_high - session_low)

    features = pd.DataFrame({
          "first_half_return": first_half_return,
          "ret_from_30":       ret_from_30,
          "volatility":        volatility,
          "avg_hl_range":      avg_hl_range,
          "up_ratio":          up_ratio,
          "close_rel_range":   close_rel_range,
      })
    return features

# ---------------------------------------------------------------------------
# Tier 2 — additional OHLC features (hypothesized, test in Phase 2)
# ---------------------------------------------------------------------------

def _ohlc_features_tier2(bars: pd.DataFrame) -> pd.DataFrame:
    """Tier-2 OHLC features — add incrementally, keep only if they improve CV Sharpe.

    Returns DataFrame indexed by ``session`` with columns:
      - max_drawdown       max peak-to-trough decline in close prices
      - vol_trend          std(log_ret bars 25-49) / std(log_ret bars 0-24)
      - price_slope        linear regression slope of close prices vs bar_ix
      - range_contraction  mean(hl_range last 10 bars) / mean(hl_range first 10 bars)
      - gap_mean           mean of (open[i] - close[i-1]) gaps between consecutive bars
      - ret_x_vol          interaction: first_half_return * volatility
    """
    # TODO: max_drawdown — for each session, compute running max of close,
    #       then drawdown = close / running_max - 1, feature = min(drawdown)
    # TODO: vol_trend — split bars at midpoint, ratio of late vol to early vol
    # TODO: price_slope — scipy.stats.linregress(bar_ix, close).slope
    # TODO: range_contraction — mean hl_range of last 10 bars / first 10 bars
    # TODO: gap_mean — mean(open[i+1] - close[i]) for consecutive bars
    # TODO: ret_x_vol — multiply first_half_return by volatility
    # TODO: Return DataFrame indexed by session
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Headline features
# ---------------------------------------------------------------------------

# Keyword lists for event-type classification
REVENUE_KEYWORDS = ["revenue", "quarterly", "earnings", "estimates"]
POSITIVE_KEYWORDS = ["breakthrough", "surge", "record", "strong demand", "raises outlook"]
NEGATIVE_KEYWORDS = ["misses", "decline", "withdraws", "warns", "disruptions",
                      "downgrades", "recall", "loses"]
CONTRACT_KEYWORDS = ["secures", "contract"]
EXPANSION_KEYWORDS = ["expands operations", "opens new", "new office"]

# Regex patterns for template normalization and slot extraction.
# Order matters in _extract_template: $ and % must run before generic number match.
_DOLLAR_PATTERN = re.compile(r'\$(\d+(?:[.,]\d+)?)\s*([MB]?)\b')
_PERCENT_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*%')
_CO_PATTERN = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
_NUMBER_PATTERN = re.compile(r'\b\d+(?:\.\d+)?\b')
_WS_PATTERN = re.compile(r'\s+')


def _extract_template(text: str) -> str:
    """Normalize a headline to its template form by replacing entities/values."""
    t = _DOLLAR_PATTERN.sub('$#', text)
    t = _PERCENT_PATTERN.sub('#%', t)
    t = _CO_PATTERN.sub('[CO]', t)
    t = _NUMBER_PATTERN.sub('#', t)
    return _WS_PATTERN.sub(' ', t).strip()


def _extract_companies(text: str) -> list[str]:
    """Extract all capitalized entity names. First one is typically the speaker."""
    return _CO_PATTERN.findall(text)


def _extract_dollar_amounts(text: str) -> list[float]:
    """Extract dollar amounts from a headline, normalized to millions.

    Examples: "$500M" -> 500.0, "$1.5B" -> 1500.0, "$250" -> 0.00025
    """
    amounts = []
    for match in _DOLLAR_PATTERN.finditer(text):
        raw_val, unit = match.groups()
        val = float(raw_val.replace(',', ''))
        if unit == 'B':
            val *= 1000
        elif unit != 'M':
            val /= 1_000_000
        amounts.append(val)
    return amounts


def _extract_percents(text: str) -> list[float]:
    """Extract all percent values from a headline as numeric floats."""
    return [float(m.group(1)) for m in _PERCENT_PATTERN.finditer(text)]


def _headline_features(headlines: pd.DataFrame) -> pd.DataFrame:
    """Compute per-session headline features grouped in 5 families.

    Returns DataFrame indexed by ``session`` with ~23 columns:
      Structural:   headline_count, unique_template_count, template_diversity,
                    has_any_headline
      Timing:       first_headline_bar, last_headline_bar, median_headline_bar,
                    headlines_first10, headlines_last10
      Identity:     main_co_mention_fraction, unique_co_count
      Events:       main_revenue_count, peer_revenue_count, main_positive_count,
                    peer_positive_count, main_negative_count, peer_negative_count,
                    contract_event_count, expansion_event_count
      Magnitude:    max_dollar_amount, total_dollar_amount, max_percent_value,
                    has_material_event

    Sessions not present in ``headlines`` are simply absent from the output —
    ``build_features`` fills those with zeros via ``.join().fillna(0.0)``.
    """
    hl = headlines.copy()
    hl["headline_l"] = hl["headline"].str.lower()

    # Slot/entity extraction per headline
    hl["template"] = hl["headline"].apply(_extract_template)
    hl["companies"] = hl["headline"].apply(_extract_companies)
    hl["dollars"] = hl["headline"].apply(_extract_dollar_amounts)
    hl["percents"] = hl["headline"].apply(_extract_percents)

    # Infer each session's main company = most-frequent capitalized name
    exploded = hl[["session", "companies"]].explode("companies").dropna(subset=["companies"])
    if len(exploded) > 0:
        main_co_per_session = exploded.groupby("session")["companies"].agg(
            lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else None
        )
    else:
        main_co_per_session = pd.Series(dtype=object)

    hl["main_co"] = hl["session"].map(main_co_per_session)
    hl["mentions_main"] = [
        (mc is not None) and (mc in cs)
        for mc, cs in zip(hl["main_co"], hl["companies"])
    ]

    # Keyword flags, one boolean column per family
    keyword_families = {
        "revenue": REVENUE_KEYWORDS,
        "positive": POSITIVE_KEYWORDS,
        "negative": NEGATIVE_KEYWORDS,
        "contract": CONTRACT_KEYWORDS,
        "expansion": EXPANSION_KEYWORDS,
    }
    for name, kws in keyword_families.items():
        pattern = "|".join(re.escape(k) for k in kws)
        hl[f"kw_{name}"] = hl["headline_l"].str.contains(pattern, regex=True)

    # Per-headline magnitude scalars
    hl["max_dollar"] = hl["dollars"].apply(lambda xs: max(xs) if xs else 0.0)
    hl["sum_dollar"] = hl["dollars"].apply(lambda xs: sum(xs) if xs else 0.0)
    hl["max_percent"] = hl["percents"].apply(lambda xs: max(xs) if xs else 0.0)
    hl["n_companies"] = hl["companies"].apply(len)

    g = hl.groupby("session")
    session_ix = g.size().index  # canonical index for alignment

    def _align(s: pd.Series, fill: float = 0.0) -> pd.Series:
        return s.reindex(session_ix, fill_value=fill)

    # Structural
    headline_count = g.size()
    unique_template_count = g["template"].nunique()
    template_diversity = (unique_template_count / headline_count).fillna(0.0)
    has_any_headline = (headline_count > 0).astype(int)

    # Timing
    first_headline_bar = g["bar_ix"].min()
    last_headline_bar = g["bar_ix"].max()
    median_headline_bar = g["bar_ix"].median()
    headlines_first10 = _align(hl.loc[hl["bar_ix"] < 10].groupby("session").size())
    headlines_last10 = _align(hl.loc[hl["bar_ix"] >= 40].groupby("session").size())

    # Identity
    total_co_mentions = g["n_companies"].sum()
    main_co_mentions = g["mentions_main"].sum()
    main_co_mention_fraction = (main_co_mentions / total_co_mentions.replace(0, np.nan)).fillna(0.0)
    unique_co_count = _align(
        hl[["session", "companies"]].explode("companies").dropna(subset=["companies"])
          .groupby("session")["companies"].nunique()
    )

    # Keyword events — main/peer split for revenue, positive, negative
    def _split_count(kw_col: str, main: bool) -> pd.Series:
        mask = hl[kw_col] & (hl["mentions_main"] == main)
        return _align(hl.loc[mask].groupby("session").size())

    features = pd.DataFrame({
        "headline_count":          headline_count,
        "unique_template_count":   unique_template_count,
        "template_diversity":      template_diversity,
        "has_any_headline":        has_any_headline,

        "first_headline_bar":      first_headline_bar,
        "last_headline_bar":       last_headline_bar,
        "median_headline_bar":     median_headline_bar,
        "headlines_first10":       headlines_first10,
        "headlines_last10":        headlines_last10,

        "main_co_mention_fraction": main_co_mention_fraction,
        "unique_co_count":          unique_co_count,

        "main_revenue_count":   _split_count("kw_revenue",  True),
        "peer_revenue_count":   _split_count("kw_revenue",  False),
        "main_positive_count":  _split_count("kw_positive", True),
        "peer_positive_count":  _split_count("kw_positive", False),
        "main_negative_count":  _split_count("kw_negative", True),
        "peer_negative_count":  _split_count("kw_negative", False),
        "contract_event_count":  g["kw_contract"].sum().astype(int),
        "expansion_event_count": g["kw_expansion"].sum().astype(int),

        "max_dollar_amount":     g["max_dollar"].max(),
        "total_dollar_amount":   g["sum_dollar"].sum(),
        "max_percent_value":     g["max_percent"].max(),
    })
    features["has_material_event"] = (
        (features["max_dollar_amount"] > 100) | (features["max_percent_value"] > 10)
    ).astype(int)

    return features


# ---------------------------------------------------------------------------
# Public API — called by run_pipeline.py, evaluate.py, submit.py
# ---------------------------------------------------------------------------

def build_features(
    bars: pd.DataFrame,
    headlines: pd.DataFrame | None = None,
    *,
    include_tier2: bool = False,
) -> pd.DataFrame:
    """Build session-level feature matrix from raw data.

    Parameters
    ----------
    bars : DataFrame
        OHLC bars with columns: session, bar_ix, open, high, low, close.
        May contain only the seen half (bar_ix 0-49).
    headlines : DataFrame or None
        Headlines with columns: session, bar_ix, headline.
        Pass None to skip headline features.
    include_tier2 : bool
        If True, include Tier-2 OHLC features (experimental, Phase 2).

    Returns
    -------
    DataFrame indexed by session, one row per session, columns = feature names.

    Called by
    --------
    - scripts/run_pipeline.py  (train + test)
    - src/evaluate.py          (inside CV loop, on train folds)
    - src/submit.py            (on test data)
    - notebooks/01_eda.ipynb   (for correlation analysis)
    """
    # TODO: Call _ohlc_features(bars)
    # TODO: If include_tier2, call _ohlc_features_tier2(bars) and join
    # TODO: If headlines provided, call _headline_features(headlines) and join
    # TODO: fillna(0.0), sort by session index
    # TODO: Return combined DataFrame
    features = _ohlc_features(bars)

    if include_tier2:
        features = features.join(_ohlc_features_tier2(bars))

    if headlines is not None:
        features = features.join(_headline_features(headlines))

    features = features.fillna(0.0).sort_index()
    return features


def get_target(bars_seen: pd.DataFrame, bars_unseen: pd.DataFrame) -> pd.Series:
    """Compute target variable: second-half return per session.

    target = close[bar 99] / close[bar 49] - 1

    Only usable on training data (unseen bars required).
    Training distribution: mean=+0.35%, std=2.0%, 57% positive.

    Called by
    --------
    - scripts/run_pipeline.py  (to get training labels)
    - src/evaluate.py          (inside CV to split X/y)
    """
    # TODO: halfway_close = bars_seen grouped by session, last close (bar 49)
    # TODO: end_close = bars_unseen grouped by session, last close (bar 99)
    # TODO: return end_close / halfway_close - 1
    halfway_close = get_halfway_close(bars_seen)
    end_close = get_halfway_close(bars_unseen)
    return end_close / halfway_close - 1


def get_halfway_close(bars_seen: pd.DataFrame) -> pd.Series:
    """Return the close price at bar 49 for each session.

    Needed by evaluate.compute_sharpe() to convert positions to PnL.

    Called by
    --------
    - src/evaluate.py  (Sharpe calculation)
    - src/submit.py    (sanity checks)
    """
    idx = bars_seen.groupby("session")["bar_ix"].idxmax()
    return bars_seen.loc[idx].set_index("session")["close"]
