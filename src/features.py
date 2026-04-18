"""Feature engineering pipeline (OHLC only)."""

import numpy as np
import pandas as pd


def _ohlc_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute 6 Tier-1 OHLC features per session."""
    first = bars.groupby("session").first()
    last = bars.groupby("session").last()
    feat = pd.DataFrame(index=first.index)
    feat["first_half_return"] = last["close"] / first["close"] - 1
    bar30 = bars[bars["bar_ix"] == 30].set_index("session")["close"]
    bar49 = bars[bars["bar_ix"] == 49].set_index("session")["close"]
    feat["ret_from_30"] = (bar49 / bar30 - 1).reindex(feat.index)
    log_ret = np.log(bars["close"] / bars["open"])
    feat["volatility"] = log_ret.groupby(bars["session"]).std()
    hl_range = (bars["high"] - bars["low"]) / bars["open"]
    feat["avg_hl_range"] = hl_range.groupby(bars["session"]).mean()
    is_up = (bars["close"] > bars["open"]).astype(int)
    feat["up_ratio"] = is_up.groupby(bars["session"]).mean()
    session_high = bars.groupby("session")["high"].max()
    session_low = bars.groupby("session")["low"].min()
    feat["close_rel_range"] = (bar49 - session_low) / (session_high - session_low)
    return feat


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Build session-level OHLC feature matrix."""
    feat = _ohlc_features(bars)
    feat = feat.fillna(0.0)
    feat = feat.sort_index()
    return feat


def get_target(bars_seen: pd.DataFrame, bars_unseen: pd.DataFrame) -> pd.Series:
    """Compute target: close[bar 99] / close[bar 49] - 1 per session."""
    halfway_close = bars_seen.groupby("session")["close"].last()
    end_close = bars_unseen.groupby("session")["close"].last()
    return (end_close / halfway_close - 1).rename("target_return")
