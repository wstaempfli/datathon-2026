"""FastAPI backend for the per-session visualization tool.

Serves OHLC bars + headlines per session from the parquet files in data/,
plus a scaffolded feature-overlay API reading CSVs from features/. Emits
SSE events when files in features/ change so the browser can live-reload.
"""

from __future__ import annotations

import asyncio
import json
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FEATURES_DIR = ROOT / "features"
MODELS_DIR = ROOT / "models"
STATIC_DIR = Path(__file__).resolve().parent / "static"

SPLITS = ("train", "public_test", "private_test")
SEEN_MAX_BAR = 49  # bar_ix 0..49 is always visible; 50..99 is "unseen"

DATA: dict[str, dict] = {}
FEATURES_CACHE: dict[str, pd.DataFrame] = {}
_TARGETS: dict[str, pd.DataFrame] = {}  # module-level cache: key "df" → targets DataFrame
_sse_clients: set[asyncio.Queue] = set()


def _load_split(split: str) -> dict:
    bars_seen = pd.read_parquet(DATA_DIR / f"bars_seen_{split}.parquet")
    bars_seen["seen"] = True
    frames = [bars_seen]
    unseen_path = DATA_DIR / f"bars_unseen_{split}.parquet"
    if unseen_path.exists():
        b_un = pd.read_parquet(unseen_path)
        b_un["seen"] = False
        frames.append(b_un)
    bars = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["session", "bar_ix"])
        .reset_index(drop=True)
    )

    h_frames = [pd.read_parquet(DATA_DIR / f"headlines_seen_{split}.parquet")]
    h_unseen_path = DATA_DIR / f"headlines_unseen_{split}.parquet"
    if h_unseen_path.exists():
        h_frames.append(pd.read_parquet(h_unseen_path))
    headlines = (
        pd.concat(h_frames, ignore_index=True)
        .sort_values(["session", "bar_ix"])
        .reset_index(drop=True)
    )

    sessions = sorted(int(s) for s in bars["session"].unique())

    return {
        "bars": bars.set_index("session", drop=False).sort_index(),
        "headlines": headlines.set_index("session", drop=False).sort_index(),
        "sessions": sessions,
    }


def _load_all() -> None:
    for split in SPLITS:
        DATA[split] = _load_split(split)


async def _watch_features() -> None:
    try:
        from watchfiles import awatch
    except ImportError:
        return
    try:
        async for _changes in awatch(str(FEATURES_DIR)):
            FEATURES_CACHE.clear()
            for q in list(_sse_clients):
                try:
                    q.put_nowait("reload")
                except asyncio.QueueFull:
                    pass
    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    FEATURES_DIR.mkdir(exist_ok=True)
    _load_all()
    watcher = asyncio.create_task(_watch_features())
    try:
        yield
    finally:
        watcher.cancel()


app = FastAPI(lifespan=lifespan, title="datathon-2026 viz")


def _clean_float(x):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


@app.get("/api/splits")
async def get_splits():
    return [
        {"name": s, "count": len(DATA[s]["sessions"]), "has_unseen": s == "train"}
        for s in SPLITS
    ]


@app.get("/api/sessions")
async def get_sessions(split: str):
    if split not in DATA:
        raise HTTPException(404, f"unknown split: {split}")
    return DATA[split]["sessions"]


@app.get("/api/session/{split}/{session_id}")
async def get_session(split: str, session_id: int):
    if split not in DATA:
        raise HTTPException(404, f"unknown split: {split}")

    bars_index = DATA[split]["bars"]
    if session_id not in bars_index.index:
        raise HTTPException(404, f"unknown session {session_id} in {split}")

    bars_df = bars_index.loc[[session_id]].sort_values("bar_ix")
    bars = [
        {
            "bar_ix": int(r.bar_ix),
            "open": _clean_float(r.open),
            "high": _clean_float(r.high),
            "low": _clean_float(r.low),
            "close": _clean_float(r.close),
            "seen": bool(r.seen),
        }
        for r in bars_df.itertuples(index=False)
    ]

    headlines: list[dict] = []
    h_index = DATA[split]["headlines"]
    if session_id in h_index.index:
        hdf = h_index.loc[[session_id]].sort_values("bar_ix")
        headlines = [
            {"bar_ix": int(r.bar_ix), "headline": str(r.headline)}
            for r in hdf.itertuples(index=False)
        ]

    cutoff_bar = SEEN_MAX_BAR if split == "train" else None
    target_return = None
    if split == "train":
        c49 = bars_df.loc[bars_df.bar_ix == 49, "close"]
        c99 = bars_df.loc[bars_df.bar_ix == 99, "close"]
        if len(c49) and len(c99):
            target_return = _clean_float(c99.iloc[0] / c49.iloc[0] - 1.0)

    return {
        "split": split,
        "session": session_id,
        "bars": bars,
        "headlines": headlines,
        "cutoff_bar": cutoff_bar,
        "target_return": target_return,
    }


def _load_feature_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported feature file type: {path.suffix}")


@app.get("/api/features")
async def list_features():
    out = []
    if not FEATURES_DIR.exists():
        return out
    for p in sorted(FEATURES_DIR.iterdir()):
        if p.name.startswith("."):
            continue
        if p.suffix not in (".csv", ".parquet"):
            continue
        try:
            df = FEATURES_CACHE.get(p.name)
            if df is None:
                df = _load_feature_file(p)
                FEATURES_CACHE[p.name] = df
            key_cols = {"session", "bar_ix"}
            numeric = df.select_dtypes(include="number").columns
            columns = [c for c in numeric if c not in key_cols]
            out.append(
                {
                    "name": p.name,
                    "columns": columns,
                    "per_bar": "bar_ix" in df.columns,
                    "rows": int(len(df)),
                }
            )
        except Exception as e:  # noqa: BLE001
            out.append({"name": p.name, "error": str(e)})
    return out


@app.get("/api/features/{name}/{session_id}")
async def get_feature(name: str, session_id: int):
    path = FEATURES_DIR / name
    if not path.exists() or path.suffix not in (".csv", ".parquet"):
        raise HTTPException(404, f"no feature file: {name}")
    df = FEATURES_CACHE.get(name)
    if df is None:
        df = _load_feature_file(path)
        FEATURES_CACHE[name] = df
    if "session" not in df.columns:
        raise HTTPException(400, "feature file is missing 'session' column")
    sub = df[df["session"] == session_id]
    if "bar_ix" in sub.columns:
        sub = sub.sort_values("bar_ix")
    records = json.loads(sub.to_json(orient="records"))
    return records


def _get_targets() -> pd.DataFrame:
    """Load + cache train-set targets ([session, target_return])."""
    df = _TARGETS.get("df")
    if df is None:
        # Local import so a src import error doesn't break unrelated routes.
        from src.data import compute_targets

        df = compute_targets()
        _TARGETS["df"] = df
    return df


@app.get("/api/analytics")
async def get_analytics(feature_file: str, feature_col: str):
    path = FEATURES_DIR / feature_file
    if not path.exists() or path.suffix not in (".csv", ".parquet"):
        raise HTTPException(404, f"no feature file: {feature_file}")

    df = FEATURES_CACHE.get(feature_file)
    if df is None:
        df = _load_feature_file(path)
        FEATURES_CACHE[feature_file] = df

    if "session" not in df.columns:
        raise HTTPException(400, "feature file is missing 'session' column")
    if feature_col not in df.columns:
        raise HTTPException(400, f"feature_col {feature_col!r} not in {feature_file}")

    # If the file is per-bar, collapse to one row per session (last bar value).
    sub = df[["session", feature_col]].copy()
    if "bar_ix" in df.columns:
        sub = (
            df.sort_values(["session", "bar_ix"])
            .groupby("session", as_index=False)
            .last()[["session", feature_col]]
        )

    targets = _get_targets()
    merged = sub.merge(targets, on="session", how="inner").dropna(
        subset=[feature_col, "target_return"]
    )
    n = int(len(merged))
    if n < 3:
        raise HTTPException(400, f"not enough rows after join/dropna: n={n}")

    x = merged[feature_col].to_numpy(dtype=float)
    y = merged["target_return"].to_numpy(dtype=float)
    sessions = merged["session"].to_numpy()

    # Correlation metrics. scipy is an optional dep; fall back to numpy if missing.
    try:
        from scipy import stats as _stats

        pr = _stats.pearsonr(x, y)
        sr = _stats.spearmanr(x, y)
        pearson_r = _clean_float(pr.statistic if hasattr(pr, "statistic") else pr[0])
        pearson_p = _clean_float(pr.pvalue if hasattr(pr, "pvalue") else pr[1])
        spearman_r = _clean_float(
            sr.statistic if hasattr(sr, "statistic") else sr[0]
        )
    except Exception:  # noqa: BLE001
        # Fallback: numpy pearson only; spearman via rank-pearson.
        pearson_r = _clean_float(np.corrcoef(x, y)[0, 1])
        pearson_p = None
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        spearman_r = _clean_float(np.corrcoef(rx, ry)[0, 1])

    # Deciles via qcut; duplicates="drop" handles low-cardinality features.
    deciles: list[dict] = []
    try:
        bins = pd.qcut(merged[feature_col], 10, duplicates="drop")
        grp = merged.groupby(bins, observed=True)["target_return"]
        for i, (interval, ys) in enumerate(grp, start=1):
            vals = ys.to_numpy(dtype=float)
            m = int(len(vals))
            if m == 0:
                continue
            x_mid = float((interval.left + interval.right) / 2)
            mean_y = float(vals.mean())
            std_y = float(vals.std(ddof=1)) if m > 1 else 0.0
            sem = std_y / (m ** 0.5) if m > 0 else 0.0
            ci95 = 1.96 * sem
            deciles.append(
                {
                    "bin": i,
                    "x_mid": _clean_float(x_mid),
                    "mean_y": _clean_float(mean_y),
                    "ci95": _clean_float(ci95),
                    "n": m,
                }
            )
    except Exception as e:  # noqa: BLE001
        # Non-fatal — analytics without deciles is still useful.
        deciles = []
        _ = e

    # Scatter points. Cap payload at 5000 rows for safety (train is ~1000 anyway).
    pts_df = merged
    if len(pts_df) > 5000:
        pts_df = pts_df.sample(5000, random_state=0).sort_values("session")
    points = [
        {
            "session": int(s),
            "x": _clean_float(xv),
            "y": _clean_float(yv),
        }
        for s, xv, yv in zip(
            pts_df["session"].to_numpy(),
            pts_df[feature_col].to_numpy(),
            pts_df["target_return"].to_numpy(),
        )
    ]

    # Feature importance (top 10 by gain if the file exists).
    importance: list[dict] = []
    imp_path = MODELS_DIR / "feature_importance.csv"
    if imp_path.exists():
        try:
            imp = pd.read_csv(imp_path)
            # Tolerant schema: first col = feature name, look for a gain-like col.
            if imp.shape[1] >= 2:
                first = imp.columns[0]
                gain_col = None
                for c in imp.columns[1:]:
                    lc = str(c).lower()
                    if "gain" in lc or "importance" in lc or lc in ("0", "value"):
                        gain_col = c
                        break
                if gain_col is None:
                    gain_col = imp.columns[1]
                imp = imp[[first, gain_col]].rename(
                    columns={first: "feature", gain_col: "gain"}
                )
                imp["gain"] = pd.to_numeric(imp["gain"], errors="coerce")
                imp = imp.dropna(subset=["gain"]).sort_values("gain", ascending=False)
                importance = [
                    {"feature": str(r.feature), "gain": _clean_float(r.gain)}
                    for r in imp.head(10).itertuples(index=False)
                ]
        except Exception:  # noqa: BLE001
            importance = []

    # Research sidecar (Gate 2 ablation + pre-computed histogram overlay).
    # Written by scripts/rskew_ablation.py and the researcher's notebook at
    # features/<stem>_research.json. Absent → hist/ablation are null so the UI
    # hides those two cards.
    hist_overlay: dict | None = None
    ablation: dict | None = None
    research_path = FEATURES_DIR / f"{Path(feature_file).stem}_research.json"
    if research_path.exists():
        try:
            research = json.loads(research_path.read_text())
            hist_overlay = research.get("hist")
            ablation = research.get("ablation")
        except Exception:  # noqa: BLE001
            pass

    # Stability diagnostic (Gate 3): wasserstein_distance(train, split) ≤ 0.5 * std_train.
    stability: dict | None = None
    try:
        feature_train = x  # values aligned to train sessions (already filtered/dropna'd)
        std_train = float(np.std(feature_train, ddof=1)) if len(feature_train) > 1 else float("nan")
        if not (math.isnan(std_train) or std_train == 0.0):
            from scipy.stats import wasserstein_distance as _wd  # local import; scipy already used above

            # Reload the raw feature df (already cached) and reduce to one row per session.
            stab_src = df[["session", feature_col]].copy()
            if "bar_ix" in df.columns:
                stab_src = (
                    df.sort_values(["session", "bar_ix"])
                    .groupby("session", as_index=False)
                    .last()[["session", feature_col]]
                )
            stab_src = stab_src.dropna(subset=[feature_col])

            splits_out: list[dict] = []
            all_pass = True
            any_row = False
            for sp in ("public_test", "private_test"):
                split_sessions = DATA.get(sp, {}).get("sessions", [])
                if not split_sessions:
                    continue
                vals = stab_src[stab_src["session"].isin(split_sessions)][feature_col].to_numpy(
                    dtype=float
                )
                if len(vals) == 0:
                    continue
                wd_val = float(_wd(feature_train, vals))
                ratio = wd_val / std_train
                passed = bool(ratio <= 0.5)
                all_pass = all_pass and passed
                any_row = True
                splits_out.append(
                    {
                        "split": sp,
                        "wd": _clean_float(wd_val),
                        "ratio": _clean_float(ratio),
                        "pass": passed,
                    }
                )
            stability = {
                "std_train": _clean_float(std_train),
                "splits": splits_out,
                "threshold_ratio": 0.5,
                "pass": bool(all_pass) if any_row else False,
            }
    except Exception:  # noqa: BLE001
        stability = None

    return {
        "feature": feature_col,
        "feature_file": feature_file,
        "n": n,
        "points": points,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "deciles": deciles,
        "importance": importance,
        "stability": stability,
        "hist": hist_overlay,
        "ablation": ablation,
    }


@app.get("/api/metrics")
async def get_metrics():
    log_path = ROOT / "src" / "feature_log.md"
    if not log_path.exists():
        return {"feature_log_raw": ""}
    text = log_path.read_text(encoding="utf-8", errors="replace")

    # Parse markdown table rows: | v1 | ... | cv sharpe | ... |
    latest_version: str | None = None
    latest_cv: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("|-") or line.startswith("|:"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        # Skip header row
        if cells[0].lower() in ("version", ""):
            continue
        # Table shape from feature_log.md:
        # Version | Date | Features | Feature Count | CV Sharpe | Notes
        cv = cells[4]
        if cv.upper() in ("TBD", "N/A", "-", ""):
            continue
        latest_version = cells[0]
        latest_cv = cv

    if latest_version is not None:
        return {"latest_version": latest_version, "cv_sharpe": latest_cv}
    # Fallback: return truncated raw log.
    return {"feature_log_raw": text[:2048]}


@app.get("/api/events")
async def sse(request: Request):
    async def gen() -> AsyncGenerator[str, None]:
        q: asyncio.Queue = asyncio.Queue(maxsize=32)
        _sse_clients.add(q)
        try:
            yield "event: ready\ndata: ok\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"event: {msg}\ndata: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _sse_clients.discard(q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
