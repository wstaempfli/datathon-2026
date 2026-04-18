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

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FEATURES_DIR = ROOT / "features"
STATIC_DIR = Path(__file__).resolve().parent / "static"

SPLITS = ("train", "public_test", "private_test")
SEEN_MAX_BAR = 49  # bar_ix 0..49 is always visible; 50..99 is "unseen"

DATA: dict[str, dict] = {}
FEATURES_CACHE: dict[str, pd.DataFrame] = {}
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
            columns = [c for c in df.columns if c not in key_cols]
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
