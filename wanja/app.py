from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HALF = 50  # seen = bars 0..49, unseen = bars 50..99

BAR_FILES = {
    "train_seen": "bars_seen_train.parquet",
    "train_unseen": "bars_unseen_train.parquet",
    "public_test_seen": "bars_seen_public_test.parquet",
    "private_test_seen": "bars_seen_private_test.parquet",
}
HEADLINE_FILES = {
    "train_seen": "headlines_seen_train.parquet",
    "train_unseen": "headlines_unseen_train.parquet",
    "public_test_seen": "headlines_seen_public_test.parquet",
    "private_test_seen": "headlines_seen_private_test.parquet",
}

STOPWORDS = set(
    """a an the and or of in on for to with by from at as is are was were be been being
    has have had it its this that these those but not no if then so than into about over
    after before between during under above up down out off more most some any all each
    other such will would can could may might do does did done new announces announced
    announcement report reports reported reporting company companies""".split()
)


# ---------- Loaders (cached) ----------

@st.cache_data(show_spinner=False)
def load_bars(split: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / BAR_FILES[split])
    return df.sort_values(["session", "bar_ix"], kind="mergesort").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_headlines(split: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / HEADLINE_FILES[split])


@st.cache_data(show_spinner=False)
def train_full_bars() -> pd.DataFrame:
    seen = load_bars("train_seen").assign(half="seen")
    unseen = load_bars("train_unseen").assign(half="unseen")
    full = pd.concat([seen, unseen], ignore_index=True)
    return full.sort_values(["session", "bar_ix"], kind="mergesort").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def summarize_seen(split_seen: str) -> pd.DataFrame:
    """Per-session summary computed from the first-half (seen) bars only."""
    df = load_bars(split_seen)
    g = df.groupby("session", sort=False)
    first = g.head(1).set_index("session")["open"].rename("open_start")
    last = g.tail(1).set_index("session")["close"].rename("close_half")
    hi = g["high"].max().rename("high_half")
    lo = g["low"].min().rename("low_half")
    # log-return-based volatility on the first half
    logret = np.log(df["close"]).groupby(df["session"]).diff()
    vol = logret.groupby(df["session"]).std().rename("vol_half")
    out = pd.concat([first, last, hi, lo, vol], axis=1).reset_index()
    out["first_half_return"] = out["close_half"] / out["open_start"] - 1
    out["range_half"] = out["high_half"] - out["low_half"]
    return out


@st.cache_data(show_spinner=False)
def summarize_train_full() -> pd.DataFrame:
    """Per-session summary for train (has second half)."""
    full = train_full_bars()
    g = full.groupby("session", sort=False)
    close_end = g.tail(1).set_index("session")["close"].rename("close_end")
    close_half = full[full["bar_ix"] == HALF - 1].set_index("session")["close"].rename("close_half")
    # second half volatility
    unseen = load_bars("train_unseen")
    ur = np.log(unseen["close"]).groupby(unseen["session"]).diff()
    vol_u = ur.groupby(unseen["session"]).std().rename("vol_second_half")
    out = pd.concat([close_half, close_end, vol_u], axis=1).reset_index()
    out["second_half_return"] = out["close_end"] / out["close_half"] - 1
    out["full_return"] = out["close_end"] - 1.0
    return out


@st.cache_data(show_spinner=False)
def headline_counts(split: str) -> pd.DataFrame:
    hl = load_headlines(split)
    return hl.groupby("session").size().rename("n_headlines").reset_index()


# ---------- Helpers ----------

def sharpe(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, dtype=float)
    s = pnl.std()
    if not np.isfinite(s) or s == 0:
        return float("nan")
    return float(pnl.mean() / s * 16)


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


CAP_ENTITY_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b")


def extract_entities(text: str) -> list[str]:
    # Drop sentence-start capitalized single words by requiring 2+ tokens
    ents = []
    for m in CAP_ENTITY_RE.finditer(text):
        phrase = m.group(1)
        if " " in phrase:  # multi-word only
            ents.append(phrase)
    return ents


# ---------- Pages ----------

def page_overview():
    st.header("Dataset overview")
    st.caption(
        "Each session = 1 synthetic stock. Seen = first 50 bars, unseen = next 50. "
        "Prices are normalized so open of bar 0 = 1.0."
    )

    rows = []
    for split, fname in BAR_FILES.items():
        df = load_bars(split)
        rows.append(
            {
                "split": split,
                "rows": len(df),
                "sessions": df["session"].nunique(),
                "bars/session (min-max)": f"{df.groupby('session').size().min()}–{df.groupby('session').size().max()}",
                "session id range": f"{df['session'].min()}–{df['session'].max()}",
            }
        )
    st.subheader("Bar files")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    rows_h = []
    for split, fname in HEADLINE_FILES.items():
        hl = load_headlines(split)
        hc = hl.groupby("session").size()
        rows_h.append(
            {
                "split": split,
                "rows": len(hl),
                "sessions with ≥1 headline": hl["session"].nunique(),
                "headlines/session (min-med-max)": f"{hc.min()}–{int(hc.median())}–{hc.max()}",
            }
        )
    st.subheader("Headline files")
    st.dataframe(pd.DataFrame(rows_h), hide_index=True, use_container_width=True)

    # Sessions without any headlines
    st.subheader("Coverage check: sessions missing headlines")
    cov_rows = []
    for split in ["train_seen", "public_test_seen", "private_test_seen"]:
        b_sess = set(load_bars(split)["session"].unique())
        h_sess = set(load_headlines(split)["session"].unique())
        missing = b_sess - h_sess
        cov_rows.append({"split": split, "bar sessions": len(b_sess), "headline sessions": len(h_sess), "missing headlines": len(missing)})
    # train unseen headlines
    b_sess = set(load_bars("train_unseen")["session"].unique())
    h_sess = set(load_headlines("train_unseen")["session"].unique())
    cov_rows.append({"split": "train_unseen", "bar sessions": len(b_sess), "headline sessions": len(h_sess), "missing headlines": len(b_sess - h_sess)})
    st.dataframe(pd.DataFrame(cov_rows), hide_index=True, use_container_width=True)


def page_session_explorer():
    st.header("Session explorer")

    split = st.selectbox(
        "Split",
        ["train", "public_test", "private_test"],
        help="Train shows the full session; tests show only the first half.",
    )
    seen_key = {"train": "train_seen", "public_test": "public_test_seen", "private_test": "private_test_seen"}[split]
    seen = load_bars(seen_key)
    sess_ids = seen["session"].unique()

    col1, col2 = st.columns([1, 3])
    with col1:
        session_id = st.number_input(
            "Session ID",
            min_value=int(sess_ids.min()),
            max_value=int(sess_ids.max()),
            value=int(sess_ids[0]),
            step=1,
        )
        if st.button("Random session"):
            st.session_state["_rand_session"] = int(np.random.choice(sess_ids))
        if "_rand_session" in st.session_state:
            session_id = st.session_state["_rand_session"]
            st.caption(f"Random pick: {session_id}")

    sess_seen = seen[seen["session"] == session_id].sort_values("bar_ix")
    if split == "train":
        unseen = load_bars("train_unseen")
        sess_unseen = unseen[unseen["session"] == session_id].sort_values("bar_ix")
    else:
        sess_unseen = pd.DataFrame(columns=sess_seen.columns)

    if sess_seen.empty:
        st.warning("No data for that session id.")
        return

    # Load headlines for this session
    if split == "train":
        hl = pd.concat(
            [
                load_headlines("train_seen").assign(half="seen"),
                load_headlines("train_unseen").assign(half="unseen"),
            ],
            ignore_index=True,
        )
    else:
        hl = load_headlines(seen_key).assign(half="seen")
    sess_hl = hl[hl["session"] == session_id].sort_values("bar_ix")

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=sess_seen["bar_ix"], open=sess_seen["open"], high=sess_seen["high"],
            low=sess_seen["low"], close=sess_seen["close"], name="seen",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        )
    )
    if not sess_unseen.empty:
        fig.add_trace(
            go.Candlestick(
                x=sess_unseen["bar_ix"], open=sess_unseen["open"], high=sess_unseen["high"],
                low=sess_unseen["low"], close=sess_unseen["close"], name="unseen (hidden at test time)",
                increasing_line_color="#80cbc4", decreasing_line_color="#ef9a9a",
                opacity=0.6,
            )
        )
    fig.add_vline(x=HALF - 0.5, line_width=1, line_dash="dash", line_color="gray")
    fig.add_annotation(x=HALF - 0.5, y=1, yref="paper", text="halfway →", showarrow=False, xanchor="left")

    # headline markers at the high of that bar
    bar_high = sess_seen.set_index("bar_ix")["high"].to_dict()
    if not sess_unseen.empty:
        bar_high.update(sess_unseen.set_index("bar_ix")["high"].to_dict())
    if not sess_hl.empty:
        marker_y = [bar_high.get(b, 1.0) * 1.01 for b in sess_hl["bar_ix"]]
        fig.add_trace(
            go.Scatter(
                x=sess_hl["bar_ix"], y=marker_y, mode="markers",
                marker=dict(symbol="triangle-down", size=9, color="#2962ff"),
                text=sess_hl["headline"], hovertemplate="bar %{x}: %{text}<extra></extra>",
                name="headline",
            )
        )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=500,
        title=f"Session {session_id}",
        legend=dict(orientation="h", y=1.08),
    )
    with col2:
        st.plotly_chart(fig, use_container_width=True)

    # Key stats
    close_half = sess_seen[sess_seen["bar_ix"] == HALF - 1]["close"].iloc[0]
    first_open = sess_seen[sess_seen["bar_ix"] == 0]["open"].iloc[0]
    cols = st.columns(4)
    cols[0].metric("Close @ halfway", f"{close_half:.4f}")
    cols[1].metric("First-half return", f"{(close_half/first_open - 1)*100:.2f}%")
    if split == "train" and not sess_unseen.empty:
        close_end = sess_unseen[sess_unseen["bar_ix"] == 2 * HALF - 1]["close"].iloc[0]
        cols[2].metric("Close @ end", f"{close_end:.4f}")
        cols[3].metric("Second-half return (target)", f"{(close_end/close_half - 1)*100:.2f}%")
    else:
        cols[2].metric("Close @ end", "hidden")
        cols[3].metric("Second-half return", "hidden")

    st.subheader("Headlines in this session")
    if sess_hl.empty:
        st.info("No headlines for this session.")
    else:
        st.dataframe(
            sess_hl[["bar_ix", "half", "headline"]].reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )


def page_price_dynamics():
    st.header("Price dynamics")

    split = st.selectbox(
        "Split",
        ["train", "public_test", "private_test"],
        help="Overlay and distributions across many sessions.",
        key="pd_split",
    )
    seen_key = {"train": "train_seen", "public_test": "public_test_seen", "private_test": "private_test_seen"}[split]

    if split == "train":
        bars = train_full_bars()
        x_max = 2 * HALF - 1
    else:
        bars = load_bars(seen_key)
        x_max = HALF - 1

    n_sample = st.slider("Sessions to overlay", 20, 500, 100, step=20)
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(bars["session"].unique(), size=n_sample, replace=False)
    sub = bars[bars["session"].isin(sample_ids)]
    fig = px.line(
        sub, x="bar_ix", y="close", color="session",
        line_group="session", render_mode="webgl",
        labels={"close": "close (normalized)"},
    )
    fig.update_traces(opacity=0.25, line_width=1)
    fig.update_layout(showlegend=False, height=420, title=f"Overlay of {n_sample} normalized session paths")
    fig.add_vline(x=HALF - 0.5, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    # Distributions
    summ_seen = summarize_seen(seen_key)
    col1, col2 = st.columns(2)
    with col1:
        f = px.histogram(summ_seen, x="first_half_return", nbins=60,
                         title="First-half return (close@halfway / 1.0 − 1)")
        f.update_layout(height=340)
        st.plotly_chart(f, use_container_width=True)
    with col2:
        f = px.histogram(summ_seen, x="vol_half", nbins=60,
                         title="First-half volatility (std of log returns)")
        f.update_layout(height=340)
        st.plotly_chart(f, use_container_width=True)

    if split == "train":
        full_summ = summarize_train_full().merge(summ_seen, on="session", suffixes=("", "_seen"))
        col3, col4 = st.columns(2)
        with col3:
            f = px.histogram(full_summ, x="second_half_return", nbins=60,
                             title="Second-half return (the target PnL per unit long)")
            f.update_layout(height=340)
            st.plotly_chart(f, use_container_width=True)
        with col4:
            corr = full_summ[["first_half_return", "second_half_return"]].corr().iloc[0, 1]
            f = px.scatter(
                full_summ, x="first_half_return", y="second_half_return",
                trendline="ols", opacity=0.6,
                title=f"First vs second half return (Pearson r = {corr:.3f})",
            )
            f.update_layout(height=340)
            st.plotly_chart(f, use_container_width=True)


def page_target_signal():
    st.header("Target signal & baselines (train)")
    st.caption(
        "The target is `pnl = position * (close_end/close_half − 1)`. "
        "Score = `mean(pnl)/std(pnl) * 16`. These are baselines computed on the 1000 train sessions — "
        "treat the numbers as orientation, not validation."
    )

    full = summarize_train_full()
    seen = summarize_seen("train_seen")
    df = full.merge(seen, on="session")

    r2 = df["second_half_return"].to_numpy()
    r1 = df["first_half_return"].to_numpy()
    v1 = df["vol_half"].to_numpy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean 2nd-half return", f"{r2.mean()*100:.3f}%")
    col2.metric("Std 2nd-half return", f"{r2.std()*100:.3f}%")
    col3.metric("Train sessions", f"{len(df)}")

    f = px.histogram(df, x="second_half_return", nbins=80, title="Distribution of the target return")
    f.add_vline(x=0, line_dash="dash")
    f.update_layout(height=340)
    st.plotly_chart(f, use_container_width=True)

    rng = np.random.default_rng(0)
    strategies = {
        "always long (+1)": np.ones_like(r2),
        "always short (−1)": -np.ones_like(r2),
        "random ±1": rng.choice([-1.0, 1.0], size=len(r2)),
        "momentum: sign(first-half)": np.sign(r1),
        "mean-rev: −sign(first-half)": -np.sign(r1),
        "momentum / vol": np.where(v1 > 0, np.sign(r1) / np.where(v1 > 0, v1, 1e-9), 0.0),
        "first-half return (linear)": r1,
        "−first-half return (linear)": -r1,
    }
    rows = []
    for name, pos in strategies.items():
        pnl = pos * r2
        rows.append({"strategy": name, "mean_pnl": pnl.mean(), "std_pnl": pnl.std(), "sharpe*16": sharpe(pnl)})
    st.subheader("Baseline Sharpe scores on train")
    st.dataframe(
        pd.DataFrame(rows).sort_values("sharpe*16", ascending=False).reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )

    st.subheader("First-half features vs target")
    feat = st.selectbox("Feature", ["first_half_return", "vol_half", "range_half", "high_half", "low_half"])
    corr = np.corrcoef(df[feat], df["second_half_return"])[0, 1]
    f = px.scatter(df, x=feat, y="second_half_return", trendline="ols", opacity=0.6,
                   title=f"{feat} vs second-half return (Pearson r = {corr:.3f})")
    f.add_hline(y=0, line_dash="dash", line_color="gray")
    f.update_layout(height=420)
    st.plotly_chart(f, use_container_width=True)


@st.cache_data(show_spinner=False)
def compute_top_tokens(split: str, top_n: int = 40) -> pd.DataFrame:
    hl = load_headlines(split)
    c = Counter()
    for text in hl["headline"]:
        for tok in tokenize(text):
            if tok in STOPWORDS or len(tok) < 3:
                continue
            c[tok] += 1
    return pd.DataFrame(c.most_common(top_n), columns=["token", "count"])


@st.cache_data(show_spinner=False)
def compute_top_entities(split: str, top_n: int = 30) -> pd.DataFrame:
    hl = load_headlines(split)
    c = Counter()
    for text in hl["headline"]:
        for ent in extract_entities(text):
            c[ent] += 1
    return pd.DataFrame(c.most_common(top_n), columns=["entity", "count"])


def page_headlines():
    st.header("Headlines (structural)")

    split = st.selectbox(
        "Split", list(HEADLINE_FILES.keys()), index=0, key="hl_split",
    )
    hl = load_headlines(split)
    hl = hl.assign(n_words=hl["headline"].str.split().str.len(),
                   n_chars=hl["headline"].str.len())

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(hl):,}")
    col2.metric("Sessions with headlines", f"{hl['session'].nunique():,}")
    col3.metric("Median words/headline", f"{int(hl['n_words'].median())}")

    cc = hl.groupby("session").size().rename("n_headlines").reset_index()
    col_a, col_b = st.columns(2)
    with col_a:
        f = px.histogram(cc, x="n_headlines", nbins=30, title="Headlines per session")
        f.update_layout(height=340)
        st.plotly_chart(f, use_container_width=True)
    with col_b:
        bc = hl.groupby("bar_ix").size().reset_index(name="n")
        f = px.bar(bc, x="bar_ix", y="n", title="Headlines per bar_ix (across all sessions)")
        f.update_layout(height=340)
        if split in ("train_seen", "public_test_seen", "private_test_seen"):
            f.add_vline(x=HALF - 0.5, line_dash="dash", line_color="gray")
        st.plotly_chart(f, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        f = px.histogram(hl, x="n_words", nbins=30, title="Headline length (words)")
        f.update_layout(height=340)
        st.plotly_chart(f, use_container_width=True)
    with col_d:
        f = px.histogram(hl, x="n_chars", nbins=40, title="Headline length (chars)")
        f.update_layout(height=340)
        st.plotly_chart(f, use_container_width=True)

    st.subheader("Top tokens & multi-word entities")
    col_e, col_f = st.columns(2)
    with col_e:
        top_tokens = compute_top_tokens(split, top_n=40)
        f = px.bar(top_tokens.head(25), x="count", y="token", orientation="h",
                   title="Top 25 tokens (stopword-filtered)")
        f.update_layout(height=560, yaxis=dict(autorange="reversed"))
        st.plotly_chart(f, use_container_width=True)
    with col_f:
        top_ents = compute_top_entities(split, top_n=30)
        f = px.bar(top_ents.head(25), x="count", y="entity", orientation="h",
                   title="Top 25 capitalized multi-word entities (proxy for company names)")
        f.update_layout(height=560, yaxis=dict(autorange="reversed"))
        st.plotly_chart(f, use_container_width=True)

    st.subheader("Search")
    q = st.text_input("Substring filter (case-insensitive)", "")
    if q:
        matches = hl[hl["headline"].str.contains(q, case=False, regex=False)]
        st.caption(f"{len(matches):,} matches across {matches['session'].nunique():,} sessions")
        st.dataframe(matches[["session", "bar_ix", "headline"]].head(200),
                     use_container_width=True, hide_index=True)


def page_parity():
    st.header("Train vs test parity")
    st.caption(
        "Are the seen-half distributions comparable across splits? "
        "Large differences here are early warning for distribution shift."
    )
    tr = summarize_seen("train_seen").assign(split="train_seen")
    pub = summarize_seen("public_test_seen").assign(split="public_test_seen")
    pri = summarize_seen("private_test_seen").assign(split="private_test_seen")
    allsumm = pd.concat([tr, pub, pri], ignore_index=True)

    for metric in ["first_half_return", "vol_half", "range_half", "high_half", "low_half"]:
        f = px.histogram(
            allsumm, x=metric, color="split", barmode="overlay", opacity=0.55,
            nbins=60, histnorm="probability density", title=metric,
        )
        f.update_layout(height=320, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(f, use_container_width=True)

    st.subheader("Summary stats")
    stats = (
        allsumm.groupby("split")[["first_half_return", "vol_half", "range_half"]]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )
    st.dataframe(stats, use_container_width=True)

    # Headlines-per-session comparison
    st.subheader("Headlines per session")
    h_tr = headline_counts("train_seen").assign(split="train_seen")
    h_pub = headline_counts("public_test_seen").assign(split="public_test_seen")
    h_pri = headline_counts("private_test_seen").assign(split="private_test_seen")
    allh = pd.concat([h_tr, h_pub, h_pri], ignore_index=True)
    f = px.histogram(allh, x="n_headlines", color="split", barmode="overlay",
                     opacity=0.55, nbins=30, histnorm="probability density")
    f.update_layout(height=340)
    st.plotly_chart(f, use_container_width=True)


# ---------- App ----------

def main():
    st.set_page_config(page_title="Datathon 2026 EDA", layout="wide")
    st.sidebar.title("Datathon 2026 EDA")
    st.sidebar.caption("Zurich 2026 · market close prediction")

    pages = {
        "Overview": page_overview,
        "Session explorer": page_session_explorer,
        "Price dynamics": page_price_dynamics,
        "Target signal": page_target_signal,
        "Headlines": page_headlines,
        "Train vs test parity": page_parity,
    }
    choice = st.sidebar.radio("Page", list(pages.keys()))
    pages[choice]()


if __name__ == "__main__":
    main()
