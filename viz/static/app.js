// Per-session visualization. Uses the global LightweightCharts from CDN.

const UP = '#26a69a';
const DOWN = '#ef5350';
const UP_DIM = 'rgba(38, 166, 154, 0.45)';
const DOWN_DIM = 'rgba(239, 83, 80, 0.45)';
const ACCENT = '#4da3ff';

const BAR_BASE = 1700000000;
const SECS_PER_BAR = 3600;
const barToTime = (b) => BAR_BASE + b * SECS_PER_BAR;
const timeToBar = (t) => Math.round((t - BAR_BASE) / SECS_PER_BAR);

const state = {
  split: 'train',
  sessions: [],
  sessionId: null,
  headlinesByBar: new Map(),
  headlines: [],
  bars: [],
};

const $split = document.getElementById('split');
const $sid = document.getElementById('session-id');
const $meta = document.getElementById('meta');
const $headlinesList = document.getElementById('headlines-list');
const $headlinesCount = document.getElementById('headlines-count');
const $featureList = document.getElementById('feature-list');
const $chart = document.getElementById('chart');

// ── chart ────────────────────────────────────────────────────────────

const chart = LightweightCharts.createChart($chart, {
  layout: {
    background: { type: 'solid', color: '#0b0e13' },
    textColor: '#d7e0ea',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
  grid: {
    vertLines: { color: 'rgba(31, 42, 58, 0.6)' },
    horzLines: { color: 'rgba(31, 42, 58, 0.6)' },
  },
  rightPriceScale: {
    borderColor: '#1f2a3a',
    scaleMargins: { top: 0.1, bottom: 0.1 },
  },
  timeScale: {
    borderColor: '#1f2a3a',
    timeVisible: false,
    secondsVisible: false,
    tickMarkFormatter: (time) => `${timeToBar(time)}`,
  },
  crosshair: {
    mode: LightweightCharts.CrosshairMode.Normal,
    vertLine: { color: 'rgba(77, 163, 255, 0.6)', width: 1, style: LightweightCharts.LineStyle.Solid, labelBackgroundColor: '#4da3ff' },
    horzLine: { color: 'rgba(77, 163, 255, 0.6)', width: 1, style: LightweightCharts.LineStyle.Solid, labelBackgroundColor: '#4da3ff' },
  },
  localization: {
    timeFormatter: (time) => `bar ${timeToBar(time)}`,
    priceFormatter: (p) => p.toFixed(4),
  },
  autoSize: true,
});

const candleSeries = chart.addCandlestickSeries({
  upColor: UP,
  downColor: DOWN,
  borderUpColor: UP,
  borderDownColor: DOWN,
  wickUpColor: UP,
  wickDownColor: DOWN,
  priceFormat: { type: 'price', precision: 4, minMove: 0.0001 },
});

const cutoffSeries = chart.addLineSeries({
  color: ACCENT,
  lineWidth: 2,
  lineStyle: LightweightCharts.LineStyle.Dashed,
  priceLineVisible: false,
  lastValueVisible: false,
  crosshairMarkerVisible: false,
});

// ── data helpers ─────────────────────────────────────────────────────

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}: ${await r.text()}`);
  return r.json();
}

function fmtPct(x) {
  const s = (x * 100).toFixed(2);
  return `${x >= 0 ? '+' : ''}${s}%`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[c]);
}

// ── rendering ────────────────────────────────────────────────────────

function renderChart(d) {
  state.bars = d.bars;
  state.headlines = d.headlines;
  state.headlinesByBar = new Map();
  for (const h of d.headlines) {
    if (!state.headlinesByBar.has(h.bar_ix)) state.headlinesByBar.set(h.bar_ix, []);
    state.headlinesByBar.get(h.bar_ix).push(h.headline);
  }

  const candles = d.bars.map((b) => {
    const up = b.close >= b.open;
    const c = b.seen ? (up ? UP : DOWN) : (up ? UP_DIM : DOWN_DIM);
    return {
      time: barToTime(b.bar_ix),
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
      color: c,
      borderColor: c,
      wickColor: c,
    };
  });
  candleSeries.setData(candles);

  if (d.cutoff_bar !== null && d.cutoff_bar !== undefined) {
    const lows = d.bars.map((b) => b.low).filter((x) => x != null);
    const highs = d.bars.map((b) => b.high).filter((x) => x != null);
    const lo = Math.min(...lows);
    const hi = Math.max(...highs);
    const pad = (hi - lo) * 0.5 || 0.01;
    // two points 1s apart → near-vertical line at the bar 49/50 boundary
    const tA = barToTime(d.cutoff_bar) + SECS_PER_BAR / 2;
    cutoffSeries.setData([
      { time: tA, value: lo - pad },
      { time: tA + 1, value: hi + pad },
    ]);
  } else {
    cutoffSeries.setData([]);
  }

  const markers = d.headlines.map((h) => ({
    time: barToTime(h.bar_ix),
    position: 'aboveBar',
    color: ACCENT,
    shape: 'circle',
    size: 1,
  }));
  candleSeries.setMarkers(markers);

  chart.timeScale().fitContent();

  // meta
  const parts = [
    `<span class="chip">${escapeHtml(d.split)}</span>`,
    `<span class="chip">session <strong style="color:var(--text)">${d.session}</strong></span>`,
    `<span class="chip">${d.bars.length} bars</span>`,
    `<span class="chip">${d.headlines.length} headlines</span>`,
  ];
  if (d.target_return !== null && d.target_return !== undefined) {
    const cls = d.target_return >= 0 ? 'pos' : 'neg';
    parts.push(`<span class="chip ${cls}">target_return ${fmtPct(d.target_return)}</span>`);
  }
  $meta.innerHTML = parts.join('');

  renderHeadlines(null);
}

function renderHeadlines(activeBar) {
  $headlinesCount.textContent = state.headlines.length ? `(${state.headlines.length})` : '';
  if (!state.headlines.length) {
    $headlinesList.innerHTML = '<div class="empty">No headlines in this session.</div>';
    return;
  }
  const rows = state.headlines.map((h) => {
    const active = activeBar !== null && h.bar_ix === activeBar;
    return `<div class="h-row${active ? ' active' : ''}" data-bar="${h.bar_ix}">
      <span class="h-bar">bar ${h.bar_ix}</span>
      <span class="h-txt">${escapeHtml(h.headline)}</span>
    </div>`;
  });
  $headlinesList.innerHTML = rows.join('');
  if (activeBar !== null) {
    const el = $headlinesList.querySelector('.h-row.active');
    if (el) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
}

chart.subscribeCrosshairMove((param) => {
  if (!param.time) { renderHeadlines(null); return; }
  renderHeadlines(timeToBar(param.time));
});

// ── state / navigation ──────────────────────────────────────────────

function writeHash() {
  const h = `#split=${state.split}&session=${state.sessionId}`;
  if (location.hash !== h) history.replaceState(null, '', h);
}

function readHash() {
  const h = location.hash.replace(/^#/, '');
  const p = new URLSearchParams(h);
  if (p.has('split')) state.split = p.get('split');
  if (p.has('session')) {
    const sid = parseInt(p.get('session'), 10);
    if (!isNaN(sid)) state.sessionId = sid;
  }
}

async function loadSplits() {
  const splits = await fetchJSON('/api/splits');
  $split.innerHTML = splits
    .map((s) => `<option value="${s.name}">${s.name} · ${s.count.toLocaleString()}</option>`)
    .join('');
  $split.value = state.split;
}

async function loadSessions() {
  state.sessions = await fetchJSON(`/api/sessions?split=${state.split}`);
  if (state.sessionId === null || !state.sessions.includes(state.sessionId)) {
    state.sessionId = state.sessions[0];
  }
  $sid.min = state.sessions[0];
  $sid.max = state.sessions[state.sessions.length - 1];
  $sid.value = state.sessionId;
}

async function loadSession() {
  if (state.sessionId === null) return;
  $sid.value = state.sessionId;
  writeHash();
  try {
    const d = await fetchJSON(`/api/session/${state.split}/${state.sessionId}`);
    renderChart(d);
  } catch (e) {
    $meta.innerHTML = `<span class="chip neg">error: ${escapeHtml(e.message)}</span>`;
  }
}

function gotoOffset(delta) {
  const idx = state.sessions.indexOf(state.sessionId);
  if (idx === -1) return;
  const ni = Math.max(0, Math.min(state.sessions.length - 1, idx + delta));
  if (ni === idx) return;
  state.sessionId = state.sessions[ni];
  loadSession();
}

// ── features sidebar (scaffold) ─────────────────────────────────────

async function refreshFeatures() {
  try {
    const features = await fetchJSON('/api/features');
    if (!features.length) {
      $featureList.innerHTML = '<div class="empty">Drop CSV or Parquet files into <code>features/</code> to overlay them here. Required columns: <code>session</code>, optionally <code>bar_ix</code>, plus feature columns.</div>';
      return;
    }
    $featureList.innerHTML = features.map((f) => {
      if (f.error) {
        return `<div class="f-file err"><div class="f-name">${escapeHtml(f.name)}</div><div class="empty">${escapeHtml(f.error)}</div></div>`;
      }
      const cols = (f.columns || []).map((c) =>
        `<label><input type="checkbox" disabled /> ${escapeHtml(c)}</label>`
      ).join('');
      return `<div class="f-file">
        <div class="f-name">${escapeHtml(f.name)}</div>
        <div class="muted" style="font-size:10px;color:var(--muted);margin-bottom:4px;">${f.per_bar ? 'per-bar' : 'per-session'} · ${f.rows} rows</div>
        ${cols || '<div class="empty">no feature columns</div>'}
      </div>`;
    }).join('');
  } catch (e) {
    $featureList.innerHTML = `<div class="empty">features api error: ${escapeHtml(e.message)}</div>`;
  }
}

// ── wire-up ─────────────────────────────────────────────────────────

$split.addEventListener('change', async () => {
  state.split = $split.value;
  state.sessionId = null;
  await loadSessions();
  await loadSession();
});

let sidDebounce;
$sid.addEventListener('input', () => {
  const v = parseInt($sid.value, 10);
  if (isNaN(v)) return;
  clearTimeout(sidDebounce);
  sidDebounce = setTimeout(() => {
    if (state.sessions.includes(v)) {
      state.sessionId = v;
      loadSession();
    }
  }, 180);
});

document.getElementById('btn-prev').addEventListener('click', () => gotoOffset(-1));
document.getElementById('btn-next').addEventListener('click', () => gotoOffset(+1));
document.getElementById('btn-random').addEventListener('click', () => {
  const i = Math.floor(Math.random() * state.sessions.length);
  state.sessionId = state.sessions[i];
  loadSession();
});

$headlinesList.addEventListener('click', (e) => {
  const row = e.target.closest('.h-row');
  if (!row) return;
  const bar = parseInt(row.dataset.bar, 10);
  if (!isNaN(bar)) {
    chart.timeScale().scrollToPosition(0, false);
    // flash the active row
    renderHeadlines(bar);
  }
});

document.addEventListener('keydown', (e) => {
  const tag = e.target.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === 'ArrowLeft') { e.preventDefault(); gotoOffset(-1); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); gotoOffset(+1); }
  else if (e.key === 'r' || e.key === 'R') { loadSession(); }
  else if (e.key === '1') { state.split = 'train'; $split.value = 'train'; state.sessionId = null; loadSessions().then(loadSession); }
  else if (e.key === '2') { state.split = 'public_test'; $split.value = 'public_test'; state.sessionId = null; loadSessions().then(loadSession); }
  else if (e.key === '3') { state.split = 'private_test'; $split.value = 'private_test'; state.sessionId = null; loadSessions().then(loadSession); }
});

// SSE for features live-reload
function connectSSE() {
  const es = new EventSource('/api/events');
  es.addEventListener('reload', () => {
    refreshFeatures();
  });
  es.addEventListener('ready', () => {});
  es.onerror = () => {
    // EventSource will auto-reconnect
  };
}

// ── init ────────────────────────────────────────────────────────────

(async function init() {
  readHash();
  await loadSplits();
  $split.value = state.split;
  await loadSessions();
  await loadSession();
  await refreshFeatures();
  connectSSE();
})();
