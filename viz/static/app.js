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
  // Overlay state: key = `${fileName}::${columnName}` → {series, color, perBar, fileName, columnName}.
  overlays: new Map(),
  // Persisted set of overlay keys that should be active. Used to restore on nav + SSE reload.
  activeKeys: new Set(),
  // Whether we've applied the default-on MA once for this session (first load).
  defaultsApplied: false,
};

const PALETTE = ['#f59e0b', '#a855f7', '#22d3ee', '#f472b6', '#84cc16', '#eab308', '#fb7185', '#60a5fa', '#f97316', '#10b981'];
const colorForKey = (key) => {
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) | 0;
  return PALETTE[Math.abs(h) % PALETTE.length];
};

// Features whose magnitude roughly matches OHLC price go on the main (right) scale.
// Others (rsi, vol, returns, sentiment, predictions) get their own left scale.
const PRICE_LIKE = /^(close|open|high|low|ma_|ema_|sma_|vwap|mid)/i;
const onPriceScale = (col) => PRICE_LIKE.test(col);

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
  leftPriceScale: {
    visible: true,
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
  const ov = Array.from(state.activeKeys).join(',');
  const parts = [`split=${state.split}`, `session=${state.sessionId}`];
  if (ov) parts.push(`ov=${encodeURIComponent(ov)}`);
  const h = '#' + parts.join('&');
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
  if (p.has('ov')) {
    const raw = decodeURIComponent(p.get('ov'));
    for (const k of raw.split(',').filter(Boolean)) state.activeKeys.add(k);
    state.defaultsApplied = true;  // honor what the URL says, don't overwrite with defaults.
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
    // Re-draw any active overlays for the new session.
    await refreshAllOverlays();
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

// ── features sidebar + overlays ─────────────────────────────────────

// File-level metadata from /api/features (name → {columns, per_bar, rows}).
let featureFiles = [];
const keyOf = (name, col) => `${name}::${col}`;

async function refreshFeatures() {
  try {
    featureFiles = await fetchJSON('/api/features');
    if (!featureFiles.length) {
      $featureList.innerHTML = '<div class="empty">Drop CSV or Parquet files into <code>features/</code> to overlay them here. Required columns: <code>session</code>, optionally <code>bar_ix</code>, plus numeric feature columns.</div>';
      return;
    }
    // Apply default-on (MA) exactly once, before we render, so checkboxes render checked.
    if (!state.defaultsApplied) {
      for (const f of featureFiles) {
        if (!f.columns) continue;
        for (const col of f.columns) {
          if (/^ma_(5|10|20)$/.test(col)) state.activeKeys.add(keyOf(f.name, col));
        }
      }
      state.defaultsApplied = true;
    }
    $featureList.innerHTML = featureFiles.map((f) => {
      if (f.error) {
        return `<div class="f-file err"><div class="f-name">${escapeHtml(f.name)}</div><div class="empty">${escapeHtml(f.error)}</div></div>`;
      }
      const cols = (f.columns || []).map((c) => {
        const k = keyOf(f.name, c);
        const checked = state.activeKeys.has(k) ? ' checked' : '';
        const sw = `<span class="swatch" style="background:${colorForKey(k)}"></span>`;
        return `<label class="f-col">${sw}<input type="checkbox" data-key="${escapeHtml(k)}" data-file="${escapeHtml(f.name)}" data-col="${escapeHtml(c)}" data-perbar="${f.per_bar ? 1 : 0}"${checked} /> ${escapeHtml(c)}</label>`;
      }).join('');
      return `<div class="f-file">
        <div class="f-name">${escapeHtml(f.name)}</div>
        <div class="muted" style="font-size:10px;color:var(--muted);margin-bottom:4px;">${f.per_bar ? 'per-bar' : 'per-session'} · ${f.rows} rows</div>
        ${cols || '<div class="empty">no numeric feature columns</div>'}
      </div>`;
    }).join('');
    // Wire listeners for each checkbox.
    $featureList.querySelectorAll('input[type="checkbox"][data-key]').forEach((cb) => {
      cb.addEventListener('change', () => onToggleFeature(cb));
    });
    // Refresh any currently-active overlays for the loaded session.
    await refreshAllOverlays();
  } catch (e) {
    $featureList.innerHTML = `<div class="empty">features api error: ${escapeHtml(e.message)}</div>`;
  }
}

function onToggleFeature(cb) {
  const k = cb.dataset.key;
  if (cb.checked) {
    state.activeKeys.add(k);
    addOverlay(cb.dataset.file, cb.dataset.col, cb.dataset.perbar === '1');
  } else {
    state.activeKeys.delete(k);
    removeOverlay(k);
  }
  writeHash();
}

async function addOverlay(fileName, colName, perBar) {
  const key = keyOf(fileName, colName);
  if (state.overlays.has(key)) removeOverlay(key);  // avoid dupes
  const color = colorForKey(key);
  const series = chart.addLineSeries({
    color,
    lineWidth: perBar ? 2 : 1,
    priceLineVisible: false,
    lastValueVisible: false,
    crosshairMarkerVisible: false,
    priceScaleId: onPriceScale(colName) ? 'right' : 'left',
  });
  state.overlays.set(key, { series, color, fileName, columnName: colName, perBar });
  await populateOverlay(key);
}

function removeOverlay(key) {
  const o = state.overlays.get(key);
  if (!o) return;
  try { chart.removeSeries(o.series); } catch (_) {}
  state.overlays.delete(key);
}

async function populateOverlay(key) {
  const o = state.overlays.get(key);
  if (!o || state.sessionId === null) return;
  try {
    const rows = await fetchJSON(`/api/features/${encodeURIComponent(o.fileName)}/${state.sessionId}`);
    let data;
    if (o.perBar) {
      data = rows
        .filter((r) => r[o.columnName] !== null && r[o.columnName] !== undefined && !Number.isNaN(+r[o.columnName]))
        .map((r) => ({ time: barToTime(+r.bar_ix), value: +r[o.columnName] }));
    } else {
      // per-session scalar: horizontal line across all bars in the session.
      const v = rows.length ? +rows[0][o.columnName] : NaN;
      if (!Number.isFinite(v) || !state.bars.length) { data = []; }
      else {
        const b0 = state.bars[0].bar_ix;
        const b1 = state.bars[state.bars.length - 1].bar_ix;
        data = [{ time: barToTime(b0), value: v }, { time: barToTime(b1), value: v }];
      }
    }
    o.series.setData(data);
  } catch (e) {
    console.warn('overlay fetch failed', key, e);
    o.series.setData([]);
  }
}

async function refreshAllOverlays() {
  // Add any activeKeys that don't have a series yet (e.g. after nav).
  for (const k of state.activeKeys) {
    if (state.overlays.has(k)) continue;
    const [fileName, colName] = k.split('::');
    const f = featureFiles.find((x) => x.name === fileName);
    if (!f || !f.columns || !f.columns.includes(colName)) continue;
    await addOverlay(fileName, colName, !!f.per_bar);
  }
  // Repopulate existing series for the current session.
  await Promise.all(Array.from(state.overlays.keys()).map(populateOverlay));
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

// SSE for features live-reload. Preserve active overlay keys across reloads so
// toggling state survives pipeline reruns. Existing series get removed (their
// underlying file may have changed) and re-added inside refreshFeatures().
function connectSSE() {
  const es = new EventSource('/api/events');
  es.addEventListener('reload', () => {
    for (const k of Array.from(state.overlays.keys())) removeOverlay(k);
    refreshFeatures();
    // If on analytics, re-fetch with the current selection.
    if (analytics.active) analytics.loadCurrent();
  });
  es.addEventListener('ready', () => {});
  es.onerror = () => {
    // EventSource will auto-reconnect
  };
}

// ── analytics view ──────────────────────────────────────────────────

const analytics = {
  active: false,
  files: [],  // [{name, columns, per_bar, rows}]
  selFile: null,
  selCol: null,
  scatter: null,
  deciles: null,
  histChart: null,
  ablationChart: null,
  loadSeq: 0,
  init() {
    if (this._inited) return;
    this._inited = true;
    // Restore persisted selection.
    try {
      this.selFile = localStorage.getItem('an.selFile') || null;
      this.selCol = localStorage.getItem('an.selCol') || null;
    } catch (_) {}
    this.$file = document.getElementById('an-file');
    this.$col = document.getElementById('an-col');
    this.$file.addEventListener('change', () => {
      this.selFile = this.$file.value;
      this.persist();
      this.populateCols();
      this.loadCurrent();
    });
    this.$col.addEventListener('change', () => {
      this.selCol = this.$col.value;
      this.persist();
      this.loadCurrent();
    });
  },
  persist() {
    try {
      if (this.selFile) localStorage.setItem('an.selFile', this.selFile);
      if (this.selCol) localStorage.setItem('an.selCol', this.selCol);
    } catch (_) {}
  },
  async refreshFiles() {
    // Per-session files only. Reuse the /api/features response.
    const all = await fetchJSON('/api/features');
    this.files = all.filter((f) => !f.per_bar && !f.error && (f.columns || []).length);
    // Default: prefer fh_return/fh_return if present and nothing persisted.
    if (!this.selFile || !this.files.some((f) => f.name === this.selFile)) {
      const fh = this.files.find((f) => f.name === 'fh_return.parquet');
      this.selFile = fh ? fh.name : (this.files[0] ? this.files[0].name : null);
      this.selCol = null;
    }
    this.$file.innerHTML = this.files
      .map((f) => `<option value="${f.name}">${f.name}</option>`)
      .join('');
    if (this.selFile) this.$file.value = this.selFile;
    this.populateCols();
  },
  populateCols() {
    const f = this.files.find((x) => x.name === this.selFile);
    const cols = f ? (f.columns || []) : [];
    this.$col.innerHTML = cols.map((c) => `<option value="${c}">${c}</option>`).join('');
    if (!this.selCol || !cols.includes(this.selCol)) {
      // Prefer fh_return col, else first numeric.
      this.selCol = cols.includes('fh_return') ? 'fh_return' : (cols[0] || null);
    }
    if (this.selCol) this.$col.value = this.selCol;
    this.persist();
  },
  async loadCurrent() {
    if (!this.selFile || !this.selCol) return;
    const seq = ++this.loadSeq;
    try {
      const q = `feature_file=${encodeURIComponent(this.selFile)}&feature_col=${encodeURIComponent(this.selCol)}`;
      const d = await fetchJSON(`/api/analytics?${q}`);
      if (seq !== this.loadSeq) return;  // stale
      this.render(d);
    } catch (e) {
      if (seq !== this.loadSeq) return;
      document.getElementById('an-meta').innerHTML =
        `<span class="chip neg">analytics error: ${escapeHtml(e.message)}</span>`;
      this.renderEmpty();
    }
  },
  renderEmpty() {
    if (this.scatter) { this.scatter.destroy(); this.scatter = null; }
    if (this.deciles) { this.deciles.destroy(); this.deciles = null; }
    if (this.histChart) { this.histChart.destroy(); this.histChart = null; }
    if (this.ablationChart) { this.ablationChart.destroy(); this.ablationChart = null; }
    document.getElementById('an-scatter-stats').textContent = '';
    document.getElementById('an-metrics').innerHTML = '';
    document.getElementById('an-importance').innerHTML = '';
    document.getElementById('an-hist-stats').textContent = '';
    document.getElementById('an-ablation-stats').textContent = '';
  },
  render(d) {
    const fmt = (v, nd = 4) => (v === null || v === undefined || Number.isNaN(v))
      ? '—' : Number(v).toFixed(nd);
    // Header stats + card header.
    const rStr = fmt(d.pearson_r, 3);
    const pStr = d.pearson_p === null || d.pearson_p === undefined
      ? '—' : (d.pearson_p < 1e-4 ? d.pearson_p.toExponential(2) : d.pearson_p.toFixed(4));
    document.getElementById('an-meta').innerHTML =
      `<span class="chip">n=${d.n}</span>` +
      `<span class="chip">r=${rStr}</span>` +
      `<span class="chip">p=${pStr}</span>`;
    document.getElementById('an-scatter-stats').textContent =
      `Pearson r = ${rStr}   p = ${pStr}   n = ${d.n}`;

    // Regression line: simple OLS on the returned points.
    const pts = (d.points || []).filter((p) => p.x !== null && p.y !== null);
    let lineData = [];
    if (pts.length >= 2) {
      let sx = 0, sy = 0, sxx = 0, sxy = 0;
      for (const p of pts) { sx += p.x; sy += p.y; sxx += p.x * p.x; sxy += p.x * p.y; }
      const nP = pts.length;
      const denom = nP * sxx - sx * sx;
      if (Math.abs(denom) > 1e-18) {
        const slope = (nP * sxy - sx * sy) / denom;
        const intercept = (sy - slope * sx) / nP;
        let xmin = Infinity, xmax = -Infinity;
        for (const p of pts) { if (p.x < xmin) xmin = p.x; if (p.x > xmax) xmax = p.x; }
        lineData = [
          { x: xmin, y: intercept + slope * xmin },
          { x: xmax, y: intercept + slope * xmax },
        ];
      }
    }

    // Scatter chart
    const scatterCfg = {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'sessions',
            data: pts.map((p) => ({ x: p.x, y: p.y })),
            backgroundColor: 'rgba(77, 163, 255, 0.55)',
            borderColor: 'rgba(77, 163, 255, 0.9)',
            pointRadius: 2.5,
            pointHoverRadius: 4,
          },
          {
            label: 'OLS fit',
            type: 'line',
            data: lineData,
            borderColor: '#f59e0b',
            backgroundColor: '#f59e0b',
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
            showLine: true,
            order: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        parsing: false,
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: d.feature, color: '#7a8699' },
            ticks: { color: '#7a8699' },
            grid: { color: 'rgba(31, 42, 58, 0.6)' },
          },
          y: {
            title: { display: true, text: 'target_return', color: '#7a8699' },
            ticks: { color: '#7a8699' },
            grid: { color: 'rgba(31, 42, 58, 0.6)' },
          },
        },
        plugins: {
          legend: { labels: { color: '#d7e0ea', boxWidth: 10, boxHeight: 10 } },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const { x, y } = ctx.parsed;
                return `${d.feature}=${x.toFixed(4)}  target=${y.toFixed(4)}`;
              },
            },
          },
        },
      },
    };
    const scCanvas = document.getElementById('an-scatter');
    if (this.scatter) this.scatter.destroy();
    this.scatter = new Chart(scCanvas, scatterCfg);

    // Deciles chart: bar with manually-drawn vertical error bars via a custom plugin.
    // Avoid adding yet another CDN plugin — the drawn path is more robust.
    const decBins = d.deciles || [];
    const errorBarPlugin = {
      id: 'errBars',
      afterDatasetsDraw(chart) {
        const meta = chart.getDatasetMeta(0);
        if (!meta || !meta.data) return;
        const { ctx, scales } = chart;
        ctx.save();
        ctx.strokeStyle = 'rgba(215, 224, 234, 0.7)';
        ctx.lineWidth = 1;
        meta.data.forEach((bar, i) => {
          const b = decBins[i];
          if (!b || b.ci95 == null) return;
          const x = bar.x;
          const yTop = scales.y.getPixelForValue(b.mean_y + b.ci95);
          const yBot = scales.y.getPixelForValue(b.mean_y - b.ci95);
          const cap = 4;
          ctx.beginPath();
          ctx.moveTo(x, yTop); ctx.lineTo(x, yBot);
          ctx.moveTo(x - cap, yTop); ctx.lineTo(x + cap, yTop);
          ctx.moveTo(x - cap, yBot); ctx.lineTo(x + cap, yBot);
          ctx.stroke();
        });
        ctx.restore();
      },
    };
    const decCfg = {
      type: 'bar',
      data: {
        labels: decBins.map((b) => String(b.bin)),
        datasets: [{
          label: 'mean target_return',
          data: decBins.map((b) => b.mean_y),
          backgroundColor: decBins.map((b) => (b.mean_y >= 0 ? 'rgba(38, 166, 154, 0.6)' : 'rgba(239, 83, 80, 0.6)')),
          borderColor: decBins.map((b) => (b.mean_y >= 0 ? '#26a69a' : '#ef5350')),
          borderWidth: 1,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: {
            title: { display: true, text: 'decile', color: '#7a8699' },
            ticks: { color: '#7a8699' },
            grid: { display: false },
          },
          y: {
            title: { display: true, text: 'mean target_return', color: '#7a8699' },
            ticks: { color: '#7a8699' },
            grid: { color: 'rgba(31, 42, 58, 0.6)' },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const b = decBins[ctx.dataIndex];
                const parts = [
                  `mean = ${b.mean_y == null ? '—' : b.mean_y.toFixed(4)}`,
                  `ci95 = ±${b.ci95 == null ? '—' : b.ci95.toFixed(4)}`,
                  `x_mid = ${b.x_mid == null ? '—' : b.x_mid.toFixed(4)}`,
                  `n = ${b.n}`,
                ];
                return parts;
              },
            },
          },
        },
      },
      plugins: [errorBarPlugin],
    };
    const decCanvas = document.getElementById('an-deciles');
    if (this.deciles) this.deciles.destroy();
    this.deciles = new Chart(decCanvas, decCfg);

    // Metrics table
    const metricsRows = [
      ['Pearson r', fmt(d.pearson_r, 4)],
      ['p-value', pStr],
      ['Spearman ρ', fmt(d.spearman_r, 4)],
      ['n', String(d.n)],
    ];
    document.getElementById('an-metrics').innerHTML =
      `<thead><tr><th>metric</th><th class="num">value</th></tr></thead><tbody>` +
      metricsRows.map(([k, v]) => `<tr><td>${escapeHtml(k)}</td><td class="num">${escapeHtml(v)}</td></tr>`).join('') +
      `</tbody>`;

    // Importance table
    const imp = d.importance || [];
    if (!imp.length) {
      document.getElementById('an-importance').innerHTML =
        `<tbody><tr><td class="muted"><em>no models/feature_importance.csv yet</em></td></tr></tbody>`;
    } else {
      document.getElementById('an-importance').innerHTML =
        `<thead><tr><th>feature</th><th class="num">gain</th></tr></thead><tbody>` +
        imp.map((r) => `<tr><td>${escapeHtml(r.feature)}</td><td class="num">${fmt(r.gain, 6)}</td></tr>`).join('') +
        `</tbody>`;
    }

    // Stability table (Gate 3: wasserstein_distance(train, split) / std_train ≤ 0.5)
    const stabEl = document.getElementById('an-stability');
    const stabStats = document.getElementById('an-stability-stats');
    const stab = d.stability;
    if (!stab) {
      if (stabEl) stabEl.innerHTML =
        `<tbody><tr><td class="muted"><em>needs feature values across all three splits</em></td></tr></tbody>`;
      if (stabStats) { stabStats.textContent = ''; stabStats.className = 'card-stats muted'; }
    } else {
      const fmtWd = (v) => {
        if (v === null || v === undefined || Number.isNaN(v)) return '—';
        const a = Math.abs(v);
        if (a > 0 && a < 1e-3) return v.toExponential(2);
        return Number(v).toFixed(4);
      };
      const fmtRatio = (v) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toFixed(3);
      const rows = (stab.splits || []).map((r) => {
        const cls = r.pass ? '' : ' class="neg"';
        const mark = r.pass ? '✓' : '✗';
        return `<tr${cls}><td>${escapeHtml(r.split)}</td><td class="num">${fmtWd(r.wd)}</td><td class="num">${fmtRatio(r.ratio)}</td><td>${mark}</td></tr>`;
      }).join('');
      const thr = (stab.threshold_ratio === null || stab.threshold_ratio === undefined)
        ? '0.50' : Number(stab.threshold_ratio).toFixed(2);
      const overallCls = stab.pass ? ' class="pos"' : ' class="neg"';
      const overallMark = stab.pass ? '✓' : '✗';
      const footer = `<tr${overallCls}><td colspan="3"><em>overall · threshold: ratio ≤ ${thr}</em></td><td>${overallMark}</td></tr>`;
      const body = rows || `<tr><td colspan="4" class="muted"><em>no test split coverage</em></td></tr>`;
      stabEl.innerHTML =
        `<thead><tr><th>split</th><th class="num">Wasserstein</th><th class="num">WD / std</th><th>pass</th></tr></thead>` +
        `<tbody>${body}${footer}</tbody>`;
      if (stabStats) {
        stabStats.textContent = `std_train = ${fmtWd(stab.std_train)}`;
        stabStats.className = 'card-stats ' + (stab.pass ? 'pos' : 'neg');
      }
    }

    this.renderHist(d.hist);
    this.renderAblation(d.ablation);
  },
  renderHist(hist) {
    const canvas = document.getElementById('an-hist');
    const stats = document.getElementById('an-hist-stats');
    if (this.histChart) { this.histChart.destroy(); this.histChart = null; }
    if (!hist || !hist.bin_edges || !hist.train) {
      stats.textContent = 'no research sidecar — run the notebook to populate';
      stats.className = 'card-stats muted';
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    const edges = hist.bin_edges;
    const centers = edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);
    const mkSet = (label, data, color) => ({
      label, data, borderColor: color,
      backgroundColor: color.replace('1)', '0.15)'),
      borderWidth: 2, pointRadius: 0, tension: 0.25, fill: true,
    });
    this.histChart = new Chart(canvas, {
      type: 'line',
      data: {
        labels: centers.map((c) => c.toPrecision(3)),
        datasets: [
          mkSet('train',   hist.train,   'rgba(38, 166, 154, 1)'),
          mkSet('public',  hist.public,  'rgba(245, 158, 11, 1)'),
          mkSet('private', hist.private, 'rgba(168, 85, 247, 1)'),
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: {
          x: { title: { display: true, text: 'feature value', color: '#7a8699' },
               ticks: { color: '#7a8699', maxTicksLimit: 10 }, grid: { display: false } },
          y: { title: { display: true, text: 'density', color: '#7a8699' },
               ticks: { color: '#7a8699' }, grid: { color: 'rgba(31, 42, 58, 0.6)' } },
        },
        plugins: { legend: { labels: { color: '#d7e0ea', boxWidth: 10, boxHeight: 10 } } },
      },
    });
    stats.textContent = `${edges.length - 1} bins · visual check for regime drift`;
    stats.className = 'card-stats muted';
  },
  renderAblation(ab) {
    const canvas = document.getElementById('an-ablation');
    const stats = document.getElementById('an-ablation-stats');
    if (this.ablationChart) { this.ablationChart.destroy(); this.ablationChart = null; }
    if (!ab || !ab.per_fold_sharpe_without || !ab.per_fold_sharpe_with) {
      stats.textContent = 'no ablation data — run scripts/rskew_ablation.py';
      stats.className = 'card-stats muted';
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    const folds = ab.per_fold_sharpe_without.map((_, i) => `fold ${i + 1}`);
    const pass = !!ab.gate2;
    this.ablationChart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: folds,
        datasets: [
          { label: 'without',
            data: ab.per_fold_sharpe_without,
            backgroundColor: 'rgba(122, 134, 153, 0.6)',
            borderColor: '#7a8699', borderWidth: 1 },
          { label: 'with rskew',
            data: ab.per_fold_sharpe_with,
            backgroundColor: 'rgba(77, 163, 255, 0.6)',
            borderColor: '#4da3ff', borderWidth: 1 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: {
          x: { ticks: { color: '#7a8699' }, grid: { display: false } },
          y: { title: { display: true, text: 'fold Sharpe', color: '#7a8699' },
               ticks: { color: '#7a8699' }, grid: { color: 'rgba(31, 42, 58, 0.6)' } },
        },
        plugins: {
          legend: { labels: { color: '#d7e0ea', boxWidth: 10, boxHeight: 10 } },
          tooltip: {
            callbacks: {
              afterBody: (items) => {
                const i = items[0]?.dataIndex;
                if (i == null) return '';
                const lift = ab.per_fold_lift ? ab.per_fold_lift[i] : null;
                return lift == null ? '' : `lift = ${lift >= 0 ? '+' : ''}${lift.toFixed(3)}`;
              },
            },
          },
        },
      },
    });
    const liftStr = (ab.lift_mean >= 0 ? '+' : '') + Number(ab.lift_mean).toFixed(3);
    const winN = Math.round((ab.win_rate || 0) * (ab.n_splits || folds.length));
    const mark = pass ? '✓' : '✗';
    stats.textContent = `lift = ${liftStr}   ·   wins ${winN}/${ab.n_splits || folds.length}   ·   gate2 ${mark}`;
    stats.className = 'card-stats ' + (pass ? 'pos' : 'neg');
  },
};

function setView(view) {
  const isAn = view === 'analytics';
  analytics.active = isAn;
  document.getElementById('view-session').classList.toggle('active', !isAn);
  document.getElementById('view-analytics').classList.toggle('active', isAn);
  document.getElementById('main-session').style.display = isAn ? 'none' : '';
  document.getElementById('main-analytics').style.display = isAn ? '' : 'none';
  document.getElementById('session-controls').style.display = isAn ? 'none' : '';
  document.getElementById('analytics-controls').style.display = isAn ? '' : 'none';
  if (isAn) {
    // Ensure init runs before refreshFiles touches this.$file / this.$col
    // (button click can race ahead of the main init IIFE's awaits).
    analytics.init();
    analytics.refreshFiles().then(() => analytics.loadCurrent());
  }
}

document.getElementById('view-session').addEventListener('click', () => setView('session'));
document.getElementById('view-analytics').addEventListener('click', () => setView('analytics'));

// ── init ────────────────────────────────────────────────────────────

(async function init() {
  readHash();
  // Wire analytics controls synchronously, before any awaits — otherwise a
  // fast click on the "Analytics" tab can race past the awaits below and
  // hit refreshFiles() while this.$file is still undefined.
  analytics.init();
  await loadSplits();
  $split.value = state.split;
  await loadSessions();
  await loadSession();
  await refreshFeatures();
  connectSSE();
})();
