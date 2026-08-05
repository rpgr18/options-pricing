/* Formatting, DOM helpers and theme-token access. */

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else if (k === 'dataset') for (const [dk, dv] of Object.entries(v)) node.dataset[dk] = dv;
    else node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat(4)) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

/**
 * Append children, dropping the empty ones.
 *
 * Native `append(null)` inserts the literal text "null", so any conditional
 * child (`cond ? node : null`) has to be filtered before it reaches the DOM.
 * `el()` already does this for its own children; use this for the cases that
 * append to an existing node.
 */
export function mount(parent, ...children) {
  for (const c of children.flat(4)) {
    if (c === null || c === undefined || c === false) continue;
    parent.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return parent;
}

/* ---- theme tokens ---------------------------------------------------- */

export function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* Categorical hues are assigned in this fixed order and never cycled. Past the
   fifth series a chart must fold the tail into "other" or facet -- see
   SERIES_LIMIT below. */
export const SERIES_VARS = ['--series-1', '--series-2', '--series-3', '--series-4', '--series-5'];
export const SERIES_LIMIT = SERIES_VARS.length;

export function seriesColor(i) {
  return token(SERIES_VARS[i % SERIES_VARS.length]);
}

export function sequentialRamp() {
  return ['--seq-0', '--seq-1', '--seq-2', '--seq-3', '--seq-4', '--seq-5', '--seq-6', '--seq-7'].map(token);
}

export function divergingRamp() {
  return ['--div-neg-3', '--div-neg-2', '--div-neg-1', '--div-mid', '--div-pos-1', '--div-pos-2', '--div-pos-3'].map(token);
}

function parseHex(hex) {
  const h = hex.replace('#', '').trim();
  const full = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
  return [parseInt(full.slice(0, 2), 16), parseInt(full.slice(2, 4), 16), parseInt(full.slice(4, 6), 16)];
}

/** Sample a ramp at t in [0,1] with linear interpolation between stops. */
export function rampAt(stops, t) {
  if (!Number.isFinite(t)) return stops[0];
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = parseHex(stops[i]);
  const b = parseHex(stops[i + 1]);
  const c = a.map((v, j) => Math.round(v + (b[j] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/* ---- number formatting ----------------------------------------------- */

export function fmt(v, digits = 4) {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a === 0) return (0).toFixed(Math.min(digits, 2));
  if (a < 1e-4 || a >= 1e7) return v.toExponential(2).replace('e', 'e');
  return v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function fmtSig(v, sig = 4) {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-4 || a >= 1e7) return v.toExponential(Math.max(1, sig - 2));
  const digits = Math.max(0, sig - 1 - Math.floor(Math.log10(a)));
  return v.toLocaleString(undefined, { maximumFractionDigits: Math.min(digits, 10) });
}

export function fmtMoney(v, digits = 4) {
  if (!Number.isFinite(v)) return '—';
  return (v < 0 ? '−$' : '$') + Math.abs(v).toLocaleString(undefined, {
    minimumFractionDigits: digits, maximumFractionDigits: digits,
  });
}

export function fmtPct(v, digits = 2) {
  if (!Number.isFinite(v)) return '—';
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmtSigned(v, digits = 4) {
  if (!Number.isFinite(v)) return '—';
  const s = Math.abs(v).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
  return (v < 0 ? '−' : '+') + s;
}

export function fmtMs(v) {
  if (!Number.isFinite(v)) return '—';
  if (v < 1) return `${v.toFixed(2)} ms`;
  if (v < 1000) return `${v.toFixed(0)} ms`;
  return `${(v / 1000).toFixed(2)} s`;
}

export function fmtInt(v) {
  if (!Number.isFinite(v)) return '—';
  return Math.round(v).toLocaleString();
}

export function fmtCompact(v) {
  if (!Number.isFinite(v)) return '—';
  return Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(v);
}

/** Nice tick values covering [lo, hi]. */
export function ticks(lo, hi, count = 6) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi === lo) return [lo];
  const span = hi - lo;
  const raw = span / Math.max(count, 2);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) {
    out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
  }
  return out;
}

/** Decade ticks for a log axis. */
export function logTicks(lo, hi) {
  const out = [];
  const start = Math.floor(Math.log10(lo));
  const end = Math.ceil(Math.log10(hi));
  for (let e = start; e <= end; e += 1) {
    const base = Math.pow(10, e);
    if (base >= lo * 0.999 && base <= hi * 1.001) out.push(base);
  }
  if (out.length < 3) {
    for (let e = start; e <= end; e += 1) {
      for (const m of [2, 5]) {
        const v = m * Math.pow(10, e);
        if (v >= lo && v <= hi) out.push(v);
      }
    }
    out.sort((a, b) => a - b);
  }
  return out;
}

export function fmtAxis(v, span) {
  if (!Number.isFinite(v)) return '';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(0);
  const step = span / 6;
  const decimals = step >= 10 ? 0 : step >= 1 ? 1 : step >= 0.1 ? 2 : step >= 0.01 ? 3 : 4;
  return v.toFixed(decimals);
}

export function debounce(fn, ms = 180) {
  let handle = 0;
  return (...args) => {
    clearTimeout(handle);
    handle = setTimeout(() => fn(...args), ms);
  };
}

export function daysToYears(days) { return days / 365; }
export function yearsToDays(years) { return years * 365; }

/** Build a table element from a header spec and row data. */
export function buildTable(columns, rows, opts = {}) {
  const thead = el('thead', {}, el('tr', {}, columns.map((c) => el('th', { title: c.title || '' }, c.label))));
  const tbody = el('tbody', {}, rows.map((r) => el('tr', { class: r._class || '' },
    columns.map((c) => {
      const raw = typeof c.get === 'function' ? c.get(r) : r[c.key];
      const cls = [c.mono === false ? '' : 'num', typeof c.cls === 'function' ? c.cls(r) : c.cls || ''].filter(Boolean).join(' ');
      return el('td', { class: cls }, raw instanceof Node ? raw : (raw ?? '—'));
    }))));
  const table = el('table', { class: 'data' }, thead, tbody);
  return opts.wrap === false ? table : el('div', { class: 'table-wrap', style: opts.maxHeight ? { maxHeight: opts.maxHeight, overflowY: 'auto' } : {} }, table);
}
