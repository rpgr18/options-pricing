/* Bootstrap: wire the rail inputs to shared state, manage tabs, mount views. */

import { $, $$, el, clear, fmtSig, debounce } from './util.js';
import { api, ApiError, setStatus } from './api.js';
import { state, update, subscribe, SCENARIOS } from './state.js';

import { PricerView } from './views/pricer.js';
import { GreeksView } from './views/greeks.js';
import { SurfaceView } from './views/surface.js';
import { ConvergenceView } from './views/convergence.js';
import { ChainView } from './views/chain.js';
import { StrategyView } from './views/strategy.js';

/* ---------------------------------------------------------------- toasts */

const seenRecently = new Map();

export function toast(message, kind = 'error', detail = null) {
  // Suppress a repeat of the same message inside 4s: a bad input can otherwise
  // fire one toast per view.
  const now = Date.now();
  if (seenRecently.get(message) > now - 4000) return;
  seenRecently.set(message, now);

  const node = el('div', { class: `toast ${kind === 'error' ? '' : kind}` },
    el('div', {},
      el('div', {}, message),
      detail ? el('div', { class: 'dim', style: { marginTop: '4px', fontSize: '11px' } }, detail) : null),
    el('button', { type: 'button', 'aria-label': 'Dismiss', onclick: (e) => e.target.closest('.toast').remove() }, '×'));
  $('#toast-stack').append(node);
  setTimeout(() => node.remove(), kind === 'error' ? 9000 : 5000);
}

export function reportError(err, where) {
  if (err instanceof ApiError) {
    setStatus('err', 'error');
    toast(err.message, 'error', [where, err.detail].filter(Boolean).join(' · '));
  } else {
    setStatus('err', 'error');
    toast(`${where}: ${err && err.message ? err.message : err}`, 'error');
    console.error(where, err);
  }
}

/* ------------------------------------------------------------------ rail */

function bindRail() {
  const pct = (id, key) => {
    const input = $(id);
    input.addEventListener('input', () => {
      const v = Number(input.value);
      if (Number.isFinite(v)) update({ [key]: v / 100 });
    });
    return input;
  };
  const plain = (id, key, transform = (v) => v) => {
    const input = $(id);
    input.addEventListener('input', () => {
      const v = Number(input.value);
      if (Number.isFinite(v)) update({ [key]: transform(v) });
    });
    return input;
  };

  const inS = plain('#in-S', 'S');
  const inK = plain('#in-K', 'K');
  const inSigma = pct('#in-sigma', 'sigma');
  const inR = pct('#in-r', 'r');
  const inQ = pct('#in-q', 'q');
  const inSteps = plain('#in-steps', 'steps');
  const inPaths = plain('#in-paths', 'paths');

  // Days and years are two views of one field; whichever is typed drives T.
  const inDays = $('#in-days');
  const inT = $('#in-T');
  inDays.addEventListener('input', () => {
    const v = Number(inDays.value);
    if (Number.isFinite(v) && v >= 0) update({ T: v / 365 });
  });
  inT.addEventListener('input', () => {
    const v = Number(inT.value);
    if (Number.isFinite(v) && v >= 0) update({ T: v });
  });

  $('#in-lattice-method').addEventListener('change', (e) => update({ lattice_method: e.target.value }));

  $$('[data-group]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const group = btn.dataset.group;
      $$(`[data-group="${group}"]`).forEach((b) => {
        const on = b === btn;
        b.classList.toggle('is-active', on);
        b.setAttribute('aria-checked', String(on));
      });
      if (group === 'type') update({ is_call: btn.dataset.value === 'call' });
      if (group === 'exercise') update({ american: btn.dataset.value === 'american' });
    });
  });

  const chips = $('#scenario-chips');
  SCENARIOS.forEach((sc) => {
    chips.append(el('button', {
      class: 'chip', type: 'button',
      onclick: () => { update(sc.patch, 'scenario'); syncRail(); },
    }, sc.label));
  });

  return { inS, inK, inSigma, inR, inQ, inSteps, inPaths, inDays, inT };
}

let railFields;

/** Push state back into the inputs (after a scenario chip, say). */
function syncRail() {
  const f = railFields;
  const set = (input, v) => { if (document.activeElement !== input) input.value = v; };
  set(f.inS, state.S);
  set(f.inK, state.K);
  set(f.inSigma, +(state.sigma * 100).toFixed(4));
  set(f.inR, +(state.r * 100).toFixed(4));
  set(f.inQ, +(state.q * 100).toFixed(4));
  set(f.inSteps, state.steps);
  set(f.inPaths, state.paths);
  set(f.inDays, +(state.T * 365).toFixed(2));
  set(f.inT, +state.T.toFixed(6));

  $$('[data-group="type"]').forEach((b) => {
    const on = b.dataset.value === (state.is_call ? 'call' : 'put');
    b.classList.toggle('is-active', on);
    b.setAttribute('aria-checked', String(on));
  });
  $$('[data-group="exercise"]').forEach((b) => {
    const on = b.dataset.value === (state.american ? 'american' : 'european');
    b.classList.toggle('is-active', on);
    b.setAttribute('aria-checked', String(on));
  });
  $('#in-lattice-method').value = state.lattice_method;

  const fwd = state.S * Math.exp((state.r - state.q) * state.T);
  const k = fwd > 0 ? Math.log(state.K / fwd) : NaN;
  $('#out-moneyness').textContent = Number.isFinite(k)
    ? `${k >= 0 ? '+' : '−'}${Math.abs(k).toFixed(4)} log`
    : '—';
}

/* ---------------------------------------------------------------- views */

const VIEWS = {
  pricer: PricerView,
  greeks: GreeksView,
  surface: SurfaceView,
  convergence: ConvergenceView,
  chain: ChainView,
  strategy: StrategyView,
};

const instances = new Map();
let activeName = 'pricer';

function ensure(name) {
  if (instances.has(name)) return instances.get(name);
  const host = $(`#view-${name}`);
  const inst = new VIEWS[name](host, { toast, reportError });
  instances.set(name, inst);
  return inst;
}

function activate(name) {
  activeName = name;
  $$('.tabs button').forEach((b) => {
    const on = b.dataset.view === name;
    b.classList.toggle('is-active', on);
    b.setAttribute('aria-selected', String(on));
  });
  $$('.view').forEach((v) => v.classList.toggle('is-active', v.id === `view-${name}`));
  const inst = ensure(name);
  if (inst.dirty !== false) inst.refresh().catch((e) => reportError(e, name));
  if (location.hash.slice(1) !== name) history.replaceState(null, '', `#${name}`);
}

const propagate = debounce(() => {
  syncRail();
  for (const [name, inst] of instances) {
    if (name === activeName) inst.refresh().catch((e) => reportError(e, name));
    else inst.dirty = true;
  }
}, 220);

/* ---------------------------------------------------------------- theme */

function bindTheme() {
  const btn = $('#theme-toggle');
  const label = $('#theme-label');
  const apply = (theme) => {
    document.documentElement.dataset.theme = theme;
    label.textContent = theme === 'dark' ? 'Light' : 'Dark';
    btn.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`);
    try { localStorage.setItem('wb-theme', theme); } catch { /* private mode */ }
    // Charts read their colours from CSS custom properties at draw time, so a
    // theme change needs a redraw, not a re-fetch.
    for (const inst of instances.values()) if (inst.redraw) inst.redraw();
  };
  let initial = 'dark';
  try { initial = localStorage.getItem('wb-theme') || 'dark'; } catch { /* ignore */ }
  apply(initial);
  btn.addEventListener('click', () => apply(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
}

/* ----------------------------------------------------------------- boot */

async function boot() {
  railFields = bindRail();
  bindTheme();
  syncRail();

  $$('.tabs button').forEach((b) => b.addEventListener('click', () => activate(b.dataset.view)));
  subscribe(propagate);

  try {
    const h = await api.health();
    setStatus('ok', `ready · numpy ${h.numpy}`);
  } catch (e) {
    reportError(e, 'health check');
  }

  const initial = location.hash.slice(1);
  activate(VIEWS[initial] ? initial : 'pricer');

  window.addEventListener('hashchange', () => {
    const name = location.hash.slice(1);
    if (VIEWS[name] && name !== activeName) activate(name);
  });
}

boot();
