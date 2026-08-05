/* Thin JSON API client with in-flight de-duplication and readable errors. */

import { $ } from './util.js';

let pending = 0;
const inflight = new Map();

function setStatus(state, text) {
  const chip = $('#status-chip');
  const label = $('#status-text');
  if (!chip || !label) return;
  chip.dataset.state = state;
  label.textContent = text;
}

export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

function busy(delta, label) {
  pending += delta;
  if (pending > 0) setStatus('busy', label || 'computing…');
  else setStatus('ok', 'ready');
}

/**
 * POST a JSON body to an endpoint.
 * Identical concurrent requests share one round trip, which matters because
 * several views react to the same contract-input change at once.
 */
export async function post(path, body = {}, opts = {}) {
  const key = path + JSON.stringify(body);
  if (inflight.has(key)) return inflight.get(key);

  const label = opts.label || 'computing…';
  const run = (async () => {
    busy(1, label);
    try {
      const res = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      let payload;
      const text = await res.text();
      try {
        payload = text ? JSON.parse(text) : {};
      } catch {
        throw new ApiError(`${path} returned a non-JSON response (HTTP ${res.status})`, res.status, text.slice(0, 400));
      }
      if (!res.ok) {
        throw new ApiError(payload.error || `HTTP ${res.status}`, res.status, payload.detail);
      }
      return payload;
    } catch (err) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(
        'Could not reach the pricing server. Is it still running?',
        0,
        err && err.message ? err.message : String(err),
      );
    } finally {
      busy(-1);
      inflight.delete(key);
    }
  })();

  inflight.set(key, run);
  return run;
}

export const api = {
  health: () => post('/api/health'),
  price: (b) => post('/api/price', b, { label: 'pricing…' }),
  greekSurface: (b) => post('/api/greek-surface', b, { label: 'building surface…' }),
  greekProfile: (b) => post('/api/greek-profile', b, { label: 'profiling…' }),
  impliedVol: (b) => post('/api/implied-vol', b, { label: 'inverting…' }),
  convergence: (b) => post('/api/convergence', b, { label: 'running convergence study…' }),
  chain: (b) => post('/api/chain', b, { label: 'loading chain…' }),
  surface: (b) => post('/api/surface', b, { label: 'calibrating surface…' }),
  surfaceCompare: (b) => post('/api/surface-compare', b, { label: 'comparing interpolators…' }),
  strategy: (b) => post('/api/strategy', b, { label: 'valuing strategy…' }),
  exerciseBoundary: (b) => post('/api/exercise-boundary', b, { label: 'solving boundary…' }),
};

export { setStatus };
