/* Shared contract state.

   Every view reads the same contract, edited in one control surface in the left
   rail rather than per-card. Changing an input re-renders whichever view is
   active; inactive views are marked dirty and re-render when opened, so a change
   never costs work in a view nobody is looking at.
*/

const listeners = new Set();

export const state = {
  S: 100,
  K: 100,
  T: 1,
  sigma: 0.25,
  r: 0.043,
  q: 0.01,
  is_call: true,
  american: false,
  steps: 500,
  paths: 200000,
  lattice_method: 'crr',
};

/** The request body shared by most endpoints. */
export function contract(extra = {}) {
  return {
    S: state.S,
    K: state.K,
    T: state.T,
    r: state.r,
    q: state.q,
    sigma: state.sigma,
    is_call: state.is_call,
    ...extra,
  };
}

export function numerics(extra = {}) {
  return {
    american: state.american,
    steps: state.steps,
    paths: state.paths,
    lattice_method: state.lattice_method,
    ...extra,
  };
}

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function update(patch, origin = 'input') {
  let changed = false;
  for (const [k, v] of Object.entries(patch)) {
    if (state[k] !== v) { state[k] = v; changed = true; }
  }
  if (changed) for (const fn of listeners) fn(state, origin);
  return changed;
}

/** A short signature of the state, for cache keys. */
export function signature() {
  return JSON.stringify([state.S, state.K, state.T, state.sigma, state.r, state.q,
    state.is_call, state.american, state.steps, state.paths, state.lattice_method]);
}

export const SCENARIOS = [
  { label: 'ATM 1y', patch: { S: 100, K: 100, T: 1, sigma: 0.25, r: 0.043, q: 0.01 } },
  { label: 'Deep OTM call', patch: { S: 100, K: 145, T: 0.5, sigma: 0.3, is_call: true } },
  { label: 'Near expiry', patch: { S: 100, K: 100, T: 3 / 365, sigma: 0.45 } },
  { label: 'High vol', patch: { S: 100, K: 100, T: 1, sigma: 0.85 } },
  { label: 'Low vol', patch: { S: 100, K: 100, T: 1, sigma: 0.08 } },
  { label: 'Deep ITM put', patch: { S: 100, K: 135, T: 1, sigma: 0.22, is_call: false } },
  { label: 'American put', patch: { S: 100, K: 110, T: 1, sigma: 0.3, r: 0.06, q: 0, is_call: false, american: true } },
  { label: 'LEAPS', patch: { S: 100, K: 120, T: 2.5, sigma: 0.28 } },
];
