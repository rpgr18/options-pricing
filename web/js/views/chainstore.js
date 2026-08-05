/* The loaded option chain, shared between the Chain and Vol Surface views.

   The chain lives here rather than in either view because both need it and
   neither owns it: the Chain view loads and inspects it, the Surface view
   calibrates to it. The server keeps the full payload cached under `chain_id`,
   so refitting a surface with different filters never re-hits the data source.
*/

let current = null;
const listeners = new Set();

export function getChain() { return current; }

export function setChain(payload) {
  current = payload;
  for (const fn of listeners) fn(payload);
}

export function onChain(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
