/* Pricer: hero premium, full Greek tiles, and a side-by-side engine comparison. */

import { el, clear, buildTable, fmt, fmtSig, fmtMoney, fmtSigned, fmtMs, fmtInt, fmtPct,
         seriesColor, token } from '../util.js';
import { api } from '../api.js';
import { state, contract, numerics, signature } from '../state.js';
import { SmallMultiples, withTableToggle } from '../chart.js';

const FIRST_ORDER = ['delta', 'gamma', 'vega', 'theta', 'rho', 'epsilon'];
const HIGHER_ORDER = ['vanna', 'volga', 'charm', 'speed', 'zomma', 'color', 'dual_delta', 'dual_gamma'];

const GREEK_BLURB = {
  delta: 'Hedge ratio: shares of underlying per option.',
  gamma: 'How fast delta moves — the cost of rehedging.',
  vega: 'Exposure to a change in implied volatility.',
  theta: 'Value lost to the passage of one day.',
  rho: 'Exposure to the risk-free rate.',
  epsilon: 'Exposure to the dividend yield.',
  vanna: 'Delta drift as volatility moves; skew exposure.',
  volga: 'Vega convexity; the value of a volatility smile.',
  charm: 'Delta drift as time passes — weekend hedge slippage.',
  speed: 'Third-order spot sensitivity of gamma.',
  zomma: 'Gamma sensitivity to volatility.',
  color: 'Gamma decay per day.',
  dual_delta: 'Sensitivity to the strike; the risk-neutral CDF.',
  dual_gamma: 'Risk-neutral density at the strike.',
};

export class PricerView {
  constructor(host, ctx) {
    this.host = host;
    this.ctx = ctx;
    this.dirty = true;
    this.profileAxis = 'spot';
    this.showHigherOrder = false;
    this._build();
  }

  _build() {
    clear(this.host);

    this.notice = el('div');

    this.heroValue = el('div', { class: 'hero-value' }, '—');
    this.heroSub = el('div', { class: 'hero-sub' });
    this.heroLabel = el('div', { class: 'hero-label' }, 'Premium');
    this.heroExtra = el('div', { class: 'tiles', style: { marginTop: '14px' } });

    const heroCard = el('div', { class: 'card' },
      this.heroLabel, this.heroValue, this.heroSub, this.heroExtra);

    this.modelTools = el('div', { class: 'card-tools' });
    this.modelTable = el('div');
    const modelCard = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          el('h3', {}, 'Engine comparison'),
          el('p', {}, 'Same contract through every engine. Error is measured against the reference '
            + 'price for this exercise style, in basis points of premium.')),
        this.modelTools),
      this.modelTable);

    this.greekTiles = el('div', { class: 'tiles' });
    this.higherToggle = el('button', {
      class: 'btn btn-ghost btn-sm', type: 'button',
      onclick: () => { this.showHigherOrder = !this.showHigherOrder; this._renderGreeks(); },
    }, 'Higher order');
    const greekCard = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          el('h3', {}, 'Analytic Greeks'),
          el('p', {}, 'Closed-form Black-Scholes-Merton, in trading units.')),
        el('div', { class: 'card-tools' }, this.higherToggle)),
      this.greekTiles);

    this.profileHost = el('div');
    this.axisSeg = el('div', { class: 'segmented' },
      ...['spot', 'tenor', 'vol'].map((a) => el('button', {
        type: 'button',
        class: a === 'spot' ? 'is-active' : '',
        onclick: () => {
          this.profileAxis = a;
          [...this.axisSeg.children].forEach((b) => b.classList.toggle('is-active', b.textContent === a));
          this._loadProfile();
        },
      }, a)));
    this.profileTools = el('div', { class: 'card-tools' }, this.axisSeg);
    const profileCard = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          el('h3', {}, 'Greek profiles'),
          this.profileNote = el('p', {}, 'How each Greek moves along one axis, with the others held fixed.')),
        this.profileTools),
      this.profileHost,
      el('p', { class: 'axis-note' },
        'Faceted rather than overlaid, with one y-axis each: delta spans 0 to 1 while gamma is order '
        + '0.01, so a shared axis would flatten gamma into the baseline and a second axis would imply '
        + 'a relationship that is not in the data. The x-axis is shared, so the shapes line up.'));

    this.host.append(
      this.notice,
      el('div', { class: 'grid grid-hero' }, heroCard, modelCard),
      greekCard,
      profileCard,
    );

    this.profileChart = new SmallMultiples(this.profileHost, { height: 176, columns: 2 });
    this.profileTable = withTableToggle(this.profileChart, this.profileTools);
    profileCard.append(this.profileTable.holder);
  }

  async refresh() {
    const sig = signature();
    this.dirty = false;
    try {
      const [price, profile] = await Promise.all([
        api.price({ ...contract(), ...numerics() }),
        this._profileRequest(),
      ]);
      if (sig !== signature()) return;   // superseded by a newer edit
      this.data = price;
      this._renderHero();
      this._renderModels();
      this._renderGreeks();
      this._renderProfile(profile);
    } finally {
      this.profileHost.classList.remove('is-stale');
    }
  }

  _profileRequest() {
    const c = contract();
    const body = { ...c, axis: this.profileAxis, greeks: ['delta', 'gamma', 'vega', 'theta'] };
    return api.greekProfile(body);
  }

  async _loadProfile() {
    try {
      this.profileHost.classList.add('is-stale');
      this._renderProfile(await this._profileRequest());
    } catch (e) {
      this.ctx.reportError(e, 'greek profile');
    } finally {
      this.profileHost.classList.remove('is-stale');
    }
  }

  _renderHero() {
    const d = this.data;
    const bsModel = d.models[0];
    const headline = state.american
      ? d.models.find((m) => m.key.startsWith('binomial')) || bsModel
      : bsModel;

    this.heroLabel.textContent = state.american
      ? `${state.is_call ? 'Call' : 'Put'} premium · American`
      : `${state.is_call ? 'Call' : 'Put'} premium · European`;
    this.heroValue.textContent = fmtMoney(headline.price, 4);

    clear(this.heroSub).append(
      el('span', {}, `${state.is_call ? 'Call' : 'Put'} · K ${fmtSig(state.K, 6)} · `
        + `${(state.T * 365).toFixed(0)}d · σ ${fmtPct(state.sigma, 1)}`),
    );

    const tiles = [
      { label: 'Intrinsic', value: fmtMoney(d.intrinsic, 2), unit: 'exercise value now' },
      { label: 'Time value', value: fmtMoney(d.time_value, 4), unit: 'premium above intrinsic' },
      { label: 'Forward', value: fmtSig(d.forward, 6), unit: `S·e^((r−q)T)` },
      { label: 'Log-moneyness', value: fmtSigned(d.moneyness, 4), unit: 'ln(K/F)' },
    ];
    if (state.american) {
      const eep = (d.models.find((m) => m.key === 'binomial_smooth') || {}).early_exercise_premium;
      tiles.push({ label: 'Early exercise', value: fmtMoney(eep, 4), unit: 'premium over European' });
    } else {
      tiles.push({ label: 'Put-call parity', value: fmtSig(d.parity_gap, 2), unit: 'residual — should be ~0' });
    }

    clear(this.heroExtra).append(...tiles.map((t) => el('div', { class: 'tile' },
      el('div', { class: 'tile-label' }, t.label),
      el('div', { class: 'tile-value' }, t.value),
      el('div', { class: 'tile-unit' }, t.unit))));

    clear(this.notice);
    if (state.american) {
      this.notice.append(el('div', { class: 'notice info' }, el('div', {},
        el('strong', {}, 'American exercise. '),
        'Black-Scholes is shown for reference only — it prices European exercise, so the gap to the '
        + 'lattice price is the early-exercise premium, not an error. The reference price is '
        + `${d.reference.label}.`)));
    }
    if (state.T * 365 < 1.5) {
      this.notice.append(el('div', { class: 'notice warn' }, el('div', {},
        el('strong', {}, 'Very short expiry. '),
        'Gamma and theta diverge as T→0 and Monte Carlo standard errors widen relative to the '
        + 'premium. Treat the higher-order Greeks as indicative here.')));
    }
  }

  _renderModels() {
    const d = this.data;
    const columns = [
      {
        label: 'Engine',
        mono: false,
        get: (m) => el('span', {},
          el('span', { class: 'swatch', style: { background: this._familyColor(m.family) } }),
          m.label),
      },
      { label: 'Price', get: (m) => fmtSig(m.price, 8) },
      {
        label: 'Error (bp)',
        get: (m) => (m.error_bp === null || m.error_bp === undefined ? '—' : fmtSigned(m.error_bp, 3)),
        cls: (m) => (m.error_bp === null || m.error_bp === undefined ? 'dim'
          : Math.abs(m.error_bp) < 1 ? 'pos' : Math.abs(m.error_bp) > 50 ? 'neg' : ''),
      },
      { label: '95% interval', get: (m) => (m.ci_low === undefined ? '—' : `${fmtSig(m.ci_low, 6)} … ${fmtSig(m.ci_high, 6)}`) },
      { label: 'Resolution', get: (m) => (m.steps ? `${fmtInt(m.steps)} steps` : m.paths ? `${fmtCompactSafe(m.paths)} paths` : 'exact') },
      { label: 'Time', get: (m) => fmtMs(m.ms) },
      {
        label: 'Notes',
        mono: false,
        get: (m) => {
          const bits = [];
          if (m.efficiency && Number.isFinite(m.efficiency)) {
            bits.push(el('span', { class: 'tag good' }, `${fmtSig(m.efficiency, 3)}× efficiency`));
          }
          if (m.exact) bits.push(el('span', { class: 'tag neutral' }, 'exact'));
          if (m.note) bits.push(el('span', { class: 'dim' }, m.note));
          return bits.length ? el('span', { class: 'row row-tight' }, ...bits) : '—';
        },
      },
    ];
    clear(this.modelTable).append(
      buildTable(columns, d.models),
      el('p', { class: 'axis-note' },
        `Reference: ${d.reference.label} = ${fmtSig(d.reference.price, 8)}. `
        + 'Efficiency compares the variance of the estimator against plain Monte Carlo using the '
        + 'same number of payoff evaluations.'),
    );
  }

  _familyColor(family) {
    return { analytic: seriesColor(0), lattice: seriesColor(1), 'monte-carlo': seriesColor(2) }[family] || seriesColor(4);
  }

  _renderGreeks() {
    const d = this.data;
    if (!d) return;
    this.higherToggle.classList.toggle('is-active', this.showHigherOrder);
    const names = this.showHigherOrder ? [...FIRST_ORDER, ...HIGHER_ORDER] : FIRST_ORDER;
    const disp = d.greeks_display || {};
    const units = d.greek_units || {};

    clear(this.greekTiles).append(...names.map((name) => {
      const raw = d.greeks[name];
      const shown = disp[name] !== undefined ? disp[name] : raw;
      const cls = shown < 0 ? 'tile is-neg' : shown > 0 ? 'tile is-pos' : 'tile';
      return el('div', { class: cls, title: GREEK_BLURB[name] || '' },
        el('div', { class: 'tile-label' }, name.replace('_', ' ')),
        el('div', { class: 'tile-value' }, fmtSig(shown, 5)),
        el('div', { class: 'tile-unit' }, units[name] || ''));
    }));
  }

  _renderProfile(p) {
    const axisLabel = { spot: 'spot', tenor: 'years to expiry', vol: 'volatility' }[p.axis];
    const markers = [];
    if (p.axis === 'spot') {
      markers.push({ x: state.K, label: `K`, color: token('--border-strong') });
      if (Math.abs(state.S - state.K) / Math.max(state.S, 1) > 0.02) {
        markers.push({ x: state.S, label: `S`, color: token('--text-muted') });
      }
    }
    this.profileChart.setData({
      x: p.x,
      panels: p.series.map((s, i) => ({
        key: s.key,
        title: s.label,
        subtitle: s.unit,
        values: s.values,
        color: seriesColor(i),
        area: true,
        fmt: (v) => fmtSig(v, 5),
      })),
      markers,
      xName: axisLabel,
      xTitle: (v) => `${axisLabel} ${fmtSig(v, 5)}`,
    });
    this.profileTable.refresh();
    this.profileNote.textContent = p.axis === 'vol'
      ? 'Volatility sweep at fixed spot and tenor. Vega peaks near the money and collapses in the wings.'
      : p.axis === 'tenor'
        ? 'Tenor sweep. Gamma and theta blow up as expiry approaches; that divergence is real, not numerical.'
        : 'Spot sweep at fixed tenor and volatility. Strike and spot are marked.';
  }

  redraw() {
    if (this.profileChart) this.profileChart.draw();
    if (this.data) { this._renderModels(); this._renderGreeks(); }
  }
}

function fmtCompactSafe(v) {
  if (!Number.isFinite(v)) return '—';
  return Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(v);
}
