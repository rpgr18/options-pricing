/* Strategy: multi-leg payoff, mark-to-market P&L at intermediate dates, and
   aggregate position Greeks. */

import { el, clear, buildTable, fmtSig, fmtSigned, fmtMoney, fmtPct, seriesColor, token } from '../util.js';
import { api } from '../api.js';
import { state, signature } from '../state.js';
import { LineChart, withTableToggle } from '../chart.js';

const PRESETS = [
  ['long_call', 'Long call'],
  ['long_put', 'Long put'],
  ['covered_call', 'Covered call'],
  ['bull_call_spread', 'Bull call spread'],
  ['bear_put_spread', 'Bear put spread'],
  ['straddle', 'Long straddle'],
  ['short_strangle', 'Short strangle'],
  ['iron_condor', 'Iron condor'],
  ['butterfly', 'Call butterfly'],
  ['calendar_ratio', 'Call ratio 1x2'],
];

export class StrategyView {
  constructor(host, ctx) {
    this.host = host;
    this.ctx = ctx;
    this.dirty = true;
    this.legs = null;
    this.preset = 'iron_condor';
    this._build();
  }

  _build() {
    clear(this.host);
    this.notice = el('div');

    this.presetChips = el('div', { class: 'chip-row' },
      ...PRESETS.map(([k, l]) => el('button', {
        class: 'chip', type: 'button',
        onclick: () => { this.preset = k; this.legs = null; this.refresh().catch((e) => this.ctx.reportError(e, 'strategy')); },
      }, l)));

    this.legsBox = el('div');
    this.addBtn = el('button', {
      class: 'btn btn-ghost btn-sm', type: 'button',
      onclick: () => {
        this.legs = [...(this.legs || []), { kind: 'call', quantity: 1, strike: Math.round(state.S), sigma: state.sigma }];
        this._renderLegs();
        this._recalc();
      },
    }, '+ Add leg');

    const builderCard = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          el('h3', {}, 'Position'),
          el('p', {}, 'Quantities are signed: negative is short. Each leg carries its own implied '
            + 'volatility, so a skewed structure can be marked with the skew it was actually traded at.')),
        el('div', { class: 'card-tools' }, this.addBtn)),
      el('div', { class: 'panel-title' }, 'Presets'),
      this.presetChips,
      this.legsBox);

    this.payoffHost = el('div');
    this.payoffLegend = el('div');
    this.payoffTools = el('div', { class: 'card-tools' });
    this.payoffCard = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          el('h3', {}, 'Profit and loss'),
          el('p', {}, 'The kinked line is P&L at expiry. The smooth curves are the position revalued at '
            + 'earlier dates — which is what the position actually does, and often a very different '
            + 'shape from the expiry diagram.')),
        this.payoffTools),
      this.payoffHost, this.payoffLegend);

    this.statsBox = el('div');
    this.greekHost = el('div');
    this.greekLegend = el('div');
    this.greekCard = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          el('h3', {}, 'Aggregate Greeks across spot'),
          el('p', {}, 'Position Greeks summed over legs, at the current tenor. Where delta crosses zero '
            + 'the position is momentarily hedged; gamma tells you how long that lasts.')),
        el('div', { class: 'card-tools' })),
      this.greekHost, this.greekLegend);

    this.host.append(
      this.notice,
      el('div', { class: 'grid grid-2' }, builderCard, el('div', { class: 'card' },
        el('div', { class: 'card-head' }, el('div', {}, el('h3', {}, 'Position summary'))),
        this.statsBox)),
      this.payoffCard,
      this.greekCard,
    );

    this.payoff = new LineChart(this.payoffHost, {
      height: 340, legendHost: this.payoffLegend, xLabel: 'spot at valuation', yLabel: 'profit / loss',
    });
    this.greeks = new LineChart(this.greekHost, {
      height: 300, legendHost: this.greekLegend, xLabel: 'spot', yLabel: 'position greek',
    });
    this.payoffTable = withTableToggle(this.payoff, this.payoffTools);
    this.payoffCard.append(this.payoffTable.holder);
  }

  async refresh() {
    this.dirty = false;
    const sig = signature();
    const T = Math.max(state.T, 1 / 365);
    const body = {
      S: state.S, T, r: state.r, q: state.q, sigma: state.sigma,
      horizons: [T, T * 0.6, T * 0.25, 0],
    };
    if (this.legs) body.legs = this.legs;
    else body.preset = this.preset;

    this.payoff.setStale(true);
    try {
      const res = await api.strategy(body);
      if (sig !== signature()) return;
      this.data = res;
      this.legs = res.legs;
      this._renderLegs();
      this._renderStats();
      this._renderPayoff();
      this._renderGreeks();
    } finally {
      this.payoff.setStale(false);
    }
  }

  _recalc() {
    this.refresh().catch((e) => this.ctx.reportError(e, 'strategy'));
  }

  _renderLegs() {
    const legs = this.legs || [];
    const rows = legs.map((leg, i) => {
      const kindSel = el('select', {
        onchange: (e) => { this.legs[i].kind = e.target.value; this._recalc(); },
      }, ...['call', 'put', 'underlying'].map((k) => el('option', { value: k, selected: k === leg.kind }, k)));

      const qty = el('input', {
        type: 'number', value: String(leg.quantity), step: '1', style: { width: '68px' },
        onchange: (e) => { this.legs[i].quantity = Number(e.target.value) || 0; this._recalc(); },
      });
      const strike = el('input', {
        type: 'number', value: String(leg.strike || 0), step: '0.5', style: { width: '84px' },
        disabled: leg.kind === 'underlying',
        onchange: (e) => { this.legs[i].strike = Number(e.target.value) || 0; this._recalc(); },
      });
      const vol = el('input', {
        type: 'number', value: (leg.sigma * 100).toFixed(2), step: '0.5', style: { width: '76px' },
        disabled: leg.kind === 'underlying',
        onchange: (e) => { this.legs[i].sigma = (Number(e.target.value) || 0) / 100; this._recalc(); },
      });
      const del = el('button', {
        class: 'btn btn-ghost btn-sm', type: 'button',
        onclick: () => { this.legs.splice(i, 1); if (!this.legs.length) this.legs = null; this._recalc(); },
      }, '×');

      return el('tr', {},
        el('td', {}, kindSel),
        el('td', {}, qty),
        el('td', {}, strike),
        el('td', {}, vol),
        el('td', { class: 'num' }, leg.premium !== null && leg.premium !== undefined ? fmtSig(leg.premium, 4) : el('span', { class: 'dim' }, 'theoretical')),
        el('td', {}, del));
    });

    clear(this.legsBox).append(
      el('div', { class: 'table-wrap', style: { marginTop: '12px' } },
        el('table', { class: 'data legs-table' },
          el('thead', {}, el('tr', {},
            el('th', {}, 'Instrument'), el('th', {}, 'Qty'), el('th', {}, 'Strike'),
            el('th', {}, 'Vol %'), el('th', {}, 'Entry'), el('th', {}, ''))),
          el('tbody', {}, rows))));
  }

  _renderStats() {
    const d = this.data;
    const g = d.net_greeks;
    const units = d.greek_units || {};

    const tiles = [
      { label: d.position === 'credit' ? 'Net credit' : 'Net debit', value: fmtMoney(Math.abs(d.entry_cost), 4),
        unit: d.position === 'credit' ? 'received to open' : 'paid to open' },
      { label: 'Max profit', value: d.max_profit === null ? 'unbounded' : fmtMoney(d.max_profit, 2),
        unit: d.max_profit === null ? 'a naked long tail' : 'at expiry' },
      { label: 'Max loss', value: d.max_loss === null ? 'unbounded' : fmtMoney(d.max_loss, 2),
        unit: d.max_loss === null ? 'a naked short tail' : 'at expiry' },
      { label: 'Breakevens', value: d.breakevens.length ? d.breakevens.map((b) => fmtSig(b, 5)).join('  ·  ') : 'none',
        unit: 'expiry P&L crosses zero' },
    ];

    clear(this.statsBox).append(
      el('div', { class: 'tiles' }, ...tiles.map((t) => el('div', { class: 'tile' },
        el('div', { class: 'tile-label' }, t.label),
        el('div', { class: 'tile-value' }, t.value),
        el('div', { class: 'tile-unit' }, t.unit)))),
      el('div', { class: 'panel-title', style: { marginTop: '14px' } }, 'Net position Greeks at spot'),
      el('div', { class: 'tiles' }, ...Object.entries(g).map(([name, v]) => el('div', { class: `tile ${v < 0 ? 'is-neg' : v > 0 ? 'is-pos' : ''}` },
        el('div', { class: 'tile-label' }, name),
        el('div', { class: 'tile-value' }, fmtSig(v, 4)),
        el('div', { class: 'tile-unit' }, units[name] || '')))),
    );

    clear(this.notice);
    if (d.max_loss === null) {
      this.notice.append(el('div', { class: 'notice warn' }, el('div', {},
        el('strong', {}, 'Unbounded loss. '),
        'This position has a naked short tail, so the loss is not capped by the grid — the payoff '
        + 'keeps going past the plotted range. The chart shows a window, not the whole risk.')));
    }
  }

  _renderPayoff() {
    const d = this.data;
    const curves = d.curves;
    const series = curves.map((c, i) => ({
      key: `c${i}`,
      label: c.label,
      shortLabel: c.days_out === 0 ? 'now' : c.label,
      values: c.pnl,
      color: seriesColor(i),
      area: i === curves.length - 1,
      areaPos: token('--series-3'),
      areaNeg: token('--critical'),
      width: i === curves.length - 1 ? 2.4 : 2,
    }));

    const markers = [
      { x: d.spot0, label: `spot ${fmtSig(d.spot0, 5)}`, color: token('--text-muted') },
      ...d.breakevens.map((b) => ({ x: b, label: `BE ${fmtSig(b, 5)}`, color: token('--border-strong') })),
    ];

    this.payoff.setData({
      x: d.spots,
      series,
      yZero: true,
      markers,
      xName: 'spot',
      xTitle: (v) => `spot ${fmtSig(v, 5)}`,
      valueFmt: (v) => fmtMoney(v, 3),
      yFmt: (v) => fmtMoney(v, 0),
    });
    this.payoffTable.refresh();
  }

  _renderGreeks() {
    const d = this.data;
    // Five Greeks is exactly the palette width; nothing is cycled.
    const names = Object.keys(d.greeks);
    this.greeks.setData({
      x: d.spots,
      series: names.map((name, i) => ({
        key: name,
        label: `${name} (${(d.greek_units || {})[name] || ''})`,
        shortLabel: name,
        values: d.greeks[name],
        color: seriesColor(i),
      })),
      yZero: true,
      markers: [{ x: d.spot0, label: `spot ${fmtSig(d.spot0, 5)}`, color: token('--text-muted') }],
      xName: 'spot',
      xTitle: (v) => `spot ${fmtSig(v, 5)}`,
      valueFmt: (v) => fmtSig(v, 5),
    });
  }

  redraw() {
    if (this.payoff) this.payoff.draw();
    if (this.greeks) this.greeks.draw();
  }
}
