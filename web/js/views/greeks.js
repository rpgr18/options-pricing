/* Greeks Lab: a Greek surface over (strike x tenor), as 3-D or heatmap. */

import { el, clear, fmtSig, fmtSigned, token, seriesColor } from '../util.js';
import { api } from '../api.js';
import { state, contract, signature } from '../state.js';
import { HeatMap, LineChart, withTableToggle } from '../chart.js';
import { Surface3D } from '../surface3d.js';

const GREEKS = [
  ['price', 'Price'], ['delta', 'Delta'], ['gamma', 'Gamma'], ['vega', 'Vega'],
  ['theta', 'Theta'], ['rho', 'Rho'], ['vanna', 'Vanna'], ['volga', 'Volga'],
  ['charm', 'Charm'], ['speed', 'Speed'], ['zomma', 'Zomma'], ['color', 'Color'],
  ['dual_delta', 'Dual delta'], ['dual_gamma', 'Dual gamma'],
];

const INSIGHT = {
  gamma: 'Gamma is a ridge along the strike that sharpens as expiry approaches — the same ridge that makes short-dated at-the-money options expensive to hedge.',
  vega: 'Vega grows with the square root of time and peaks near the money, so long-dated options carry the volatility exposure.',
  theta: 'Theta is most negative for short-dated at-the-money options, where the ridge is steepest.',
  vanna: 'Vanna changes sign across the strike: it is the Greek a risk reversal is built to trade.',
  charm: 'Charm shows delta drifting purely from time passing — why a hedge set on Friday is wrong on Monday.',
  volga: 'Volga is positive in both wings and negative near the money, which is why strangles are long volatility-of-volatility.',
  color: 'Color is gamma decay: how quickly the gamma ridge sharpens as expiry approaches.',
  dual_gamma: 'Dual gamma is the discounted risk-neutral density at the strike — the market-implied distribution of the terminal price.',
};

export class GreeksView {
  constructor(host, ctx) {
    this.host = host;
    this.ctx = ctx;
    this.dirty = true;
    this.greek = 'gamma';
    this.mode = '3d';
    this.zScale = 0.62;
    this._build();
  }

  _build() {
    clear(this.host);

    this.picker = el('select', {
      onchange: (e) => { this.greek = e.target.value; this.refresh().catch((x) => this.ctx.reportError(x, 'greek surface')); },
    }, ...GREEKS.map(([v, l]) => el('option', { value: v, selected: v === this.greek }, l)));

    this.modeSeg = el('div', { class: 'segmented' },
      el('button', { type: 'button', class: 'is-active', onclick: () => this._setMode('3d') }, '3-D'),
      el('button', { type: 'button', onclick: () => this._setMode('heat') }, 'Heatmap'));

    this.wireBtn = el('button', {
      class: 'btn btn-ghost btn-sm is-active', type: 'button',
      onclick: () => {
        this.surface.showWire = !this.surface.showWire;
        this.wireBtn.classList.toggle('is-active', this.surface.showWire);
        this.surface.draw();
      },
    }, 'Wireframe');

    this.resetBtn = el('button', {
      class: 'btn btn-ghost btn-sm', type: 'button',
      onclick: () => this.surface.resetView(),
    }, 'Reset view');

    this.reliefInput = el('input', {
      type: 'range', min: '0.15', max: '1.2', step: '0.05', value: String(this.zScale),
      style: { width: '96px' },
      oninput: (e) => { this.zScale = Number(e.target.value); this.surface.setView({ zScale: this.zScale }); },
    });

    this.surfTools = el('div', { class: 'card-tools' },
      el('span', { class: 'inline-label' }, 'Greek'), this.picker,
      this.modeSeg, this.wireBtn,
      el('span', { class: 'inline-label' }, 'relief'), this.reliefInput,
      this.resetBtn);

    this.surfHost = el('div');
    this.heatHost = el('div', { style: { display: 'none' } });
    this.legendHost = el('div');
    this.insight = el('p', { class: 'axis-note' });

    this.surfCard = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          this.title = el('h3', {}, 'Gamma surface'),
          this.subtitle = el('p', {}, '')),
        this.surfTools),
      this.surfHost, this.heatHost, this.legendHost, this.insight);

    this.sliceHost = el('div');
    this.sliceLegend = el('div');
    this.sliceTools = el('div', { class: 'card-tools' });
    const sliceCard = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          el('h3', {}, 'Term slices at the current strike'),
          el('p', {}, 'The same Greek as a function of spot, at four tenors. This is the 3-D surface '
            + 'cut into readable curves — the view you can actually take numbers off.')),
        this.sliceTools),
      this.sliceHost, this.sliceLegend);

    this.host.append(this.surfCard, sliceCard);

    this.surface = new Surface3D(this.surfHost, { height: 440 });
    this.surface.setView({ zScale: this.zScale });
    this.heat = new HeatMap(this.heatHost, { height: 400 });
    this.slice = new LineChart(this.sliceHost, {
      height: 300, legendHost: this.sliceLegend, xLabel: 'spot', yLabel: 'greek',
    });

    this.surfTable = withTableToggle(() => (this.mode === '3d' ? this.surface : this.heat), this.surfTools);
    this.surfCard.append(this.surfTable.holder);
    this.sliceTable = withTableToggle(this.slice, this.sliceTools);
    sliceCard.append(this.sliceTable.holder);
  }

  _setMode(mode) {
    this.mode = mode;
    [...this.modeSeg.children].forEach((b, i) => b.classList.toggle('is-active', i === (mode === '3d' ? 0 : 1)));
    this.surfHost.style.display = mode === '3d' ? '' : 'none';
    this.heatHost.style.display = mode === 'heat' ? '' : 'none';
    [this.wireBtn, this.resetBtn, this.reliefInput].forEach((n) => { n.style.display = mode === '3d' ? '' : 'none'; });
    clear(this.legendHost).append(mode === '3d' ? this.surface.scaleLegend() : this.heat.scaleLegend());
    if (mode === '3d') this.surface.draw(); else this.heat.draw();
    // The toggle resolves its target lazily, so it follows the active renderer.
    this.surfTable.refresh();
  }

  async refresh() {
    this.dirty = false;
    const sig = signature();
    const c = contract();
    const span = 0.42;

    const [surf, slices] = await Promise.all([
      api.greekSurface({
        ...c,
        greek: this.greek,
        strike_low: state.S * (1 - span),
        strike_high: state.S * (1 + span),
        tenor_low: 7 / 365,
        tenor_high: Math.max(state.T, 0.5),
        n_strikes: 54,
        n_tenors: 44,
      }),
      this._sliceData(c),
    ]);
    if (sig !== signature()) return;

    this.data = surf;
    const label = (GREEKS.find(([v]) => v === surf.greek) || [, surf.greek])[1];
    this.title.textContent = `${label} surface`;
    this.subtitle.textContent = `${label} over strike and tenor, at spot ${fmtSig(state.S, 6)} and `
      + `σ ${(state.sigma * 100).toFixed(1)}%. Units: ${surf.unit}. The tenor axis starts at one week: `
      + `gamma and theta diverge as T→0, and a single one-day cell would set the colour scale for the `
      + `whole surface. The term-slice chart below goes right down to a week.`;
    this.insight.textContent = INSIGHT[surf.greek]
      || `${label} across the strike-tenor plane. Drag to orbit; the colour ramp and the height both encode ${label.toLowerCase()}.`;

    const common = {
      xs: surf.strikes,
      ys: surf.tenors,
      z: surf.values,
      diverging: surf.diverging,
      xName: 'strike',
      yName: 'years to expiry',
      zName: `${label} (${surf.unit})`,
      valueName: `${label} (${surf.unit})`,
      xFmt: (v) => fmtSig(v, 4),
      yFmt: (v) => (v < 0.1 ? `${(v * 365).toFixed(0)}d` : `${v.toFixed(2)}y`),
      zFmt: (v) => fmtSig(v, 4),
      min: surf.min,
      max: surf.max,
      markers: [{ x: state.S, label: `S ${fmtSig(state.S, 5)}` }],
    };
    this.surface.setData(common);
    this.heat.setData(common);
    clear(this.legendHost).append(this.mode === '3d' ? this.surface.scaleLegend() : this.heat.scaleLegend());

    this._renderSlices(slices, label, surf.unit);
  }

  _sliceData(c) {
    const tenors = [Math.max(state.T, 0.5), Math.max(state.T, 0.5) * 0.5, Math.max(state.T, 0.5) * 0.2, 7 / 365];
    return Promise.all(tenors.map((T) => api.greekProfile({
      ...c, T, axis: 'spot', greeks: [this.greek],
      low: state.S * 0.58, high: state.S * 1.42, n: 161,
    }).then((r) => ({ T, r }))));
  }

  _renderSlices(slices, label, unit) {
    const first = slices[0].r;
    this.slice.opts.yLabel = `${label} (${unit})`;
    this.slice.setData({
      x: first.x,
      series: slices.map((s, i) => ({
        key: `t${i}`,
        label: s.T < 0.08 ? `${(s.T * 365).toFixed(0)} days` : `${s.T.toFixed(2)} years`,
        shortLabel: s.T < 0.08 ? `${(s.T * 365).toFixed(0)}d` : `${s.T.toFixed(2)}y`,
        values: s.r.series[0].values,
        color: seriesColor(i),
      })),
      yZero: true,
      markers: [
        { x: state.K, label: `K ${fmtSig(state.K, 5)}`, color: token('--border-strong') },
        { x: state.S, label: `S ${fmtSig(state.S, 5)}`, color: token('--text-muted') },
      ],
      xName: 'spot',
      xTitle: (v) => `spot ${fmtSig(v, 5)}`,
      valueFmt: (v) => fmtSig(v, 5),
    });
    this.sliceTable.refresh();
  }

  redraw() {
    if (this.surface) this.surface.draw();
    if (this.heat) this.heat.draw();
    if (this.slice) this.slice.draw();
    if (this.data) clear(this.legendHost).append(this.mode === '3d' ? this.surface.scaleLegend() : this.heat.scaleLegend());
  }
}
