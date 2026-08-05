/* Vol Surface: calibrate an IV surface to a chain, inspect it, and compare
   interpolators on held-out error and arbitrage admissibility. */

import { el, clear, mount, buildTable, fmtSig, fmtSigned, fmtPct, fmtMs, fmtInt,
         seriesColor, token } from '../util.js';
import { api } from '../api.js';
import { LineChart, withTableToggle } from '../chart.js';
import { Surface3D } from '../surface3d.js';
import { getChain, setChain, onChain } from './chainstore.js';

const METHOD_LABEL = {
  svi: 'Raw SVI (per expiry)',
  ssvi: 'SSVI (global, arb-free)',
  cubic: 'Cubic spline',
  rbf: 'Thin-plate RBF',
};

const METHOD_BLURB = {
  svi: 'Five parameters per expiry, calibrated by the Zeliade quasi-explicit method. Fits each smile '
     + 'tightly but nothing couples the slices, so sparse chains can produce calendar arbitrage.',
  ssvi: 'One (ρ, η, γ) triple shared across the whole surface on top of an ATM variance curve. Three '
      + 'global shape parameters cannot chase noise, and the Gatheral-Jacquier conditions are enforced.',
  cubic: 'A natural cubic spline through the quoted smile in total variance. Interpolates exactly — which '
       + 'is the problem: it reproduces quote noise and offers no arbitrage guarantees.',
  rbf: 'A thin-plate spline over all (log-moneyness, expiry) points at once, with a ridge term. Global and '
     + 'expiry-coupled, so it smooths across maturities rather than within them.',
};

export class SurfaceView {
  constructor(host, ctx) {
    this.host = host;
    this.ctx = ctx;
    this.dirty = true;
    this.method = 'svi';
    this.mode = 'iv';
    this.sliceIdx = 0;
    this.filters = { min_open_interest: 0, max_spread_frac: 0.35, otm_only: true, max_abs_k: 1.0 };
    this._build();
    this._unsub = onChain(() => { this.dirty = true; if (this.host.classList.contains('is-active')) this.refresh().catch((e) => this.ctx.reportError(e, 'surface')); });
  }

  _build() {
    clear(this.host);
    this.notice = el('div');

    this.methodSeg = el('div', { class: 'segmented' },
      ...Object.entries(METHOD_LABEL).map(([k, l]) => el('button', {
        type: 'button', class: k === this.method ? 'is-active' : '',
        onclick: () => { this.method = k; this._syncMethod(); this.refresh().catch((e) => this.ctx.reportError(e, 'surface')); },
      }, k.toUpperCase())));

    this.modeSeg = el('div', { class: 'segmented' },
      el('button', { type: 'button', class: 'is-active', onclick: () => this._setMode('iv') }, 'Implied vol'),
      el('button', { type: 'button', onclick: () => this._setMode('g') }, 'Butterfly g(k)'));

    this.quotesBtn = el('button', {
      class: 'btn btn-ghost btn-sm is-active', type: 'button',
      onclick: () => {
        this.surface.showQuotes = !this.surface.showQuotes;
        this.quotesBtn.classList.toggle('is-active', this.surface.showQuotes);
        this.surface.draw();
      },
    }, 'Quotes');

    this.extrapBtn = el('button', {
      class: 'btn btn-ghost btn-sm', type: 'button',
      title: 'Also draw the region beyond the quoted strikes, where the model is extrapolating',
      onclick: () => {
        this.surface.showExtrapolated = !this.surface.showExtrapolated;
        this.extrapBtn.classList.toggle('is-active', this.surface.showExtrapolated);
        this.surface.draw();
        clear(this.surfLegend).append(this.surface.scaleLegend());
      },
    }, 'Extrapolation');

    this.surfTools = el('div', { class: 'card-tools' }, this.modeSeg, this.quotesBtn, this.extrapBtn,
      el('button', { class: 'btn btn-ghost btn-sm', type: 'button', onclick: () => this.surface.resetView() }, 'Reset view'));

    this.surfHost = el('div');
    this.surfLegend = el('div');
    this.surfCard = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          this.surfTitle = el('h3', {}, 'Implied volatility surface'),
          this.surfSub = el('p', {}, 'Calibrating…')),
        this.surfTools),
      this.surfHost, this.surfLegend);

    this.methodCard = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          el('h3', {}, 'Interpolator'),
          this.methodBlurb = el('p', {}, '')),
        el('div', { class: 'card-tools' }, this.methodSeg)),
      this.diagBox = el('div'));

    this.sliceSelect = el('select', { onchange: (e) => { this.sliceIdx = Number(e.target.value); this._renderSlice(); } });
    this.smileHost = el('div');
    this.smileLegend = el('div');
    this.smileTools = el('div', { class: 'card-tools' }, el('span', { class: 'inline-label' }, 'Expiry'), this.sliceSelect);
    this.smileCard = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          el('h3', {}, 'Smile, fitted vs quoted'),
          el('p', {}, 'One expiry slice. Dots are the implied vols this app inverted from mid prices; '
            + 'the line is the calibrated model. Residuals are the fit error in vol points.')),
        this.smileTools),
      this.smileHost, this.smileLegend);

    this.densityHost = el('div');
    this.densityLegend = el('div');
    this.densityCard = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          el('h3', {}, 'Risk-neutral density and butterfly diagnostic'),
          el('p', {}, 'The density implied by the fitted slice via Breeden-Litzenberger, alongside '
            + "Durrleman's g(k). Where g dips below zero the surface implies a negative density — "
            + 'a butterfly arbitrage, not merely an ugly fit.')),
        el('div', { class: 'card-tools' })),
      this.densityHost, this.densityLegend);

    this.compareBtn = el('button', {
      class: 'btn btn-primary btn-sm', type: 'button',
      onclick: () => this._runCompare(),
    }, 'Run comparison');
    this.compareBox = el('div', {}, el('div', { class: 'empty' },
      'Fits all four interpolators to the same quotes and scores them on held-out error and arbitrage admissibility.',
      el('div', { style: { marginTop: '10px' } }, this.compareBtn)));
    const compareCard = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          el('h3', {}, 'Interpolator shootout'),
          el('p', {}, 'Held-out RMSE is the column that matters. In-sample error rewards threading '
            + 'through noise — the cubic spline scores near zero there and still generalizes worst.')),
        el('div', { class: 'card-tools' })),
      this.compareBox);

    this.host.append(
      this.notice,
      el('div', { class: 'grid grid-2' }, this.surfCard, this.methodCard),
      el('div', { class: 'grid grid-2' }, this.smileCard, this.densityCard),
      compareCard,
    );

    this.surface = new Surface3D(this.surfHost, { height: 420 });
    this.smile = new LineChart(this.smileHost, {
      height: 280, legendHost: this.smileLegend, xLabel: 'log-moneyness ln(K/F)', yLabel: 'implied vol',
    });
    this.density = new LineChart(this.densityHost, {
      height: 280, legendHost: this.densityLegend, xLabel: 'log-moneyness ln(K/F)',
    });

    this.surfTable = withTableToggle(this.surface, this.surfTools);
    this.surfCard.append(this.surfTable.holder);
    this.smileTable = withTableToggle(this.smile, this.smileTools);
    this.smileCard.append(this.smileTable.holder);

    this._syncMethod();
  }

  _syncMethod() {
    [...this.methodSeg.children].forEach((b) => b.classList.toggle('is-active', b.textContent === this.method.toUpperCase()));
    this.methodBlurb.textContent = METHOD_BLURB[this.method];
  }

  _setMode(mode) {
    this.mode = mode;
    [...this.modeSeg.children].forEach((b, i) => b.classList.toggle('is-active', i === (mode === 'iv' ? 0 : 1)));
    this._renderSurface();
  }

  async refresh() {
    this.dirty = false;
    let chain = getChain();
    if (!chain) {
      // Load the synthetic chain rather than sending the user to another tab: it
      // is the default data source and it arrives in a couple of hundred ms.
      clear(this.notice).append(el('div', { class: 'notice info' }, el('div', { class: 'row' },
        el('div', { class: 'spinner' }),
        'No chain loaded yet — fetching the synthetic SSVI chain to calibrate against.')));
      const payload = await api.chain({ source: 'demo', r: 0.043, q: 0.008 });
      setChain(payload);
      chain = payload;
    }
    clear(this.notice);

    const body = { chain_id: chain.chain_id, method: this.method, ...this.filters };
    this.data = await api.surface(body);

    this._renderSurface();
    this._renderDiagnostics();
    this._renderSliceOptions();
    this._renderSlice();
  }

  _renderSurface() {
    const d = this.data;
    if (!d) return;
    const g = d.grid;
    const isG = this.mode === 'g';
    const z = isG ? g.g : g.iv;

    this.surface.setData({
      xs: g.k,
      ys: g.T,
      z,
      covered: g.covered,
      diverging: isG,
      points: isG || !this.surface.showQuotes ? null : g.quotes.map((q) => ({ x: q.k, y: q.T, z: q.iv })),
      xName: 'log-moneyness ln(K/F)',
      yName: 'years to expiry',
      zName: isG ? 'Durrleman g(k)' : 'implied vol',
      xFmt: (v) => fmtSigned(v, 2),
      yFmt: (v) => (v < 0.12 ? `${(v * 365).toFixed(0)}d` : `${v.toFixed(2)}y`),
      zFmt: isG ? ((v) => fmtSig(v, 3)) : ((v) => `${(v * 100).toFixed(1)}%`),
    });
    clear(this.surfLegend).append(this.surface.scaleLegend());

    this.surfTitle.textContent = isG ? 'Butterfly diagnostic surface' : 'Implied volatility surface';
    this.surfSub.textContent = isG
      ? 'Durrleman g(k) across the surface. Red is negative — a butterfly arbitrage. Grey is zero.'
      : `${d.ticker} · ${METHOD_LABEL[d.method]} · ${fmtInt(d.n_quotes)} quotes, `
        + `${d.diagnostics.n_slices} expiries, calibrated in ${fmtMs(d.fit_ms)}.`;
    this.surfTable.refresh();
  }

  _renderDiagnostics() {
    const d = this.data;
    const diag = d.diagnostics;
    const fit = diag.fit;

    const flag = (ok, label) => el('span', { class: `tag ${ok ? 'good' : 'bad'}` }, `${ok ? '✓' : '✕'} ${label}`);

    const tiles = [
      { label: 'In-sample RMSE', value: `${fit.rmse_vol_pts.toFixed(3)}`, unit: 'volatility points' },
      { label: 'Max abs error', value: `${fit.max_abs_vol_pts.toFixed(3)}`, unit: 'volatility points' },
      { label: 'Quotes fitted', value: fmtInt(d.n_quotes), unit: `${diag.n_slices} expiries` },
    ];
    if (d.truth) {
      tiles.push({
        label: 'Error vs true surface',
        value: d.truth.rmse_vol_pts.toFixed(3),
        unit: `vol points, vs the ${d.truth.kind.toUpperCase()} that generated the quotes`,
      });
    }

    mount(clear(this.diagBox),
      el('div', { class: 'tiles' }, ...tiles.map((t) => el('div', { class: 'tile' },
        el('div', { class: 'tile-label' }, t.label),
        el('div', { class: 'tile-value' }, t.value),
        el('div', { class: 'tile-unit' }, t.unit)))),

      el('div', { class: 'row', style: { marginTop: '12px' } },
        flag(diag.butterfly.ok, 'no butterfly arbitrage'),
        flag(diag.calendar.ok, 'no calendar arbitrage')),

      el('p', { class: 'axis-note' },
        `min g(k) = ${fmtSigned(diag.butterfly.min_g, 4)}`,
        diag.butterfly.ok ? ' (non-negative everywhere scanned). ' : ' — negative, so the fitted density goes negative somewhere. ',
        `min ∂w/∂T = ${fmtSigned(diag.calendar.min_dw_dT, 5)}`,
        diag.calendar.ok ? ' (total variance is non-decreasing in T). '
          : ` at k = ${fmtSigned(diag.calendar.at_k, 2)}, T = ${fmtSig(diag.calendar.at_T, 3)} — total variance falls with maturity there. `),

      diag.ssvi ? this._ssviBox(diag.ssvi) : null,
      this._sliceTable(diag),
    );

    if (!diag.butterfly.ok || !diag.calendar.ok) {
      clear(this.notice).append(el('div', { class: 'notice warn' }, el('div', {},
        el('strong', {}, 'This fit admits arbitrage. '),
        this.method === 'svi'
          ? 'Raw SVI fits each expiry independently, so on a sparse chain adjacent slices can imply '
            + 'falling total variance. Switch to SSVI to see a surface that is arbitrage-free by construction — '
            + 'at the cost of a looser fit to each individual smile.'
          : 'Neither the cubic spline nor the RBF interpolant enforces any no-arbitrage condition; '
            + 'they are here precisely so the diagnostic can catch them out. SSVI is the admissible option.')));
    } else {
      clear(this.notice);
    }
  }

  _ssviBox(s) {
    return el('div', { style: { marginTop: '12px' } },
      el('div', { class: 'panel-title' }, 'SSVI parameters'),
      el('div', { class: 'row' },
        el('span', { class: 'tag neutral mono' }, `ρ = ${fmtSigned(s.rho, 4)}`),
        el('span', { class: 'tag neutral mono' }, `η = ${fmtSig(s.eta, 4)}`),
        el('span', { class: 'tag neutral mono' }, `γ = ${fmtSig(s.gamma, 4)}`),
        el('span', { class: `tag ${s.admissibility.butterfly_ok ? 'good' : 'bad'}` },
          `θφ(1+|ρ|) = ${fmtSig(s.admissibility.butterfly_cond_1, 3)} ≤ 4`)),
      el('p', { class: 'axis-note' },
        'ρ is the spot/vol correlation that generates the skew; η scales the smile; γ controls how fast '
        + 'curvature decays with maturity. The condition shown is Gatheral-Jacquier\'s sufficient test '
        + 'for the whole surface being free of butterfly arbitrage.'));
  }

  _sliceTable(diag) {
    const cols = [
      { label: 'Expiry', mono: false, get: (s) => (s.T < 0.12 ? `${(s.T * 365).toFixed(0)}d` : `${s.T.toFixed(3)}y`) },
      { label: 'Quotes', get: (s) => fmtInt(s.n_quotes) },
      { label: 'ATM vol', get: (s) => fmtPct(s.atm_vol, 2) },
      { label: 'min g(k)', get: (s) => fmtSigned(s.min_g, 4), cls: (s) => (s.butterfly_ok ? 'pos' : 'neg') },
      { label: 'at k', get: (s) => fmtSigned(s.k_at_min_g, 3) },
    ];
    if (diag.slices.some((s) => s.svi)) {
      cols.push(
        { label: 'a', get: (s) => (s.svi ? fmtSigned(s.svi.a, 5) : '—') },
        { label: 'b', get: (s) => (s.svi ? fmtSig(s.svi.b, 4) : '—') },
        { label: 'ρ', get: (s) => (s.svi ? fmtSigned(s.svi.rho, 3) : '—') },
        { label: 'm', get: (s) => (s.svi ? fmtSigned(s.svi.m, 4) : '—') },
        { label: 'σ', get: (s) => (s.svi ? fmtSig(s.svi.sigma, 4) : '—') },
      );
    }
    return el('div', { style: { marginTop: '12px' } },
      el('div', { class: 'panel-title' }, 'Per-expiry slices'),
      buildTable(cols, diag.slices, { maxHeight: '260px' }));
  }

  _renderSliceOptions() {
    const smiles = this.data.smiles;
    clear(this.sliceSelect).append(...smiles.map((s, i) => el('option', { value: String(i) },
      s.T < 0.12 ? `${(s.T * 365).toFixed(0)} days` : `${s.T.toFixed(3)} years`)));
    this.sliceIdx = Math.min(this.sliceIdx, smiles.length - 1);
    this.sliceSelect.value = String(this.sliceIdx);
  }

  _renderSlice() {
    const s = this.data.smiles[this.sliceIdx];
    if (!s) return;

    // Quotes are scattered in k, so they get their own aligned series: the model
    // curve is sampled on a dense grid, the dots only where quotes exist.
    const quoteSeries = new Array(s.k.length).fill(null);
    for (const q of s.quotes) {
      let best = 0; let bd = Infinity;
      for (let i = 0; i < s.k.length; i += 1) {
        const dd = Math.abs(s.k[i] - q.k);
        if (dd < bd) { bd = dd; best = i; }
      }
      quoteSeries[best] = q.iv;
    }

    this.smile.setData({
      x: s.k,
      series: [
        { key: 'fit', label: `${METHOD_LABEL[this.data.method]} fit`, shortLabel: 'fit', values: s.iv, color: seriesColor(0) },
        { key: 'quotes', label: 'Inverted from mid price', shortLabel: 'quotes', values: quoteSeries, color: seriesColor(1), width: 0, markers: true },
      ],
      xName: 'ln(K/F)',
      xTitle: (v) => `ln(K/F) ${fmtSigned(v, 4)}`,
      valueFmt: (v) => (Number.isFinite(v) ? fmtPct(v, 2) : '—'),
      yFmt: (v) => `${(v * 100).toFixed(0)}%`,
      markers: [{ x: 0, label: 'ATM forward', color: token('--border-strong') }],
    });
    this.smileTable.refresh();

    const gMin = Math.min(...s.g);
    this.density.setData({
      x: s.k,
      series: [
        { key: 'density', label: 'Risk-neutral density', shortLabel: 'density', values: s.density, color: seriesColor(2), area: true },
        { key: 'g', label: "Durrleman g(k)", shortLabel: 'g(k)', values: s.g, color: gMin < 0 ? token('--critical') : seriesColor(3) },
      ],
      yZero: true,
      xName: 'ln(K/F)',
      xTitle: (v) => `ln(K/F) ${fmtSigned(v, 4)}`,
      valueFmt: (v) => fmtSig(v, 4),
      markers: [{ x: 0, label: 'ATM forward', color: token('--border-strong') }],
    });
  }

  async _runCompare() {
    const chain = getChain();
    if (!chain) return;
    clear(this.compareBox).append(el('div', { class: 'empty' },
      el('div', { class: 'spinner' }), 'Fitting four interpolators with k-fold refits — this takes a few seconds.'));
    try {
      const res = await api.surfaceCompare({ chain_id: chain.chain_id, ...this.filters });

      const ok = res.rows.filter((r) => !r.error && Number.isFinite(r.holdout_rmse));
      const bestHoldout = ok.length ? Math.min(...ok.map((r) => r.holdout_rmse)) : null;
      const withTruth = res.rows.filter((r) => !r.error && Number.isFinite(r.truth_rmse));
      const bestTruth = withTruth.length ? Math.min(...withTruth.map((r) => r.truth_rmse)) : null;

      const cols = [
        { label: 'Interpolator', mono: false, get: (r) => el('span', {},
          el('span', { class: 'swatch', style: { background: seriesColor(['svi', 'ssvi', 'cubic', 'rbf'].indexOf(r.method)) } }),
          METHOD_LABEL[r.method] || r.method) },
        { label: 'In-sample', get: (r) => (r.error ? '—' : r.in_sample_rmse.toFixed(3)) },
        { label: 'Held out', get: (r) => (r.error ? '—' : (r.holdout_rmse ?? NaN).toFixed(3)),
          cls: (r) => (!r.error && r.holdout_rmse === bestHoldout ? 'pos' : '') },
        res.has_truth ? { label: 'vs true surface', get: (r) => (r.error || r.truth_rmse === undefined ? '—' : r.truth_rmse.toFixed(3)),
          cls: (r) => (!r.error && r.truth_rmse === bestTruth ? 'pos' : '') } : null,
        { label: 'Butterfly', mono: false, get: (r) => (r.error ? '—'
          : el('span', { class: `tag ${r.butterfly_ok ? 'good' : 'bad'}` }, `${r.butterfly_ok ? '✓' : '✕'} ${fmtSigned(r.min_g, 3)}`)) },
        { label: 'Calendar', mono: false, get: (r) => (r.error ? '—'
          : el('span', { class: `tag ${r.calendar_ok ? 'good' : 'bad'}` }, `${r.calendar_ok ? '✓' : '✕'} ${fmtSigned(r.min_dw_dT, 4)}`)) },
        { label: 'Fit time', get: (r) => (r.error ? '—' : fmtMs(r.fit_ms)) },
      ].filter(Boolean);

      clear(this.compareBox).append(
        buildTable(cols, res.rows),
        el('p', { class: 'axis-note' }, `All errors in volatility points on ${fmtInt(res.n_quotes)} quotes. `, res.note),
        el('div', { class: 'row', style: { marginTop: '10px' } }, this.compareBtn),
      );
    } catch (e) {
      clear(this.compareBox).append(
        el('div', { class: 'notice bad' }, el('div', {}, e.message || String(e))),
        el('div', { class: 'row' }, this.compareBtn));
    }
  }

  redraw() {
    if (this.surface) this.surface.draw();
    if (this.smile) this.smile.draw();
    if (this.density) this.density.draw();
    if (this.data) clear(this.surfLegend).append(this.surface.scaleLegend());
  }
}
