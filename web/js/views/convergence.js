/* Convergence: measured error rates and the cost of hitting a target accuracy. */

import { el, clear, buildTable, fmtSig, fmtSigned, fmtMs, fmtInt, fmtCompact,
         seriesColor, token } from '../util.js';
import { api } from '../api.js';
import { state, contract, signature } from '../state.js';
import { LineChart, BarChart, withTableToggle } from '../chart.js';

const LATTICE_CHOICES = [
  ['crr', 'CRR'],
  ['crr_smooth', 'CRR + smoothing'],
  ['crr_richardson', 'CRR + smooth + Richardson'],
  ['jarrow_rudd', 'Jarrow-Rudd'],
  ['tian', 'Tian'],
  ['leisen_reimer', 'Leisen-Reimer'],
  ['trinomial', 'Trinomial'],
];

// Short forms for the endpoint labels; the legend carries the full name.
const SHORT = {
  crr: 'CRR', crr_smooth: 'CRR+sm', crr_richardson: 'CRR+sm+Rich',
  jarrow_rudd: 'J-R', tian: 'Tian', leisen_reimer: 'L-R', trinomial: 'Trinomial',
  plain: 'plain', antithetic: 'anti', control: 'control', both: 'anti+ctrl', qmc: 'QMC',
};

const MC_CHOICES = [
  ['plain', 'Plain MC'],
  ['antithetic', 'Antithetic'],
  ['control', 'Control variate'],
  ['both', 'Antithetic + control'],
  ['qmc', 'Halton QMC'],
];

export class ConvergenceView {
  constructor(host, ctx) {
    this.host = host;
    this.ctx = ctx;
    this.dirty = true;
    // Capped at five so categorical hues are never cycled.
    this.latticeEngines = new Set(['crr', 'crr_smooth', 'leisen_reimer', 'trinomial']);
    this.mcEngines = new Set(['plain', 'antithetic', 'both', 'qmc']);
    this.nMax = 800;
    this.pathsMax = 262144;
    this.targetBp = 1;
    this._build();
  }

  _build() {
    clear(this.host);
    this.notice = el('div');

    this.runBtn = el('button', { class: 'btn btn-primary', type: 'button', onclick: () => this.refresh().catch((e) => this.ctx.reportError(e, 'convergence')) }, 'Run study');

    this.latticeChips = el('div', { class: 'chip-row' });
    this.mcChips = el('div', { class: 'chip-row' });
    this._renderChips();

    const controls = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          el('h3', {}, 'Study configuration'),
          el('p', {}, 'Each engine is walked over a geometric grid of discretization sizes; the '
            + 'convergence order is the fitted slope of log|error| against log(n).')),
        el('div', { class: 'card-tools' }, this.runBtn)),
      el('div', { class: 'panel-title' }, 'Lattice engines (max 5)'),
      this.latticeChips,
      el('div', { class: 'panel-title', style: { marginTop: '12px' } }, 'Monte Carlo estimators (max 5)'),
      this.mcChips,
      el('div', { class: 'row', style: { marginTop: '12px' } },
        el('label', { class: 'inline-label' }, 'max steps'),
        el('input', { type: 'number', value: String(this.nMax), min: '16', max: '4000', step: '100',
          style: { width: '92px' }, oninput: (e) => { this.nMax = Number(e.target.value) || 800; } }),
        el('label', { class: 'inline-label' }, 'max paths'),
        el('input', { type: 'number', value: String(this.pathsMax), min: '1024', max: '2000000', step: '10000',
          style: { width: '112px' }, oninput: (e) => { this.pathsMax = Number(e.target.value) || 262144; } }),
        el('label', { class: 'inline-label' }, 'target accuracy (bp)'),
        el('input', { type: 'number', value: String(this.targetBp), min: '0.01', max: '1000', step: '0.5',
          style: { width: '80px' }, oninput: (e) => { this.targetBp = Number(e.target.value) || 1; } })));

    this.latticeHost = el('div');
    this.latticeLegend = el('div');
    this.latticeTools = el('div', { class: 'card-tools' });
    this.latticeFit = el('div');
    this.latticeCard = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          el('h3', {}, 'Lattice convergence'),
          this.latticeNote = el('p', {}, '')),
        this.latticeTools),
      this.latticeHost, this.latticeLegend, this.latticeFit);

    this.mcHost = el('div');
    this.mcLegend = el('div');
    this.mcTools = el('div', { class: 'card-tools' });
    this.mcFit = el('div');
    this.mcCard = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          el('h3', {}, 'Monte Carlo convergence'),
          this.mcNote = el('p', {}, '')),
        this.mcTools),
      this.mcHost, this.mcLegend, this.mcFit);

    this.shootHost = el('div');
    this.shootTable = el('div');
    this.shootCard = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          el('h3', {}, 'Cost to reach the target accuracy'),
          this.shootNote = el('p', {}, 'Wall time at the smallest discretization that lands inside the '
            + 'tolerance. Monte Carlo rows must also fit their 95% interval inside it, so a lucky draw '
            + 'does not count as convergence.')),
        el('div', { class: 'card-tools' })),
      this.shootHost, this.shootTable);

    this.host.append(this.notice, controls, this.latticeCard, this.mcCard, this.shootCard);

    this.latticeChart = new LineChart(this.latticeHost, {
      height: 320, legendHost: this.latticeLegend, xLabel: 'steps (log)', yLabel: '|error| (log)',
    });
    this.mcChart = new LineChart(this.mcHost, {
      height: 320, legendHost: this.mcLegend, xLabel: 'paths (log)', yLabel: '|error| (log)',
    });
    this.shootChart = new BarChart(this.shootHost, { xLabel: 'milliseconds (log)' });

    this.latticeTableToggle = withTableToggle(this.latticeChart, this.latticeTools);
    this.latticeCard.append(this.latticeTableToggle.holder);
    this.mcTableToggle = withTableToggle(this.mcChart, this.mcTools);
    this.mcCard.append(this.mcTableToggle.holder);
  }

  _renderChips() {
    const build = (host, choices, set) => {
      clear(host);
      for (const [key, label] of choices) {
        const on = set.has(key);
        host.append(el('button', {
          class: 'chip', type: 'button',
          style: on ? { borderColor: 'var(--accent)', background: 'var(--accent-soft)', color: 'var(--text-primary)' } : {},
          'aria-pressed': String(on),
          onclick: () => {
            if (set.has(key)) set.delete(key);
            else if (set.size >= 5) {
              this.ctx.toast('Five series is the cap — categorical hues are never cycled past the palette.', 'info');
              return;
            } else set.add(key);
            this._renderChips();
          },
        }, label));
      }
    };
    build(this.latticeChips, LATTICE_CHOICES, this.latticeEngines);
    build(this.mcChips, MC_CHOICES, this.mcEngines);
  }

  async refresh() {
    this.dirty = false;
    const sig = signature();
    [this.latticeChart, this.mcChart].forEach((c) => c.setStale(true));
    try {
      const res = await api.convergence({
        ...contract(),
        american: state.american,
        lattice_engines: [...this.latticeEngines],
        mc_engines: [...this.mcEngines],
        n_max: this.nMax,
        paths_max: this.pathsMax,
        target_bp: this.targetBp,
      });
      if (sig !== signature()) return;
      this.data = res;
      this._render();
    } finally {
      [this.latticeChart, this.mcChart].forEach((c) => c.setStale(false));
    }
  }

  _render() {
    const d = this.data;
    clear(this.notice);
    if (state.american) {
      this.notice.append(el('div', { class: 'notice info' }, el('div', {},
        el('strong', {}, 'American exercise. '),
        'There is no closed form, so error is measured against a high-accuracy lattice '
        + '(Leisen-Reimer at 6001 steps with Richardson extrapolation). The Monte Carlo panel is '
        + 'European-only — Longstaff-Schwartz is biased low by construction, which makes a '
        + 'convergence-rate fit against a fixed reference misleading.')));
    }

    this._renderLattice(d.lattice);
    this._renderMC(d.monte_carlo);
    this._renderShootout(d.shootout);
  }

  _renderLattice(L) {
    this.latticeCard.style.display = L ? '' : 'none';
    if (!L) return;
    const xs = L.series[0] ? L.series[0].points.map((p) => p.n) : [];
    this.latticeChart.setData({
      x: xs,
      xLog: true,
      yLog: true,
      series: L.series.map((s, i) => ({
        key: s.key,
        label: `${s.label} — order ${s.fit.order.toFixed(2)}`,
        shortLabel: SHORT[s.key] || s.label,
        values: s.points.map((p) => Math.abs(p.error)),
        color: seriesColor(i),
        markers: xs.length <= 30,
        fmt: (v) => fmtSig(v, 3),
      })),
      xName: 'steps',
      xTitle: (v) => `${fmtInt(v)} steps`,
      xTableFmt: (v) => fmtInt(v),
      valueFmt: (v) => fmtSig(v, 3),
    });
    this.latticeTableToggle.refresh();
    this.latticeNote.textContent = `Absolute pricing error against ${L.reference_label} `
      + `= ${fmtSig(L.reference, 8)}. Both axes are logarithmic, so a straight line is a power law and `
      + `its slope is the convergence order.`;

    const cols = [
      { label: 'Engine', mono: false, get: (s, i) => el('span', {},
        el('span', { class: 'swatch', style: { background: seriesColor(L.series.indexOf(s)) } }), s.label) },
      { label: 'Fitted order', get: (s) => s.fit.order.toFixed(3), cls: (s) => (s.fit.order > 1.5 ? 'pos' : '') },
      { label: 'R² of fit', get: (s) => s.fit.r_squared.toFixed(3),
        cls: (s) => (s.fit.r_squared < 0.8 ? 'neg' : '') },
      { label: `Error at n=${fmtInt(this.nMax)}`, get: (s) => fmtSigned(s.final_error, 3) },
      { label: 'Total time', get: (s) => fmtMs(s.total_ms) },
    ];
    clear(this.latticeFit).append(
      buildTable(cols, L.series),
      el('p', { class: 'axis-note' }, L.note));
  }

  _renderMC(M) {
    this.mcCard.style.display = M ? '' : 'none';
    if (!M) return;
    const xs = M.series[0] ? M.series[0].points.map((p) => p.n) : [];
    this.mcChart.setData({
      x: xs,
      xLog: true,
      yLog: true,
      series: M.series.map((s, i) => ({
        key: s.key,
        label: `${s.label} — order ${s.fit.order.toFixed(2)}`,
        shortLabel: SHORT[s.key] || s.label,
        values: s.points.map((p) => Math.abs(p.error)),
        color: seriesColor(i),
        markers: true,
        fmt: (v) => fmtSig(v, 3),
      })),
      xName: 'paths',
      xTitle: (v) => `${fmtCompact(v)} paths`,
      xTableFmt: (v) => fmtInt(v),
      valueFmt: (v) => fmtSig(v, 3),
    });
    this.mcTableToggle.refresh();
    this.mcNote.textContent = `Absolute error against the closed form = ${fmtSig(M.reference, 8)}. `
      + `Theoretical order for pseudorandom Monte Carlo is 0.50 regardless of variance reduction.`;

    const cols = [
      { label: 'Estimator', mono: false, get: (s) => el('span', {},
        el('span', { class: 'swatch', style: { background: seriesColor(M.series.indexOf(s)) } }), s.label) },
      { label: 'Fitted order', get: (s) => s.fit.order.toFixed(3), cls: (s) => (s.fit.order > 0.8 ? 'pos' : '') },
      { label: 'R² of fit', get: (s) => s.fit.r_squared.toFixed(3) },
      { label: 'Std-err order', get: (s) => s.se_fit.order.toFixed(3) },
      { label: 'Final error', get: (s) => fmtSigned(s.final_error, 4) },
      { label: 'Efficiency', get: (s) => {
        const last = s.points[s.points.length - 1];
        return last && Number.isFinite(last.efficiency) ? `${fmtSig(last.efficiency, 3)}×` : '—';
      } },
      { label: 'Total time', get: (s) => fmtMs(s.total_ms) },
    ];
    clear(this.mcFit).append(
      buildTable(cols, M.series),
      el('p', { class: 'axis-note' }, M.note,
        ' The standard-error column is the cleaner measurement: a single error point is one draw from a '
        + 'distribution, whereas the standard error is an estimate of that distribution\'s width.'));
  }

  _renderShootout(S) {
    if (!S) { this.shootCard.style.display = 'none'; return; }
    this.shootCard.style.display = '';
    const familyColor = (f) => (f === 'lattice' ? seriesColor(1) : seriesColor(2));
    this.shootChart.setData({
      xLog: true,
      valueFmt: (v) => fmtMs(v),
      bars: S.rows.map((r) => ({
        label: r.label,
        value: r.reached ? Math.max(r.ms, 0.01) : NaN,
        color: familyColor(r.family),
        note: r.reached ? `${fmtInt(r.n)} ${r.family === 'lattice' ? 'steps' : 'paths'}` : 'never reached the target',
      })),
    });

    const cols = [
      { label: 'Engine', mono: false, get: (r) => el('span', {},
        el('span', { class: 'swatch', style: { background: familyColor(r.family) } }), r.label) },
      { label: 'Reached', mono: false, get: (r) => el('span', { class: `tag ${r.reached ? 'good' : 'bad'}` }, r.reached ? '✓' : '✕') },
      { label: 'Resolution', get: (r) => (r.reached ? `${fmtInt(r.n)} ${r.family === 'lattice' ? 'steps' : 'paths'}` : '—') },
      { label: 'Time', get: (r) => (r.reached ? fmtMs(r.ms) : '—') },
      { label: 'Error there', get: (r) => (r.reached ? fmtSigned(r.error, 3) : '—') },
    ];
    clear(this.shootTable).append(
      buildTable(cols, S.rows),
      el('p', { class: 'axis-note' },
        `Target ${S.target_bp} bp of premium = ${fmtSig(S.tolerance, 3)} in price terms, against a `
        + `reference of ${fmtSig(S.reference, 8)}. `, S.note));
  }

  redraw() {
    [this.latticeChart, this.mcChart, this.shootChart].forEach((c) => c && c.draw());
    if (this.data) this._render();
  }
}
