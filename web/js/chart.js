/* ==========================================================================
   Canvas chart primitives: LineChart, HeatMap, BarChart.

   Shared conventions, applied by every chart here:
     - thin marks (2px lines, >=8px markers), hairline solid grid one shade off
       the surface, never dashed
     - a crosshair + tooltip hover layer by default, with a >=24px hit band
     - a legend whenever there are two or more series, plus selective direct
       labels on series endpoints -- never a value on every point
     - a table-view twin, so no value is reachable only by hovering
     - the previous render is held at reduced opacity during a refetch rather
       than collapsing to a skeleton
   ========================================================================== */

import { el, clear, token, seriesColor, rampAt, sequentialRamp, divergingRamp,
         ticks, logTicks, fmtAxis, fmtSig, buildTable } from './util.js';

const FONT = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
const FONT_SM = '10px system-ui, -apple-system, "Segoe UI", sans-serif';

class BaseChart {
  constructor(host, opts = {}) {
    this.host = host;
    this.opts = opts;
    this.height = opts.height || 300;
    this.pad = Object.assign({ top: 14, right: 18, bottom: 34, left: 56 }, opts.pad);

    host.classList.add('chart-host');
    clear(host);
    this.canvas = el('canvas');
    this.tooltip = el('div', { class: 'chart-tooltip' });
    host.append(this.canvas, this.tooltip);
    this.ctx = this.canvas.getContext('2d');

    this.hover = null;
    this._bindPointer();

    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(host);
    this._resize();
  }

  destroy() { this._ro.disconnect(); }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(this.host.clientWidth, 200);
    const h = this.height;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.canvas.style.height = `${h}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = w;
    this.h = h;
    this.draw();
  }

  get plot() {
    return {
      x: this.pad.left,
      y: this.pad.top,
      w: Math.max(this.w - this.pad.left - this.pad.right, 10),
      h: Math.max(this.h - this.pad.top - this.pad.bottom, 10),
    };
  }

  _bindPointer() {
    const move = (ev) => {
      const rect = this.canvas.getBoundingClientRect();
      this._onHover(ev.clientX - rect.left, ev.clientY - rect.top);
    };
    this.canvas.addEventListener('pointermove', move);
    this.canvas.addEventListener('pointerdown', move);
    this.canvas.addEventListener('pointerleave', () => {
      this.hover = null;
      this.tooltip.classList.remove('is-visible');
      this.draw();
    });
  }

  _onHover() {}

  setStale(on) { this.host.classList.toggle('is-stale', !!on); }

  _showTooltip(px, py, title, rows) {
    const t = this.tooltip;
    clear(t);
    if (title) t.append(el('div', { class: 'tt-title' }, title));
    for (const r of rows) {
      t.append(el('div', { class: 'tt-row' },
        el('span', { class: 'tt-key' },
          r.color ? el('span', { class: 'swatch', style: { background: r.color } }) : null,
          r.label),
        el('span', { class: 'tt-val' }, r.value)));
    }
    t.classList.add('is-visible');
    // Keep the bubble inside the host box.
    const tw = t.offsetWidth;
    const left = Math.min(Math.max(px, tw / 2 + 4), this.w - tw / 2 - 4);
    t.style.left = `${left}px`;
    t.style.top = `${Math.max(py - 12, 34)}px`;
  }

  _grid(ctx, xs, ys, xScale, yScale) {
    const p = this.plot;
    ctx.save();
    ctx.strokeStyle = token('--grid');
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const t of ys) {
      const y = Math.round(yScale(t)) + 0.5;
      ctx.moveTo(p.x, y); ctx.lineTo(p.x + p.w, y);
    }
    for (const t of xs) {
      const x = Math.round(xScale(t)) + 0.5;
      ctx.moveTo(x, p.y); ctx.lineTo(x, p.y + p.h);
    }
    ctx.stroke();
    ctx.restore();
  }

  _axes(ctx, xs, ys, xScale, yScale, xFmt, yFmt) {
    const p = this.plot;
    ctx.save();
    ctx.strokeStyle = token('--axis');
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(p.x, p.y + p.h + 0.5); ctx.lineTo(p.x + p.w, p.y + p.h + 0.5);
    ctx.moveTo(p.x - 0.5, p.y); ctx.lineTo(p.x - 0.5, p.y + p.h);
    ctx.stroke();

    ctx.fillStyle = token('--text-muted');
    ctx.font = FONT;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const t of ys) ctx.fillText(yFmt(t), p.x - 8, yScale(t));
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const t of xs) ctx.fillText(xFmt(t), xScale(t), p.y + p.h + 8);

    if (this.opts.xLabel) {
      ctx.fillStyle = token('--text-muted');
      ctx.font = FONT_SM;
      ctx.textAlign = 'right';
      ctx.fillText(this.opts.xLabel, p.x + p.w, p.y + p.h + 22);
    }
    if (this.opts.yLabel) {
      ctx.save();
      ctx.translate(12, p.y);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.font = FONT_SM;
      ctx.fillText(this.opts.yLabel, 0, 0);
      ctx.restore();
    }
    ctx.restore();
  }
}

/* ------------------------------------------------------------------ line */

export class LineChart extends BaseChart {
  constructor(host, opts = {}) {
    super(host, Object.assign({ height: 300 }, opts));
    this.data = { x: [], series: [] };
    this.hidden = new Set();
    this.legendHost = null;
  }

  /**
   * data = {
   *   x: number[],                        shared x values
   *   series: [{ key, label, values[], color?, dash?, area?, markers?, width? }],
   *   xLog, yLog, yZero, markers: [{x, label, color}], bands: [{series, lo[], hi[]}]
   * }
   */
  setData(data) {
    this.data = Object.assign({ x: [], series: [] }, data);
    // Reserve room on the right for endpoint labels when they will be drawn.
    // The width has to be measured, not guessed: "Binomial Leisen-Reimer" is
    // three times the width of "delta", and a fixed gutter clips one or wastes
    // space on the other.
    const n = this.data.series.length;
    const base = this.opts.pad?.right ?? 18;
    const labelled = n >= 2 && n <= 4 && this.data.directLabels !== false;
    if (labelled) {
      this.ctx.font = FONT_SM;
      const widest = Math.max(...this.data.series.map(
        (s) => this.ctx.measureText(s.shortLabel || s.label).width,
      ));
      // Cap the gutter at a third of the canvas: past that, drop the labels and
      // let the legend carry identity rather than squeezing the plot.
      const wanted = Math.ceil(widest) + 12;
      if (wanted > this.w / 3) {
        this.data.directLabels = false;
        this.pad.right = base;
      } else {
        this.pad.right = base + wanted;
      }
    } else {
      this.pad.right = base;
    }
    this._renderLegend();
    this.draw();
  }

  _visible() {
    return this.data.series.filter((s) => !this.hidden.has(s.key));
  }

  _renderLegend() {
    if (!this.opts.legendHost) return;
    const host = this.opts.legendHost;
    clear(host);
    const series = this.data.series;
    // A single series needs no legend box: the card title names it.
    if (series.length < 2) return;
    host.className = 'legend';
    series.forEach((s, i) => {
      const color = s.color || seriesColor(i);
      const off = this.hidden.has(s.key);
      host.append(el('button', {
        class: `legend-item${off ? ' is-off' : ''}`,
        type: 'button',
        'aria-pressed': String(!off),
        onclick: () => {
          if (this.hidden.has(s.key)) this.hidden.delete(s.key);
          else this.hidden.add(s.key);
          this._renderLegend();
          this.draw();
        },
      }, el('span', { class: 'swatch line', style: { background: color } }), s.label));
    });
  }

  _scales() {
    const p = this.plot;
    const d = this.data;
    const vis = this._visible();
    const xs = d.x.filter(Number.isFinite);
    let xlo = d.xMin ?? Math.min(...xs);
    let xhi = d.xMax ?? Math.max(...xs);
    if (!(xhi > xlo)) { xlo -= 1; xhi += 1; }

    let vals = [];
    for (const s of vis) for (const v of s.values) if (Number.isFinite(v)) vals.push(v);
    for (const b of d.bands || []) {
      if (this.hidden.has(b.series)) continue;
      for (const v of b.lo) if (Number.isFinite(v)) vals.push(v);
      for (const v of b.hi) if (Number.isFinite(v)) vals.push(v);
    }
    if (!vals.length) vals = [0, 1];
    let ylo = d.yMin ?? Math.min(...vals);
    let yhi = d.yMax ?? Math.max(...vals);
    if (d.yZero && !d.yLog) { ylo = Math.min(ylo, 0); yhi = Math.max(yhi, 0); }
    if (!(yhi > ylo)) { const c = ylo; ylo = c - 1; yhi = c + 1; }
    if (!d.yLog) {
      const padY = (yhi - ylo) * 0.08;
      ylo -= padY; yhi += padY;
    }

    const xScale = d.xLog
      ? (v) => p.x + (Math.log10(Math.max(v, 1e-300)) - Math.log10(xlo)) / (Math.log10(xhi) - Math.log10(xlo)) * p.w
      : (v) => p.x + (v - xlo) / (xhi - xlo) * p.w;
    const yScale = d.yLog
      ? (v) => p.y + p.h - (Math.log10(Math.max(v, 1e-300)) - Math.log10(ylo)) / (Math.log10(yhi) - Math.log10(ylo)) * p.h
      : (v) => p.y + p.h - (v - ylo) / (yhi - ylo) * p.h;

    return { xlo, xhi, ylo, yhi, xScale, yScale };
  }

  draw() {
    const ctx = this.ctx;
    if (!ctx || !this.w) return;
    ctx.clearRect(0, 0, this.w, this.h);
    const d = this.data;
    const p = this.plot;
    // The base constructor sizes the canvas (and so draws) before a subclass has
    // assigned its own data, so every draw path has to tolerate an empty state.
    if (!d || !d.x || !d.x.length) {
      ctx.fillStyle = token('--text-muted');
      ctx.font = FONT;
      ctx.textAlign = 'center';
      ctx.fillText('no data', this.w / 2, this.h / 2);
      return;
    }

    const { xlo, xhi, ylo, yhi, xScale, yScale } = this._scales();
    const xt = d.xLog ? logTicks(xlo, xhi) : ticks(xlo, xhi, d.xTickCount || 6);
    const yt = d.yLog ? logTicks(ylo, yhi) : ticks(ylo, yhi, 6);
    const xFmt = d.xFmt || ((v) => (d.xLog ? fmtSig(v, 2) : fmtAxis(v, xhi - xlo)));
    const yFmt = d.yFmt || ((v) => (d.yLog ? v.toExponential(0) : fmtAxis(v, yhi - ylo)));

    this._grid(ctx, xt, yt, xScale, yScale);

    // Zero rule, drawn one shade stronger than the grid when zero is in range.
    if (!d.yLog && ylo < 0 && yhi > 0) {
      ctx.save();
      ctx.strokeStyle = token('--axis');
      ctx.lineWidth = 1;
      ctx.beginPath();
      const y0 = Math.round(yScale(0)) + 0.5;
      ctx.moveTo(p.x, y0); ctx.lineTo(p.x + p.w, y0);
      ctx.stroke();
      ctx.restore();
    }

    // Vertical reference markers (spot, strike, breakevens...). Their labels sit
    // at the top of the plot, so nearby markers -- spot and forward are often a
    // dollar apart -- get stacked rather than overprinted.
    const placedLabels = [];
    for (const m of d.markers || []) {
      if (m.x < xlo || m.x > xhi) continue;
      const x = Math.round(xScale(m.x)) + 0.5;
      ctx.save();
      ctx.strokeStyle = m.color || token('--border-strong');
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, p.y); ctx.lineTo(x, p.y + p.h);
      ctx.stroke();
      if (m.label) {
        ctx.fillStyle = m.color || token('--text-muted');
        ctx.font = FONT_SM;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        const width = ctx.measureText(m.label).width;
        let row = 0;
        while (placedLabels.some((q) => q.row === row && x < q.right + 6 && x + width > q.left - 6)) row += 1;
        placedLabels.push({ row, left: x + 4, right: x + 4 + width });
        ctx.fillText(m.label, x + 4, p.y + 2 + row * 12);
      }
      ctx.restore();
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(p.x, p.y - 4, p.w, p.h + 8);
    ctx.clip();

    // Uncertainty bands first, so lines sit on top.
    for (const b of d.bands || []) {
      if (this.hidden.has(b.series)) continue;
      const idx = d.series.findIndex((s) => s.key === b.series);
      const color = b.color || (idx >= 0 ? (d.series[idx].color || seriesColor(idx)) : token('--series-1'));
      ctx.save();
      ctx.globalAlpha = 0.16;
      ctx.fillStyle = color;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < d.x.length; i += 1) {
        if (!Number.isFinite(b.hi[i])) continue;
        const X = xScale(d.x[i]); const Y = yScale(b.hi[i]);
        started ? ctx.lineTo(X, Y) : (ctx.moveTo(X, Y), started = true);
      }
      for (let i = d.x.length - 1; i >= 0; i -= 1) {
        if (!Number.isFinite(b.lo[i])) continue;
        ctx.lineTo(xScale(d.x[i]), yScale(b.lo[i]));
      }
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    const vis = this._visible();
    d.series.forEach((s, i) => {
      if (this.hidden.has(s.key)) return;
      const color = s.color || seriesColor(i);

      if (s.area) {
        // Split the fill at zero so sign reads as polarity, not as one blob.
        const y0 = yScale(Math.max(Math.min(0, yhi), ylo));
        for (const sign of [1, -1]) {
          ctx.save();
          ctx.beginPath();
          ctx.rect(p.x, sign > 0 ? p.y : y0, p.w, sign > 0 ? Math.max(y0 - p.y, 0) : Math.max(p.y + p.h - y0, 0));
          ctx.clip();
          ctx.globalAlpha = 0.15;
          ctx.fillStyle = sign > 0 ? (s.areaPos || color) : (s.areaNeg || color);
          ctx.beginPath();
          ctx.moveTo(xScale(d.x[0]), y0);
          for (let j = 0; j < d.x.length; j += 1) {
            if (Number.isFinite(s.values[j])) ctx.lineTo(xScale(d.x[j]), yScale(s.values[j]));
          }
          ctx.lineTo(xScale(d.x[d.x.length - 1]), y0);
          ctx.closePath();
          ctx.fill();
          ctx.restore();
        }
      }

      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = s.width || 2;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      let started = false;
      for (let j = 0; j < d.x.length; j += 1) {
        const v = s.values[j];
        if (!Number.isFinite(v) || (d.yLog && v <= 0)) { started = false; continue; }
        const X = xScale(d.x[j]); const Y = yScale(v);
        if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
      }
      ctx.stroke();

      if (s.markers) {
        for (let j = 0; j < d.x.length; j += 1) {
          const v = s.values[j];
          if (!Number.isFinite(v) || (d.yLog && v <= 0)) continue;
          const X = xScale(d.x[j]); const Y = yScale(v);
          // 2px surface ring instead of a border, so overlapping dots separate.
          ctx.beginPath();
          ctx.arc(X, Y, 4.6, 0, Math.PI * 2);
          ctx.fillStyle = token('--surface-1');
          ctx.fill();
          ctx.beginPath();
          ctx.arc(X, Y, 3.2, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
        }
      }
      ctx.restore();
    });
    ctx.restore();

    // Selective direct labels: series endpoints only, and only when few enough
    // series are visible that the labels cannot collide into noise.
    if (vis.length >= 2 && vis.length <= 4 && d.directLabels !== false) {
      ctx.save();
      ctx.font = FONT_SM;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      const placed = [];
      for (const s of vis) {
        const i = d.series.indexOf(s);
        let last = -1;
        for (let j = d.x.length - 1; j >= 0; j -= 1) {
          if (Number.isFinite(s.values[j]) && !(d.yLog && s.values[j] <= 0)) { last = j; break; }
        }
        if (last < 0) continue;
        let y = yScale(s.values[last]);
        while (placed.some((py) => Math.abs(py - y) < 11)) y += 11;
        if (y < p.y || y > p.y + p.h + 10) continue;
        placed.push(y);
        ctx.fillStyle = s.color || seriesColor(i);
        const x = Math.min(xScale(d.x[last]) + 6, this.w - 4);
        ctx.fillText(s.shortLabel || s.label, x, y);
      }
      ctx.restore();
    }

    this._axes(ctx, xt, yt, xScale, yScale, xFmt, yFmt);

    // Crosshair
    if (this.hover) {
      const { i, px } = this.hover;
      ctx.save();
      ctx.strokeStyle = token('--border-strong');
      ctx.lineWidth = 1;
      ctx.beginPath();
      const x = Math.round(px) + 0.5;
      ctx.moveTo(x, p.y); ctx.lineTo(x, p.y + p.h);
      ctx.stroke();
      for (const s of vis) {
        const v = s.values[i];
        if (!Number.isFinite(v) || (d.yLog && v <= 0)) continue;
        const idx = d.series.indexOf(s);
        ctx.beginPath();
        ctx.arc(px, yScale(v), 5.4, 0, Math.PI * 2);
        ctx.fillStyle = token('--surface-1');
        ctx.fill();
        ctx.beginPath();
        ctx.arc(px, yScale(v), 3.6, 0, Math.PI * 2);
        ctx.fillStyle = s.color || seriesColor(idx);
        ctx.fill();
      }
      ctx.restore();
    }
  }

  _onHover(mx, my) {
    const d = this.data;
    const p = this.plot;
    if (!d || !d.x || !d.x.length || my < p.y - 10 || my > p.y + p.h + 10) {
      if (this.hover) { this.hover = null; this.tooltip.classList.remove('is-visible'); this.draw(); }
      return;
    }
    const { xScale, yScale } = this._scales();
    let best = 0; let bestDist = Infinity;
    for (let i = 0; i < d.x.length; i += 1) {
      const dist = Math.abs(xScale(d.x[i]) - mx);
      if (dist < bestDist) { bestDist = dist; best = i; }
    }
    this.hover = { i: best, px: xScale(d.x[best]) };
    const rows = this._visible().map((s) => {
      const idx = d.series.indexOf(s);
      return {
        label: s.label,
        color: s.color || seriesColor(idx),
        value: (s.fmt || d.valueFmt || ((v) => fmtSig(v, 5)))(s.values[best]),
      };
    });
    const title = (d.xTitle || ((v) => `${d.xName || 'x'} ${fmtSig(v, 5)}`))(d.x[best]);
    let ty = p.y + 8;
    const vals = this._visible().map((s) => s.values[best]).filter(Number.isFinite);
    if (vals.length) ty = Math.min(...vals.map((v) => yScale(v)));
    this._showTooltip(this.hover.px, ty, title, rows);
    this.draw();
  }

  /** The table-view twin: every plotted value, reachable without hovering. */
  tableView() {
    const d = this.data;
    if (!d || !d.x || !d.x.length) return el('div', { class: 'empty' }, 'no data yet');
    const cols = [{ label: d.xName || 'x', get: (r) => r.x, mono: false }];
    d.series.forEach((s, i) => cols.push({
      label: s.label,
      get: (r) => (s.fmt || ((v) => fmtSig(v, 5)))(s.values[r.i]),
    }));
    const step = Math.max(1, Math.ceil(d.x.length / 200));
    const rows = [];
    for (let i = 0; i < d.x.length; i += step) {
      rows.push({ i, x: (d.xTableFmt || ((v) => fmtSig(v, 5)))(d.x[i]) });
    }
    return buildTable(cols, rows, { maxHeight: '340px' });
  }
}

/* --------------------------------------------------------------- heatmap */

export class HeatMap extends BaseChart {
  constructor(host, opts = {}) {
    super(host, Object.assign({ height: 340, pad: { top: 14, right: 18, bottom: 40, left: 66 } }, opts));
    this.data = null;
  }

  /** data = { xs, ys, z: number[nx][ny], diverging, min, max, xName, yName, valueName } */
  setData(data) {
    this.data = data;
    this.draw();
  }

  _colorFor(v) {
    const d = this.data;
    if (!Number.isFinite(v)) return token('--surface-2');
    if (d.diverging) {
      const m = Math.max(Math.abs(d.min), Math.abs(d.max)) || 1;
      return rampAt(divergingRamp(), 0.5 + 0.5 * (v / m));
    }
    const span = (d.max - d.min) || 1;
    return rampAt(sequentialRamp(), (v - d.min) / span);
  }

  draw() {
    const ctx = this.ctx;
    if (!ctx || !this.w) return;
    ctx.clearRect(0, 0, this.w, this.h);
    const d = this.data;
    if (!d || !d.xs.length) return;
    const p = this.plot;

    const nx = d.xs.length;
    const ny = d.ys.length;
    const cw = p.w / nx;
    const ch = p.h / ny;

    for (let i = 0; i < nx; i += 1) {
      for (let j = 0; j < ny; j += 1) {
        ctx.fillStyle = this._colorFor(d.z[i][j]);
        // Cells butt up against each other; the ramp itself carries the edge.
        ctx.fillRect(p.x + i * cw, p.y + p.h - (j + 1) * ch, Math.ceil(cw) + 0.5, Math.ceil(ch) + 0.5);
      }
    }

    const xScale = (v) => p.x + (v - d.xs[0]) / (d.xs[nx - 1] - d.xs[0]) * p.w;
    const yScale = (v) => p.y + p.h - (v - d.ys[0]) / (d.ys[ny - 1] - d.ys[0]) * p.h;

    for (const m of d.markers || []) {
      const x = Math.round(xScale(m.x)) + 0.5;
      if (x < p.x || x > p.x + p.w) continue;
      ctx.save();
      ctx.strokeStyle = token('--text-primary');
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, p.y); ctx.lineTo(x, p.y + p.h);
      ctx.stroke();
      if (m.label) {
        ctx.globalAlpha = 1;
        ctx.fillStyle = token('--text-primary');
        ctx.font = FONT_SM;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(m.label, x + 4, p.y + 3);
      }
      ctx.restore();
    }

    const xt = ticks(d.xs[0], d.xs[nx - 1], 7);
    const yt = ticks(d.ys[0], d.ys[ny - 1], 6);
    this._axes(ctx, xt, yt, xScale, yScale,
      (v) => fmtAxis(v, d.xs[nx - 1] - d.xs[0]),
      (v) => fmtAxis(v, d.ys[ny - 1] - d.ys[0]));

    if (this.hover) {
      const { i, j } = this.hover;
      ctx.save();
      ctx.strokeStyle = token('--text-primary');
      ctx.lineWidth = 2;
      ctx.strokeRect(p.x + i * cw, p.y + p.h - (j + 1) * ch, cw, ch);
      ctx.restore();
    }
  }

  _onHover(mx, my) {
    const d = this.data;
    if (!d) return;
    const p = this.plot;
    const nx = d.xs.length; const ny = d.ys.length;
    const i = Math.floor((mx - p.x) / (p.w / nx));
    const j = Math.floor((p.y + p.h - my) / (p.h / ny));
    if (i < 0 || j < 0 || i >= nx || j >= ny) {
      this.hover = null; this.tooltip.classList.remove('is-visible'); this.draw();
      return;
    }
    this.hover = { i, j };
    const v = d.z[i][j];
    this._showTooltip(
      p.x + (i + 0.5) * (p.w / nx),
      p.y + p.h - (j + 0.7) * (p.h / ny),
      null,
      [
        { label: d.xName || 'x', value: fmtSig(d.xs[i], 5) },
        { label: d.yName || 'y', value: fmtSig(d.ys[j], 5) },
        { label: d.valueName || 'value', value: fmtSig(v, 5), color: this._colorFor(v) },
      ],
    );
    this.draw();
  }

  scaleLegend() {
    const d = this.data;
    if (!d) return el('div');
    const stops = d.diverging ? divergingRamp() : sequentialRamp();
    const css = stops.map((c, i) => `${c} ${(i / (stops.length - 1) * 100).toFixed(1)}%`).join(', ');
    const lo = d.diverging ? -Math.max(Math.abs(d.min), Math.abs(d.max)) : d.min;
    const hi = d.diverging ? Math.max(Math.abs(d.min), Math.abs(d.max)) : d.max;
    return el('div', { class: 'scale-legend' },
      el('span', {}, d.valueName || 'value'),
      el('span', { class: 'ramp-label' }, fmtSig(lo, 3)),
      el('div', { class: 'ramp', style: { background: `linear-gradient(90deg, ${css})` } }),
      el('span', { class: 'ramp-label' }, fmtSig(hi, 3)),
      d.diverging ? el('span', { class: 'dim' }, '· neutral grey at zero') : null);
  }

  tableView() {
    const d = this.data;
    if (!d) return el('div');
    const stepX = Math.max(1, Math.ceil(d.xs.length / 24));
    const stepY = Math.max(1, Math.ceil(d.ys.length / 40));
    const cols = [{ label: `${d.yName || 'y'} \\ ${d.xName || 'x'}`, get: (r) => fmtSig(d.ys[r.j], 4), mono: false }];
    for (let i = 0; i < d.xs.length; i += stepX) {
      cols.push({ label: fmtSig(d.xs[i], 4), get: (r) => fmtSig(d.z[i][r.j], 4) });
    }
    const rows = [];
    for (let j = d.ys.length - 1; j >= 0; j -= stepY) rows.push({ j });
    return buildTable(cols, rows, { maxHeight: '340px' });
  }
}

/* ------------------------------------------------------------------- bar */

export class BarChart extends BaseChart {
  constructor(host, opts = {}) {
    super(host, Object.assign({ height: 280, pad: { top: 14, right: 18, bottom: 34, left: 150 } }, opts));
    this.data = { bars: [] };
  }

  /** data = { bars: [{ label, value, color?, note? }], xLog, valueFmt } */
  setData(data) {
    this.data = Object.assign({ bars: [] }, data);
    this.height = Math.max(120, 20 + this.data.bars.length * 26 + this.pad.bottom);
    this._resize();
  }

  draw() {
    const ctx = this.ctx;
    if (!ctx || !this.w) return;
    ctx.clearRect(0, 0, this.w, this.h);
    const bars = (this.data && this.data.bars) || [];
    const p = this.plot;
    if (!bars.length) return;

    const vals = bars.map((b) => b.value).filter((v) => Number.isFinite(v) && v > 0);
    const lo = this.data.xLog ? Math.min(...vals) / 2 : 0;
    const hi = Math.max(...(vals.length ? vals : [1])) * 1.12;
    const xScale = this.data.xLog
      ? (v) => p.x + (Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * p.w
      : (v) => p.x + (v - lo) / (hi - lo) * p.w;

    const xt = this.data.xLog ? logTicks(lo, hi) : ticks(lo, hi, 6);
    ctx.save();
    ctx.strokeStyle = token('--grid');
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const t of xt) { const x = Math.round(xScale(t)) + 0.5; ctx.moveTo(x, p.y); ctx.lineTo(x, p.y + p.h); }
    ctx.stroke();
    ctx.restore();

    const slot = p.h / bars.length;
    const bh = Math.min(slot - 8, 18);   // the 2px+ gap between adjacent bars
    const radius = 4;

    bars.forEach((b, i) => {
      const y = p.y + i * slot + (slot - bh) / 2;
      const color = b.color || seriesColor(0);
      ctx.save();
      ctx.fillStyle = token('--text-secondary');
      ctx.font = FONT;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(b.label, p.x - 10, y + bh / 2);

      if (!Number.isFinite(b.value)) {
        ctx.fillStyle = token('--text-muted');
        ctx.textAlign = 'left';
        ctx.fillText(b.note || 'did not reach target', p.x + 6, y + bh / 2);
        ctx.restore();
        return;
      }

      const x1 = xScale(b.value);
      ctx.fillStyle = color;
      ctx.beginPath();
      // Rounded data-end, square against the baseline.
      ctx.moveTo(p.x, y);
      ctx.lineTo(Math.max(x1 - radius, p.x), y);
      ctx.quadraticCurveTo(x1, y, x1, y + radius);
      ctx.lineTo(x1, y + bh - radius);
      ctx.quadraticCurveTo(x1, y + bh, Math.max(x1 - radius, p.x), y + bh);
      ctx.lineTo(p.x, y + bh);
      ctx.closePath();
      ctx.fill();

      const label = (this.data.valueFmt || ((v) => fmtSig(v, 3)))(b.value);
      ctx.font = FONT_SM;
      // Only put the label inside the bar when it fits with padding.
      const tw = ctx.measureText(label).width;
      if (x1 - p.x > tw + 16) {
        ctx.fillStyle = token('--surface-1');
        ctx.textAlign = 'right';
        ctx.fillText(label, x1 - 7, y + bh / 2);
      } else {
        ctx.fillStyle = token('--text-secondary');
        ctx.textAlign = 'left';
        ctx.fillText(label, x1 + 7, y + bh / 2);
      }
      ctx.restore();
    });

    ctx.save();
    ctx.strokeStyle = token('--axis');
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(p.x - 0.5, p.y); ctx.lineTo(p.x - 0.5, p.y + p.h);
    ctx.stroke();
    ctx.fillStyle = token('--text-muted');
    ctx.font = FONT;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const t of xt) ctx.fillText(fmtSig(t, 2), xScale(t), p.y + p.h + 8);
    if (this.opts.xLabel) {
      ctx.textAlign = 'right';
      ctx.font = FONT_SM;
      ctx.fillText(this.opts.xLabel, p.x + p.w, p.y + p.h + 21);
    }
    ctx.restore();
  }

  _onHover(mx, my) {
    const bars = (this.data && this.data.bars) || [];
    const p = this.plot;
    const slot = p.h / Math.max(bars.length, 1);
    const i = Math.floor((my - p.y) / slot);
    if (i < 0 || i >= bars.length) {
      this.tooltip.classList.remove('is-visible');
      return;
    }
    const b = bars[i];
    this._showTooltip(Math.max(mx, p.x + 40), p.y + i * slot + 4, b.label, [
      { label: this.opts.xLabel || 'value', value: (this.data.valueFmt || ((v) => fmtSig(v, 4)))(b.value), color: b.color },
      ...(b.note ? [{ label: 'note', value: b.note }] : []),
    ]);
  }
}

/* ------------------------------------------------------- small multiples */

/**
 * A grid of independently scaled single-series line charts.
 *
 * This is the answer to "several measures, wildly different magnitudes". A
 * single plot would need two y-axes (which invents correlations that are not in
 * the data) or would flatten the small series into the baseline. Faceting keeps
 * one axis per chart and one shared x, so the shapes are comparable and the
 * magnitudes are not conflated. Each facet has exactly one series, so no facet
 * needs a legend -- its own title names it.
 */
export class SmallMultiples {
  constructor(host, opts = {}) {
    this.host = host;
    this.opts = Object.assign({ height: 172, columns: 2 }, opts);
    this.charts = new Map();
    host.classList.add('spark-grid');
    host.style.setProperty('--spark-cols', String(this.opts.columns));
  }

  /** panels = [{ key, title, subtitle, values, color, fmt }], plus shared x. */
  setData({ x, panels, xName, xTitle, markers, xFmt }) {
    // Rebuild only when the panel set changes; otherwise reuse the canvases so
    // resizing and hovering stay smooth.
    const keys = panels.map((p) => p.key).join('|');
    if (keys !== this._keys) {
      clear(this.host);
      this.charts.clear();
      for (const p of panels) {
        const card = el('div', { class: 'spark' },
          el('div', { class: 'spark-head' },
            el('span', { class: 'spark-title' },
              el('span', { class: 'swatch line', style: { background: p.color } }), p.title),
            p.subtitle ? el('span', { class: 'spark-sub' }, p.subtitle) : null));
        const plotHost = el('div');
        card.append(plotHost);
        this.host.append(card);
        this.charts.set(p.key, {
          chart: new LineChart(plotHost, {
            height: this.opts.height,
            pad: { top: 10, right: 14, bottom: 26, left: 52 },
          }),
          valueEl: null,
        });
      }
      this._keys = keys;
    }

    for (const p of panels) {
      const entry = this.charts.get(p.key);
      if (!entry) continue;
      entry.chart.setData({
        x,
        series: [{ key: p.key, label: p.title, values: p.values, color: p.color, area: p.area }],
        yZero: true,
        markers,
        xName,
        xTitle,
        xFmt,
        valueFmt: p.fmt || ((v) => fmtSig(v, 5)),
        yFmt: p.yFmt,
        directLabels: false,
      });
    }
    this.panels = panels;
    this.shared = { x, xName, xTitle };
  }

  draw() { for (const { chart } of this.charts.values()) chart.draw(); }

  tableView() {
    const { x, xName } = this.shared || { x: [], xName: 'x' };
    const cols = [{ label: xName || 'x', get: (r) => fmtSig(x[r.i], 5), mono: false }];
    for (const p of this.panels || []) {
      cols.push({ label: p.title, get: (r) => (p.fmt || ((v) => fmtSig(v, 5)))(p.values[r.i]) });
    }
    const step = Math.max(1, Math.ceil(x.length / 200));
    const rows = [];
    for (let i = 0; i < x.length; i += step) rows.push({ i });
    return buildTable(cols, rows, { maxHeight: '340px' });
  }
}

/* ------------------------------------------------------------- utilities */

/**
 * Add a table-view toggle for a chart, satisfying the twin rule: every value on
 * a chart must also be reachable without hovering.
 *
 * `target` may be a chart or a getter, so a card that swaps between two
 * renderers (3-D and heatmap, say) keeps one button pointed at whichever is live.
 */
export function withTableToggle(target, tools) {
  const resolve = () => (typeof target === 'function' ? target() : target);
  const holder = el('div', { style: { display: 'none', marginTop: '12px' } });
  const btn = el('button', { class: 'btn btn-ghost btn-sm', type: 'button' }, 'Table');
  const show = () => { clear(holder).append(resolve().tableView()); };
  btn.addEventListener('click', () => {
    if (holder.style.display !== 'none') {
      holder.style.display = 'none';
      btn.classList.remove('is-active');
    } else {
      show();
      holder.style.display = 'block';
      btn.classList.add('is-active');
    }
  });
  if (tools) tools.append(btn);
  const refresh = () => { if (holder.style.display !== 'none') show(); };
  return { holder, btn, refresh };
}
