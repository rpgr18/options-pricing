/* ==========================================================================
   Interactive 3-D surface renderer on a 2-D canvas.

   No WebGL and no dependencies: the surface is a grid of quads projected with
   an orthographic camera, depth-sorted back to front (painter's algorithm) and
   filled from the same sequential/diverging ramps the 2-D charts use, modulated
   by a diffuse shading term from each quad's normal. Quads are drawn with a
   hairline seam in their own fill colour, which closes the gaps antialiasing
   would otherwise leave between neighbours.

   Interaction: drag to orbit, wheel to zoom, hover to read a value. Because
   colour alone cannot be read precisely off a 3-D surface, every caller pairs
   this with a scale legend and a table view.
   ========================================================================== */

import { el, clear, token, rampAt, sequentialRamp, divergingRamp, fmtSig } from './util.js';

const FONT = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
const FONT_SM = '10px system-ui, -apple-system, "Segoe UI", sans-serif';

function shade(rgb, factor) {
  const m = /rgb\((\d+),\s*(\d+),\s*(\d+)\)/.exec(rgb);
  if (!m) return rgb;
  const c = [1, 2, 3].map((i) => Math.max(0, Math.min(255, Math.round(Number(m[i]) * factor))));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export class Surface3D {
  constructor(host, opts = {}) {
    this.host = host;
    this.opts = Object.assign({ height: 420, zLabel: 'value' }, opts);
    host.classList.add('surface-host');
    clear(host);

    this.canvas = el('canvas');
    this.tooltip = el('div', { class: 'chart-tooltip' });
    this.hint = el('div', { class: 'surface-hint' }, 'drag to orbit · wheel to zoom');
    host.append(this.canvas, this.tooltip, this.hint);
    this.ctx = this.canvas.getContext('2d');

    this.yaw = -0.72;
    this.pitch = 0.52;
    this.zoom = 1;
    this.zScale = 0.62;
    this.data = null;
    this.hover = null;
    this.showWire = true;
    this.showQuotes = true;
    // Quoted strike ranges widen with maturity, so the region a surface is
    // actually fitted on is a cone, not a rectangle. Rendering the rectangle
    // means a 4-day option 70% out of the money -- pure extrapolation -- gets
    // the same visual weight as the fitted core, and its extreme values
    // dominate both the shape and the eye. Off by default; toggleable.
    this.showExtrapolated = false;

    this._bind();
    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(host);
    this._resize();
  }

  destroy() { this._ro.disconnect(); }

  /**
   * data = {
   *   xs, ys,                 grid axes (xs -> horizontal, ys -> depth)
   *   z: number[nx][ny],      surface height / colour value
   *   covered?: bool[nx][ny], false marks extrapolated cells, drawn muted
   *   diverging?: bool,
   *   xName, yName, zName, xFmt, yFmt, zFmt,
   *   points?: [{x, y, z}]    scatter overlay, e.g. the market quotes
   * }
   */
  setData(data) {
    this.data = data;
    if (data) {
      const flat = [];
      for (let i = 0; i < data.z.length; i += 1) {
        for (let j = 0; j < data.z[i].length; j += 1) {
          const v = data.z[i][j];
          const cov = !data.covered || data.covered[i][j];
          if (Number.isFinite(v) && cov) flat.push(v);
        }
      }
      // Scale from the covered region only: an extrapolated corner should not
      // set the colour range for the whole surface.
      if (flat.length) {
        this.vmin = data.vmin ?? Math.min(...flat);
        this.vmax = data.vmax ?? Math.max(...flat);
      } else {
        this.vmin = 0; this.vmax = 1;
      }
      if (this.vmax <= this.vmin) this.vmax = this.vmin + 1e-9;
    }
    this.draw();
  }

  setView({ yaw, pitch, zoom, zScale } = {}) {
    if (yaw !== undefined) this.yaw = yaw;
    if (pitch !== undefined) this.pitch = pitch;
    if (zoom !== undefined) this.zoom = zoom;
    if (zScale !== undefined) this.zScale = zScale;
    this.draw();
  }

  resetView() { this.setView({ yaw: -0.72, pitch: 0.52, zoom: 1 }); }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(this.host.clientWidth, 240);
    const h = this.opts.height;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.canvas.style.height = `${h}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = w; this.h = h;
    this.draw();
  }

  _bind() {
    let dragging = false;
    let lastX = 0; let lastY = 0;

    this.canvas.addEventListener('pointerdown', (ev) => {
      dragging = true;
      lastX = ev.clientX; lastY = ev.clientY;
      this.host.classList.add('is-dragging');
      this.canvas.setPointerCapture(ev.pointerId);
    });
    this.canvas.addEventListener('pointerup', (ev) => {
      dragging = false;
      this.host.classList.remove('is-dragging');
      try { this.canvas.releasePointerCapture(ev.pointerId); } catch { /* already released */ }
    });
    this.canvas.addEventListener('pointermove', (ev) => {
      if (dragging) {
        this.yaw += (ev.clientX - lastX) * 0.008;
        this.pitch = Math.max(-0.12, Math.min(1.4, this.pitch + (ev.clientY - lastY) * 0.006));
        lastX = ev.clientX; lastY = ev.clientY;
        this.tooltip.classList.remove('is-visible');
        this.hover = null;
        this.draw();
      } else {
        const rect = this.canvas.getBoundingClientRect();
        this._pick(ev.clientX - rect.left, ev.clientY - rect.top);
      }
    });
    this.canvas.addEventListener('pointerleave', () => {
      this.hover = null;
      this.tooltip.classList.remove('is-visible');
      this.draw();
    });
    this.canvas.addEventListener('wheel', (ev) => {
      ev.preventDefault();
      this.zoom = Math.max(0.45, Math.min(2.6, this.zoom * (ev.deltaY > 0 ? 0.92 : 1.08)));
      this.draw();
    }, { passive: false });
  }

  /* ---- projection ---- */

  _project(u, v, t) {
    // u, v, t are already in [-1,1] x [-1,1] x [-0.5,0.5] model space.
    const cy = Math.cos(this.yaw); const sy = Math.sin(this.yaw);
    const cp = Math.cos(this.pitch); const sp = Math.sin(this.pitch);
    const X = u * cy - v * sy;
    const Y = u * sy + v * cy;
    const depth = Y * cp + t * sp;
    const up = -Y * sp + t * cp;
    // 0.26 and the upward offset leave room for the floor tick labels and axis
    // names, which sit outside the unit square and would otherwise clip.
    const s = Math.min(this.w, this.h * 1.7) * 0.26 * this.zoom;
    return {
      sx: this.w / 2 + X * s,
      sy: this.h / 2 - up * s - this.h * 0.07,
      depth,
    };
  }

  /** Project a grid index + value straight to screen space. */
  _pm(i, j, v) {
    const m = this._model(i, j, v);
    return this._project(m.u, m.v, m.t);
  }

  _model(i, j, v) {
    const d = this.data;
    const nx = d.xs.length; const ny = d.ys.length;
    const u = nx > 1 ? (i / (nx - 1)) * 2 - 1 : 0;
    const vv = ny > 1 ? (j / (ny - 1)) * 2 - 1 : 0;
    const t = ((v - this.vmin) / (this.vmax - this.vmin) - 0.5) * 2 * this.zScale;
    return { u, v: vv, t: Number.isFinite(t) ? Math.max(-1.4, Math.min(1.4, t)) : 0 };
  }

  _colorFor(v) {
    if (!Number.isFinite(v)) return token('--surface-2');
    if (this.data.diverging) {
      const m = Math.max(Math.abs(this.vmin), Math.abs(this.vmax)) || 1;
      return rampAt(divergingRamp(), 0.5 + 0.5 * (v / m));
    }
    return rampAt(sequentialRamp(), (v - this.vmin) / (this.vmax - this.vmin));
  }

  /* ---- draw ---- */

  draw() {
    const ctx = this.ctx;
    if (!ctx || !this.w) return;
    ctx.clearRect(0, 0, this.w, this.h);
    const d = this.data;
    if (!d || !d.xs || !d.xs.length) {
      ctx.fillStyle = token('--text-muted');
      ctx.font = FONT;
      ctx.textAlign = 'center';
      ctx.fillText('no surface', this.w / 2, this.h / 2);
      return;
    }

    this._drawFloor(ctx);

    const nx = d.xs.length; const ny = d.ys.length;
    const quads = [];
    for (let i = 0; i < nx - 1; i += 1) {
      for (let j = 0; j < ny - 1; j += 1) {
        const vs = [d.z[i][j], d.z[i + 1][j], d.z[i + 1][j + 1], d.z[i][j + 1]];
        if (vs.some((v) => !Number.isFinite(v))) continue;
        const pts = [
          this._pm(i, j, vs[0]),
          this._pm(i + 1, j, vs[1]),
          this._pm(i + 1, j + 1, vs[2]),
          this._pm(i, j + 1, vs[3]),
        ];
        const anyCovered = !d.covered || d.covered[i][j] || d.covered[i + 1][j] || d.covered[i + 1][j + 1] || d.covered[i][j + 1];
        const covered = !d.covered || (d.covered[i][j] && d.covered[i + 1][j] && d.covered[i + 1][j + 1] && d.covered[i][j + 1]);
        if (!anyCovered && !this.showExtrapolated) continue;
        // Screen-space cross product gives the facing term without needing the
        // full world normal; it is enough for a stable diffuse shade.
        const ax = pts[1].sx - pts[0].sx; const ay = pts[1].sy - pts[0].sy;
        const bx = pts[3].sx - pts[0].sx; const by = pts[3].sy - pts[0].sy;
        const area = ax * by - ay * bx;
        quads.push({
          pts,
          depth: (pts[0].depth + pts[1].depth + pts[2].depth + pts[3].depth) / 4,
          value: (vs[0] + vs[1] + vs[2] + vs[3]) / 4,
          slope: Math.abs(area) < 1e-9 ? 0 : Math.min(1, Math.abs(area) / 900),
          covered,
        });
      }
    }
    quads.sort((a, b) => b.depth - a.depth);

    for (const qd of quads) {
      const base = this._colorFor(qd.value);
      // Steeper quads present less area to the viewer: darken them slightly so
      // the shape reads even where the colour value barely changes.
      const lit = 0.80 + 0.34 * qd.slope;
      ctx.fillStyle = qd.covered ? shade(base, lit) : shade(base, lit * 0.52);
      ctx.strokeStyle = ctx.fillStyle;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(qd.pts[0].sx, qd.pts[0].sy);
      for (let k = 1; k < 4; k += 1) ctx.lineTo(qd.pts[k].sx, qd.pts[k].sy);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }

    if (this.showWire) this._drawWire(ctx);
    if (this.showQuotes && d.points) this._drawPoints(ctx);
    this._drawAxes(ctx);
    if (this.hover) this._drawHoverMark(ctx);
  }

  _drawFloor(ctx) {
    const d = this.data;
    const t = -this.zScale * 1.08;
    const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([u, v]) => this._project(u, v, t));
    ctx.save();
    ctx.fillStyle = token('--surface-2');
    ctx.globalAlpha = 0.55;
    ctx.beginPath();
    ctx.moveTo(corners[0].sx, corners[0].sy);
    for (let i = 1; i < 4; i += 1) ctx.lineTo(corners[i].sx, corners[i].sy);
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1;

    ctx.strokeStyle = token('--grid');
    ctx.lineWidth = 1;
    ctx.beginPath();
    const N = 8;
    for (let g = 0; g <= N; g += 1) {
      const a = (g / N) * 2 - 1;
      const p1 = this._project(a, -1, t); const p2 = this._project(a, 1, t);
      const p3 = this._project(-1, a, t); const p4 = this._project(1, a, t);
      ctx.moveTo(p1.sx, p1.sy); ctx.lineTo(p2.sx, p2.sy);
      ctx.moveTo(p3.sx, p3.sy); ctx.lineTo(p4.sx, p4.sy);
    }
    ctx.stroke();
    ctx.restore();
  }

  _drawWire(ctx) {
    const d = this.data;
    const nx = d.xs.length; const ny = d.ys.length;
    // Thin out the wireframe on dense grids so it stays a hint, not a mesh blob.
    const stepI = Math.max(1, Math.round(nx / 16));
    const stepJ = Math.max(1, Math.round(ny / 12));
    ctx.save();
    ctx.strokeStyle = token('--text-primary');
    ctx.globalAlpha = 0.13;
    ctx.lineWidth = 1;
    ctx.beginPath();
    const shown = (i, j) => (!d.covered || this.showExtrapolated || d.covered[i][j]);
    for (let i = 0; i < nx; i += stepI) {
      let started = false;
      for (let j = 0; j < ny; j += 1) {
        const v = d.z[i][j];
        if (!Number.isFinite(v) || !shown(i, j)) { started = false; continue; }
        const p = this._pm(i, j, v);
        started ? ctx.lineTo(p.sx, p.sy) : (ctx.moveTo(p.sx, p.sy), started = true);
      }
    }
    for (let j = 0; j < ny; j += stepJ) {
      let started = false;
      for (let i = 0; i < nx; i += 1) {
        const v = d.z[i][j];
        if (!Number.isFinite(v) || !shown(i, j)) { started = false; continue; }
        const p = this._pm(i, j, v);
        started ? ctx.lineTo(p.sx, p.sy) : (ctx.moveTo(p.sx, p.sy), started = true);
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  _drawPoints(ctx) {
    const d = this.data;
    const xs = d.xs; const ys = d.ys;
    const xlo = xs[0]; const xhi = xs[xs.length - 1];
    const ylo = ys[0]; const yhi = ys[ys.length - 1];
    ctx.save();
    for (const pt of d.points) {
      if (pt.x < xlo || pt.x > xhi || pt.y < ylo || pt.y > yhi) continue;
      const i = ((pt.x - xlo) / (xhi - xlo)) * (xs.length - 1);
      const j = ((pt.y - ylo) / (yhi - ylo)) * (ys.length - 1);
      const p = this._pm(i, j, pt.z);
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, 3.6, 0, Math.PI * 2);
      ctx.fillStyle = token('--surface-1');
      ctx.fill();
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, 2.2, 0, Math.PI * 2);
      ctx.fillStyle = token('--text-primary');
      ctx.fill();
    }
    ctx.restore();
  }

  _drawAxes(ctx) {
    const d = this.data;
    const t = -this.zScale * 1.08;
    ctx.save();
    ctx.font = FONT_SM;
    ctx.fillStyle = token('--text-muted');
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const xFmt = d.xFmt || ((v) => fmtSig(v, 3));
    const yFmt = d.yFmt || ((v) => fmtSig(v, 3));

    // Label the two floor edges nearest the camera.
    const frontV = Math.cos(this.yaw) > 0 ? -1 : 1;
    const frontU = Math.sin(this.yaw) > 0 ? 1 : -1;

    for (let g = 0; g <= 4; g += 1) {
      const a = (g / 4) * 2 - 1;
      const px = this._project(a, frontV * 1.14, t);
      const val = d.xs[0] + ((a + 1) / 2) * (d.xs[d.xs.length - 1] - d.xs[0]);
      ctx.fillText(xFmt(val), px.sx, px.sy);

      const py = this._project(frontU * 1.14, a, t);
      const valY = d.ys[0] + ((a + 1) / 2) * (d.ys[d.ys.length - 1] - d.ys[0]);
      ctx.fillText(yFmt(valY), py.sx, py.sy);
    }

    ctx.fillStyle = token('--text-secondary');
    ctx.font = FONT;
    const cx = this._project(0, frontV * 1.34, t);
    ctx.fillText(d.xName || '', cx.sx, cx.sy);
    const cyp = this._project(frontU * 1.34, 0, t);
    ctx.fillText(d.yName || '', cyp.sx, cyp.sy);

    // Vertical scale at the rear corner.
    const corner = [-frontU, -frontV];
    ctx.textAlign = 'right';
    ctx.font = FONT_SM;
    ctx.fillStyle = token('--text-muted');
    const zFmt = d.zFmt || ((v) => fmtSig(v, 3));
    const span = this.vmax - this.vmin;
    for (let g = 0; g <= 3; g += 1) {
      const frac = g / 3;
      let val = this.vmin + frac * span;
      // Snap values that are zero to within rounding of the axis range. Deep
      // out-of-the-money gamma is ~1e-55, and printing that as a tick label is
      // noise, not precision.
      if (Math.abs(val) < span * 1e-9) val = 0;
      const tt = (frac - 0.5) * 2 * this.zScale;
      const p = this._project(corner[0] * 1.06, corner[1] * 1.06, tt);
      ctx.fillText(zFmt(val), p.sx - 4, p.sy);
    }
    ctx.restore();
  }

  _drawHoverMark(ctx) {
    const { i, j, v } = this.hover;
    const m = this._model(i, j, v);
    const p = this._project(m.u, m.v, m.t);
    const floor = this._project(m.u, m.v, -this.zScale * 1.08);
    ctx.save();
    ctx.strokeStyle = token('--text-primary');
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(floor.sx, floor.sy);
    ctx.lineTo(p.sx, p.sy);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.arc(p.sx, p.sy, 5.4, 0, Math.PI * 2);
    ctx.fillStyle = token('--surface-1');
    ctx.fill();
    ctx.beginPath();
    ctx.arc(p.sx, p.sy, 3.4, 0, Math.PI * 2);
    ctx.fillStyle = token('--text-primary');
    ctx.fill();
    ctx.restore();
  }

  _pick(mx, my) {
    const d = this.data;
    if (!d) return;
    const nx = d.xs.length; const ny = d.ys.length;
    // Subsample the search on big grids: the hit target stays generous either way.
    const stepI = Math.max(1, Math.round(nx / 60));
    const stepJ = Math.max(1, Math.round(ny / 60));
    let best = null; let bestD = 24 * 24;
    for (let i = 0; i < nx; i += stepI) {
      for (let j = 0; j < ny; j += stepJ) {
        const v = d.z[i][j];
        if (!Number.isFinite(v)) continue;
        if (d.covered && !this.showExtrapolated && !d.covered[i][j]) continue;
        const p = this._pm(i, j, v);
        const dist = (p.sx - mx) ** 2 + (p.sy - my) ** 2;
        if (dist < bestD) { bestD = dist; best = { i, j, v, p }; }
      }
    }
    if (!best) {
      this.hover = null;
      this.tooltip.classList.remove('is-visible');
      this.draw();
      return;
    }
    this.hover = best;
    const covered = !d.covered || d.covered[best.i][best.j];
    const rows = [
      { label: d.xName || 'x', value: (d.xFmt || ((v) => fmtSig(v, 5)))(d.xs[best.i]) },
      { label: d.yName || 'y', value: (d.yFmt || ((v) => fmtSig(v, 5)))(d.ys[best.j]) },
      { label: d.zName || 'value', value: (d.zFmt || ((v) => fmtSig(v, 5)))(best.v), color: this._colorFor(best.v) },
    ];
    if (!covered) rows.push({ label: 'region', value: 'extrapolated' });
    const t = this.tooltip;
    clear(t);
    for (const r of rows) {
      t.append(el('div', { class: 'tt-row' },
        el('span', { class: 'tt-key' }, r.color ? el('span', { class: 'swatch', style: { background: r.color } }) : null, r.label),
        el('span', { class: 'tt-val' }, r.value)));
    }
    t.classList.add('is-visible');
    const tw = t.offsetWidth;
    t.style.left = `${Math.min(Math.max(best.p.sx, tw / 2 + 4), this.w - tw / 2 - 4)}px`;
    t.style.top = `${Math.max(best.p.sy - 12, 28)}px`;
    this.draw();
  }

  scaleLegend(label) {
    const stops = this.data && this.data.diverging ? divergingRamp() : sequentialRamp();
    const css = stops.map((c, i) => `${c} ${(i / (stops.length - 1) * 100).toFixed(1)}%`).join(', ');
    const fmt = (this.data && this.data.zFmt) || ((v) => fmtSig(v, 3));
    const lo = this.data && this.data.diverging ? -Math.max(Math.abs(this.vmin), Math.abs(this.vmax)) : this.vmin;
    const hi = this.data && this.data.diverging ? Math.max(Math.abs(this.vmin), Math.abs(this.vmax)) : this.vmax;
    return el('div', { class: 'scale-legend' },
      el('span', {}, label || (this.data && this.data.zName) || 'value'),
      el('span', { class: 'ramp-label' }, fmt(lo)),
      el('div', { class: 'ramp', style: { background: `linear-gradient(90deg, ${css})` } }),
      el('span', { class: 'ramp-label' }, fmt(hi)),
      this.data && this.data.covered
        ? el('span', { class: 'dim' }, this.showExtrapolated
          ? '· dimmed = extrapolated beyond the quoted strikes'
          : '· drawn only where strikes are actually quoted')
        : null);
  }

  tableView() {
    const d = this.data;
    if (!d) return el('div');
    const stepX = Math.max(1, Math.ceil(d.xs.length / 20));
    const stepY = Math.max(1, Math.ceil(d.ys.length / 34));
    const xFmt = d.xFmt || ((v) => fmtSig(v, 4));
    const yFmt = d.yFmt || ((v) => fmtSig(v, 4));
    const zFmt = d.zFmt || ((v) => fmtSig(v, 4));

    const thead = el('thead', {}, el('tr', {},
      el('th', {}, `${d.yName || 'y'} \\ ${d.xName || 'x'}`),
      ...Array.from({ length: Math.ceil(d.xs.length / stepX) }, (_, n) => el('th', {}, xFmt(d.xs[n * stepX])))));
    const rows = [];
    for (let j = d.ys.length - 1; j >= 0; j -= stepY) {
      rows.push(el('tr', {},
        el('td', {}, yFmt(d.ys[j])),
        ...Array.from({ length: Math.ceil(d.xs.length / stepX) }, (_, n) => {
          const i = n * stepX;
          const cov = !d.covered || d.covered[i][j];
          return el('td', { class: `num${cov ? '' : ' dim'}` }, zFmt(d.z[i][j]));
        })));
    }
    return el('div', { class: 'table-wrap', style: { maxHeight: '340px', overflowY: 'auto' } },
      el('table', { class: 'data' }, thead, el('tbody', {}, rows)));
  }
}
