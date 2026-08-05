/* Option Chain: load a chain, invert every mid to implied vol, and compare that
   inversion against the feed's own published implied vol. */

import { el, clear, buildTable, fmtSig, fmtSigned, fmtPct, fmtInt, fmtMoney,
         seriesColor, token } from '../util.js';
import { api } from '../api.js';
import { update } from '../state.js';
import { LineChart, withTableToggle } from '../chart.js';
import { setChain, getChain } from './chainstore.js';

export class ChainView {
  constructor(host, ctx) {
    this.host = host;
    this.ctx = ctx;
    this.dirty = true;
    this.expiryIdx = 0;
    this.typeFilter = 'both';
    this._build();
  }

  _build() {
    clear(this.host);
    this.notice = el('div');

    this.tickerInput = el('input', {
      type: 'text', value: 'SPY', placeholder: 'ticker', style: { width: '110px' },
      onkeydown: (e) => { if (e.key === 'Enter') this._load('yahoo'); },
    });
    this.rInput = el('input', { type: 'number', value: '4.3', step: '0.1', style: { width: '74px' } });
    this.qInput = el('input', { type: 'number', value: '0.8', step: '0.1', style: { width: '74px' } });

    this.demoBtn = el('button', { class: 'btn btn-primary btn-sm', type: 'button', onclick: () => this._load('demo') }, 'Load synthetic chain');
    this.liveBtn = el('button', { class: 'btn btn-ghost btn-sm', type: 'button', onclick: () => this._load('yahoo') }, 'Fetch live chain');

    const sourceCard = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          el('h3', {}, 'Data source'),
          el('p', {}, 'The synthetic chain is generated from an arbitrage-free SSVI surface, priced with '
            + 'Black-Scholes, then rounded to exchange ticks and wrapped in a spread — so the true '
            + 'volatility of every quote is known and the inversion can be scored exactly. The live '
            + 'fetch pulls delayed quotes from Yahoo Finance.')),
        el('div', { class: 'card-tools' }, this.demoBtn, this.liveBtn)),
      el('div', { class: 'row' },
        el('span', { class: 'inline-label' }, 'ticker'), this.tickerInput,
        el('span', { class: 'inline-label' }, 'rate %'), this.rInput,
        el('span', { class: 'inline-label' }, 'div yield %'), this.qInput),
      this.summary = el('div', { style: { marginTop: '12px' } }));

    this.expirySelect = el('select', { onchange: (e) => { this.expiryIdx = Number(e.target.value); this._renderExpiry(); } });
    this.typeSeg = el('div', { class: 'segmented' },
      ...[['both', 'Both'], ['call', 'Calls'], ['put', 'Puts']].map(([v, l]) => el('button', {
        type: 'button', class: v === 'both' ? 'is-active' : '',
        onclick: () => {
          this.typeFilter = v;
          [...this.typeSeg.children].forEach((b) => b.classList.toggle('is-active', b.textContent === l));
          this._renderExpiry();
        },
      }, l)));

    this.smileHost = el('div');
    this.smileLegend = el('div');
    this.smileTools = el('div', { class: 'card-tools' },
      el('span', { class: 'inline-label' }, 'expiry'), this.expirySelect, this.typeSeg);
    this.smileCard = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          el('h3', {}, 'Implied volatility by strike'),
          this.smileNote = el('p', {}, '')),
        this.smileTools),
      this.smileHost, this.smileLegend);

    this.tableBox = el('div');
    this.chainCard = el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          el('h3', {}, 'Chain'),
          el('p', {}, 'Click any row to load that contract into the pricer. "Solved IV" is inverted from '
            + 'the bid/ask mid by this app; "feed IV" is the provider\'s own number.')),
        el('div', { class: 'card-tools' })),
      this.tableBox);

    this.host.append(this.notice, sourceCard, this.smileCard, this.chainCard);

    this.smile = new LineChart(this.smileHost, {
      height: 300, legendHost: this.smileLegend, xLabel: 'strike', yLabel: 'implied volatility',
    });
    this.smileTable = withTableToggle(this.smile, this.smileTools);
    this.smileCard.append(this.smileTable.holder);
  }

  async refresh() {
    this.dirty = false;
    if (!getChain()) await this._load('demo');
    else this._renderAll();
  }

  async _load(source) {
    const btn = source === 'demo' ? this.demoBtn : this.liveBtn;
    const original = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Loading…';
    clear(this.notice);
    try {
      const body = {
        source,
        r: (Number(this.rInput.value) || 0) / 100,
        q: (Number(this.qInput.value) || 0) / 100,
      };
      if (source === 'yahoo') body.ticker = this.tickerInput.value.trim();
      const payload = await api.chain(body);
      this.expiryIdx = 0;
      setChain(payload);
      this._renderAll();
      this.ctx.toast(
        `Loaded ${payload.chain.ticker}: ${fmtInt(payload.stats.rows)} contracts across `
        + `${payload.chain.expiries.length} expiries.`, 'good');
    } catch (e) {
      if (source === 'yahoo') {
        clear(this.notice).append(el('div', { class: 'notice bad' }, el('div', {},
          el('strong', {}, 'Live fetch failed. '), e.message || String(e),
          el('div', { style: { marginTop: '6px' } },
            'Yahoo gates its option endpoints and rate-limits aggressively, so this can fail even for a '
            + 'valid ticker. The synthetic chain is always available and is the better demonstration '
            + 'anyway, since it comes with ground truth.'))));
      } else {
        this.ctx.reportError(e, 'chain');
      }
    } finally {
      btn.disabled = false;
      btn.textContent = original;
    }
  }

  _renderAll() {
    const payload = getChain();
    if (!payload) return;
    this.payload = payload;
    this._renderSummary();
    this._renderExpiryOptions();
    this._renderExpiry();
  }

  _renderSummary() {
    const { chain, stats } = this.payload;
    const tiles = [
      { label: 'Underlying', value: chain.ticker, unit: chain.name || '' },
      { label: 'Spot', value: fmtSig(chain.spot, 6), unit: chain.currency || '' },
      { label: 'Contracts', value: fmtInt(stats.rows), unit: `${chain.expiries.length} expiries` },
      { label: 'IV solve rate', value: fmtPct(stats.solve_rate, 1), unit: `${fmtInt(stats.solved)} inverted` },
    ];
    if (stats.iv_vs_feed_rmse_vol_pts !== null && stats.iv_vs_feed_rmse_vol_pts !== undefined) {
      tiles.push({
        label: 'Vs feed IV',
        value: stats.iv_vs_feed_rmse_vol_pts.toFixed(3),
        unit: `vol points RMSE over ${fmtInt(stats.iv_vs_feed_n)} quotes`,
      });
    }

    clear(this.summary).append(
      el('div', { class: 'tiles' }, ...tiles.map((t) => el('div', { class: 'tile' },
        el('div', { class: 'tile-label' }, t.label),
        el('div', { class: 'tile-value' }, t.value),
        el('div', { class: 'tile-unit' }, t.unit)))),
      el('p', { class: 'axis-note' }, chain.note),
    );

    clear(this.notice);
    if (chain.synthetic) {
      const adm = chain.truth && chain.truth.admissibility;
      this.notice.append(el('div', { class: 'notice info' }, el('div', {},
        el('strong', {}, 'Synthetic data. '),
        'These are not market quotes. They come from an SSVI surface with ρ = ',
        el('span', { class: 'mono' }, fmtSigned(chain.truth.params.rho, 3)),
        ', η = ', el('span', { class: 'mono' }, fmtSig(chain.truth.params.eta, 3)),
        ', γ = ', el('span', { class: 'mono' }, fmtSig(chain.truth.params.gamma, 3)),
        adm ? `, which satisfies the no-arbitrage conditions (θφ(1+|ρ|) = ${fmtSig(adm.butterfly_cond_1, 3)} ≤ 4).` : '.',
        ' Because the generating surface is known, the Vol Surface tab can score each interpolator '
        + 'against the truth rather than only against the quotes.')));
    } else if (chain.warnings && chain.warnings.length) {
      this.notice.append(el('div', { class: 'notice warn' }, el('div', {},
        el('strong', {}, 'Partial fetch. '), chain.warnings.join(' · '))));
    }
  }

  _renderExpiryOptions() {
    const exps = this.payload.chain.expiries;
    clear(this.expirySelect).append(...exps.map((e, i) => el('option', { value: String(i) },
      `${e.label} · ${e.days}d`)));
    this.expiryIdx = Math.min(this.expiryIdx, exps.length - 1);
    this.expirySelect.value = String(this.expiryIdx);
  }

  _renderExpiry() {
    const chain = this.payload.chain;
    const exp = chain.expiries[this.expiryIdx];
    if (!exp) return;

    const rows = exp.rows.filter((r) => this.typeFilter === 'both' || r.type === this.typeFilter);
    const strikes = [...new Set(rows.map((r) => r.strike))].sort((a, b) => a - b);

    const pick = (type, field) => strikes.map((k) => {
      const r = rows.find((x) => x.strike === k && x.type === type);
      return r && r[field] !== null && r[field] !== undefined ? r[field] : null;
    });

    const series = [];
    let slot = 0;
    if (this.typeFilter !== 'put') {
      series.push({ key: 'call_solved', label: 'Call IV (solved from mid)', shortLabel: 'call', values: pick('call', 'iv_solved'), color: seriesColor(slot++), markers: true });
    }
    if (this.typeFilter !== 'call') {
      series.push({ key: 'put_solved', label: 'Put IV (solved from mid)', shortLabel: 'put', values: pick('put', 'iv_solved'), color: seriesColor(slot++), markers: true });
    }
    const feed = strikes.map((k) => {
      const r = rows.find((x) => x.strike === k && x.iv_market);
      return r ? r.iv_market : null;
    });
    if (feed.some((v) => v !== null)) {
      series.push({ key: 'feed', label: 'Feed IV', shortLabel: 'feed', values: feed, color: seriesColor(slot++), width: 1 });
    }
    if (chain.synthetic) {
      const truth = strikes.map((k) => {
        const r = rows.find((x) => x.strike === k && x.iv_truth);
        return r ? r.iv_truth : null;
      });
      series.push({ key: 'truth', label: 'True IV (SSVI)', shortLabel: 'truth', values: truth, color: seriesColor(slot++), width: 2 });
    }

    this.smile.setData({
      x: strikes,
      series,
      xName: 'strike',
      xTitle: (v) => `strike ${fmtSig(v, 6)}`,
      valueFmt: (v) => (Number.isFinite(v) ? fmtPct(v, 2) : '—'),
      yFmt: (v) => `${(v * 100).toFixed(0)}%`,
      markers: [
        { x: chain.spot, label: `spot ${fmtSig(chain.spot, 5)}`, color: token('--text-muted') },
        { x: exp.forward, label: `fwd ${fmtSig(exp.forward, 5)}`, color: token('--border-strong') },
      ],
    });
    this.smileTable.refresh();

    this.smileNote.textContent = `${exp.label} · ${exp.days} days · T = ${exp.T.toFixed(5)}y · `
      + `forward ${fmtSig(exp.forward, 6)}. Calls and puts at the same strike should imply the same `
      + `volatility; where they diverge, the bid/ask spread is wider than the information content.`;

    this._renderTable(exp, rows);
  }

  _renderTable(exp, rows) {
    const sorted = [...rows].sort((a, b) => a.strike - b.strike || a.type.localeCompare(b.type));
    const cols = [
      { label: 'Strike', get: (r) => fmtSig(r.strike, 6) },
      { label: 'Type', mono: false, get: (r) => el('span', { class: `tag ${r.type === 'call' ? 'neutral' : 'neutral'}` }, r.type) },
      { label: 'Bid', get: (r) => (r.bid === null || r.bid === undefined ? '—' : fmtSig(r.bid, 2)) },
      { label: 'Ask', get: (r) => (r.ask === null || r.ask === undefined ? '—' : fmtSig(r.ask, 2)) },
      { label: 'Mid', get: (r) => (r.mid ? fmtSig(r.mid, 4) : '—') },
      { label: 'Spread', get: (r) => {
        if (!r.mid || r.bid === null || r.ask === null || r.bid === undefined || r.ask === undefined) return '—';
        return fmtPct((r.ask - r.bid) / r.mid, 1);
      }, cls: (r) => {
        if (!r.mid || r.bid === null || r.ask === null) return 'dim';
        return (r.ask - r.bid) / r.mid > 0.35 ? 'neg' : '';
      } },
      { label: 'Solved IV', get: (r) => (r.iv_solved ? fmtPct(r.iv_solved, 2) : '—'),
        cls: (r) => (r.iv_solved ? '' : 'dim') },
      { label: 'Feed IV', get: (r) => (r.iv_market ? fmtPct(r.iv_market, 2) : '—') },
      { label: 'Δ vs feed', get: (r) => (r.iv_solved && r.iv_market ? fmtSigned((r.iv_solved - r.iv_market) * 100, 2) : '—'),
        cls: (r) => {
          if (!r.iv_solved || !r.iv_market) return 'dim';
          return Math.abs(r.iv_solved - r.iv_market) > 0.02 ? 'neg' : '';
        } },
      this.payload.chain.synthetic
        ? { label: 'Δ vs truth', get: (r) => (r.iv_solved && r.iv_truth ? fmtSigned((r.iv_solved - r.iv_truth) * 100, 2) : '—') }
        : null,
      { label: 'Volume', get: (r) => fmtInt(r.volume || 0) },
      { label: 'OI', get: (r) => fmtInt(r.open_interest || 0) },
      { label: 'Newton iters', get: (r) => (r.iv_iterations ? fmtInt(r.iv_iterations) : '—') },
    ].filter(Boolean);

    const table = buildTable(cols, sorted, { maxHeight: '460px' });
    // Clicking a row loads that contract into the shared pricer state.
    [...table.querySelectorAll('tbody tr')].forEach((tr, i) => {
      const r = sorted[i];
      tr.style.cursor = 'pointer';
      tr.title = 'Load this contract into the pricer';
      tr.addEventListener('click', () => {
        update({
          S: this.payload.chain.spot,
          K: r.strike,
          T: exp.T,
          r: this.payload.chain.r,
          q: this.payload.chain.q,
          is_call: r.type === 'call',
          ...(r.iv_solved ? { sigma: r.iv_solved } : {}),
        }, 'chain');
        this.ctx.toast(`Loaded ${r.type} K=${fmtSig(r.strike, 6)} ${exp.days}d into the pricer.`, 'info');
      });
    });

    clear(this.tableBox).append(table,
      el('p', { class: 'axis-note' },
        'Wide spreads are highlighted: a quote whose spread exceeds 35% of its mid carries little '
        + 'volatility information, which is why the surface fitter filters on exactly that.'));
  }

  redraw() {
    if (this.smile) this.smile.draw();
  }
}
