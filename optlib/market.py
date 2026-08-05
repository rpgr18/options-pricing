"""Option chain sources: a synthetic SSVI generator and a live Yahoo Finance fetch.

The synthetic generator is not a toy. It uses **SSVI** (Gatheral-Jacquier 2014),

    w(k, theta) = theta/2 * { 1 + rho*phi(theta)*k
                              + sqrt((phi(theta)*k + rho)^2 + (1 - rho^2)) }

with the power-law phi(theta) = eta / (theta^gamma * (1+theta)^(1-gamma)), which
is provably free of butterfly and calendar arbitrage under explicit conditions
on (eta, rho, gamma) that are checked below. That gives the app two things a
live feed cannot: a surface that is *known* to be admissible, and a ground-truth
volatility for every quote.

Prices are generated from that surface, then rounded to a real tick and wrapped
in a bid/ask spread, so inverting the mid back to implied vol reproduces the
actual difficulty of the task -- discretization noise that blows up in the wings
where vega is small. The gap between the recovered surface and the known truth
is therefore a genuine measurement, reported as `truth_rmse_vol_pts`.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import numpy as np

from . import blackscholes as bs
from .implied_vol import implied_vol

DAYS_PER_YEAR = 365.0
DEFAULT_EXPIRY_DAYS = (7, 14, 30, 60, 91, 182, 273, 365, 547, 730)


# --------------------------------------------------------------------------
# SSVI
# --------------------------------------------------------------------------

@dataclass
class SSVIParams:
    """SSVI surface parameters plus an ATM total-variance term structure."""

    rho: float = -0.62          # spot/vol correlation: equity skew is negative
    eta: float = 1.15           # overall smile curvature level
    gamma: float = 0.42         # how fast curvature decays with maturity
    atm_vol_short: float = 0.31  # ATM vol at the front
    atm_vol_long: float = 0.215  # ATM vol asymptote
    atm_decay: float = 1.35      # years over which the term structure relaxes

    def theta(self, T):
        """ATM total variance theta(T) = sigma_atm(T)^2 * T (non-decreasing in T)."""
        T = np.asarray(T, dtype=float)
        vol = self.atm_vol_long + (self.atm_vol_short - self.atm_vol_long) * np.exp(-T / self.atm_decay)
        return vol * vol * T

    def phi(self, theta):
        theta = np.maximum(np.asarray(theta, dtype=float), 1e-12)
        return self.eta / (np.power(theta, self.gamma) * np.power(1.0 + theta, 1.0 - self.gamma))

    def total_variance(self, k, T):
        k = np.asarray(k, dtype=float)
        th = self.theta(T)
        ph = self.phi(th)
        x = ph * k
        return 0.5 * th * (1.0 + self.rho * x + np.sqrt((x + self.rho) ** 2 + (1.0 - self.rho ** 2)))

    def iv(self, k, T):
        T = np.asarray(T, dtype=float)
        return np.sqrt(self.total_variance(k, T) / np.maximum(T, 1e-12))

    def admissibility(self, T_grid=None) -> dict:
        """Check the Gatheral-Jacquier no-arbitrage conditions on this surface."""
        if T_grid is None:
            T_grid = np.linspace(1.0 / 365.0, 3.0, 240)
        th = self.theta(T_grid)
        ph = self.phi(th)
        one = float(np.max(th * ph * (1.0 + abs(self.rho))))
        two = float(np.max(th * ph * ph * (1.0 + abs(self.rho))))
        # Calendar condition: theta non-decreasing and theta*phi(theta) non-decreasing.
        d_theta = float(np.min(np.diff(th)))
        d_thetaphi = float(np.min(np.diff(th * ph)))
        return {
            "butterfly_cond_1": one,
            "butterfly_cond_2": two,
            "butterfly_ok": bool(one <= 4.0 + 1e-9 and two <= 4.0 + 1e-9),
            "min_d_theta": d_theta,
            "min_d_theta_phi": d_thetaphi,
            "calendar_ok": bool(d_theta >= -1e-12 and d_thetaphi >= -1e-9),
            "note": "theta*phi*(1+|rho|) <= 4 and theta*phi^2*(1+|rho|) <= 4 give no butterfly arbitrage.",
        }


# --------------------------------------------------------------------------
# Chain assembly
# --------------------------------------------------------------------------

def _tick_for(price: float) -> float:
    """US listed option tick: a penny under $3, a nickel above."""
    return 0.01 if price < 3.0 else 0.05


def _solve_chain_ivs(rows: list[dict], spot: float, r: float, q: float, T: float) -> None:
    """Fill iv_solved / iv_error in place by inverting each mid price."""
    for row in rows:
        mid = row.get("mid")
        if mid is None or not math.isfinite(mid) or mid <= 0.0:
            row["iv_solved"] = None
            row["iv_status"] = "no mid"
            continue
        res = implied_vol(mid, spot, row["strike"], T, r, q, row["type"] == "call")
        row["iv_solved"] = float(res.vol) if math.isfinite(res.vol) else None
        row["iv_iterations"] = res.iterations
        row["iv_status"] = res.reason if res.converged else f"failed: {res.reason}"
        if row["iv_solved"] is not None and row.get("iv_market"):
            row["iv_error"] = row["iv_solved"] - row["iv_market"]


def generate_demo_chain(
    spot: float = 100.0,
    r: float = 0.043,
    q: float = 0.008,
    expiry_days: tuple[int, ...] = DEFAULT_EXPIRY_DAYS,
    params: SSVIParams | None = None,
    n_strikes: int = 17,
    strike_span: float = 0.45,
    spread_bps: float = 220.0,
    noise_vol_pts: float = 0.35,
    seed: int = 20260805,
) -> dict:
    """Build a fully synthetic but realistically quoted option chain from SSVI."""
    p = params or SSVIParams()
    rng = np.random.default_rng(seed)
    today = _dt.date.today()

    expiries = []
    truth_pts = []
    for days in expiry_days:
        T = days / DAYS_PER_YEAR
        fwd = spot * math.exp((r - q) * T)
        # Strikes spaced in log-moneyness, then snapped to a plausible grid.
        span = strike_span * math.sqrt(max(T, 1.0 / 12.0))
        ks = np.linspace(-span, span, n_strikes)
        raw_strikes = fwd * np.exp(ks)
        inc = 1.0 if spot < 50 else (2.5 if spot < 200 else 5.0)
        strikes = np.unique(np.round(raw_strikes / inc) * inc)
        strikes = strikes[strikes > 0.05 * spot]

        rows = []
        for K in strikes:
            k = math.log(K / fwd)
            iv_true = float(p.iv(k, T))
            # A small idiosyncratic quote error, largest in the illiquid wings.
            wing = 1.0 + 2.5 * abs(k)
            iv_q = iv_true + rng.normal(0.0, noise_vol_pts / 100.0) * wing
            iv_q = max(iv_q, 0.02)

            for kind, is_call in (("call", True), ("put", False)):
                theo = float(np.asarray(bs.price(spot, K, T, r, q, iv_q, is_call)).reshape(()))
                half = max(theo * spread_bps / 20000.0, _tick_for(theo) * 0.5)
                tick = _tick_for(theo)
                bid = max(math.floor((theo - half) / tick) * tick, 0.0)
                ask = math.ceil((theo + half) / tick) * tick
                if ask <= bid:
                    ask = bid + tick
                mid = 0.5 * (bid + ask)
                moneyness = abs(k)
                rows.append(
                    {
                        "strike": float(K),
                        "type": kind,
                        "bid": round(bid, 2),
                        "ask": round(ask, 2),
                        "mid": round(mid, 4),
                        "last": round(mid, 2),
                        "volume": int(max(0, rng.poisson(max(900 * math.exp(-6 * moneyness), 0.4)))),
                        "open_interest": int(max(0, rng.poisson(max(5200 * math.exp(-4 * moneyness), 2)))),
                        "iv_market": round(iv_q, 6),
                        "iv_truth": round(iv_true, 6),
                        "in_the_money": bool((spot > K) if is_call else (spot < K)),
                    }
                )
            truth_pts.append((T, k, iv_true))

        _solve_chain_ivs(rows, spot, r, q, T)
        expiries.append(
            {
                "label": (today + _dt.timedelta(days=days)).isoformat(),
                "days": int(days),
                "T": T,
                "forward": fwd,
                "rows": rows,
            }
        )

    return {
        "source": "demo",
        "synthetic": True,
        "ticker": "DEMO",
        "name": "Synthetic SSVI surface",
        "spot": spot,
        "r": r,
        "q": q,
        "as_of": _dt.datetime.now().isoformat(timespec="seconds"),
        "currency": "USD",
        "expiries": expiries,
        "truth": {
            "kind": "ssvi",
            "params": p.__dict__,
            "admissibility": p.admissibility(),
        },
        "note": (
            "Synthetic data generated from an arbitrage-free SSVI surface, priced with "
            "Black-Scholes, then rounded to exchange ticks and wrapped in a bid/ask "
            "spread. Not market data."
        ),
    }


# --------------------------------------------------------------------------
# Yahoo Finance
# --------------------------------------------------------------------------

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
_YAHOO_TIMEOUT = 12.0
_cookie_cache: dict[str, str] = {}


def _http_get(url: str, timeout: float = _YAHOO_TIMEOUT) -> tuple[bytes, dict]:
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json,text/html,*/*"})
    if _cookie_cache.get("cookie"):
        req.add_header("Cookie", _cookie_cache["cookie"])
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), dict(resp.headers)


def _prime_yahoo_session() -> str | None:
    """Yahoo gates its JSON APIs behind a cookie plus a 'crumb' token."""
    if _cookie_cache.get("crumb"):
        return _cookie_cache["crumb"]
    try:
        req = urllib.request.Request("https://fc.yahoo.com/", headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=_YAHOO_TIMEOUT) as resp:
                cookies = resp.headers.get_all("Set-Cookie") or []
        except urllib.error.HTTPError as e:  # 404 still hands out the cookie
            cookies = e.headers.get_all("Set-Cookie") or []
        jar = "; ".join(c.split(";")[0] for c in cookies)
        if not jar:
            return None
        _cookie_cache["cookie"] = jar
        body, _ = _http_get("https://query2.finance.yahoo.com/v1/test/getcrumb")
        crumb = body.decode("utf-8", "replace").strip()
        if crumb and len(crumb) < 40 and "<" not in crumb:
            _cookie_cache["crumb"] = crumb
            return crumb
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return None


def _yahoo_options_url(ticker: str, date_epoch: int | None, crumb: str | None) -> str:
    url = f"https://query2.finance.yahoo.com/v7/finance/options/{urllib.parse.quote(ticker)}"
    parts = []
    if date_epoch:
        parts.append(f"date={int(date_epoch)}")
    if crumb:
        parts.append(f"crumb={urllib.parse.quote(crumb)}")
    return url + ("?" + "&".join(parts) if parts else "")


def _fetch_yahoo_json(ticker: str, date_epoch: int | None = None) -> dict:
    last_err: Exception | None = None
    # Try bare first; only pay for the cookie/crumb handshake if that is refused.
    for crumb in (None, _prime_yahoo_session()):
        try:
            body, _ = _http_get(_yahoo_options_url(ticker, date_epoch, crumb))
            payload = json.loads(body)
            chain = (payload.get("optionChain") or {}).get("result") or []
            if chain:
                return chain[0]
            err = (payload.get("optionChain") or {}).get("error")
            last_err = RuntimeError(str(err) if err else "empty result")
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as e:
            last_err = e
    raise RuntimeError(f"Yahoo Finance request failed: {last_err}")


def _row_from_yahoo(c: dict, kind: str) -> dict:
    bid, ask = c.get("bid"), c.get("ask")
    mid = None
    if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and ask > 0 and bid >= 0 and ask >= bid:
        mid = 0.5 * (bid + ask)
    if mid is None or mid <= 0:
        lp = c.get("lastPrice")
        mid = float(lp) if isinstance(lp, (int, float)) and lp > 0 else None
    iv = c.get("impliedVolatility")
    return {
        "strike": float(c.get("strike", float("nan"))),
        "type": kind,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "last": c.get("lastPrice"),
        "volume": c.get("volume") or 0,
        "open_interest": c.get("openInterest") or 0,
        # Yahoo reports its own IV; keeping it lets the UI compare our inversion
        # against a third party rather than only against itself.
        "iv_market": float(iv) if isinstance(iv, (int, float)) and iv > 0 else None,
        "in_the_money": bool(c.get("inTheMoney", False)),
    }


def fetch_yahoo_chain(
    ticker: str,
    r: float = 0.043,
    q: float = 0.0,
    max_expiries: int = 8,
) -> dict:
    """Fetch a live option chain from Yahoo Finance and invert mids to implied vol.

    Raises RuntimeError with a readable message if the feed is unavailable, so
    the UI can fall back to the synthetic chain and say so.
    """
    ticker = ticker.strip().upper()
    if not ticker or len(ticker) > 12 or not all(ch.isalnum() or ch in ".-^" for ch in ticker):
        raise ValueError(f"implausible ticker {ticker!r}")

    root = _fetch_yahoo_json(ticker)
    quote = root.get("quote") or {}
    spot = quote.get("regularMarketPrice") or quote.get("previousClose")
    if not isinstance(spot, (int, float)) or spot <= 0:
        raise RuntimeError(f"no usable spot price for {ticker}")
    spot = float(spot)

    all_dates = [int(d) for d in (root.get("expirationDates") or [])]
    if not all_dates:
        raise RuntimeError(f"{ticker} has no listed option expiries")

    div_yield = quote.get("trailingAnnualDividendYield")
    if q == 0.0 and isinstance(div_yield, (int, float)) and 0.0 < div_yield < 0.25:
        q = float(div_yield)

    now = _dt.datetime.now(_dt.timezone.utc)
    chosen = all_dates[:max_expiries]
    expiries = []
    warnings: list[str] = []

    for i, epoch in enumerate(chosen):
        try:
            node = root if i == 0 else _fetch_yahoo_json(ticker, epoch)
            opts = (node.get("options") or [{}])[0]
        except RuntimeError as e:
            warnings.append(f"skipped expiry {epoch}: {e}")
            continue

        exp_dt = _dt.datetime.fromtimestamp(epoch, _dt.timezone.utc).replace(hour=21, minute=0)
        T = max((exp_dt - now).total_seconds() / (365.0 * 86400.0), 1.0 / (365.0 * 24.0))

        rows = [_row_from_yahoo(c, "call") for c in (opts.get("calls") or [])]
        rows += [_row_from_yahoo(c, "put") for c in (opts.get("puts") or [])]
        rows = [x for x in rows if math.isfinite(x["strike"]) and x["strike"] > 0]
        if not rows:
            continue
        _solve_chain_ivs(rows, spot, r, q, T)
        expiries.append(
            {
                "label": exp_dt.date().isoformat(),
                "days": int(round(T * 365.0)),
                "T": T,
                "forward": spot * math.exp((r - q) * T),
                "rows": rows,
            }
        )

    if not expiries:
        raise RuntimeError(f"fetched {ticker} but every expiry came back empty")

    return {
        "source": "yahoo",
        "synthetic": False,
        "ticker": ticker,
        "name": quote.get("shortName") or ticker,
        "spot": spot,
        "r": r,
        "q": q,
        "as_of": _dt.datetime.now().isoformat(timespec="seconds"),
        "currency": quote.get("currency") or "USD",
        "expiries": expiries,
        "truth": None,
        "warnings": warnings,
        "note": (
            "Live delayed quotes from Yahoo Finance. Implied vols labelled 'solved' are "
            "inverted from the bid/ask mid by this app; 'market' is Yahoo's own figure."
        ),
    }


def chain_liquidity_filter(
    chain: dict,
    min_volume: int = 0,
    min_open_interest: int = 0,
    max_spread_frac: float = 0.35,
    otm_only: bool = True,
    max_abs_k: float = 1.0,
) -> list[dict]:
    """Select the quotes worth fitting a surface to.

    Defaults keep out-of-the-money options only. That is standard practice: OTM
    quotes carry nearly all the volatility information, while deep-in-the-money
    options are mostly intrinsic value, so a tick of price error there maps to a
    huge vol error and would dominate the fit.
    """
    out: list[dict] = []
    spot = chain["spot"]
    for exp in chain["expiries"]:
        T = exp["T"]
        fwd = exp["forward"]
        for row in exp["rows"]:
            iv = row.get("iv_solved")
            if iv is None or not math.isfinite(iv) or iv <= 0.005 or iv > 5.0:
                continue
            if (row.get("volume") or 0) < min_volume:
                continue
            if (row.get("open_interest") or 0) < min_open_interest:
                continue
            bid, ask, mid = row.get("bid"), row.get("ask"), row.get("mid")
            if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and mid:
                if bid <= 0 or (ask - bid) / max(mid, 1e-9) > max_spread_frac:
                    continue
            k = math.log(row["strike"] / fwd)
            if abs(k) > max_abs_k:
                continue
            if otm_only:
                is_otm = (row["type"] == "call" and row["strike"] >= spot) or (
                    row["type"] == "put" and row["strike"] <= spot
                )
                if not is_otm:
                    continue
            out.append(
                {
                    "T": T,
                    "strike": row["strike"],
                    "iv": iv,
                    "type": row["type"],
                    "k": k,
                    # Vega-like weighting: a quote is trusted in proportion to how
                    # much its price actually moves when vol moves.
                    "weight": float(max(np.exp(-2.0 * k * k), 0.05)),
                    "iv_truth": row.get("iv_truth"),
                    "expiry_label": exp["label"],
                }
            )
    return out
