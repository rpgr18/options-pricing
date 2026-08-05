"""Local JSON API and static file server for the options pricing workbench.

Deliberately built on `http.server` rather than a framework: the whole point of
this project is that `python3 run.py` works on a bare interpreter with NumPy and
nothing else. The surface area is small enough that a router dict and a JSON
codec cover it.

The server binds to the loopback interface only.
"""

from __future__ import annotations

import json
import math
import mimetypes
import os
import posixpath
import socket
import threading
import time
import traceback
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from optlib import blackscholes as bs
from optlib import convergence as conv
from optlib import lattice, market, montecarlo as mc, strategy
from optlib.implied_vol import implied_vol
from optlib.surface import METHODS as SURFACE_METHODS
from optlib.surface import VolSurface, quotes_from_chain

WEB_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
MAX_BODY = 8 * 1024 * 1024

# Chains are cached server-side so that refitting a surface with different
# filters does not re-hit the upstream data provider.
_chain_cache: dict[str, dict] = {}
_chain_order: list[str] = []
_chain_lock = threading.Lock()
_CHAIN_CACHE_MAX = 8


class ApiError(Exception):
    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------
# Input coercion
# --------------------------------------------------------------------------

def _num(body: dict, key: str, default=None, lo=None, hi=None, required=False) -> float:
    raw = body.get(key, default)
    if raw is None:
        if required:
            raise ApiError(f"missing required numeric field {key!r}")
        raise ApiError(f"field {key!r} has no value and no default")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        raise ApiError(f"field {key!r} must be a number, got {raw!r}") from None
    if not math.isfinite(val):
        raise ApiError(f"field {key!r} must be finite, got {raw!r}")
    if lo is not None and val < lo:
        raise ApiError(f"field {key!r} must be >= {lo} (got {val})")
    if hi is not None and val > hi:
        raise ApiError(f"field {key!r} must be <= {hi} (got {val})")
    return val


def _int(body: dict, key: str, default: int, lo: int, hi: int) -> int:
    val = int(_num(body, key, default))
    return max(lo, min(hi, val))


def _bool(body: dict, key: str, default: bool = False) -> bool:
    raw = body.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return bool(raw)


def _contract(body: dict) -> dict:
    """The common option-contract block shared by most endpoints."""
    return {
        "S": _num(body, "S", 100.0, lo=1e-6, hi=1e9),
        "K": _num(body, "K", 100.0, lo=1e-6, hi=1e9),
        "T": _num(body, "T", 1.0, lo=0.0, hi=100.0),
        "r": _num(body, "r", 0.043, lo=-1.0, hi=1.0),
        "q": _num(body, "q", 0.0, lo=-1.0, hi=1.0),
        "sigma": _num(body, "sigma", 0.25, lo=1e-6, hi=10.0),
        "is_call": _bool(body, "is_call", True),
    }


def _clean(obj):
    """Make a value JSON-safe: NaN/Inf become null, NumPy scalars become Python."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

def api_health(_body: dict) -> dict:
    return {"ok": True, "numpy": np.__version__, "surface_methods": list(SURFACE_METHODS)}


def api_price(body: dict) -> dict:
    c = _contract(body)
    american = _bool(body, "american", False)
    steps = _int(body, "steps", 500, 4, 20_000)
    paths = _int(body, "paths", 200_000, 256, 5_000_000)
    lattice_method = str(body.get("lattice_method", "crr"))
    if lattice_method not in lattice.BINOMIAL_METHODS:
        raise ApiError(f"lattice_method must be one of {lattice.BINOMIAL_METHODS}")

    out: dict = {"inputs": {**c, "american": american, "steps": steps, "paths": paths}}
    S, K, T, r, q, sigma, is_call = (c["S"], c["K"], c["T"], c["r"], c["q"], c["sigma"], c["is_call"])
    out["forward"] = S * math.exp((r - q) * T)
    out["moneyness"] = math.log(K / out["forward"]) if out["forward"] > 0 else None

    models = []

    # --- Black-Scholes (European only; shown as a reference for American) ---
    t0 = time.perf_counter()
    g = bs.greeks(S, K, T, r, q, sigma, is_call)
    bs_ms = (time.perf_counter() - t0) * 1e3
    models.append({
        "key": "black_scholes",
        "label": "Black-Scholes-Merton",
        "family": "analytic",
        "price": g["price"],
        "ms": bs_ms,
        "exact": not american,
        "note": "Closed form; European exercise only." if american else "Closed form.",
        "greeks": {k: v for k, v in g.items() if not k.endswith("_display")},
        "greeks_display": {k[: -len("_display")]: v for k, v in g.items() if k.endswith("_display")},
    })
    out["greeks"] = models[0]["greeks"]
    out["greeks_display"] = models[0]["greeks_display"]
    out["greek_units"] = {k: v[1] for k, v in bs.GREEK_DISPLAY.items()}
    out["parity_gap"] = bs.parity_gap(S, K, T, r, q, sigma)

    # --- Lattice ---
    for key, label, kw in (
        (lattice_method, f"Binomial {lattice_method.replace('_', '-').title()}", {"method": lattice_method}),
        ("smooth", f"Binomial {lattice_method} + smoothing + Richardson",
         {"method": lattice_method, "smoothing": True, "richardson": True}),
    ):
        t0 = time.perf_counter()
        res = lattice.binomial(S, K, T, r, q, sigma, steps, is_call, american, **kw)
        ms = (time.perf_counter() - t0) * 1e3
        models.append({
            "key": f"binomial_{key}",
            "label": label,
            "family": "lattice",
            "price": res.price,
            "ms": ms,
            "steps": res.steps,
            "greeks": {"delta": res.delta, "gamma": res.gamma, "theta": res.theta, "vega": res.vega, "rho": res.rho},
            "early_exercise_premium": res.early_exercise_premium,
        })

    t0 = time.perf_counter()
    tri = lattice.trinomial(S, K, T, r, q, sigma, max(steps // 2, 4), is_call, american)
    models.append({
        "key": "trinomial",
        "label": "Trinomial (Boyle)",
        "family": "lattice",
        "price": tri.price,
        "ms": (time.perf_counter() - t0) * 1e3,
        "steps": tri.steps,
        "greeks": {"delta": tri.delta, "gamma": tri.gamma, "theta": tri.theta},
        "early_exercise_premium": tri.early_exercise_premium,
    })

    # --- Monte Carlo ---
    if american:
        t0 = time.perf_counter()
        lsm = mc.longstaff_schwartz(S, K, T, r, q, sigma, min(paths, 400_000),
                                    _int(body, "lsm_steps", 50, 4, 400), is_call,
                                    degree=_int(body, "lsm_degree", 3, 1, 6))
        models.append({
            "key": "lsm",
            "label": "Longstaff-Schwartz LSMC",
            "family": "monte-carlo",
            "price": lsm.price,
            "ms": (time.perf_counter() - t0) * 1e3,
            "paths": lsm.paths,
            "std_error": lsm.std_error,
            "ci_low": lsm.ci_low,
            "ci_high": lsm.ci_high,
            "early_exercise_premium": lsm.early_exercise_premium,
            "note": "LSMC is biased low: the exercise rule is estimated from the same paths.",
        })
    else:
        for key, label, kw in (
            ("mc", "Monte Carlo (antithetic + control)", {"antithetic": True, "control_variate": True}),
            ("qmc", "Quasi-MC (Halton, randomized)",
             {"antithetic": True, "control_variate": True, "sampler": "halton"}),
        ):
            t0 = time.perf_counter()
            res = mc.european(S, K, T, r, q, sigma, paths, is_call, **kw)
            models.append({
                "key": key,
                "label": label,
                "family": "monte-carlo",
                "price": res.price,
                "ms": (time.perf_counter() - t0) * 1e3,
                "paths": res.paths,
                "std_error": res.std_error,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "efficiency": res.variance_reduction,
                "control_beta": res.control_beta,
                "greeks": {"delta": res.delta, "gamma": res.gamma, "vega": res.vega,
                           "theta": res.theta, "rho": res.rho},
                "greek_std_errors": {"delta": res.delta_se, "vega": res.vega_se},
            })

    # Error column, against whichever price is the truth for this exercise style.
    if american:
        ref = lattice.american_reference(S, K, T, r, q, sigma, is_call)
        out["reference"] = {"price": ref, "label": "Leisen-Reimer n=6001 + Richardson"}
    else:
        ref = models[0]["price"]
        out["reference"] = {"price": ref, "label": "Black-Scholes closed form"}
    for m in models:
        if american and m["key"] == "black_scholes":
            m["error"] = None
            m["error_bp"] = None
            continue
        m["error"] = m["price"] - ref
        m["error_bp"] = (m["price"] - ref) / ref * 10_000.0 if ref else None

    out["models"] = models
    out["intrinsic"] = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    out["time_value"] = models[0]["price"] - out["intrinsic"]
    return out


def api_greek_surface(body: dict) -> dict:
    c = _contract(body)
    name = str(body.get("greek", "delta"))
    if name not in bs.GREEK_REGISTRY:
        raise ApiError(f"unknown greek {name!r}; expected one of {sorted(bs.GREEK_REGISTRY)}")

    n_k = _int(body, "n_strikes", 52, 6, 160)
    n_t = _int(body, "n_tenors", 44, 6, 160)
    k_lo = _num(body, "strike_low", c["S"] * 0.6, lo=1e-6)
    k_hi = _num(body, "strike_high", c["S"] * 1.4, lo=1e-6)
    if k_hi <= k_lo:
        raise ApiError("strike_high must exceed strike_low")
    t_lo = _num(body, "tenor_low", 1.0 / 365.0, lo=1e-6)
    t_hi = _num(body, "tenor_high", max(c["T"], 1.0), lo=1e-6)
    if t_hi <= t_lo:
        raise ApiError("tenor_high must exceed tenor_low")

    strikes = np.linspace(k_lo, k_hi, n_k)
    tenors = np.linspace(t_lo, t_hi, n_t)
    KK, TT = np.meshgrid(strikes, tenors, indexing="ij")
    Z = np.asarray(bs.evaluate(name, c["S"], KK, TT, c["r"], c["q"], c["sigma"], c["is_call"]), dtype=float)

    scale, unit = bs.GREEK_DISPLAY[name]
    Zs = Z * scale
    finite = Zs[np.isfinite(Zs)]
    return {
        "greek": name,
        "unit": unit,
        "scale": scale,
        "strikes": strikes.tolist(),
        "tenors": tenors.tolist(),
        "values": Zs.tolist(),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "diverging": bool(finite.size and finite.min() < 0.0 < finite.max()),
        "inputs": c,
        "spot": c["S"],
    }


def api_greek_profile(body: dict) -> dict:
    """Greeks along a single axis (spot, tenor or vol) for the line charts."""
    c = _contract(body)
    axis = str(body.get("axis", "spot"))
    names = body.get("greeks") or ["delta", "gamma", "vega", "theta"]
    names = [n for n in names if n in bs.GREEK_REGISTRY][:8]
    if not names:
        raise ApiError("no valid greeks requested")
    n = _int(body, "n", 161, 11, 801)

    if axis == "spot":
        lo = _num(body, "low", c["S"] * 0.55, lo=1e-6)
        hi = _num(body, "high", c["S"] * 1.45, lo=1e-6)
        xs = np.linspace(lo, hi, n)
        args = lambda x: (x, c["K"], c["T"], c["r"], c["q"], c["sigma"])
    elif axis == "tenor":
        lo = _num(body, "low", 1.0 / 365.0, lo=1e-9)
        hi = _num(body, "high", max(c["T"], 1.0), lo=1e-9)
        xs = np.linspace(lo, hi, n)
        args = lambda x: (c["S"], c["K"], x, c["r"], c["q"], c["sigma"])
    elif axis == "vol":
        lo = _num(body, "low", 0.02, lo=1e-6)
        hi = _num(body, "high", 1.2, lo=1e-6)
        xs = np.linspace(lo, hi, n)
        args = lambda x: (c["S"], c["K"], c["T"], c["r"], c["q"], x)
    else:
        raise ApiError("axis must be 'spot', 'tenor' or 'vol'")
    if hi <= lo:
        raise ApiError("high must exceed low")

    series = []
    for name in names:
        scale, unit = bs.GREEK_DISPLAY[name]
        vals = np.asarray(bs.evaluate(name, *args(xs), c["is_call"]), dtype=float) * scale
        series.append({"key": name, "label": name.replace("_", " "), "unit": unit, "values": vals.tolist()})
    return {"axis": axis, "x": xs.tolist(), "series": series, "inputs": c, "spot": c["S"]}


def api_implied_vol(body: dict) -> dict:
    c = _contract(body)
    price = _num(body, "price", required=True, lo=0.0)
    res = implied_vol(price, c["S"], c["K"], c["T"], c["r"], c["q"], c["is_call"])
    out = res.as_dict()
    out["inputs"] = {**c, "price": price}
    if math.isfinite(res.vol):
        out["greeks"] = bs.greeks(c["S"], c["K"], c["T"], c["r"], c["q"], res.vol, c["is_call"])
        out["reprice"] = float(np.asarray(bs.price(c["S"], c["K"], c["T"], c["r"], c["q"], res.vol, c["is_call"])).reshape(()))
    dfq, dfr = math.exp(-c["q"] * c["T"]), math.exp(-c["r"] * c["T"])
    fwd = c["S"] * dfq - c["K"] * dfr
    out["bounds"] = {
        "lower": max(fwd, 0.0) if c["is_call"] else max(-fwd, 0.0),
        "upper": c["S"] * dfq if c["is_call"] else c["K"] * dfr,
    }
    return out


def api_convergence(body: dict) -> dict:
    c = _contract(body)
    american = _bool(body, "american", False)
    which = str(body.get("study", "all"))
    args = (c["S"], c["K"], c["T"], c["r"], c["q"], c["sigma"], c["is_call"])
    out: dict = {"inputs": {**c, "american": american}}

    if which in ("all", "lattice"):
        engines = tuple(body.get("lattice_engines") or ("crr", "crr_smooth", "leisen_reimer", "trinomial"))
        out["lattice"] = conv.lattice_convergence(
            *args, american=american, engines=engines,
            n_max=_int(body, "n_max", 800, 16, 4000),
            n_points=_int(body, "n_points", 26, 6, 60),
        )
    if which in ("all", "mc") and not american:
        engines = tuple(body.get("mc_engines") or ("plain", "antithetic", "both", "qmc"))
        out["monte_carlo"] = conv.mc_convergence(
            *args, engines=engines,
            n_max=_int(body, "paths_max", 262_144, 1024, 2_000_000),
            n_points=_int(body, "paths_points", 10, 4, 16),
        )
    if which in ("all", "shootout"):
        out["shootout"] = conv.engine_shootout(
            *args, american=american, target_bp=_num(body, "target_bp", 1.0, lo=0.01, hi=1000.0)
        )
    return out


def _cache_chain(chain: dict) -> str:
    chain_id = f"{chain['source']}:{chain['ticker']}:{int(time.time() * 1000)}"
    with _chain_lock:
        _chain_cache[chain_id] = chain
        _chain_order.append(chain_id)
        while len(_chain_order) > _CHAIN_CACHE_MAX:
            _chain_cache.pop(_chain_order.pop(0), None)
    return chain_id


def api_chain(body: dict) -> dict:
    source = str(body.get("source", "demo")).lower()
    r = _num(body, "r", 0.043, lo=-1.0, hi=1.0)
    q = _num(body, "q", 0.0, lo=-1.0, hi=1.0)

    if source == "yahoo":
        ticker = str(body.get("ticker", "")).strip()
        if not ticker:
            raise ApiError("a ticker is required for the yahoo source")
        try:
            chain = market.fetch_yahoo_chain(ticker, r, q, _int(body, "max_expiries", 8, 1, 20))
        except (RuntimeError, ValueError) as e:
            raise ApiError(
                f"Could not fetch {ticker} from Yahoo Finance: {e}. "
                "The synthetic SSVI chain is always available as a fallback.",
                status=HTTPStatus.BAD_GATEWAY,
            ) from None
    elif source == "demo":
        p = market.SSVIParams()
        for field in ("rho", "eta", "gamma", "atm_vol_short", "atm_vol_long", "atm_decay"):
            if field in body:
                setattr(p, field, _num(body, field))
        chain = market.generate_demo_chain(
            spot=_num(body, "spot", 100.0, lo=1e-3),
            r=r, q=_num(body, "q", 0.008, lo=-1.0, hi=1.0), params=p,
            n_strikes=_int(body, "n_strikes", 17, 5, 61),
            spread_bps=_num(body, "spread_bps", 220.0, lo=0.0, hi=5000.0),
            noise_vol_pts=_num(body, "noise_vol_pts", 0.35, lo=0.0, hi=20.0),
            seed=_int(body, "seed", 20260805, 0, 2**31 - 1),
        )
    else:
        raise ApiError("source must be 'demo' or 'yahoo'")

    chain_id = _cache_chain(chain)
    # Summary statistics on how well our own inversion agrees with the feed's IV.
    diffs = [
        row["iv_solved"] - row["iv_market"]
        for exp in chain["expiries"] for row in exp["rows"]
        if row.get("iv_solved") and row.get("iv_market")
    ]
    total = sum(len(e["rows"]) for e in chain["expiries"])
    solved = sum(1 for e in chain["expiries"] for row in e["rows"] if row.get("iv_solved"))
    return {
        "chain_id": chain_id,
        "chain": chain,
        "stats": {
            "rows": total,
            "solved": solved,
            "solve_rate": solved / total if total else 0.0,
            "iv_vs_feed_rmse_vol_pts": float(np.sqrt(np.mean(np.square(diffs))) * 100.0) if diffs else None,
            "iv_vs_feed_n": len(diffs),
        },
    }


def _resolve_chain(body: dict) -> dict:
    chain_id = body.get("chain_id")
    if chain_id:
        with _chain_lock:
            chain = _chain_cache.get(str(chain_id))
        if chain is None:
            raise ApiError("chain_id is unknown or has expired; re-fetch the chain", HTTPStatus.GONE)
        return chain
    if isinstance(body.get("chain"), dict):
        return body["chain"]
    raise ApiError("provide either chain_id or an inline chain")


def api_surface(body: dict) -> dict:
    chain = _resolve_chain(body)
    method = str(body.get("method", "svi"))
    if method not in SURFACE_METHODS:
        raise ApiError(f"method must be one of {list(SURFACE_METHODS)}")

    selected = market.chain_liquidity_filter(
        chain,
        min_volume=_int(body, "min_volume", 0, 0, 10**7),
        min_open_interest=_int(body, "min_open_interest", 0, 0, 10**7),
        max_spread_frac=_num(body, "max_spread_frac", 0.35, lo=0.0, hi=10.0),
        otm_only=_bool(body, "otm_only", True),
        max_abs_k=_num(body, "max_abs_k", 1.0, lo=0.01, hi=5.0),
    )
    if len(selected) < 6:
        raise ApiError(
            f"only {len(selected)} quotes survived the liquidity filter; loosen it "
            "(lower the open-interest floor or widen the spread limit) to fit a surface"
        )

    quotes = quotes_from_chain(selected, chain["spot"], chain["r"], chain["q"])
    t0 = time.perf_counter()
    surf = VolSurface(quotes, method)
    fit_ms = (time.perf_counter() - t0) * 1e3

    diag = surf.diagnostics(with_holdout=_bool(body, "holdout", False))
    grid = surf.grid(_int(body, "n_k", 48, 8, 120), _int(body, "n_T", 36, 6, 90))

    smiles = []
    for sl in surf.slices:
        sm = surf.smile(sl.T)
        sm["quotes"] = [
            {"k": float(k), "iv": float(np.sqrt(w / sl.T)), "strike": float(st)}
            for k, w, st in zip(sl.k, sl.w, sl.strikes)
        ]
        sm["label"] = f"{sl.T * 365.0:.0f}d"
        smiles.append(sm)

    out = {
        "method": method,
        "fit_ms": fit_ms,
        "n_quotes": len(quotes),
        "spot": chain["spot"],
        "r": chain["r"],
        "q": chain["q"],
        "ticker": chain["ticker"],
        "synthetic": chain.get("synthetic", False),
        "diagnostics": diag,
        "grid": grid,
        "smiles": smiles,
    }

    # When the chain is synthetic we know the true surface, so the fit can be
    # scored against ground truth rather than only against its own inputs.
    truth = [(s["k"], s["T"], s["iv_truth"]) for s in selected if s.get("iv_truth")]
    if truth and chain.get("truth"):
        kk = np.array([t[0] for t in truth])
        TT = np.array([t[1] for t in truth])
        tv = np.array([t[2] for t in truth])
        err = (np.asarray(surf.iv(kk, TT)) - tv) * 100.0
        out["truth"] = {
            "kind": chain["truth"]["kind"],
            "params": chain["truth"]["params"],
            "rmse_vol_pts": float(np.sqrt(np.mean(err * err))),
            "max_abs_vol_pts": float(np.max(np.abs(err))),
            "n": int(err.size),
        }
    return out


def api_surface_compare(body: dict) -> dict:
    """Fit every interpolator to the same quotes and score them side by side."""
    chain = _resolve_chain(body)
    selected = market.chain_liquidity_filter(
        chain,
        min_volume=_int(body, "min_volume", 0, 0, 10**7),
        min_open_interest=_int(body, "min_open_interest", 0, 0, 10**7),
        max_spread_frac=_num(body, "max_spread_frac", 0.35, lo=0.0, hi=10.0),
        otm_only=_bool(body, "otm_only", True),
        max_abs_k=_num(body, "max_abs_k", 1.0, lo=0.01, hi=5.0),
    )
    if len(selected) < 8:
        raise ApiError(f"only {len(selected)} quotes survived the filter; need at least 8 to compare")

    quotes = quotes_from_chain(selected, chain["spot"], chain["r"], chain["q"])
    truth = np.array([s["iv_truth"] for s in selected]) if all(s.get("iv_truth") for s in selected) else None
    kk = np.array([s["k"] for s in selected])
    TT = np.array([s["T"] for s in selected])

    rows = []
    for method in SURFACE_METHODS:
        try:
            t0 = time.perf_counter()
            surf = VolSurface(quotes, method)
            fit_ms = (time.perf_counter() - t0) * 1e3
            fq = surf.fit_quality(with_holdout=True)
            d = surf.diagnostics()
            row = {
                "method": method,
                "fit_ms": fit_ms,
                "in_sample_rmse": fq["rmse_vol_pts"],
                "holdout_rmse": fq.get("holdout_rmse_vol_pts"),
                "holdout_n": fq.get("holdout_n"),
                "max_abs": fq["max_abs_vol_pts"],
                "min_g": d["butterfly"]["min_g"],
                "butterfly_ok": d["butterfly"]["ok"],
                "min_dw_dT": d["calendar"]["min_dw_dT"],
                "calendar_ok": d["calendar"]["ok"],
            }
            if truth is not None:
                err = (np.asarray(surf.iv(kk, TT)) - truth) * 100.0
                row["truth_rmse"] = float(np.sqrt(np.mean(err * err)))
            rows.append(row)
        except (ValueError, np.linalg.LinAlgError) as e:
            rows.append({"method": method, "error": str(e)})

    return {
        "n_quotes": len(quotes),
        "has_truth": truth is not None,
        "rows": rows,
        "note": (
            "Held-out RMSE is the column that matters. In-sample error rewards "
            "interpolation through noise; the cubic spline scores ~0 there by "
            "construction and still generalizes worst."
        ),
    }


def api_strategy(body: dict) -> dict:
    S0 = _num(body, "S", 100.0, lo=1e-6)
    T0 = _num(body, "T", 0.25, lo=0.0, hi=100.0)
    r = _num(body, "r", 0.043, lo=-1.0, hi=1.0)
    q = _num(body, "q", 0.0, lo=-1.0, hi=1.0)
    sigma = _num(body, "sigma", 0.25, lo=1e-6, hi=10.0)

    raw = body.get("legs")
    if body.get("preset"):
        pre = strategy.presets(S0, T0, sigma)
        if body["preset"] not in pre:
            raise ApiError(f"unknown preset {body['preset']!r}; expected one of {sorted(pre)}")
        raw = pre[body["preset"]]
    if not isinstance(raw, list) or not raw:
        raise ApiError("provide a non-empty 'legs' array or a 'preset' name")
    if len(raw) > 12:
        raise ApiError("at most 12 legs")

    legs = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ApiError(f"leg {i} must be an object")
        try:
            leg = strategy.Leg(
                kind=str(item.get("kind", "call")),
                quantity=float(item.get("quantity", 1)),
                strike=float(item.get("strike", 0.0) or 0.0),
                sigma=float(item.get("sigma", sigma) or sigma),
                premium=None if item.get("premium") in (None, "") else float(item["premium"]),
                multiplier=float(item.get("multiplier", 1.0)),
            )
            leg.validate()
        except (TypeError, ValueError) as e:
            raise ApiError(f"leg {i}: {e}") from None
        legs.append(leg)

    horizons = body.get("horizons")
    if horizons is not None:
        try:
            horizons = sorted({max(0.0, min(float(h), T0)) for h in horizons}, reverse=True)
        except (TypeError, ValueError):
            raise ApiError("horizons must be a list of numbers") from None
        if not horizons:
            horizons = None

    out = strategy.evaluate(legs, S0, T0, r, q, horizons=horizons,
                            n_spots=_int(body, "n_spots", 181, 21, 801),
                            spot_span=_num(body, "spot_span", 0.45, lo=0.02, hi=0.95))
    out["legs"] = [
        {"kind": l.kind, "quantity": l.quantity, "strike": l.strike, "sigma": l.sigma,
         "premium": l.premium, "multiplier": l.multiplier} for l in legs
    ]
    out["inputs"] = {"S": S0, "T": T0, "r": r, "q": q, "sigma": sigma}
    out["greek_units"] = {k: bs.GREEK_DISPLAY[k][1] for k in strategy.AGGREGATE_GREEKS}
    return out


def api_presets(body: dict) -> dict:
    S0 = _num(body, "S", 100.0, lo=1e-6)
    T0 = _num(body, "T", 0.25, lo=1e-9)
    sigma = _num(body, "sigma", 0.25, lo=1e-6)
    return {"presets": strategy.presets(S0, T0, sigma)}


def api_exercise_boundary(body: dict) -> dict:
    """The early-exercise boundary of an American option, from the lattice."""
    c = _contract(body)
    res = lattice.binomial(
        c["S"], c["K"], c["T"], c["r"], c["q"], c["sigma"],
        _int(body, "steps", 600, 20, 5000), c["is_call"], True, "crr", want_boundary=True,
    )
    euro = float(np.asarray(bs.price(c["S"], c["K"], c["T"], c["r"], c["q"], c["sigma"], c["is_call"])).reshape(()))
    return {
        "boundary": [{"t": t, "T_remaining": c["T"] - t, "s": s} for t, s in res.boundary],
        "american": res.price,
        "european": euro,
        "early_exercise_premium": res.price - euro,
        "steps": res.steps,
        "inputs": c,
    }


ROUTES = {
    "/api/health": api_health,
    "/api/price": api_price,
    "/api/greek-surface": api_greek_surface,
    "/api/greek-profile": api_greek_profile,
    "/api/implied-vol": api_implied_vol,
    "/api/convergence": api_convergence,
    "/api/chain": api_chain,
    "/api/surface": api_surface,
    "/api/surface-compare": api_surface_compare,
    "/api/strategy": api_strategy,
    "/api/presets": api_presets,
    "/api/exercise-boundary": api_exercise_boundary,
}


# --------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "OptionsWorkbench/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter, and on one line
        if self.server.verbose:
            print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

    # ---- responses ----

    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # Local tool: never let a stale bundle survive a code edit.
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(_clean(payload), allow_nan=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str, detail: str | None = None) -> None:
        self._send_json(status, {"error": message, "detail": detail, "status": int(status)})

    # ---- routing ----

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path.startswith("/api/"):
            params = {k: (v[0] if len(v) == 1 else v) for k, v in urllib.parse.parse_qs(parsed.query).items()}
            self._dispatch(path, params)
            return
        self._serve_static(path)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if not path.startswith("/api/"):
            self._error(HTTPStatus.NOT_FOUND, f"no POST route for {path}")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
            return
        if length > MAX_BODY:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"request body over {MAX_BODY} bytes")
            return
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError as e:
            self._error(HTTPStatus.BAD_REQUEST, f"malformed JSON body: {e}")
            return
        if not isinstance(body, dict):
            self._error(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
            return
        self._dispatch(path, body)

    def _dispatch(self, path: str, body: dict) -> None:
        fn = ROUTES.get(path)
        if fn is None:
            self._error(HTTPStatus.NOT_FOUND, f"unknown endpoint {path}",
                        detail=f"available: {', '.join(sorted(ROUTES))}")
            return
        t0 = time.perf_counter()
        try:
            payload = fn(body)
        except ApiError as e:
            self._error(e.status, str(e))
            return
        except Exception as e:  # a bug, not bad input: report it usefully
            traceback.print_exc()
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(e).__name__}: {e}",
                        detail="See the server console for the full traceback.")
            return
        payload["_server_ms"] = (time.perf_counter() - t0) * 1e3
        self._send_json(HTTPStatus.OK, payload)

    def _serve_static(self, path: str) -> None:
        # Normalize and confine to WEB_ROOT: no traversal out of the web directory.
        clean = posixpath.normpath(urllib.parse.unquote(path))
        if clean in ("/", "", "."):
            clean = "/index.html"
        parts = [p for p in clean.split("/") if p not in ("", ".", "..")]
        target = os.path.join(WEB_ROOT, *parts)
        real_root = os.path.realpath(WEB_ROOT)
        real_target = os.path.realpath(target)
        if not (real_target == real_root or real_target.startswith(real_root + os.sep)):
            self._error(HTTPStatus.FORBIDDEN, "path outside the web root")
            return
        if not os.path.isfile(real_target):
            self._error(HTTPStatus.NOT_FOUND, f"no such file {clean}")
            return
        ctype, _ = mimetypes.guess_type(real_target)
        if real_target.endswith(".js"):
            ctype = "text/javascript; charset=utf-8"
        elif real_target.endswith(".css"):
            ctype = "text/css; charset=utf-8"
        try:
            with open(real_target, "rb") as fh:
                data = fh.read()
        except OSError as e:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"could not read {clean}: {e}")
            return
        self._send(HTTPStatus.OK, data, ctype or "application/octet-stream")


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, verbose: bool = False, family=socket.AF_INET):
        self.address_family = family
        super().__init__(addr, handler)
        self.verbose = verbose


def serve(host: str = "127.0.0.1", port: int = 8770, verbose: bool = False) -> Server:
    mimetypes.add_type("text/javascript", ".js")
    return Server((host, port), Handler, verbose)


def serve_ipv6_loopback(port: int, verbose: bool = False) -> Server | None:
    """Best-effort second listener on [::1].

    `localhost` resolves to ::1 before 127.0.0.1 for a lot of clients, so an
    IPv4-only bind makes the app look dead to them even though it is running.
    Binding both loopback addresses -- and only loopback -- fixes that without
    exposing the server to the network. Returns None if IPv6 is unavailable.
    """
    try:
        srv = Server(("::1", port), Handler, verbose, family=socket.AF_INET6)
    except OSError:
        return None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
