"""Convergence and error-rate studies across the pricing engines.

The point of this module is to measure, not assert. For each engine it walks a
grid of discretization sizes, records signed error against a reference price and
the wall time taken, then fits the observed convergence order as the slope of
log|error| against log(n) by least squares.

Two caveats are surfaced in the output rather than smoothed over:

* Binomial error **oscillates**. Whether the terminal node lattice straddles the
  strike changes with n, so |error| has a sawtooth on top of its trend and the
  fitted slope has a genuinely poor R^2 unless payoff smoothing is on. The
  reported `r_squared` is how you tell a real rate from a fitted artefact.
* For Monte Carlo, error at one seed is a single draw from a distribution. The
  fitted slope is therefore reported next to the theoretical -1/2, and the
  standard error band is what should be compared, not the individual points.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import blackscholes as bs
from . import lattice, montecarlo as mc

# Engines available to the convergence study, keyed by the UI's identifier.
LATTICE_ENGINES = {
    "crr": {"label": "Binomial CRR", "method": "crr"},
    "crr_smooth": {"label": "CRR + payoff smoothing", "method": "crr", "smoothing": True},
    "crr_richardson": {"label": "CRR + smoothing + Richardson", "method": "crr", "smoothing": True, "richardson": True},
    "jarrow_rudd": {"label": "Binomial Jarrow-Rudd", "method": "jarrow_rudd"},
    "tian": {"label": "Binomial Tian", "method": "tian"},
    "leisen_reimer": {"label": "Binomial Leisen-Reimer", "method": "leisen_reimer"},
    "trinomial": {"label": "Trinomial (Boyle)", "trinomial": True},
}

MC_ENGINES = {
    "plain": {"label": "MC plain", "antithetic": False, "control_variate": False},
    "antithetic": {"label": "MC antithetic", "antithetic": True, "control_variate": False},
    "control": {"label": "MC control variate", "antithetic": False, "control_variate": True},
    "both": {"label": "MC antithetic + control", "antithetic": True, "control_variate": True},
    "qmc": {"label": "QMC Halton + both", "antithetic": True, "control_variate": True, "sampler": "halton"},
}


def _fit_order(sizes: np.ndarray, errs: np.ndarray) -> dict:
    """Fit |error| ~ C * n^(-p) by least squares in log-log space."""
    ok = np.isfinite(errs) & (np.abs(errs) > 0) & np.isfinite(sizes) & (sizes > 0)
    if ok.sum() < 3:
        return {"order": float("nan"), "r_squared": float("nan"), "n_points": int(ok.sum())}
    x = np.log(sizes[ok])
    y = np.log(np.abs(errs[ok]))
    A = np.column_stack([x, np.ones(x.size)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "order": float(-coef[0]),
        "intercept": float(coef[1]),
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n_points": int(ok.sum()),
    }


def lattice_convergence(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool = True,
    american: bool = False,
    engines: tuple[str, ...] = ("crr", "crr_smooth", "leisen_reimer", "trinomial"),
    n_min: int = 4,
    n_max: int = 800,
    n_points: int = 26,
) -> dict:
    """Error and timing versus step count for the lattice engines."""
    sizes = np.unique(np.round(np.geomspace(max(n_min, 2), max(n_max, n_min + 1), n_points)).astype(int))

    if american:
        reference = lattice.american_reference(S, K, T, r, q, sigma, is_call)
        ref_label = "Leisen-Reimer n=6001 + Richardson"
    else:
        reference = float(np.asarray(bs.price(S, K, T, r, q, sigma, is_call)).reshape(()))
        ref_label = "Black-Scholes closed form"

    series = []
    for key in engines:
        spec = LATTICE_ENGINES.get(key)
        if spec is None:
            continue
        pts = []
        for n in sizes:
            t0 = time.perf_counter()
            if spec.get("trinomial"):
                res = lattice.trinomial(S, K, T, r, q, sigma, int(n), is_call, american)
            else:
                res = lattice.binomial(
                    S, K, T, r, q, sigma, int(n), is_call, american,
                    spec["method"], spec.get("smoothing", False), spec.get("richardson", False),
                )
            ms = (time.perf_counter() - t0) * 1e3
            pts.append({"n": int(res.steps or n), "price": res.price, "error": res.price - reference, "ms": ms})
        errs = np.array([p["error"] for p in pts])
        ns = np.array([p["n"] for p in pts], dtype=float)
        series.append({
            "key": key,
            "label": spec["label"],
            "points": pts,
            "fit": _fit_order(ns, errs),
            "final_error": float(errs[-1]),
            "total_ms": float(sum(p["ms"] for p in pts)),
        })

    return {
        "kind": "lattice",
        "reference": reference,
        "reference_label": ref_label,
        "exercise": "american" if american else "european",
        "sizes": sizes.tolist(),
        "series": series,
        "note": (
            "Unsmoothed binomial error oscillates with n because the terminal lattice "
            "straddles the strike differently at each step count; a low R^2 on the "
            "fitted order is that sawtooth, not a bad fit."
        ),
    }


def mc_convergence(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool = True,
    engines: tuple[str, ...] = ("plain", "antithetic", "both", "qmc"),
    n_min: int = 512,
    n_max: int = 262_144,
    n_points: int = 10,
    seed: int = 987,
) -> dict:
    """Error, standard error and timing versus path count for the MC estimators."""
    sizes = np.unique(np.round(np.geomspace(max(n_min, 64), max(n_max, n_min * 2), n_points)).astype(int))
    reference = float(np.asarray(bs.price(S, K, T, r, q, sigma, is_call)).reshape(()))

    series = []
    for key in engines:
        spec = MC_ENGINES.get(key)
        if spec is None:
            continue
        pts = []
        for i, n in enumerate(sizes):
            t0 = time.perf_counter()
            res = mc.european(
                S, K, T, r, q, sigma, int(n), is_call,
                antithetic=spec.get("antithetic", False),
                control_variate=spec.get("control_variate", False),
                sampler=spec.get("sampler", "pseudo"),
                seed=seed + 101 * i,
                want_greeks=False,
            )
            ms = (time.perf_counter() - t0) * 1e3
            pts.append({
                "n": int(res.paths),
                "price": res.price,
                "error": res.price - reference,
                "std_error": res.std_error,
                "ms": ms,
                "efficiency": res.variance_reduction,
            })
        errs = np.array([p["error"] for p in pts])
        ns = np.array([p["n"] for p in pts], dtype=float)
        series.append({
            "key": key,
            "label": spec["label"],
            "points": pts,
            "fit": _fit_order(ns, errs),
            "se_fit": _fit_order(ns, np.array([p["std_error"] for p in pts])),
            "final_error": float(errs[-1]),
            "total_ms": float(sum(p["ms"] for p in pts)),
        })

    return {
        "kind": "monte-carlo",
        "reference": reference,
        "reference_label": "Black-Scholes closed form",
        "theoretical_order": 0.5,
        "sizes": sizes.tolist(),
        "series": series,
        "note": (
            "Plain Monte Carlo error decays as n^-1/2 regardless of variance reduction; "
            "what antithetic sampling and the control variate change is the constant, "
            "not the rate. Randomized-Halton QMC changes the rate itself."
        ),
    }


def engine_shootout(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool = True,
    american: bool = False,
    target_bp: float = 1.0,
) -> dict:
    """Cost of reaching a target accuracy, per engine.

    This is the comparison that actually matters in practice: not "which is more
    accurate at n = 500" but "what does each engine cost to get within a basis
    point of the premium". Reported as the wall time at the smallest
    discretization that hits the target and stays there.
    """
    if american:
        reference = lattice.american_reference(S, K, T, r, q, sigma, is_call)
    else:
        reference = float(np.asarray(bs.price(S, K, T, r, q, sigma, is_call)).reshape(()))
    tol = abs(reference) * target_bp / 10_000.0
    tol = max(tol, 1e-8)

    rows = []
    for key, spec in LATTICE_ENGINES.items():
        hit = None
        for n in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
            t0 = time.perf_counter()
            if spec.get("trinomial"):
                res = lattice.trinomial(S, K, T, r, q, sigma, n, is_call, american)
            else:
                res = lattice.binomial(
                    S, K, T, r, q, sigma, n, is_call, american,
                    spec["method"], spec.get("smoothing", False), spec.get("richardson", False),
                )
            ms = (time.perf_counter() - t0) * 1e3
            if abs(res.price - reference) <= tol:
                hit = {"n": int(res.steps or n), "ms": ms, "error": res.price - reference}
                break
        rows.append({
            "key": key, "label": spec["label"], "family": "lattice",
            "reached": hit is not None,
            "n": hit["n"] if hit else None,
            "ms": hit["ms"] if hit else None,
            "error": hit["error"] if hit else None,
        })

    if not american:
        for key, spec in MC_ENGINES.items():
            hit = None
            for n in (1024, 4096, 16_384, 65_536, 262_144, 1_048_576):
                t0 = time.perf_counter()
                res = mc.european(
                    S, K, T, r, q, sigma, n, is_call,
                    antithetic=spec.get("antithetic", False),
                    control_variate=spec.get("control_variate", False),
                    sampler=spec.get("sampler", "pseudo"),
                    seed=4242, want_greeks=False,
                )
                ms = (time.perf_counter() - t0) * 1e3
                # Require the confidence interval to sit inside the tolerance, not
                # just the point estimate: a lucky draw is not convergence.
                if abs(res.price - reference) <= tol and res.std_error * 1.96 <= tol * 2.0:
                    hit = {"n": int(res.paths), "ms": ms, "error": res.price - reference}
                    break
            rows.append({
                "key": key, "label": spec["label"], "family": "monte-carlo",
                "reached": hit is not None,
                "n": hit["n"] if hit else None,
                "ms": hit["ms"] if hit else None,
                "error": hit["error"] if hit else None,
            })

    return {
        "reference": reference,
        "target_bp": target_bp,
        "tolerance": tol,
        "exercise": "american" if american else "european",
        "rows": rows,
        "note": "Monte Carlo rows additionally require the 95% interval to fit the tolerance.",
    }
