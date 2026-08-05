"""Generalized Black-Scholes-Merton pricing and analytic Greeks.

Everything is vectorized over any broadcastable combination of S, K, T, r, q,
sigma so the Greeks-surface endpoint can evaluate a whole (strike x tenor) grid
in one call.

Conventions
-----------
`q` is a continuous dividend yield. Setting q = r reproduces Black-76 on a
futures/forward underlying; setting q = r_foreign gives Garman-Kohlhagen for FX.

Raw Greeks are per unit of their argument (vega per 1.00 of vol, theta per
year, rho per 1.00 of rate). The `greeks()` helper additionally reports the
trading conventions (vega per vol point, theta per calendar day, rho per bp)
under separate keys so display code never has to guess a scale factor.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .normal import norm_cdf, norm_pdf

# Below these thresholds the diffusion is degenerate and the closed form is
# replaced by its limit (discounted intrinsic on the forward).
_MIN_T = 1e-12
_MIN_VOL = 1e-12
_MIN_S = 1e-300


def _prep(S, K, T, r, q, sigma):
    S, K, T, r, q, sigma = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (S, K, T, r, q, sigma))
    )
    return S, K, T, r, q, sigma


def _d1_d2(S, K, T, r, q, sigma):
    """d1, d2 and sigma*sqrt(T), with degenerate cells clamped (not fixed).

    Callers must mask the degenerate cells themselves; clamping only keeps the
    intermediate arithmetic finite so NumPy does not emit warnings.
    """
    sqrtT = np.sqrt(np.maximum(T, 0.0))
    vs = np.maximum(sigma * sqrtT, _MIN_VOL)
    with np.errstate(divide="ignore", invalid="ignore"):
        moneyness = np.log(np.maximum(S, _MIN_S) / np.maximum(K, _MIN_S))
    d1 = (moneyness + (r - q + 0.5 * sigma * sigma) * T) / vs
    return d1, d1 - vs, vs, sqrtT


def _degenerate(T, sigma):
    return (T <= _MIN_T) | (sigma <= _MIN_VOL)


def price(S, K, T, r, q, sigma, is_call=True):
    """Black-Scholes-Merton price."""
    S, K, T, r, q, sigma = _prep(S, K, T, r, q, sigma)
    d1, d2, _, _ = _d1_d2(S, K, T, r, q, sigma)

    dfq = np.exp(-q * T)
    dfr = np.exp(-r * T)

    call = S * dfq * norm_cdf(d1) - K * dfr * norm_cdf(d2)
    put = K * dfr * norm_cdf(-d2) - S * dfq * norm_cdf(-d1)
    out = np.where(is_call, call, put)

    # Degenerate limit: discounted intrinsic measured on the forward.
    fwd = S * dfq - K * dfr
    limit = np.where(is_call, np.maximum(fwd, 0.0), np.maximum(-fwd, 0.0))
    return np.where(_degenerate(T, sigma), limit, out)


def delta(S, K, T, r, q, sigma, is_call=True):
    S, K, T, r, q, sigma = _prep(S, K, T, r, q, sigma)
    d1, _, _, _ = _d1_d2(S, K, T, r, q, sigma)
    dfq = np.exp(-q * T)
    out = np.where(is_call, dfq * norm_cdf(d1), -dfq * norm_cdf(-d1))
    fwd = S * dfq - K * np.exp(-r * T)
    limit = np.where(is_call, dfq * (fwd > 0), -dfq * (fwd < 0))
    return np.where(_degenerate(T, sigma), limit, out)


def gamma(S, K, T, r, q, sigma):
    S, K, T, r, q, sigma = _prep(S, K, T, r, q, sigma)
    d1, _, vs, _ = _d1_d2(S, K, T, r, q, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.exp(-q * T) * norm_pdf(d1) / (np.maximum(S, _MIN_S) * vs)
    return np.where(_degenerate(T, sigma), 0.0, out)


def vega(S, K, T, r, q, sigma):
    """dPrice/dSigma, per 1.00 of volatility."""
    S, K, T, r, q, sigma = _prep(S, K, T, r, q, sigma)
    d1, _, _, sqrtT = _d1_d2(S, K, T, r, q, sigma)
    out = S * np.exp(-q * T) * norm_pdf(d1) * sqrtT
    return np.where(_degenerate(T, sigma), 0.0, out)


def theta(S, K, T, r, q, sigma, is_call=True):
    """dPrice/dt with calendar time moving forward, per year."""
    S, K, T, r, q, sigma = _prep(S, K, T, r, q, sigma)
    d1, d2, _, sqrtT = _d1_d2(S, K, T, r, q, sigma)
    dfq = np.exp(-q * T)
    dfr = np.exp(-r * T)

    with np.errstate(divide="ignore", invalid="ignore"):
        decay = -S * dfq * norm_pdf(d1) * sigma / (2.0 * np.maximum(sqrtT, _MIN_VOL))
    call = decay + q * S * dfq * norm_cdf(d1) - r * K * dfr * norm_cdf(d2)
    put = decay - q * S * dfq * norm_cdf(-d1) + r * K * dfr * norm_cdf(-d2)
    out = np.where(is_call, call, put)
    return np.where(_degenerate(T, sigma), 0.0, out)


def rho(S, K, T, r, q, sigma, is_call=True):
    """dPrice/dr, per 1.00 of rate."""
    S, K, T, r, q, sigma = _prep(S, K, T, r, q, sigma)
    _, d2, _, _ = _d1_d2(S, K, T, r, q, sigma)
    dfr = np.exp(-r * T)
    out = np.where(is_call, K * T * dfr * norm_cdf(d2), -K * T * dfr * norm_cdf(-d2))
    return np.where(_degenerate(T, sigma), 0.0, out)


def epsilon(S, K, T, r, q, sigma, is_call=True):
    """dPrice/dq (a.k.a. psi / dividend rho), per 1.00 of yield."""
    S, K, T, r, q, sigma = _prep(S, K, T, r, q, sigma)
    d1, _, _, _ = _d1_d2(S, K, T, r, q, sigma)
    dfq = np.exp(-q * T)
    out = np.where(is_call, -S * T * dfq * norm_cdf(d1), S * T * dfq * norm_cdf(-d1))
    return np.where(_degenerate(T, sigma), 0.0, out)


# --------------------------------------------------------------------------
# Second and third order
# --------------------------------------------------------------------------

def vanna(S, K, T, r, q, sigma):
    """dDelta/dSigma == dVega/dS."""
    S, K, T, r, q, sigma = _prep(S, K, T, r, q, sigma)
    d1, d2, _, _ = _d1_d2(S, K, T, r, q, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -np.exp(-q * T) * norm_pdf(d1) * d2 / np.maximum(sigma, _MIN_VOL)
    return np.where(_degenerate(T, sigma), 0.0, out)


def volga(S, K, T, r, q, sigma):
    """d2Price/dSigma2 (vomma)."""
    S, K, T, r, q, sigma = _prep(S, K, T, r, q, sigma)
    d1, d2, _, _ = _d1_d2(S, K, T, r, q, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = vega(S, K, T, r, q, sigma) * d1 * d2 / np.maximum(sigma, _MIN_VOL)
    return np.where(_degenerate(T, sigma), 0.0, out)


def charm(S, K, T, r, q, sigma, is_call=True):
    """dDelta/dt, per year. How fast the hedge ratio drifts as time passes."""
    S, K, T, r, q, sigma = _prep(S, K, T, r, q, sigma)
    d1, d2, vs, sqrtT = _d1_d2(S, K, T, r, q, sigma)
    dfq = np.exp(-q * T)
    with np.errstate(divide="ignore", invalid="ignore"):
        shared = -dfq * norm_pdf(d1) * (2.0 * (r - q) * T - d2 * vs) / (2.0 * np.maximum(T, _MIN_T) * vs)
    call = q * dfq * norm_cdf(d1) + shared
    put = -q * dfq * norm_cdf(-d1) + shared
    out = np.where(is_call, call, put)
    return np.where(_degenerate(T, sigma), 0.0, out)


def speed(S, K, T, r, q, sigma):
    """dGamma/dS."""
    S, K, T, r, q, sigma = _prep(S, K, T, r, q, sigma)
    d1, _, vs, _ = _d1_d2(S, K, T, r, q, sigma)
    g = gamma(S, K, T, r, q, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -g / np.maximum(S, _MIN_S) * (d1 / vs + 1.0)
    return np.where(_degenerate(T, sigma), 0.0, out)


def zomma(S, K, T, r, q, sigma):
    """dGamma/dSigma."""
    S, K, T, r, q, sigma = _prep(S, K, T, r, q, sigma)
    d1, d2, _, _ = _d1_d2(S, K, T, r, q, sigma)
    g = gamma(S, K, T, r, q, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = g * (d1 * d2 - 1.0) / np.maximum(sigma, _MIN_VOL)
    return np.where(_degenerate(T, sigma), 0.0, out)


def color(S, K, T, r, q, sigma):
    """dGamma/dt, per year."""
    S, K, T, r, q, sigma = _prep(S, K, T, r, q, sigma)
    d1, d2, vs, sqrtT = _d1_d2(S, K, T, r, q, sigma)
    dfq = np.exp(-q * T)
    Ts = np.maximum(T, _MIN_T)
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = 2.0 * q * Ts + 1.0 + d1 * (2.0 * (r - q) * Ts - d2 * vs) / vs
        # Haug's expression is dGamma/dT; negate for calendar time moving forward
        # so color, charm and theta all share one time convention.
        out = dfq * norm_pdf(d1) / (2.0 * np.maximum(S, _MIN_S) * Ts * vs) * inner
    return np.where(_degenerate(T, sigma), 0.0, out)


def dual_delta(S, K, T, r, q, sigma, is_call=True):
    """dPrice/dK. Up to a discount factor this is the risk-neutral CDF."""
    S, K, T, r, q, sigma = _prep(S, K, T, r, q, sigma)
    _, d2, _, _ = _d1_d2(S, K, T, r, q, sigma)
    dfr = np.exp(-r * T)
    out = np.where(is_call, -dfr * norm_cdf(d2), dfr * norm_cdf(-d2))
    return np.where(_degenerate(T, sigma), 0.0, out)


def dual_gamma(S, K, T, r, q, sigma):
    """d2Price/dK2 == discounted risk-neutral density at K."""
    S, K, T, r, q, sigma = _prep(S, K, T, r, q, sigma)
    _, d2, vs, _ = _d1_d2(S, K, T, r, q, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.exp(-r * T) * norm_pdf(d2) / (np.maximum(K, _MIN_S) * vs)
    return np.where(_degenerate(T, sigma), 0.0, out)


# The set exposed to the surface visualizer: name -> (callable, needs_is_call).
GREEK_REGISTRY: dict[str, tuple[Any, bool]] = {
    "price": (price, True),
    "delta": (delta, True),
    "gamma": (gamma, False),
    "vega": (vega, False),
    "theta": (theta, True),
    "rho": (rho, True),
    "epsilon": (epsilon, True),
    "vanna": (vanna, False),
    "volga": (volga, False),
    "charm": (charm, True),
    "speed": (speed, False),
    "zomma": (zomma, False),
    "color": (color, False),
    "dual_delta": (dual_delta, True),
    "dual_gamma": (dual_gamma, False),
}

# Display scaling applied on top of the raw per-unit value, plus the unit label.
GREEK_DISPLAY: dict[str, tuple[float, str]] = {
    "price": (1.0, "per contract-share"),
    "delta": (1.0, "per $1 of spot"),
    "gamma": (1.0, "delta per $1 of spot"),
    "vega": (0.01, "per +1 vol point"),
    "theta": (1.0 / 365.0, "per calendar day"),
    "rho": (0.01, "per +1% rate"),
    "epsilon": (0.01, "per +1% yield"),
    "vanna": (0.01, "delta per +1 vol point"),
    "volga": (0.0001, "vega per +1 vol point"),
    "charm": (1.0 / 365.0, "delta per calendar day"),
    "speed": (1.0, "gamma per $1 of spot"),
    "zomma": (0.01, "gamma per +1 vol point"),
    "color": (1.0 / 365.0, "gamma per calendar day"),
    "dual_delta": (1.0, "per $1 of strike"),
    "dual_gamma": (1.0, "dual delta per $1 of strike"),
}


def evaluate(name: str, S, K, T, r, q, sigma, is_call=True):
    """Evaluate one registered Greek by name."""
    try:
        fn, needs_call = GREEK_REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown greek {name!r}") from None
    return fn(S, K, T, r, q, sigma, is_call) if needs_call else fn(S, K, T, r, q, sigma)


def greeks(S, K, T, r, q, sigma, is_call=True) -> dict[str, float]:
    """Full analytic Greek set for scalar inputs, raw and display-scaled."""
    out: dict[str, float] = {}
    for name in GREEK_REGISTRY:
        raw = float(np.asarray(evaluate(name, S, K, T, r, q, sigma, is_call)).reshape(()))
        out[name] = raw
        scale, _ = GREEK_DISPLAY[name]
        if scale != 1.0:
            out[name + "_display"] = raw * scale
    return out


def parity_gap(S, K, T, r, q, sigma) -> float:
    """C - P - (S e^-qT - K e^-rT); should be ~0. Used as a self-check."""
    c = float(np.asarray(price(S, K, T, r, q, sigma, True)).reshape(()))
    p = float(np.asarray(price(S, K, T, r, q, sigma, False)).reshape(()))
    fwd = float(S) * np.exp(-float(q) * float(T)) - float(K) * np.exp(-float(r) * float(T))
    return c - p - fwd
