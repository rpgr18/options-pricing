"""Monte Carlo pricing with variance reduction, MC Greeks, and Longstaff-Schwartz.

Three families of estimator live here:

European
    Exact one-step GBM sampling of the terminal price, optionally with
    antithetic pairing, a regression-optimal control variate on the discounted
    terminal spot, and a randomized-Halton (QMC) sampler in place of
    pseudorandom draws.

Greeks
    Pathwise (infinitesimal perturbation) estimators for delta and vega, which
    are unbiased and far tighter than bumping. Gamma uses the likelihood-ratio
    weight (Z^2 - sigma*sqrt(T)*Z - 1) / (sigma^2 T S^2), because the pathwise
    method fails for a payoff whose second derivative is a point mass.

American
    Longstaff-Schwartz least-squares Monte Carlo, regressing the discounted
    continuation value on a polynomial basis over the in-the-money paths only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import blackscholes as bs
from .normal import norm_ppf

Sampler = Literal["pseudo", "halton"]
_Z95 = 1.959963984540054

# First primes, used as Halton bases (one per time dimension).
_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71)


@dataclass
class MCResult:
    price: float
    std_error: float
    ci_low: float
    ci_high: float
    paths: int
    delta: float = float("nan")
    gamma: float = float("nan")
    vega: float = float("nan")
    theta: float = float("nan")
    rho: float = float("nan")
    delta_se: float = float("nan")
    vega_se: float = float("nan")
    control_beta: float = float("nan")
    variance_reduction: float = float("nan")
    method: str = "mc"
    early_exercise_premium: float = float("nan")

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# --------------------------------------------------------------------------
# Samplers
# --------------------------------------------------------------------------

def _van_der_corput(n: int, base: int, offset: int = 1) -> np.ndarray:
    """Radical-inverse sequence, skipping the degenerate leading zero."""
    idx = np.arange(offset, offset + n, dtype=np.int64)
    out = np.zeros(n, dtype=float)
    f = 1.0 / base
    while np.any(idx > 0):
        out += f * (idx % base)
        idx //= base
        f /= base
    return out


def normal_draws(
    n: int, dims: int = 1, sampler: Sampler = "pseudo", seed: int | None = None
) -> np.ndarray:
    """(n, dims) standard normals from either a pseudorandom or a QMC source."""
    rng = np.random.default_rng(seed)
    if sampler == "pseudo":
        return rng.standard_normal((n, dims))

    if dims > len(_PRIMES):
        raise ValueError(f"halton sampler supports up to {len(_PRIMES)} dimensions")
    u = np.empty((n, dims))
    for j in range(dims):
        # Cranley-Patterson rotation: a random shift mod 1 keeps the low
        # discrepancy but makes independent replications (hence error bars) valid.
        shift = rng.random()
        u[:, j] = (_van_der_corput(n, _PRIMES[j], offset=1) + shift) % 1.0
    # Keep the inverse-CDF away from the open-interval endpoints.
    np.clip(u, 1e-12, 1.0 - 1e-12, out=u)
    return norm_ppf(u)


# --------------------------------------------------------------------------
# European
# --------------------------------------------------------------------------

def _euro_estimator(Z, S, K, T, r, q, sigma, is_call, antithetic, control_variate):
    """Build the estimator sample for one set of normals.

    Returns (sample, raw_payoff, beta). `sample` is the array whose mean is the
    price estimate: antithetic pairs are averaged *before* the variance is taken
    (a pair is one draw from the estimator's point of view, and averaging first
    is what actually cancels the odd part of the payoff), then the control
    variate correction is applied to that reduced sample.
    """
    drift = (r - q - 0.5 * sigma * sigma) * T
    vol = sigma * np.sqrt(T)
    ST = S * np.exp(drift + vol * Z)
    disc = np.exp(-r * T)
    payoff = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)
    Y = disc * payoff
    C = disc * ST

    if antithetic:
        half = Z.size // 2
        Y_s = 0.5 * (Y[:half] + Y[half:])
        C_s = 0.5 * (C[:half] + C[half:])
    else:
        Y_s, C_s = Y, C

    beta = float("nan")
    if control_variate:
        # E[e^{-rT} S_T] = S e^{-qT} exactly, so the discounted spot is a valid
        # control; the regression slope is the variance-minimizing coefficient.
        var_c = float(C_s.var(ddof=1))
        if var_c > 0.0:
            beta = float(np.cov(Y_s, C_s, ddof=1)[0, 1]) / var_c
            Y_s = Y_s - beta * (C_s - S * np.exp(-q * T))

    return Y_s, Y, beta


def european(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int = 100_000,
    is_call: bool = True,
    antithetic: bool = True,
    control_variate: bool = True,
    sampler: Sampler = "pseudo",
    seed: int | None = 12345,
    want_greeks: bool = True,
    n_replications: int = 16,
) -> MCResult:
    """Terminal-value Monte Carlo for a vanilla European option.

    With `sampler="halton"` the usual sqrt(variance/n) standard error is not
    valid — QMC points are deterministic and correlated by construction, so the
    CLT does not apply. Instead the budget is split across `n_replications`
    independently shifted Halton sets and the error is taken from the spread of
    the replication means. That is the only defensible QMC error bar, and it is
    also what makes the QMC-vs-pseudorandom comparison on the convergence tab an
    honest one.
    """
    n_paths = max(int(n_paths), 64)
    randomized = sampler == "halton"
    reps = max(int(n_replications), 2) if randomized else 1
    per_rep = max(n_paths // reps, 32)
    if antithetic:
        per_rep = max((per_rep // 2) * 2, 32)

    rep_means: list[float] = []
    samples: list[np.ndarray] = []
    raws: list[np.ndarray] = []
    Zs: list[np.ndarray] = []
    beta = float("nan")

    for k in range(reps):
        sub_seed = None if seed is None else seed + 7919 * k
        if antithetic:
            half = per_rep // 2
            zh = normal_draws(half, 1, sampler, sub_seed)[:, 0]
            Z = np.concatenate([zh, -zh])
        else:
            Z = normal_draws(per_rep, 1, sampler, sub_seed)[:, 0]
        s_k, raw_k, beta = _euro_estimator(Z, S, K, T, r, q, sigma, is_call, antithetic, control_variate)
        rep_means.append(float(s_k.mean()))
        samples.append(s_k)
        raws.append(raw_k)
        Zs.append(Z)

    Z = np.concatenate(Zs)
    sample = np.concatenate(samples)
    raw = np.concatenate(raws)
    n = Z.size

    if randomized:
        rm = np.asarray(rep_means)
        est = float(rm.mean())
        se = float(np.sqrt(rm.var(ddof=1) / reps))
    else:
        est = float(sample.mean())
        se = float(np.sqrt(sample.var(ddof=1) / sample.size))

    # Efficiency is measured against plain i.i.d. MC using the same number of
    # payoff evaluations, so antithetic pairing and the control variate both count.
    base_var_of_mean = float(raw.var(ddof=1)) / n
    vr = base_var_of_mean / (se * se) if se > 0 else float("nan")

    label = "mc-european"
    tags = [t for t, on in (("antithetic", antithetic), ("control", control_variate), ("halton-qmc", randomized)) if on]
    if tags:
        label += " [" + "+".join(tags) + "]"

    out = MCResult(
        price=est,
        std_error=se,
        ci_low=est - _Z95 * se,
        ci_high=est + _Z95 * se,
        paths=n,
        control_beta=beta,
        variance_reduction=vr,
        method=label,
    )

    if want_greeks:
        drift = (r - q - 0.5 * sigma * sigma) * T
        vol = sigma * np.sqrt(T)
        ST = S * np.exp(drift + vol * Z)
        disc = np.exp(-r * T)
        payoff = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)
        indicator = (ST > K) if is_call else (ST < K)
        sign = 1.0 if is_call else -1.0

        # Pathwise: differentiate the simulated payoff, not the estimator.
        d_delta = sign * disc * indicator * ST / S
        dST_dsigma = ST * ((np.log(ST / S) - drift) / sigma - sigma * T)
        d_vega = sign * disc * indicator * dST_dsigma

        # Likelihood ratio for the second derivative.
        w2 = (Z * Z - vol * Z - 1.0) / (sigma * sigma * T * S * S)
        d_gamma = disc * payoff * w2

        # rho: d/dr of both the drift and the discount factor, pathwise.
        d_rho = disc * (sign * indicator * ST * T - payoff * T)

        out.delta = float(d_delta.mean())
        out.delta_se = float(np.sqrt(d_delta.var(ddof=1) / n))
        out.vega = float(d_vega.mean())
        out.vega_se = float(np.sqrt(d_vega.var(ddof=1) / n))
        out.gamma = float(d_gamma.mean())
        out.rho = float(d_rho.mean())
        # Theta has no clean pathwise form once T enters the discount factor and
        # the drift together; a common-random-numbers central difference is both
        # cheap and low variance because the same Z is reused.
        h = min(1e-4, T * 0.5)
        up = _repriced(S, K, T + h, r, q, sigma, Z, is_call)
        dn = _repriced(S, K, T - h, r, q, sigma, Z, is_call)
        out.theta = -(up - dn) / (2.0 * h)  # calendar time forward = -dV/dT
    return out


def _repriced(S, K, T, r, q, sigma, Z, is_call) -> float:
    """Re-evaluate the discounted payoff on fixed normals (common random numbers)."""
    ST = S * np.exp((r - q - 0.5 * sigma * sigma) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)
    return float(np.exp(-r * T) * payoff.mean())


# --------------------------------------------------------------------------
# American (Longstaff-Schwartz)
# --------------------------------------------------------------------------

def longstaff_schwartz(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int = 50_000,
    n_steps: int = 50,
    is_call: bool = True,
    degree: int = 3,
    antithetic: bool = True,
    seed: int | None = 12345,
) -> MCResult:
    """Least-squares Monte Carlo for an American option.

    The regression uses only in-the-money paths (Longstaff-Schwartz 2001): out
    of the money the exercise decision is trivial, and including those paths
    degrades the fit exactly where the boundary matters.
    """
    n_paths = max(int(n_paths), 64)
    n_steps = max(int(n_steps), 2)
    dt = T / n_steps
    disc_step = np.exp(-r * dt)

    rng = np.random.default_rng(seed)
    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal((half, n_steps))
        Z = np.concatenate([z, -z], axis=0)
    else:
        Z = rng.standard_normal((n_paths, n_steps))
    n = Z.shape[0]

    log_inc = (r - q - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * Z
    paths = S * np.exp(np.cumsum(log_inc, axis=1))  # (n, n_steps), t = dt .. T

    def intrinsic(x):
        return np.maximum(x - K, 0.0) if is_call else np.maximum(K - x, 0.0)

    cashflow = intrinsic(paths[:, -1])

    for t in range(n_steps - 2, -1, -1):
        cashflow *= disc_step
        Sx = paths[:, t]
        ex = intrinsic(Sx)
        itm = ex > 0.0
        if itm.sum() > degree + 2:
            x = Sx[itm] / K  # scale for conditioning
            basis = np.vander(x, degree + 1, increasing=True)
            coef, *_ = np.linalg.lstsq(basis, cashflow[itm], rcond=None)
            continuation = basis @ coef
            exercise_now = ex[itm] > continuation
            idx = np.flatnonzero(itm)[exercise_now]
            cashflow[idx] = ex[idx]

    cashflow *= disc_step  # discount the first step back to t=0
    est = float(cashflow.mean())
    se = float(np.sqrt(cashflow.var(ddof=1) / n))
    euro = float(np.asarray(bs.price(S, K, T, r, q, sigma, is_call)).reshape(()))

    return MCResult(
        price=est,
        std_error=se,
        ci_low=est - _Z95 * se,
        ci_high=est + _Z95 * se,
        paths=n,
        method=f"lsm-american(deg {degree}, {n_steps} steps)",
        early_exercise_premium=est - euro,
    )
