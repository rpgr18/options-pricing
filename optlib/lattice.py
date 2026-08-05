"""Lattice pricers: binomial (four parameterizations) and Boyle trinomial.

Each engine prices European or American exercise and returns Greeks read off
the lattice itself where that is natural (delta/gamma from the first two time
slices, theta from the step-2 centre node) and by central difference otherwise.

Two accuracy refinements are available on every binomial engine because they
are what make the convergence chart interesting:

`smoothing`
    Replace the payoff at the penultimate time slice with the closed-form
    European value over the last step. The sawtooth in binomial convergence is
    caused by the kink in the terminal payoff falling between nodes; integrating
    the last step analytically removes it and turns O(1/n) with oscillation
    into smooth O(1/n).

`richardson`
    Two-point Richardson extrapolation, 2*V(n) - V(n/2). Valid once the error
    is a clean O(1/n) series, i.e. it should be paired with `smoothing`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import blackscholes as bs
from .normal import norm_cdf

BINOMIAL_METHODS = ("crr", "jarrow_rudd", "tian", "leisen_reimer")


@dataclass
class LatticeResult:
    price: float
    delta: float = float("nan")
    gamma: float = float("nan")
    theta: float = float("nan")
    vega: float = float("nan")
    rho: float = float("nan")
    steps: int = 0
    method: str = ""
    early_exercise_premium: float = float("nan")
    # Exercise boundary for American options: (time, critical spot) pairs.
    boundary: list[tuple[float, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {
            "price": self.price,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
            "steps": self.steps,
            "method": self.method,
            "early_exercise_premium": self.early_exercise_premium,
        }
        if self.boundary:
            d["boundary"] = [{"t": t, "s": s} for t, s in self.boundary]
        return d


def _payoff(S: np.ndarray, K: float, is_call: bool) -> np.ndarray:
    return np.maximum(S - K, 0.0) if is_call else np.maximum(K - S, 0.0)


def _theta_from_pde(V, dlt, gma, S, r, q, sigma) -> float:
    """Theta implied by the Black-Scholes PDE given lattice delta and gamma.

    Reading theta off the step-2 centre node only works when u*d == 1, which is
    true for CRR and Leisen-Reimer but not for Jarrow-Rudd or Tian: their trees
    are drift-shifted, so the "centre" node sits at a different spot than S and
    the naive difference quotient picks up a spurious delta term. Inverting the
    PDE instead is exact for European exercise and valid for American exercise
    anywhere in the continuation region, and it is independent of the geometry.
    """
    if not (np.isfinite(dlt) and np.isfinite(gma)):
        return float("nan")
    return float(-(0.5 * sigma * sigma * S * S * gma + (r - q) * S * dlt - r * V))


# --------------------------------------------------------------------------
# Step parameterizations
# --------------------------------------------------------------------------

def _peizer_pratt(z: float, n: int) -> float:
    """Peizer-Pratt inversion, method 2. Maps a normal quantile to a probability."""
    c = 1.0 / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))
    inner = 0.25 - 0.25 * np.exp(-((z * c) ** 2) * (n + 1.0 / 6.0))
    return 0.5 + np.sign(z) * np.sqrt(max(inner, 0.0))


def _binomial_params(method: str, S, K, T, r, q, sigma, n):
    """Return (u, d, p) for one time step under the requested scheme."""
    dt = T / n
    growth = np.exp((r - q) * dt)

    if method == "crr":
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
    elif method == "jarrow_rudd":
        # Equal-probability tree: drift is carried by the node spacing.
        drift = (r - q - 0.5 * sigma * sigma) * dt
        u = np.exp(drift + sigma * np.sqrt(dt))
        d = np.exp(drift - sigma * np.sqrt(dt))
        return u, d, 0.5
    elif method == "tian":
        # Tian (1993): matches the first three moments exactly.
        v = np.exp(sigma * sigma * dt)
        rt = np.sqrt(v * v + 2.0 * v - 3.0)
        u = 0.5 * growth * v * (v + 1.0 + rt)
        d = 0.5 * growth * v * (v + 1.0 - rt)
    elif method == "leisen_reimer":
        if n % 2 == 0:
            n += 1
            dt = T / n
            growth = np.exp((r - q) * dt)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        p = _peizer_pratt(d2, n)
        p_hat = _peizer_pratt(d1, n)
        u = growth * p_hat / p
        d = (growth - p * u) / (1.0 - p)
        return u, d, p
    else:
        raise ValueError(f"unknown binomial method {method!r}")

    p = (growth - d) / (u - d)
    return u, d, p


def _effective_steps(method: str, n: int) -> int:
    """Leisen-Reimer requires an odd number of steps."""
    if method == "leisen_reimer" and n % 2 == 0:
        return n + 1
    return n


# --------------------------------------------------------------------------
# Binomial
# --------------------------------------------------------------------------

def binomial(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n: int = 500,
    is_call: bool = True,
    american: bool = False,
    method: str = "crr",
    smoothing: bool = False,
    richardson: bool = False,
    want_boundary: bool = False,
    want_greeks: bool = True,
) -> LatticeResult:
    """Price on a binomial lattice and read Greeks off the tree.

    Vega and rho need four extra tree builds (central differences in sigma and
    r), which dominates the cost at large step counts -- pass want_greeks=False
    when only the price is wanted, as the high-accuracy reference does.
    """
    if richardson:
        # Halve the step count for the coarse leg, keeping parity requirements.
        n_fine = _effective_steps(method, max(n, 4))
        n_coarse = _effective_steps(method, max(n_fine // 2, 2))
        fine = binomial(S, K, T, r, q, sigma, n_fine, is_call, american, method, smoothing, False, want_boundary, want_greeks)
        coarse = binomial(S, K, T, r, q, sigma, n_coarse, is_call, american, method, smoothing, False, False, want_greeks)
        out = LatticeResult(
            price=2.0 * fine.price - coarse.price,
            delta=2.0 * fine.delta - coarse.delta,
            gamma=2.0 * fine.gamma - coarse.gamma,
            theta=2.0 * fine.theta - coarse.theta,
            vega=fine.vega,
            rho=fine.rho,
            steps=n_fine,
            method=method + "+richardson",
            boundary=fine.boundary,
        )
        euro = bs.price(S, K, T, r, q, sigma, is_call)
        out.early_exercise_premium = out.price - float(np.asarray(euro).reshape(())) if american else 0.0
        return out

    v0, dlt, gma, tht, boundary = _binomial_core(
        S, K, T, r, q, sigma, n, is_call, american, method, smoothing, want_boundary
    )

    vega_ = rho_ = float("nan")
    if want_greeks:
        # Vol and rate sensitivities by central difference on the same lattice.
        h_v, h_r = 1e-3, 1e-4
        v_up = _binomial_core(S, K, T, r, q, sigma + h_v, n, is_call, american, method, smoothing, False)[0]
        v_dn = _binomial_core(S, K, T, r, q, max(sigma - h_v, 1e-8), n, is_call, american, method, smoothing, False)[0]
        r_up = _binomial_core(S, K, T, r + h_r, q, sigma, n, is_call, american, method, smoothing, False)[0]
        r_dn = _binomial_core(S, K, T, r - h_r, q, sigma, n, is_call, american, method, smoothing, False)[0]
        vega_ = (v_up - v_dn) / (2.0 * h_v)
        rho_ = (r_up - r_dn) / (2.0 * h_r)

    euro_ref = float(np.asarray(bs.price(S, K, T, r, q, sigma, is_call)).reshape(()))
    return LatticeResult(
        price=v0,
        delta=dlt,
        gamma=gma,
        theta=tht,
        vega=vega_,
        rho=rho_,
        steps=_effective_steps(method, n),
        method=method + ("+smooth" if smoothing else ""),
        early_exercise_premium=(v0 - euro_ref) if american else 0.0,
        boundary=boundary,
    )


def _binomial_core(S, K, T, r, q, sigma, n, is_call, american, method, smoothing, want_boundary):
    n = _effective_steps(method, max(int(n), 2))
    dt = T / n
    u, d, p = _binomial_params(method, S, K, T, r, q, sigma, n)
    disc = np.exp(-r * dt)
    pu, pd = disc * p, disc * (1.0 - p)

    j = np.arange(n + 1)
    # Node price after j up-moves out of the current level.
    def spot_at(level: int) -> np.ndarray:
        k = np.arange(level + 1)
        return S * (u ** k) * (d ** (level - k))

    if smoothing:
        # Start one slice early with the analytic value of the final step.
        level = n - 1
        S_pen = spot_at(level)
        V = bs.price(S_pen, K, dt, r, q, sigma, is_call)
        V = np.asarray(V, dtype=float).copy()
        if american:
            V = np.maximum(V, _payoff(S_pen, K, is_call))
        start = level - 1
    else:
        V = _payoff(spot_at(n), K, is_call)
        start = n - 1

    boundary: list[tuple[float, float]] = []
    slice1 = slice2 = None

    for i in range(start, -1, -1):
        V = pu * V[1:] + pd * V[:-1]
        if american or want_boundary:
            S_i = spot_at(i)
            ex = _payoff(S_i, K, is_call)
            if american:
                exercised = ex > V
                V = np.where(exercised, ex, V)
                if want_boundary and exercised.any():
                    # The critical spot is the edge of the exercise region facing
                    # the continuation region. A call is exercised when spot is
                    # HIGH, so its region is S >= S*(t) and the boundary is the
                    # lowest exercised node; a put is the mirror image.
                    idx = np.flatnonzero(exercised)
                    node = idx.min() if is_call else idx.max()
                    boundary.append((i * dt, float(S_i[node])))
        if i == 2:
            slice2 = (V.copy(), spot_at(2))
        elif i == 1:
            slice1 = (V.copy(), spot_at(1))

    v0 = float(V[0])

    dlt = gma = tht = float("nan")
    if slice1 is not None:
        (Vd, Vu), (Sd, Su) = slice1[0], slice1[1]
        dlt = float((Vu - Vd) / (Su - Sd))
    if slice2 is not None:
        V2, S2 = slice2
        d_up = (V2[2] - V2[1]) / (S2[2] - S2[1])
        d_dn = (V2[1] - V2[0]) / (S2[1] - S2[0])
        gma = float((d_up - d_dn) / (0.5 * (S2[2] - S2[0])))
        tht = _theta_from_pde(v0, dlt, gma, S, r, q, sigma)

    boundary.reverse()
    return v0, dlt, gma, tht, boundary


# --------------------------------------------------------------------------
# Trinomial (Boyle)
# --------------------------------------------------------------------------

def trinomial(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n: int = 300,
    is_call: bool = True,
    american: bool = False,
    lam: float = 1.2247448713915889,  # sqrt(3/2): the Boyle-optimal stretch
) -> LatticeResult:
    """Boyle trinomial lattice. Converges as O(1/n^2) for European payoffs."""
    n = max(int(n), 2)
    dt = T / n
    dx = lam * sigma * np.sqrt(dt)
    nu = (r - q - 0.5 * sigma * sigma) * dt

    pu = 0.5 * ((sigma * sigma * dt + nu * nu) / (dx * dx) + nu / dx)
    pd = 0.5 * ((sigma * sigma * dt + nu * nu) / (dx * dx) - nu / dx)
    pm = 1.0 - pu - pd
    disc = np.exp(-r * dt)

    def spot_at(level: int) -> np.ndarray:
        k = np.arange(-level, level + 1)
        return S * np.exp(k * dx)

    V = _payoff(spot_at(n), K, is_call)
    slice1 = slice2 = None
    for i in range(n - 1, -1, -1):
        V = disc * (pu * V[2:] + pm * V[1:-1] + pd * V[:-2])
        if american:
            V = np.maximum(V, _payoff(spot_at(i), K, is_call))
        if i == 2:
            slice2 = (V.copy(), spot_at(2))
        elif i == 1:
            slice1 = (V.copy(), spot_at(1))

    v0 = float(V[0])
    dlt = gma = tht = float("nan")
    if slice1 is not None:
        V1, S1 = slice1
        dlt = float((V1[2] - V1[0]) / (S1[2] - S1[0]))
    if slice2 is not None:
        V2, S2 = slice2
        d_up = (V2[3] - V2[2]) / (S2[3] - S2[2])
        d_dn = (V2[2] - V2[1]) / (S2[2] - S2[1])
        gma = float((d_up - d_dn) / (0.5 * (S2[3] - S2[1])))
        tht = _theta_from_pde(v0, dlt, gma, S, r, q, sigma)

    euro_ref = float(np.asarray(bs.price(S, K, T, r, q, sigma, is_call)).reshape(()))
    return LatticeResult(
        price=v0,
        delta=dlt,
        gamma=gma,
        theta=tht,
        steps=n,
        method="trinomial",
        early_exercise_premium=(v0 - euro_ref) if american else 0.0,
    )


def american_reference(
    S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool = True, n: int = 6001
) -> float:
    """High-accuracy American value used as the convergence-study target.

    Leisen-Reimer with a large odd step count plus Richardson extrapolation;
    accurate to well under a basis point of premium for ordinary inputs.
    """
    res = binomial(
        S, K, T, r, q, sigma, n, is_call, True, "leisen_reimer",
        smoothing=True, richardson=True, want_greeks=False,
    )
    return res.price
