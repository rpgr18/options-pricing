"""Implied volatility inversion.

Strategy: check the no-arbitrage bounds first, seed with a closed-form
approximation, then run guarded Newton on total variance and fall back to
bisection if Newton leaves the bracket. Newton alone is not safe here — vega
collapses for deep out-of-the-money or nearly expired options, so an unguarded
step can fly off to a negative or absurd volatility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import blackscholes as bs

VOL_MIN = 1e-6
VOL_MAX = 6.0


@dataclass
class IVResult:
    vol: float
    iterations: int
    residual: float
    converged: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _bounds(S: float, K: float, T: float, r: float, q: float, is_call: bool) -> tuple[float, float]:
    """Arbitrage-free price interval for a vanilla European option."""
    dfq, dfr = np.exp(-q * T), np.exp(-r * T)
    if is_call:
        return max(S * dfq - K * dfr, 0.0), S * dfq
    return max(K * dfr - S * dfq, 0.0), K * dfr


def _seed(S: float, K: float, T: float, r: float, q: float, price: float, is_call: bool) -> float:
    """Corrado-Miller style seed, falling back to Brenner-Subrahmanyam.

    Both are built for near-the-money options; the clamp keeps the seed inside a
    sane band when they are pushed outside their range of validity.
    """
    dfr = np.exp(-r * T)
    Se = S * np.exp(-q * T)
    Kd = K * dfr
    # Convert to the equivalent call price so one formula covers both types.
    c = price if is_call else price + Se - Kd

    bs_seed = np.sqrt(2.0 * np.pi / T) * c / max(Se, 1e-12)
    inner = (c - 0.5 * (Se - Kd)) ** 2 - (Se - Kd) ** 2 / np.pi
    if inner > 0.0:
        cm = (np.sqrt(2.0 * np.pi / T) / (Se + Kd)) * (
            (c - 0.5 * (Se - Kd)) + np.sqrt(inner)
        )
        if np.isfinite(cm) and cm > 0.0:
            return float(np.clip(cm, 0.01, 3.0))
    return float(np.clip(bs_seed if np.isfinite(bs_seed) and bs_seed > 0 else 0.3, 0.01, 3.0))


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    q: float = 0.0,
    is_call: bool = True,
    tol: float = 1e-10,
    max_iter: int = 60,
) -> IVResult:
    """Back out Black-Scholes volatility from a single option price."""
    if not np.isfinite(price) or price <= 0.0 or T <= 0.0:
        return IVResult(float("nan"), 0, float("nan"), False, "non-positive price or expiry")

    lo_p, hi_p = _bounds(S, K, T, r, q, is_call)
    if price < lo_p - 1e-12:
        return IVResult(float("nan"), 0, price - lo_p, False, "price below intrinsic bound")
    if price > hi_p + 1e-12:
        return IVResult(float("nan"), 0, price - hi_p, False, "price above no-arbitrage cap")
    if price <= lo_p + 1e-14:
        return IVResult(VOL_MIN, 0, 0.0, True, "at the intrinsic bound")

    def f(vol: float) -> float:
        return float(np.asarray(bs.price(S, K, T, r, q, vol, is_call)).reshape(())) - price

    lo, hi = VOL_MIN, VOL_MAX
    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0.0:
        return IVResult(VOL_MIN, 0, f_lo, False, "price unreachable: below the vol floor")
    if f_hi < 0.0:
        return IVResult(VOL_MAX, 0, f_hi, False, "price unreachable: above the vol cap")

    vol = _seed(S, K, T, r, q, price, is_call)
    vol = min(max(vol, lo), hi)

    for i in range(1, max_iter + 1):
        diff = f(vol)
        if abs(diff) < tol:
            return IVResult(vol, i, diff, True, "newton")

        # Maintain the bracket from every evaluation, so the fallback is tight.
        if diff > 0.0:
            hi, f_hi = vol, diff
        else:
            lo, f_lo = vol, diff

        v = float(np.asarray(bs.vega(S, K, T, r, q, vol)).reshape(()))
        step = diff / v if v > 1e-12 else np.inf
        nxt = vol - step
        if not np.isfinite(nxt) or nxt <= lo or nxt >= hi:
            nxt = 0.5 * (lo + hi)  # bisect when Newton leaves the bracket
        if abs(nxt - vol) < 1e-15:
            return IVResult(vol, i, diff, True, "step underflow")
        vol = nxt

    return IVResult(vol, max_iter, f(vol), False, "iteration limit")


def implied_vol_array(
    prices,
    S,
    K,
    T,
    r=0.0,
    q=0.0,
    is_call=True,
) -> np.ndarray:
    """Elementwise implied vol over broadcastable inputs; NaN where inversion fails."""
    prices, S, K, T, r, q, is_call = np.broadcast_arrays(
        np.asarray(prices, dtype=float),
        np.asarray(S, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(T, dtype=float),
        np.asarray(r, dtype=float),
        np.asarray(q, dtype=float),
        np.asarray(is_call, dtype=bool),
    )
    out = np.empty(prices.shape, dtype=float)
    flat = out.reshape(-1)
    for i, (p, s, k, t, rr, qq, c) in enumerate(
        zip(prices.reshape(-1), S.reshape(-1), K.reshape(-1), T.reshape(-1), r.reshape(-1), q.reshape(-1), is_call.reshape(-1))
    ):
        flat[i] = implied_vol(float(p), float(s), float(k), float(t), float(rr), float(qq), bool(c)).vol
    return out
