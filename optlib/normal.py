"""Standard normal CDF/PDF and inverse, vectorized, with no SciPy dependency.

The CDF uses Hart's (1968) rational/continued-fraction pair as presented in
Graeme West, "Better Approximations to Cumulative Normal Functions" (2005).
It is accurate to roughly double precision across the whole real line, which
matters here: implied-vol root finding differentiates the CDF, so a cheap
Abramowitz-Stegun approximation (~1e-7) would put a visible floor on the
achievable IV accuracy and pollute the convergence study.
"""

from __future__ import annotations

import numpy as np

SQRT_2PI = 2.5066282746310002
INV_SQRT_2PI = 1.0 / SQRT_2PI

# Hart numerator / denominator coefficients, highest power last.
_HART_NUM = (
    220.206867912376,
    221.213596169931,
    112.079291497871,
    33.912866078383,
    6.37396220353165,
    0.700383064443688,
    3.52624965998911e-02,
)
_HART_DEN = (
    440.413735824752,
    793.826512519948,
    637.333633378831,
    296.564248779674,
    86.7807322029461,
    16.064177579207,
    1.75566716318264,
    8.83883476483184e-02,
)

# Below this |z| the rational form is used; above it, the continued fraction.
_HART_SPLIT = 7.071067811865475  # 5 * sqrt(2)
_HART_CUTOFF = 37.0  # beyond this the tail underflows to 0 in float64


def _horner(coeffs: tuple[float, ...], x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, coeffs[-1])
    for c in reversed(coeffs[:-1]):
        out = out * x + c
    return out


def norm_pdf(z):
    """Standard normal density."""
    z = np.asarray(z, dtype=float)
    return INV_SQRT_2PI * np.exp(-0.5 * z * z)


def norm_cdf(z):
    """Standard normal CDF, accurate to ~1e-15 for |z| < 37."""
    z = np.asarray(z, dtype=float)
    za = np.abs(z)
    small = za < _HART_SPLIT
    live = za <= _HART_CUTOFF

    # exp(-z^2/2) underflows far out in the tail; clamp so it stays finite.
    zc = np.where(live, za, 0.0)
    e = np.exp(-0.5 * zc * zc)

    rational = e * _horner(_HART_NUM, zc) / _horner(_HART_DEN, zc)

    # Continued fraction for the tail. Guard the divisions where zc == 0.
    zt = np.where(small, 1.0, zc)
    cf = zt + 0.65
    for k in (4.0, 3.0, 2.0, 1.0):
        cf = zt + k / cf
    tail = e / (cf * SQRT_2PI)

    upper = np.where(small, rational, tail)
    upper = np.where(live, upper, 0.0)

    return np.where(z > 0.0, 1.0 - upper, upper)


# Acklam's inverse-normal rational approximation (|err| < 1.15e-9), followed by
# one Halley step against the high-accuracy CDF above, which lands it at ~1e-15.
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)


def norm_ppf(p):
    """Inverse standard normal CDF (quantile function)."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan) if p.ndim else np.array(np.nan)

    lo, hi = 0.02425, 1.0 - 0.02425

    with np.errstate(divide="ignore", invalid="ignore"):
        # Lower tail
        m = (p > 0.0) & (p < lo)
        q = np.sqrt(-2.0 * np.log(np.where(m, p, 0.5)))
        low = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
        out = np.where(m, low, out)

        # Central
        m = (p >= lo) & (p <= hi)
        q = np.where(m, p, 0.5) - 0.5
        r = q * q
        cen = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
            ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        )
        out = np.where(m, cen, out)

        # Upper tail (by symmetry)
        m = (p > hi) & (p < 1.0)
        q = np.sqrt(-2.0 * np.log(np.where(m, 1.0 - p, 0.5)))
        up = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
        out = np.where(m, up, out)

    out = np.where(p <= 0.0, -np.inf, out)
    out = np.where(p >= 1.0, np.inf, out)

    # One Halley refinement where the value is finite.
    finite = np.isfinite(out)
    x = np.where(finite, out, 0.0)
    err = norm_cdf(x) - np.where(finite, p, 0.5)
    d = norm_pdf(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        step = err / np.where(d > 1e-300, d, 1.0)
        refined = x - step / (1.0 + 0.5 * x * step)
    return np.where(finite, refined, out)
