"""Implied volatility surface construction, interpolation and arbitrage checks.

The surface is always built in **total implied variance** w = sigma^2 * T over
**log-moneyness** k = log(K / F(T)). That choice is not cosmetic: the
no-arbitrage conditions, the SVI parameterizations and the Breeden-Litzenberger
density all take their simple form in (k, w) coordinates, and interpolating w
linearly in T is exactly the condition that keeps a surface calendar-arbitrage
free between two slices.

Four interpolators are provided over the same scattered quotes so they can be
compared on identical data:

`svi`
    A five-parameter raw-SVI slice per expiry, calibrated by the Zeliade
    quasi-explicit method: for a fixed (m, sigma) the fit is a *convex*
    least-squares problem in the reduced parameters (a, d, c), so only a
    two-dimensional search is left for the simplex. Fitting all five parameters
    directly is notoriously prone to local minima; reducing it this way is what
    makes the fit repeatable. Slices are fitted independently, so nothing
    couples them -- which is exactly why this method can and does produce
    calendar arbitrage on sparse chains. The diagnostics report it rather than
    hiding it.

`ssvi`
    Surface SVI (Gatheral-Jacquier 2014): one (rho, eta, gamma) triple shared by
    the whole surface on top of a per-expiry ATM total variance. Three global
    shape parameters instead of five per slice, so it cannot contort itself to
    fit noise, and it satisfies explicit sufficient conditions for being
    arbitrage-free. Higher in-sample error than raw SVI, better held-out error.

`cubic`
    A natural cubic spline through the quoted smile in (k, w), linear in k
    outside the quoted range. Interpolates exactly; no arbitrage guarantees.

`rbf`
    A thin-plate-spline radial basis interpolant over scattered (k, T) at once,
    with a ridge term for smoothing. Global and expiry-coupled, unlike the slice
    methods.

All are scored by in-sample RMSE *and* held-out RMSE, because an interpolator
that reproduces its own inputs perfectly (cubic, by construction) tells you
nothing about whether it is sensible between the quotes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .optimize import nelder_mead, projected_lsq

METHODS = ("svi", "ssvi", "cubic", "rbf")
SQRT_2PI = 2.5066282746310002


# --------------------------------------------------------------------------
# Raw SVI
# --------------------------------------------------------------------------

@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    rmse: float = float("nan")
    n_quotes: int = 0
    T: float = float("nan")

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def svi_w(k, p: SVIParams):
    """Total implied variance under raw SVI."""
    k = np.asarray(k, dtype=float)
    x = k - p.m
    return p.a + p.b * (p.rho * x + np.sqrt(x * x + p.sigma * p.sigma))


def svi_dw(k, p: SVIParams):
    k = np.asarray(k, dtype=float)
    x = k - p.m
    return p.b * (p.rho + x / np.sqrt(x * x + p.sigma * p.sigma))


def svi_d2w(k, p: SVIParams):
    k = np.asarray(k, dtype=float)
    x = k - p.m
    return p.b * p.sigma * p.sigma / np.power(x * x + p.sigma * p.sigma, 1.5)


def durrleman_g_from(w, wp, wpp, k):
    """Durrleman's function. g >= 0 everywhere on a slice <=> no butterfly arbitrage."""
    w = np.maximum(np.asarray(w, dtype=float), 1e-12)
    return (1.0 - k * wp / (2.0 * w)) ** 2 - (wp * wp / 4.0) * (1.0 / w + 0.25) + wpp / 2.0


RHO_CAP = 0.95


def fit_svi_slice(k, w, T: float = float("nan"), weights=None, polish: bool = True) -> SVIParams:
    """Calibrate one raw-SVI slice to (log-moneyness, total variance) quotes."""
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    good = np.isfinite(k) & np.isfinite(w) & (w > 0.0)
    k, w = k[good], w[good]
    n = k.size
    if weights is None:
        sw = np.ones(n)
    else:
        sw = np.asarray(weights, dtype=float)[good]
        sw = np.where(np.isfinite(sw) & (sw > 0), sw, 1e-8)

    if n < 5:
        # Not enough quotes to identify five parameters: fall back to a flat
        # slice at the weighted-average variance, which is always arb-free.
        wbar = float(np.average(w, weights=sw)) if n else 0.04
        return SVIParams(a=wbar, b=0.0, rho=0.0, m=0.0, sigma=0.1, rmse=0.0, n_quotes=n, T=T)

    w_max = float(w.max())
    rt = np.sqrt(sw)
    b_scaled = w * rt
    ones = np.ones(n)
    k_span = max(float(k.max() - k.min()), 1e-3)
    # Keep sigma from collapsing to a kink the quotes cannot resolve.
    sig_floor = 0.08 * k_span

    def inner(m: float, sig: float) -> tuple[np.ndarray, float]:
        """Solve the convex reduced problem for (a, d, c) at fixed (m, sigma)."""
        y = (k - m) / sig
        z = np.sqrt(y * y + 1.0)
        A = np.column_stack([ones, y, z]) * rt[:, None]
        four_sig = 4.0 * sig

        def project(v: np.ndarray) -> np.ndarray:
            a_, d_, c_ = float(v[0]), float(v[1]), float(v[2])
            # The Zeliade feasible domain: these are the raw-SVI constraints
            # (b >= 0, |rho| <= 1, w > 0) rewritten in the reduced coordinates
            # c = b*sigma, d = rho*b*sigma.
            c_ = min(max(c_, 0.0), four_sig)
            lim = min(c_ * RHO_CAP, four_sig - c_)
            d_ = min(max(d_, -lim), lim)
            a_ = min(max(a_, 0.0), w_max)
            return np.array([a_, d_, c_])

        v = projected_lsq(A, b_scaled, project, max_iter=120)
        resid = A @ v - b_scaled
        return v, float(np.sqrt(np.mean(resid * resid)))

    def outer(theta: np.ndarray) -> float:
        m = float(theta[0])
        sig = float(np.exp(theta[1]))  # log-parameterized to keep sigma > 0
        if not np.isfinite(m) or not np.isfinite(sig) or sig < sig_floor or sig > 10.0 or abs(m) > 3.0:
            return 1e6
        return inner(m, sig)[1]

    # A short multi-start over the outer 2-D problem removes the residual
    # sensitivity of the simplex to where it is dropped, cheaply.
    best = None
    for m0, s0 in (
        (0.0, max(0.3 * k_span, sig_floor)),
        (float(np.average(k, weights=sw)), max(k_span, 0.1)),
    ):
        theta, val, _ = nelder_mead(
            outer, [m0, np.log(s0)], step=[0.15 * k_span + 1e-3, 0.3], max_iter=160
        )
        if best is None or val < best[1]:
            best = (theta, val)

    m = float(best[0][0])
    sig = max(float(np.exp(best[0][1])), sig_floor)
    v, rmse = inner(m, sig)
    a, d, c = (float(x) for x in v)
    b = c / sig
    rho = (d / c) if c > 1e-12 else 0.0
    params = SVIParams(a=a, b=b, rho=float(np.clip(rho, -RHO_CAP, RHO_CAP)), m=m, sigma=sig, rmse=rmse, n_quotes=n, T=T)

    if polish:
        # A short simplex pass on all five parameters, penalized for leaving the
        # admissible set and for butterfly violations on the quoted range. This
        # buys back the accuracy the reduced-domain clamps cost without
        # reintroducing the local-minimum problem, because it starts from an
        # already-good point.
        base = np.array([params.a, params.b, params.rho, params.m, params.sigma])
        pad = 0.2 * k_span
        k_chk = np.linspace(k.min() - pad, k.max() + pad, 25)

        def full(theta: np.ndarray) -> float:
            a_, b_, rho_, m_, s_ = (float(t) for t in theta)
            if b_ < 0.0 or abs(rho_) > RHO_CAP or s_ < sig_floor or b_ * (1.0 + abs(rho_)) > 4.0:
                return 1e6
            trial = SVIParams(a_, b_, rho_, m_, s_)
            ww = svi_w(k, trial)
            if np.any(ww <= 0.0):
                return 1e6
            resid = (ww - w) * rt
            err = float(np.sqrt(np.mean(resid * resid)))
            g = durrleman_g_from(svi_w(k_chk, trial), svi_dw(k_chk, trial), svi_d2w(k_chk, trial), k_chk)
            gmin = float(np.min(g))
            return err + (0.0 if gmin >= 0.0 else 0.25 * w_max * (-gmin))

        theta, val, _ = nelder_mead(full, base, step=np.maximum(np.abs(base) * 0.08, 1e-4), max_iter=280)
        if val < full(base):
            cand = SVIParams(*(float(t) for t in theta), rmse=val, n_quotes=n, T=T)
            cand.rho = float(np.clip(cand.rho, -RHO_CAP, RHO_CAP))
            resid = (svi_w(k, cand) - w) * rt
            cand.rmse = float(np.sqrt(np.mean(resid * resid)))
            params = cand

    return params


# --------------------------------------------------------------------------
# SSVI
# --------------------------------------------------------------------------

@dataclass
class SSVIFit:
    """Surface SVI: three global shape parameters over an ATM variance curve.

        w(k, T) = theta/2 * {1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2)}
        phi(theta) = eta / (theta^gamma * (1 + theta)^(1 - gamma))

    `theta_nodes` is the ATM total variance at each quoted expiry, forced
    non-decreasing so the calendar condition holds by construction.
    """

    rho: float
    eta: float
    gamma: float
    expiries: np.ndarray
    theta_nodes: np.ndarray
    rmse: float = float("nan")

    def theta(self, T):
        T = np.asarray(T, dtype=float)
        Ts, th = self.expiries, self.theta_nodes
        if Ts.size == 1:
            return np.maximum(th[0] * T / Ts[0], 1e-12)
        out = np.interp(T, Ts, th)
        # Outside the quoted range hold implied vol flat, so theta stays linear
        # in T (and therefore non-decreasing).
        out = np.where(T < Ts[0], th[0] * T / Ts[0], out)
        out = np.where(T > Ts[-1], th[-1] * T / Ts[-1], out)
        return np.maximum(out, 1e-12)

    def phi(self, theta):
        theta = np.maximum(np.asarray(theta, dtype=float), 1e-12)
        return self.eta / (np.power(theta, self.gamma) * np.power(1.0 + theta, 1.0 - self.gamma))

    def total_variance(self, k, T):
        k = np.asarray(k, dtype=float)
        th = self.theta(T)
        x = self.phi(th) * k
        return 0.5 * th * (1.0 + self.rho * x + np.sqrt((x + self.rho) ** 2 + (1.0 - self.rho ** 2)))

    def admissibility(self) -> dict:
        Ts = np.linspace(float(self.expiries.min()) * 0.5, float(self.expiries.max()) * 1.5, 200)
        th = self.theta(Ts)
        ph = self.phi(th)
        c1 = float(np.max(th * ph * (1.0 + abs(self.rho))))
        c2 = float(np.max(th * ph * ph * (1.0 + abs(self.rho))))
        return {
            "butterfly_cond_1": c1,
            "butterfly_cond_2": c2,
            "butterfly_ok": bool(c1 <= 4.0 + 1e-9 and c2 <= 4.0 + 1e-9),
            "min_d_theta": float(np.min(np.diff(th))),
            "min_d_theta_phi": float(np.min(np.diff(th * ph))),
            "note": "Gatheral-Jacquier: theta*phi*(1+|rho|) <= 4 and theta*phi^2*(1+|rho|) <= 4.",
        }

    def as_dict(self) -> dict:
        return {
            "rho": self.rho,
            "eta": self.eta,
            "gamma": self.gamma,
            "rmse": self.rmse,
            "theta_nodes": [{"T": float(t), "theta": float(v), "atm_vol": float(np.sqrt(v / t))} for t, v in zip(self.expiries, self.theta_nodes)],
            "admissibility": self.admissibility(),
        }


def fit_ssvi(slices: list["SliceFit"], weights_by_slice: list[np.ndarray] | None = None) -> SSVIFit:
    """Fit SSVI globally: ATM variance per expiry, then one shared shape triple.

    Two stages, because they are nearly separable. The ATM total variance is read
    off each slice's own spline at k = 0, which is a direct observation rather
    than something to optimize; that leaves only (rho, eta, gamma) for the
    simplex, and three parameters over the whole surface is a search that
    actually converges.
    """
    Ts = np.array([s.T for s in slices], dtype=float)
    theta0 = np.array([max(float(np.asarray(s.spline(0.0)).reshape(-1)[0]), 1e-8) for s in slices])
    # Enforce a non-decreasing ATM variance term structure (the calendar condition).
    theta0 = np.maximum.accumulate(theta0)

    all_k = np.concatenate([s.k for s in slices])
    all_w = np.concatenate([s.w for s in slices])
    all_T = np.concatenate([np.full(s.k.size, s.T) for s in slices])
    if weights_by_slice is None:
        all_wt = np.ones(all_k.size)
    else:
        all_wt = np.concatenate(weights_by_slice)
    rt = np.sqrt(np.where(np.isfinite(all_wt) & (all_wt > 0), all_wt, 1e-8))

    def build(theta_nodes, rho, eta, gam) -> SSVIFit:
        return SSVIFit(rho=rho, eta=eta, gamma=gam, expiries=Ts, theta_nodes=theta_nodes)

    def err_of(theta_nodes, p) -> float:
        rho, eta, gam = p
        if not (-RHO_CAP <= rho <= RHO_CAP) or not (1e-3 < eta < 40.0) or not (0.0 < gam < 1.0):
            return 1e6
        f = build(theta_nodes, rho, eta, gam)
        adm = f.admissibility()
        pred = f.total_variance(all_k, all_T)
        if np.any(~np.isfinite(pred)) or np.any(pred <= 0):
            return 1e6
        resid = (pred - all_w) * rt
        e = float(np.sqrt(np.mean(resid * resid)))
        # Soft-enforce the sufficient no-arbitrage conditions.
        pen = max(adm["butterfly_cond_1"] - 4.0, 0.0) + max(adm["butterfly_cond_2"] - 4.0, 0.0)
        return e + 0.05 * pen

    best = None
    for rho0 in (-0.7, -0.4, -0.1):
        for eta0 in (0.5, 1.2, 2.5):
            theta, val, _ = nelder_mead(
                lambda p: err_of(theta0, p), [rho0, eta0, 0.45], step=[0.12, 0.35, 0.12], max_iter=400
            )
            if best is None or val < best[1]:
                best = (theta, val)

    rho, eta, gam = (float(x) for x in best[0])
    rho = float(np.clip(rho, -RHO_CAP, RHO_CAP))
    eta = float(np.clip(eta, 1e-3, 40.0))
    gam = float(np.clip(gam, 1e-3, 0.999))

    # Refine the ATM curve with the shape held fixed, then re-impose monotonicity.
    def theta_err(scale: np.ndarray) -> float:
        nodes = np.maximum.accumulate(np.maximum(theta0 * np.exp(scale), 1e-10))
        return err_of(nodes, (rho, eta, gam))

    scale, val, _ = nelder_mead(theta_err, np.zeros(Ts.size), step=0.05, max_iter=600)
    if val <= best[1]:
        theta_nodes = np.maximum.accumulate(np.maximum(theta0 * np.exp(scale), 1e-10))
    else:
        theta_nodes = theta0
        val = best[1]

    fit = build(theta_nodes, rho, eta, gam)
    pred = fit.total_variance(all_k, all_T)
    resid = (pred - all_w) * rt
    fit.rmse = float(np.sqrt(np.mean(resid * resid)))
    return fit


# --------------------------------------------------------------------------
# Slice interpolators
# --------------------------------------------------------------------------

def _natural_cubic(x: np.ndarray, y: np.ndarray):
    """Natural cubic spline evaluator with linear extrapolation outside [x0, xn]."""
    order = np.argsort(x)
    x, y = np.asarray(x, dtype=float)[order], np.asarray(y, dtype=float)[order]
    # Collapse duplicate abscissae, which market strike grids can produce.
    keep = np.concatenate([[True], np.diff(x) > 1e-12])
    x, y = x[keep], y[keep]
    n = x.size

    if n == 1:
        return lambda t: np.full(np.shape(np.atleast_1d(t)), y[0], dtype=float)
    if n == 2:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return lambda t: y[0] + slope * (np.atleast_1d(np.asarray(t, dtype=float)) - x[0])

    h = np.diff(x)
    A = np.zeros((n, n))
    rhs = np.zeros(n)
    A[0, 0] = A[-1, -1] = 1.0  # natural: second derivative zero at both ends
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    M = np.linalg.solve(A, rhs)

    sl_lo = (y[1] - y[0]) / h[0] - h[0] * M[1] / 6.0 - h[0] * M[0] / 3.0
    sl_hi = (y[-1] - y[-2]) / h[-1] + h[-1] * M[-2] / 6.0 + h[-1] * M[-1] / 3.0

    def ev(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        idx = np.clip(np.searchsorted(x, t) - 1, 0, n - 2)
        hi_ = h[idx]
        A_ = (x[idx + 1] - t) / hi_
        B_ = (t - x[idx]) / hi_
        out = (
            A_ * y[idx]
            + B_ * y[idx + 1]
            + ((A_ ** 3 - A_) * M[idx] + (B_ ** 3 - B_) * M[idx + 1]) * (hi_ * hi_) / 6.0
        )
        out = np.where(t < x[0], y[0] + sl_lo * (t - x[0]), out)
        out = np.where(t > x[-1], y[-1] + sl_hi * (t - x[-1]), out)
        return out

    return ev


def _thin_plate(points: np.ndarray, values: np.ndarray, smooth: float = 1e-8):
    """Thin-plate-spline interpolant over 2-D scattered points.

    Coordinates are normalized to the unit square first: k spans ~0.5 while T can
    span years, and an un-normalized radial kernel would let whichever axis has
    the larger numeric range dominate the distance metric entirely.
    """
    P = np.asarray(points, dtype=float)
    v = np.asarray(values, dtype=float)
    lo = P.min(axis=0)
    span = np.where(P.max(axis=0) - lo > 1e-12, P.max(axis=0) - lo, 1.0)
    X = (P - lo) / span
    n = X.shape[0]

    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Kmat = np.where(d > 1e-12, d * d * np.log(np.maximum(d, 1e-300)), 0.0)
    Kmat = Kmat + smooth * n * np.eye(n)

    Pm = np.column_stack([np.ones(n), X])
    M = np.zeros((n + 3, n + 3))
    M[:n, :n] = Kmat
    M[:n, n:] = Pm
    M[n:, :n] = Pm.T
    rhs = np.concatenate([v, np.zeros(3)])
    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    wts, poly = sol[:n], sol[n:]

    def ev(pts):
        Q = np.atleast_2d(np.asarray(pts, dtype=float))
        Y = (Q - lo) / span
        dd = np.linalg.norm(Y[:, None, :] - X[None, :, :], axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            phi = np.where(dd > 1e-12, dd * dd * np.log(np.maximum(dd, 1e-300)), 0.0)
        return phi @ wts + poly[0] + Y @ poly[1:]

    return ev


# --------------------------------------------------------------------------
# The surface
# --------------------------------------------------------------------------

@dataclass
class Quote:
    T: float
    k: float          # log-moneyness log(K/F)
    iv: float
    strike: float = float("nan")
    weight: float = 1.0


@dataclass
class SliceFit:
    T: float
    k: np.ndarray
    w: np.ndarray
    strikes: np.ndarray
    weights: np.ndarray
    params: SVIParams | None = None
    spline: object | None = None

    def eval_w(self, kq, method: str) -> np.ndarray:
        kq = np.atleast_1d(np.asarray(kq, dtype=float))
        if method == "svi" and self.params is not None:
            return np.asarray(svi_w(kq, self.params), dtype=float)
        return np.asarray(self.spline(kq), dtype=float)


class VolSurface:
    """A fitted implied-volatility surface with arbitrage diagnostics."""

    def __init__(self, quotes: list[Quote], method: str = "svi", rbf_smooth: float = 1e-6):
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        self.method = method
        self.quotes = [q for q in quotes if np.isfinite(q.iv) and q.iv > 0 and q.T > 0 and np.isfinite(q.k)]
        if not self.quotes:
            raise ValueError("no usable quotes: need finite positive iv and T")

        by_T: dict[float, list[Quote]] = {}
        for q in self.quotes:
            by_T.setdefault(round(q.T, 10), []).append(q)

        self.slices: list[SliceFit] = []
        for T in sorted(by_T):
            qs = sorted(by_T[T], key=lambda x: x.k)
            k = np.array([x.k for x in qs])
            iv = np.array([x.iv for x in qs])
            wt = np.array([x.weight for x in qs])
            sl = SliceFit(T=T, k=k, w=iv * iv * T, strikes=np.array([x.strike for x in qs]), weights=wt)
            sl.spline = _natural_cubic(k, sl.w)
            if method == "svi":
                sl.params = fit_svi_slice(k, sl.w, T, wt)
            self.slices.append(sl)

        self.expiries = np.array([s.T for s in self.slices])

        self.ssvi: SSVIFit | None = None
        self._rbf = None
        if method == "ssvi":
            self.ssvi = fit_ssvi(self.slices, [s.weights for s in self.slices])
        elif method == "rbf":
            pts = np.array([[q.k, q.T] for q in self.quotes])
            vals = np.array([q.iv * q.iv * q.T for q in self.quotes])
            self._rbf = _thin_plate(pts, vals, rbf_smooth)

    # ---- evaluation -----------------------------------------------------

    def total_variance(self, k, T):
        """Total implied variance w(k, T), broadcast over k and T."""
        k, T = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
        shape = k.shape
        kf, Tf = k.reshape(-1), T.reshape(-1)

        if self.method == "ssvi":
            out = np.asarray(self.ssvi.total_variance(kf, Tf), dtype=float)
        elif self.method == "rbf":
            out = np.asarray(self._rbf(np.column_stack([kf, Tf])), dtype=float)
        else:
            Ts = self.expiries
            # Evaluate every slice at every k once, then interpolate along T.
            # Vectorizing this matters: the 3-D view asks for thousands of points.
            W = np.stack([sl.eval_w(kf, self.method) for sl in self.slices], axis=0)  # (n_slices, n_pts)
            if Ts.size == 1:
                out = W[0] * Tf / Ts[0]
            else:
                j = np.clip(np.searchsorted(Ts, Tf), 1, Ts.size - 1)
                lo_T, hi_T = Ts[j - 1], Ts[j]
                cols = np.arange(kf.size)
                w_lo, w_hi = W[j - 1, cols], W[j, cols]
                lam = (Tf - lo_T) / (hi_T - lo_T)
                # Linear in T on total variance: the calendar-arbitrage-free rule.
                out = (1.0 - lam) * w_lo + lam * w_hi
                # Outside the quoted range, hold implied vol flat.
                front = Tf < Ts[0]
                back = Tf > Ts[-1]
                out = np.where(front, W[0, cols] * Tf / Ts[0], out)
                out = np.where(back, W[-1, cols] * Tf / Ts[-1], out)

        return np.maximum(out, 1e-10).reshape(shape)

    def iv(self, k, T):
        """Implied volatility at log-moneyness k and expiry T."""
        T = np.asarray(T, dtype=float)
        return np.sqrt(self.total_variance(k, T) / np.maximum(T, 1e-12))

    def iv_at_strike(self, strike, T, forward):
        return self.iv(np.log(np.asarray(strike, dtype=float) / np.asarray(forward, dtype=float)), T)

    # ---- diagnostics ----------------------------------------------------

    def durrleman_g(self, k, T):
        """Durrleman's function g(k) at fixed T, by central difference on w."""
        k = np.atleast_1d(np.asarray(k, dtype=float))
        h = 1e-4
        w = self.total_variance(k, T)
        w_up = self.total_variance(k + h, T)
        w_dn = self.total_variance(k - h, T)
        wp = (w_up - w_dn) / (2.0 * h)
        wpp = (w_up - 2.0 * w + w_dn) / (h * h)
        return durrleman_g_from(w, wp, wpp, k)

    def density(self, k, T):
        """Risk-neutral density in log-moneyness (Gatheral's form, via g)."""
        k = np.atleast_1d(np.asarray(k, dtype=float))
        w = self.total_variance(k, T)
        g = self.durrleman_g(k, T)
        d_minus = -k / np.sqrt(w) - 0.5 * np.sqrt(w)
        return g / (SQRT_2PI * np.sqrt(w)) * np.exp(-0.5 * d_minus * d_minus)

    def k_range(self, pad_frac: float = 0.15) -> tuple[float, float]:
        all_k = np.array([q.k for q in self.quotes])
        pad = pad_frac * max(float(np.ptp(all_k)), 0.1)
        return float(all_k.min() - pad), float(all_k.max() + pad)

    def diagnostics(self, n_k: int = 121, n_T: int = 40, with_holdout: bool = False) -> dict:
        """Butterfly and calendar arbitrage scan plus fit-quality statistics."""
        k_lo, k_hi = self.k_range()
        k_grid = np.linspace(k_lo, k_hi, n_k)

        per_slice = []
        worst_g = np.inf
        for sl in self.slices:
            g = self.durrleman_g(k_grid, sl.T)
            gmin = float(np.nanmin(g))
            worst_g = min(worst_g, gmin)
            atm_w = float(self.total_variance(np.array([0.0]), sl.T)[0])
            entry = {
                "T": sl.T,
                "min_g": gmin,
                "k_at_min_g": float(k_grid[int(np.nanargmin(g))]),
                "butterfly_ok": bool(gmin >= -1e-6),
                "n_quotes": int(sl.k.size),
                "atm_vol": float(np.sqrt(atm_w / sl.T)),
            }
            if sl.params is not None and self.method == "svi":
                entry["svi"] = sl.params.as_dict()
                entry["svi_b_bound"] = float(sl.params.b * (1.0 + abs(sl.params.rho)))
            per_slice.append(entry)

        # Calendar: total variance must be non-decreasing in T at every k.
        Ts = np.linspace(float(self.expiries.min()) * 0.5, float(self.expiries.max()) * 1.2, n_T)
        min_dwdt, cal_at = np.inf, (float("nan"), float("nan"))
        for kk in k_grid[::4]:
            w = self.total_variance(np.full(Ts.shape, kk), Ts)
            dd = np.diff(w) / np.diff(Ts)
            j = int(np.argmin(dd))
            if dd[j] < min_dwdt:
                min_dwdt = float(dd[j])
                cal_at = (float(kk), float(0.5 * (Ts[j] + Ts[j + 1])))

        out = {
            "method": self.method,
            "n_quotes": len(self.quotes),
            "n_slices": len(self.slices),
            "butterfly": {
                "min_g": float(worst_g),
                "ok": bool(worst_g >= -1e-6),
                "note": "Durrleman g(k) >= 0 on every slice means no butterfly arbitrage.",
            },
            "calendar": {
                "min_dw_dT": float(min_dwdt),
                "ok": bool(min_dwdt >= -1e-9),
                "at_k": cal_at[0],
                "at_T": cal_at[1],
                "note": "Total variance must be non-decreasing in T at fixed log-moneyness.",
            },
            "slices": per_slice,
            "fit": self.fit_quality(with_holdout=with_holdout),
        }
        if self.ssvi is not None:
            out["ssvi"] = self.ssvi.as_dict()
        return out

    def fit_quality(self, with_holdout: bool = True) -> dict:
        """In-sample and held-out error, in volatility points."""
        k = np.array([q.k for q in self.quotes])
        T = np.array([q.T for q in self.quotes])
        iv = np.array([q.iv for q in self.quotes])
        err = np.asarray(self.iv(k, T)) - iv
        out = {
            "rmse_vol_pts": float(np.sqrt(np.mean(err * err)) * 100.0),
            "max_abs_vol_pts": float(np.max(np.abs(err)) * 100.0),
            "mean_bias_vol_pts": float(np.mean(err) * 100.0),
        }
        if with_holdout:
            rmse, n = self._holdout_rmse()
            out["holdout_rmse_vol_pts"] = rmse
            out["holdout_n"] = n
        return out

    def _holdout_rmse(self, folds: int = 4) -> tuple[float, int]:
        """K-fold error, refitting per fold.

        Quotes are held out slice-wise on interleaved strikes, never at a slice
        endpoint (that would score extrapolation, not interpolation). Cubic
        interpolation reproduces its inputs exactly, so in-sample error cannot
        distinguish the interpolators at all -- this is the number that can.
        """
        min_needed = 7 if self.method in ("svi", "ssvi") else 4
        by_T: dict[float, list[Quote]] = {}
        for q in self.quotes:
            by_T.setdefault(round(q.T, 10), []).append(q)
        for v in by_T.values():
            v.sort(key=lambda x: x.k)

        errs: list[float] = []
        for f in range(folds):
            train: list[Quote] = []
            test: list[Quote] = []
            for qs in by_T.values():
                if len(qs) < min_needed + 2:
                    train.extend(qs)
                    continue
                for i, q in enumerate(qs):
                    if i % folds == f and 0 < i < len(qs) - 1:
                        test.append(q)
                    else:
                        train.append(q)
            if not test:
                continue
            try:
                sub = VolSurface(train, self.method)
                pred = np.asarray(sub.iv(np.array([q.k for q in test]), np.array([q.T for q in test])))
            except (ValueError, np.linalg.LinAlgError, IndexError):
                continue
            ivt = np.array([q.iv for q in test])
            errs.extend(((pred - ivt) * 100.0).tolist())

        if not errs:
            return float("nan"), 0
        e = np.asarray(errs)
        return float(np.sqrt(np.mean(e * e))), int(e.size)

    # ---- gridding for the 3-D view --------------------------------------

    def grid(self, n_k: int = 48, n_T: int = 36, T_max_mult: float = 1.0) -> dict:
        k_lo, k_hi = self.k_range(0.1)
        ks = np.linspace(k_lo, k_hi, n_k)
        Ts = np.linspace(
            max(float(self.expiries.min()) * 0.6, 1.0 / 365.0),
            float(self.expiries.max()) * T_max_mult,
            n_T,
        )
        KK, TT = np.meshgrid(ks, Ts, indexing="ij")
        IV = np.asarray(self.iv(KK, TT))
        G = np.stack([self.durrleman_g(ks, t) for t in Ts], axis=1)

        # Coverage mask: is this (k, T) actually inside the quoted region, or is
        # the surface extrapolating? Quoted strike ranges widen with maturity, so
        # the covered region is a cone, not a rectangle -- and a 4-day option at
        # 60% out of the money is pure extrapolation that would otherwise
        # dominate the colour scale and the eye.
        k_lo_by_T = np.array([s.k.min() for s in self.slices])
        k_hi_by_T = np.array([s.k.max() for s in self.slices])
        lo_i = np.interp(Ts, self.expiries, k_lo_by_T)
        hi_i = np.interp(Ts, self.expiries, k_hi_by_T)
        covered = (
            (KK >= lo_i[None, :])
            & (KK <= hi_i[None, :])
            & (TT >= self.expiries.min())
            & (TT <= self.expiries.max())
        )

        return {
            "k": ks.tolist(),
            "T": Ts.tolist(),
            "iv": IV.tolist(),
            "g": G.tolist(),
            "covered": covered.tolist(),
            "quotes": [{"k": q.k, "T": q.T, "iv": q.iv, "strike": q.strike} for q in self.quotes],
        }

    def smile(self, T: float, n: int = 121) -> dict:
        """One expiry slice, with the density and butterfly diagnostic alongside."""
        k_lo, k_hi = self.k_range(0.2)
        ks = np.linspace(k_lo, k_hi, n)
        return {
            "T": float(T),
            "k": ks.tolist(),
            "iv": np.asarray(self.iv(ks, T)).tolist(),
            "w": np.asarray(self.total_variance(ks, T)).tolist(),
            "g": np.asarray(self.durrleman_g(ks, T)).tolist(),
            "density": np.asarray(self.density(ks, T)).tolist(),
        }


def quotes_from_chain(rows: list[dict], spot: float, r: float, q: float) -> list[Quote]:
    """Build surface quotes from chain rows carrying (T, strike, iv[, weight])."""
    out: list[Quote] = []
    for row in rows:
        T = float(row["T"])
        strike = float(row["strike"])
        iv = float(row.get("iv", float("nan")))
        if not (np.isfinite(iv) and iv > 0.0 and T > 0.0 and strike > 0.0):
            continue
        fwd = spot * np.exp((r - q) * T)
        out.append(
            Quote(
                T=T,
                k=float(np.log(strike / fwd)),
                iv=iv,
                strike=strike,
                weight=float(row.get("weight", 1.0)),
            )
        )
    return out
