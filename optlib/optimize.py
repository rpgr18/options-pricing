"""Small dependency-free optimizers.

SciPy is deliberately not used (the target environment has a broken SciPy/NumPy
ABI pairing), and these two routines are all the calibration needs: a derivative
free simplex search for the low-dimensional outer problems, and projected
gradient descent for the convex inner least-squares problem with constraints.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def nelder_mead(
    f: Callable[[np.ndarray], float],
    x0: Sequence[float],
    step: Sequence[float] | float = 0.1,
    max_iter: int = 2000,
    ftol: float = 1e-12,
    xtol: float = 1e-10,
) -> tuple[np.ndarray, float, int]:
    """Nelder-Mead simplex minimization. Returns (x, f(x), iterations)."""
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    steps = np.full(n, float(step)) if np.isscalar(step) else np.asarray(step, dtype=float)

    # Build the initial simplex by perturbing one coordinate at a time.
    sim = np.empty((n + 1, n))
    sim[0] = x0
    for i in range(n):
        pt = x0.copy()
        pt[i] += steps[i] if steps[i] != 0.0 else 0.05
        sim[i + 1] = pt

    fv = np.array([f(p) for p in sim])
    alpha, gamma_, rho_, sigma_ = 1.0, 2.0, 0.5, 0.5

    it = 0
    for it in range(1, max_iter + 1):
        order = np.argsort(fv)
        sim, fv = sim[order], fv[order]

        if abs(fv[-1] - fv[0]) <= ftol * (abs(fv[0]) + ftol) and np.max(np.abs(sim[1:] - sim[0])) <= xtol:
            break

        centroid = sim[:-1].mean(axis=0)
        xr = centroid + alpha * (centroid - sim[-1])
        fr = f(xr)

        if fr < fv[0]:
            xe = centroid + gamma_ * (xr - centroid)
            fe = f(xe)
            sim[-1], fv[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fv[-2]:
            sim[-1], fv[-1] = xr, fr
        else:
            # Contract, on whichever side of the worst face is better.
            if fr < fv[-1]:
                xc = centroid + rho_ * (xr - centroid)
                fc = f(xc)
                if fc <= fr:
                    sim[-1], fv[-1] = xc, fc
                else:
                    sim[1:] = sim[0] + sigma_ * (sim[1:] - sim[0])
                    fv[1:] = [f(p) for p in sim[1:]]
            else:
                xc = centroid + rho_ * (sim[-1] - centroid)
                fc = f(xc)
                if fc < fv[-1]:
                    sim[-1], fv[-1] = xc, fc
                else:
                    sim[1:] = sim[0] + sigma_ * (sim[1:] - sim[0])
                    fv[1:] = [f(p) for p in sim[1:]]

    best = int(np.argmin(fv))
    return sim[best], float(fv[best]), it


def projected_lsq(
    A: np.ndarray,
    b: np.ndarray,
    project: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray | None = None,
    max_iter: int = 500,
    tol: float = 1e-14,
) -> np.ndarray:
    """Minimize ||Ax - b||^2 over a convex set, given a projection onto it.

    Projected gradient with the exact Lipschitz step 1/L (L = largest eigenvalue
    of A'A). The objective is convex quadratic, so this converges to the
    constrained optimum; starting from the projected unconstrained solution
    means it usually has almost nothing left to do.
    """
    G = A.T @ A
    c = A.T @ b
    L = float(np.linalg.eigvalsh(G).max())
    if not np.isfinite(L) or L <= 0.0:
        return project(np.zeros(A.shape[1]))
    lr = 1.0 / L

    if x0 is None:
        try:
            x0, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            x0 = np.zeros(A.shape[1])
    x0 = np.asarray(x0, dtype=float)

    # The unconstrained optimum is usually already feasible, in which case it is
    # the answer and the iteration is pure waste. This short-circuit matters:
    # the SVI outer search calls this tens of thousands of times.
    x = project(x0)
    if np.max(np.abs(x - x0)) < 1e-12:
        return x

    for _ in range(max_iter):
        x_new = project(x - lr * (G @ x - c))
        if np.max(np.abs(x_new - x)) < tol:
            return x_new
        x = x_new
    return x
