"""Correctness tests for the pricing engines.

Run with:  python3 -m unittest discover -s tests -v
       or:  python3 tests/test_pricing.py

The tests are mostly *relationships* rather than golden numbers, because that is
what actually catches bugs here: put-call parity, analytic Greeks against finite
differences of the price they claim to differentiate, independent engines against
each other, and published benchmark values where they exist. A golden number
locks in whatever the code did on the day it was written, including its bugs.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optlib import blackscholes as bs
from optlib import convergence as conv
from optlib import lattice, market, montecarlo as mc, strategy
from optlib.implied_vol import implied_vol
from optlib.normal import norm_cdf, norm_pdf, norm_ppf
from optlib.surface import VolSurface, fit_svi_slice, quotes_from_chain, svi_w

# A spread of contracts: at/in/out of the money, short and long dated, low and
# high vol, with and without a dividend yield.
CASES = [
    #  S,    K,     T,     r,     q,    sigma
    (100.0, 100.0, 1.00, 0.05, 0.00, 0.25),
    (100.0, 100.0, 1.00, 0.043, 0.02, 0.25),
    (100.0, 120.0, 0.50, 0.05, 0.01, 0.30),
    (100.0, 80.0, 0.50, 0.05, 0.01, 0.30),
    (100.0, 100.0, 0.02, 0.03, 0.00, 0.45),
    (100.0, 100.0, 3.00, 0.02, 0.03, 0.15),
    (50.0, 55.0, 0.25, 0.06, 0.00, 0.60),
    (2500.0, 2400.0, 0.75, 0.04, 0.015, 0.18),
    (100.0, 100.0, 1.00, 0.00, 0.00, 0.05),
    (100.0, 140.0, 0.10, 0.05, 0.00, 0.20),
]


class TestNormal(unittest.TestCase):
    def test_cdf_matches_erf(self):
        z = np.linspace(-10, 10, 401)
        ref = np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in z])
        self.assertLess(np.abs(norm_cdf(z) - ref).max(), 1e-14)

    def test_cdf_tails_and_symmetry(self):
        z = np.linspace(0.0, 38.0, 200)
        # Symmetry must hold to full precision, since d1/d2 straddle zero.
        self.assertLess(np.abs(norm_cdf(z) + norm_cdf(-z) - 1.0).max(), 1e-15)
        self.assertEqual(float(norm_cdf(-40.0)), 0.0)
        self.assertEqual(float(norm_cdf(40.0)), 1.0)

    def test_pdf_integrates_to_one(self):
        z = np.linspace(-12, 12, 200001)
        self.assertAlmostEqual(float(np.trapezoid(norm_pdf(z), z)), 1.0, places=12)

    def test_ppf_inverts_cdf(self):
        p = np.linspace(1e-10, 1 - 1e-10, 5000)
        self.assertLess(np.abs(norm_cdf(norm_ppf(p)) - p).max(), 1e-14)

    def test_ppf_edges(self):
        self.assertEqual(float(norm_ppf(0.0)), -math.inf)
        self.assertEqual(float(norm_ppf(1.0)), math.inf)
        self.assertAlmostEqual(float(norm_ppf(0.5)), 0.0, places=14)


class TestBlackScholes(unittest.TestCase):
    def test_put_call_parity(self):
        for S, K, T, r, q, sig in CASES:
            with self.subTest(S=S, K=K, T=T):
                self.assertAlmostEqual(bs.parity_gap(S, K, T, r, q, sig), 0.0, places=9)

    def test_price_within_no_arbitrage_bounds(self):
        for S, K, T, r, q, sig in CASES:
            dfq, dfr = math.exp(-q * T), math.exp(-r * T)
            c = float(bs.price(S, K, T, r, q, sig, True))
            p = float(bs.price(S, K, T, r, q, sig, False))
            with self.subTest(S=S, K=K):
                self.assertGreaterEqual(c, max(S * dfq - K * dfr, 0.0) - 1e-10)
                self.assertLessEqual(c, S * dfq + 1e-10)
                self.assertGreaterEqual(p, max(K * dfr - S * dfq, 0.0) - 1e-10)
                self.assertLessEqual(p, K * dfr + 1e-10)

    def test_monotonic_in_vol_and_strike(self):
        for S, K, T, r, q, _ in CASES:
            vols = np.linspace(0.05, 1.5, 40)
            calls = np.asarray(bs.price(S, K, T, r, q, vols, True))
            # Vega > 0, so price is strictly increasing in vol for both types.
            self.assertTrue(np.all(np.diff(calls) > 0), f"call not increasing in vol at K={K}")
            strikes = np.linspace(S * 0.4, S * 2.0, 60)
            c_k = np.asarray(bs.price(S, strikes, T, r, q, 0.3, True))
            p_k = np.asarray(bs.price(S, strikes, T, r, q, 0.3, False))
            self.assertTrue(np.all(np.diff(c_k) < 0), "call not decreasing in strike")
            self.assertTrue(np.all(np.diff(p_k) > 0), "put not increasing in strike")

    def test_first_order_greeks_match_finite_difference(self):
        """Each Greek must differentiate the price it claims to."""
        for S, K, T, r, q, sig in CASES:
            for is_call in (True, False):
                px = lambda s, k, t, rr, qq, v: float(bs.price(s, k, t, rr, qq, v, is_call))
                hS = S * 1e-5
                fd_delta = (px(S + hS, K, T, r, q, sig) - px(S - hS, K, T, r, q, sig)) / (2 * hS)
                fd_vega = (px(S, K, T, r, q, sig + 1e-6) - px(S, K, T, r, q, sig - 1e-6)) / 2e-6
                fd_theta = -(px(S, K, T + 1e-6, r, q, sig) - px(S, K, T - 1e-6, r, q, sig)) / 2e-6
                fd_rho = (px(S, K, T, r + 1e-7, q, sig) - px(S, K, T, r - 1e-7, q, sig)) / 2e-7
                fd_eps = (px(S, K, T, r, q + 1e-7, sig) - px(S, K, T, r, q - 1e-7, sig)) / 2e-7
                fd_dualdelta = (px(S, K + K * 1e-5, T, r, q, sig) - px(S, K - K * 1e-5, T, r, q, sig)) / (2 * K * 1e-5)

                with self.subTest(S=S, K=K, T=T, call=is_call):
                    scale = max(abs(px(S, K, T, r, q, sig)), 1.0)
                    self.assertAlmostEqual(float(bs.delta(S, K, T, r, q, sig, is_call)), fd_delta, delta=1e-5)
                    self.assertAlmostEqual(float(bs.vega(S, K, T, r, q, sig)), fd_vega, delta=1e-4 * scale)
                    self.assertAlmostEqual(float(bs.theta(S, K, T, r, q, sig, is_call)), fd_theta, delta=1e-3 * scale)
                    self.assertAlmostEqual(float(bs.rho(S, K, T, r, q, sig, is_call)), fd_rho, delta=1e-3 * scale)
                    self.assertAlmostEqual(float(bs.epsilon(S, K, T, r, q, sig, is_call)), fd_eps, delta=1e-3 * scale)
                    self.assertAlmostEqual(float(bs.dual_delta(S, K, T, r, q, sig, is_call)), fd_dualdelta, delta=1e-5)

    def test_second_order_greeks_match_finite_difference(self):
        for S, K, T, r, q, sig in CASES:
            hS, hv, hT = S * 2e-4, 1e-5, 1e-5
            dlt = lambda s, v, t: float(bs.delta(s, K, t, r, q, v, True))
            gma = lambda s, v, t: float(bs.gamma(s, K, t, r, q, v))
            vga = lambda s, v: float(bs.vega(s, K, T, r, q, v))

            fd_gamma = (dlt(S + hS, sig, T) - dlt(S - hS, sig, T)) / (2 * hS)
            fd_vanna = (dlt(S, sig + hv, T) - dlt(S, sig - hv, T)) / (2 * hv)
            fd_charm = -(dlt(S, sig, T + hT) - dlt(S, sig, T - hT)) / (2 * hT)
            fd_volga = (vga(S, sig + hv) - vga(S, sig - hv)) / (2 * hv)
            fd_speed = (gma(S + hS, sig, T) - gma(S - hS, sig, T)) / (2 * hS)
            fd_zomma = (gma(S, sig + hv, T) - gma(S, sig - hv, T)) / (2 * hv)
            fd_color = -(gma(S, sig, T + hT) - gma(S, sig, T - hT)) / (2 * hT)

            with self.subTest(S=S, K=K, T=T):
                tol = lambda ref: max(abs(ref) * 2e-3, 1e-7)
                self.assertAlmostEqual(float(bs.gamma(S, K, T, r, q, sig)), fd_gamma, delta=tol(fd_gamma))
                self.assertAlmostEqual(float(bs.vanna(S, K, T, r, q, sig)), fd_vanna, delta=tol(fd_vanna))
                self.assertAlmostEqual(float(bs.charm(S, K, T, r, q, sig, True)), fd_charm, delta=tol(fd_charm))
                self.assertAlmostEqual(float(bs.volga(S, K, T, r, q, sig)), fd_volga, delta=tol(fd_volga))
                self.assertAlmostEqual(float(bs.speed(S, K, T, r, q, sig)), fd_speed, delta=tol(fd_speed))
                self.assertAlmostEqual(float(bs.zomma(S, K, T, r, q, sig)), fd_zomma, delta=tol(fd_zomma))
                self.assertAlmostEqual(float(bs.color(S, K, T, r, q, sig)), fd_color, delta=tol(fd_color))

    def test_dual_gamma_is_a_density(self):
        """d2C/dK2 discounted back is the risk-neutral density; it must integrate to 1."""
        S, T, r, q, sig = 100.0, 0.75, 0.04, 0.01, 0.28
        K = np.linspace(0.05, 900.0, 400001)
        dens = np.asarray(bs.dual_gamma(S, K, T, r, q, sig)) * math.exp(r * T)
        self.assertAlmostEqual(float(np.trapezoid(dens, K)), 1.0, places=5)

    def test_degenerate_inputs(self):
        # Expiry today: discounted intrinsic on the forward.
        self.assertAlmostEqual(float(bs.price(105, 100, 0.0, 0.05, 0.0, 0.2, True)), 5.0, places=12)
        self.assertAlmostEqual(float(bs.price(95, 100, 0.0, 0.05, 0.0, 0.2, False)), 5.0, places=12)
        self.assertAlmostEqual(float(bs.price(95, 100, 0.0, 0.05, 0.0, 0.2, True)), 0.0, places=12)
        # Zero vol: deterministic forward.
        fwd = 100 * math.exp(0.05 * 1.0)
        self.assertAlmostEqual(float(bs.price(100, 90, 1.0, 0.05, 0.0, 0.0, True)),
                               (fwd - 90) * math.exp(-0.05), places=9)
        # All Greeks stay finite for a degenerate contract.
        for name in bs.GREEK_REGISTRY:
            v = float(np.asarray(bs.evaluate(name, 100, 100, 0.0, 0.05, 0.0, 0.2, True)).reshape(()))
            self.assertTrue(math.isfinite(v), f"{name} not finite at T=0")

    def test_vectorization_broadcasts(self):
        S = np.linspace(80, 120, 7)[:, None]
        T = np.array([0.1, 0.5, 1.0, 2.0])
        out = bs.evaluate("vanna", S, 100.0, T, 0.04, 0.01, 0.25, True)
        self.assertEqual(np.asarray(out).shape, (7, 4))
        # Vectorized must equal the scalar path elementwise.
        for i, s in enumerate(S[:, 0]):
            for j, t in enumerate(T):
                self.assertAlmostEqual(
                    float(np.asarray(out)[i, j]),
                    float(np.asarray(bs.vanna(s, 100.0, t, 0.04, 0.01, 0.25)).reshape(())),
                    places=12,
                )


class TestLattice(unittest.TestCase):
    def test_european_converges_to_black_scholes(self):
        for S, K, T, r, q, sig in CASES:
            exact = float(bs.price(S, K, T, r, q, sig, True))
            for method in lattice.BINOMIAL_METHODS:
                got = lattice.binomial(S, K, T, r, q, sig, 2000, True, False, method).price
                with self.subTest(method=method, K=K, T=T):
                    self.assertAlmostEqual(got, exact, delta=max(exact * 2e-3, 1e-4))

    def test_trinomial_converges_to_black_scholes(self):
        for S, K, T, r, q, sig in CASES:
            exact = float(bs.price(S, K, T, r, q, sig, False))
            got = lattice.trinomial(S, K, T, r, q, sig, 1200, False, False).price
            with self.subTest(K=K, T=T):
                self.assertAlmostEqual(got, exact, delta=max(exact * 2e-3, 1e-4))

    def test_lattice_greeks_match_analytic(self):
        S, K, T, r, q, sig = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25
        res = lattice.binomial(S, K, T, r, q, sig, 2000, True, False, "crr")
        self.assertAlmostEqual(res.delta, float(bs.delta(S, K, T, r, q, sig, True)), places=4)
        self.assertAlmostEqual(res.gamma, float(bs.gamma(S, K, T, r, q, sig)), places=4)
        self.assertAlmostEqual(res.theta, float(bs.theta(S, K, T, r, q, sig, True)), places=1)
        self.assertAlmostEqual(res.vega, float(bs.vega(S, K, T, r, q, sig)), places=1)
        self.assertAlmostEqual(res.rho, float(bs.rho(S, K, T, r, q, sig, True)), places=1)

    def test_theta_is_geometry_independent(self):
        """Every scheme must agree on theta, including the drift-shifted trees.

        Jarrow-Rudd and Tian do not satisfy u*d == 1, so reading theta off the
        step-2 centre node gives the wrong answer for them; this test is what
        pins the PDE-based derivation in place.
        """
        S, K, T, r, q, sig = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25
        ref = float(bs.theta(S, K, T, r, q, sig, True))
        for method in lattice.BINOMIAL_METHODS:
            got = lattice.binomial(S, K, T, r, q, sig, 1500, True, False, method).theta
            with self.subTest(method=method):
                self.assertAlmostEqual(got, ref, delta=0.05)

    def test_american_matches_published_benchmark(self):
        """S=K=100, T=1, r=5%, q=0, sigma=20% American put is 6.0903 in the literature."""
        self.assertAlmostEqual(lattice.american_reference(100, 100, 1.0, 0.05, 0.0, 0.20, False),
                               6.0903, places=3)

    def test_american_call_without_dividend_equals_european(self):
        """With q = 0 it is never optimal to exercise an American call early."""
        for S, K, T, r, sig in [(100, 100, 1.0, 0.05, 0.25), (100, 80, 0.5, 0.03, 0.4)]:
            euro = float(bs.price(S, K, T, r, 0.0, sig, True))
            amer = lattice.binomial(S, K, T, r, 0.0, sig, 1500, True, True, "crr").price
            with self.subTest(K=K):
                self.assertAlmostEqual(amer, euro, delta=max(euro * 2e-3, 1e-4))

    def test_american_at_least_european(self):
        """Compared on the same lattice, not against the closed form.

        The early-exercise premium can be far smaller than the lattice's own
        O(1/n) discretization error -- for a call with a small dividend yield it
        is a fraction of a basis point -- so testing the American lattice price
        against the *exact* European price measures discretization noise, not the
        property of interest. Backward induction with a max() at every node
        cannot produce less than the same induction without it, and that is an
        exact invariant at any step count.
        """
        for S, K, T, r, q, sig in CASES:
            for is_call in (True, False):
                euro = lattice.binomial(S, K, T, r, q, sig, 400, is_call, False, "crr").price
                amer = lattice.binomial(S, K, T, r, q, sig, 400, is_call, True, "crr").price
                with self.subTest(K=K, call=is_call):
                    self.assertGreaterEqual(amer, euro - 1e-12)
                    # And the converged American value beats the exact European
                    # one, to within the reference lattice's own residual error.
                    # That slack has to scale with the price: on a 2500-level
                    # underlying a 1e-6 absolute tolerance is 1e-8 relative,
                    # which is past what any lattice delivers.
                    exact_euro = float(bs.price(S, K, T, r, q, sig, is_call))
                    conv_amer = lattice.american_reference(S, K, T, r, q, sig, is_call)
                    self.assertGreaterEqual(conv_amer, exact_euro - max(exact_euro * 1e-7, 1e-6))

    def test_engines_agree_on_american(self):
        S, K, T, r, q, sig = 100.0, 105.0, 1.0, 0.06, 0.01, 0.30
        ref = lattice.american_reference(S, K, T, r, q, sig, False)
        crr = lattice.binomial(S, K, T, r, q, sig, 3000, False, True, "crr").price
        tri = lattice.trinomial(S, K, T, r, q, sig, 1500, False, True).price
        lsm = mc.longstaff_schwartz(S, K, T, r, q, sig, 120_000, 60, False, seed=7).price
        self.assertAlmostEqual(crr, ref, delta=0.005)
        self.assertAlmostEqual(tri, ref, delta=0.02)
        # LSMC is low-biased by construction: the exercise rule is estimated from
        # the same paths it is applied to. Assert the bias has the right sign.
        self.assertLess(lsm, ref + 0.05)
        self.assertGreater(lsm, ref - 0.15)

    def test_exercise_boundary_is_monotone_for_a_put(self):
        res = lattice.binomial(100, 100, 1.0, 0.05, 0.0, 0.25, 600, False, True, "crr", want_boundary=True)
        self.assertGreater(len(res.boundary), 50)
        times = [t for t, _ in res.boundary]
        spots = [s for _, s in res.boundary]
        self.assertEqual(times, sorted(times))
        # The critical spot for an American put rises toward the strike as expiry
        # approaches, and reaches it at expiry. It never exceeds the strike.
        self.assertLess(spots[0], spots[-1])
        self.assertLessEqual(max(spots), 100.0 + 1e-9)
        self.assertGreater(spots[-1], 95.0)

    def test_exercise_boundary_for_a_call_falls_toward_the_strike(self):
        """A dividend-paying American call exercises when spot is HIGH.

        Its boundary must therefore start above the strike and fall toward it,
        the mirror image of the put -- which is what pins down which edge of the
        exercise region the boundary is read from.
        """
        res = lattice.binomial(100, 100, 1.0, 0.02, 0.08, 0.25, 600, True, True,
                               "crr", want_boundary=True)
        spots = [s for _, s in res.boundary]
        self.assertGreater(len(spots), 20)
        self.assertGreater(spots[0], spots[-1])
        self.assertGreaterEqual(min(spots), 100.0 - 1e-9)

    def test_leisen_reimer_forces_odd_steps(self):
        for n in (100, 101, 500):
            res = lattice.binomial(100, 100, 1.0, 0.05, 0.0, 0.25, n, True, False, "leisen_reimer")
            self.assertEqual(res.steps % 2, 1)

    def test_smoothing_and_richardson_improve_accuracy(self):
        S, K, T, r, q, sig = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25
        exact = float(bs.price(S, K, T, r, q, sig, True))
        plain = abs(lattice.binomial(S, K, T, r, q, sig, 200, True, False, "crr").price - exact)
        rich = abs(lattice.binomial(S, K, T, r, q, sig, 200, True, False, "crr",
                                    smoothing=True, richardson=True).price - exact)
        self.assertLess(rich, plain / 10.0)


class TestMonteCarlo(unittest.TestCase):
    def test_european_within_confidence_interval(self):
        for S, K, T, r, q, sig in CASES:
            exact = float(bs.price(S, K, T, r, q, sig, True))
            res = mc.european(S, K, T, r, q, sig, 200_000, True, seed=11, want_greeks=False)
            with self.subTest(K=K, T=T):
                if exact < 1e-4:
                    # The payoff is essentially never in the money, so no finite
                    # sample resolves it and the standard error is legitimately
                    # zero. The right assertion is that MC also returns ~nothing.
                    self.assertLess(res.price, 1e-3)
                    continue
                # Four standard errors is a ~1-in-16000 false-failure rate per case.
                self.assertLess(abs(res.price - exact), max(4.0 * res.std_error, 1e-9))

    def test_variance_reduction_actually_reduces_variance(self):
        S, K, T, r, q, sig = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25
        plain = mc.european(S, K, T, r, q, sig, 200_000, True, antithetic=False,
                            control_variate=False, seed=3, want_greeks=False)
        both = mc.european(S, K, T, r, q, sig, 200_000, True, antithetic=True,
                           control_variate=True, seed=3, want_greeks=False)
        qmc = mc.european(S, K, T, r, q, sig, 200_000, True, antithetic=True,
                          control_variate=True, sampler="halton", seed=3, want_greeks=False)
        self.assertLess(both.std_error, plain.std_error)
        self.assertGreater(both.variance_reduction, 3.0)
        # Randomized QMC should beat pseudorandom by a wide margin on a 1-D payoff.
        self.assertLess(qmc.std_error, both.std_error)

    def test_pathwise_greeks(self):
        S, K, T, r, q, sig = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25
        for is_call in (True, False):
            res = mc.european(S, K, T, r, q, sig, 400_000, is_call, seed=5)
            with self.subTest(call=is_call):
                self.assertAlmostEqual(res.delta, float(bs.delta(S, K, T, r, q, sig, is_call)),
                                       delta=max(4 * res.delta_se, 1e-6))
                self.assertAlmostEqual(res.vega, float(bs.vega(S, K, T, r, q, sig)),
                                       delta=max(4 * res.vega_se, 1e-6))
                # Likelihood-ratio gamma is high variance; a loose band is honest.
                self.assertAlmostEqual(res.gamma, float(bs.gamma(S, K, T, r, q, sig)), delta=0.002)
                self.assertAlmostEqual(res.theta, float(bs.theta(S, K, T, r, q, sig, is_call)), delta=0.1)
                self.assertAlmostEqual(res.rho, float(bs.rho(S, K, T, r, q, sig, is_call)), delta=0.2)

    def test_halton_sampler_properties(self):
        z = mc.normal_draws(4096, 3, "halton", seed=1)
        self.assertEqual(z.shape, (4096, 3))
        self.assertTrue(np.all(np.isfinite(z)))
        # A low-discrepancy set should match the standard normal moments tightly.
        self.assertLess(abs(z.mean()), 0.02)
        self.assertLess(abs(z.std() - 1.0), 0.02)

    def test_seed_determinism(self):
        a = mc.european(100, 100, 1, 0.05, 0, 0.25, 20_000, seed=42, want_greeks=False).price
        b = mc.european(100, 100, 1, 0.05, 0, 0.25, 20_000, seed=42, want_greeks=False).price
        self.assertEqual(a, b)


class TestImpliedVol(unittest.TestCase):
    def test_roundtrip_recovers_input_vol(self):
        for S, K, T, r, q, sig in CASES:
            for is_call in (True, False):
                px = float(bs.price(S, K, T, r, q, sig, is_call))
                if px < 1e-12:
                    continue
                res = implied_vol(px, S, K, T, r, q, is_call)
                with self.subTest(K=K, T=T, call=is_call, sigma=sig):
                    self.assertTrue(res.converged, res.reason)
                    self.assertAlmostEqual(res.vol, sig, places=8)

    def test_extreme_vols_roundtrip(self):
        for sig in (0.01, 0.05, 0.5, 1.5, 3.0, 5.0):
            px = float(bs.price(100, 100, 1.0, 0.03, 0.0, sig, True))
            res = implied_vol(px, 100, 100, 1.0, 0.03, 0.0, True)
            with self.subTest(sigma=sig):
                self.assertAlmostEqual(res.vol, sig, places=6)

    def test_rejects_arbitrage_violating_prices(self):
        # Below intrinsic.
        r = implied_vol(0.001, 150, 100, 1.0, 0.05, 0.0, True)
        self.assertFalse(r.converged)
        self.assertIn("intrinsic", r.reason)
        # Above the no-arbitrage cap.
        r = implied_vol(500.0, 100, 100, 1.0, 0.05, 0.0, True)
        self.assertFalse(r.converged)
        # Nonsense inputs return NaN rather than raising.
        self.assertTrue(math.isnan(implied_vol(-1.0, 100, 100, 1.0, 0.0, 0.0, True).vol))
        self.assertTrue(math.isnan(implied_vol(5.0, 100, 100, 0.0, 0.0, 0.0, True).vol))

    def test_array_helper(self):
        strikes = np.array([80.0, 100.0, 120.0])
        px = bs.price(100.0, strikes, 0.5, 0.04, 0.0, 0.3, True)
        from optlib.implied_vol import implied_vol_array
        got = implied_vol_array(px, 100.0, strikes, 0.5, 0.04, 0.0, True)
        self.assertTrue(np.allclose(got, 0.3, atol=1e-8))


class TestSurface(unittest.TestCase):
    def test_svi_slice_recovers_its_own_parameters(self):
        from optlib.surface import SVIParams
        truth = SVIParams(a=0.02, b=0.15, rho=-0.55, m=0.05, sigma=0.20)
        k = np.linspace(-0.45, 0.45, 21)
        w = svi_w(k, truth)
        fit = fit_svi_slice(k, w, T=1.0)
        # Parameters are only weakly identified individually; the curve is what
        # must match, so that is what is asserted.
        self.assertLess(float(np.abs(svi_w(k, fit) - w).max()), 2e-4)

    def test_ssvi_generated_chain_is_admissible(self):
        p = market.SSVIParams()
        adm = p.admissibility()
        self.assertTrue(adm["butterfly_ok"], adm)
        self.assertTrue(adm["calendar_ok"], adm)

    def test_demo_chain_inverts_cleanly(self):
        chain = market.generate_demo_chain(n_strikes=13)
        rows = [r for e in chain["expiries"] for r in e["rows"]]
        self.assertGreater(len(rows), 100)
        solved = [r for r in rows if r["iv_solved"] is not None]
        # Every quote is generated from a real Black-Scholes price, so inversion
        # should never fail outright.
        self.assertEqual(len(solved), len(rows))
        err = np.array([r["iv_solved"] - r["iv_truth"] for r in solved])
        # Tick rounding is the only error source; a few tenths of a vol point.
        self.assertLess(float(np.sqrt(np.mean(err ** 2))), 0.02)

    def test_all_interpolators_fit_and_score(self):
        chain = market.generate_demo_chain(n_strikes=15)
        selected = market.chain_liquidity_filter(chain)
        quotes = quotes_from_chain(selected, chain["spot"], chain["r"], chain["q"])
        self.assertGreater(len(quotes), 40)
        for method in ("svi", "ssvi", "cubic", "rbf"):
            with self.subTest(method=method):
                surf = VolSurface(quotes, method)
                fq = surf.fit_quality(with_holdout=False)
                self.assertLess(fq["rmse_vol_pts"], 3.0)
                # Round-trip through the surface must reproduce a quoted IV.
                q0 = quotes[len(quotes) // 2]
                self.assertAlmostEqual(float(surf.iv(q0.k, q0.T)), q0.iv, delta=0.05)

    def test_ssvi_fit_is_arbitrage_free(self):
        chain = market.generate_demo_chain(n_strikes=15)
        selected = market.chain_liquidity_filter(chain)
        quotes = quotes_from_chain(selected, chain["spot"], chain["r"], chain["q"])
        diag = VolSurface(quotes, "ssvi").diagnostics()
        self.assertTrue(diag["butterfly"]["ok"], diag["butterfly"])
        self.assertTrue(diag["calendar"]["ok"], diag["calendar"])

    def test_ssvi_recovers_the_generating_parameters(self):
        p = market.SSVIParams()
        chain = market.generate_demo_chain(n_strikes=21, noise_vol_pts=0.0, spread_bps=1.0, params=p)
        selected = market.chain_liquidity_filter(chain)
        quotes = quotes_from_chain(selected, chain["spot"], chain["r"], chain["q"])
        fit = VolSurface(quotes, "ssvi").ssvi
        # rho drives the skew and is the well-identified parameter of the three.
        self.assertAlmostEqual(fit.rho, p.rho, delta=0.08)

    def test_density_integrates_to_one(self):
        chain = market.generate_demo_chain(n_strikes=17)
        selected = market.chain_liquidity_filter(chain)
        quotes = quotes_from_chain(selected, chain["spot"], chain["r"], chain["q"])
        surf = VolSurface(quotes, "ssvi")
        k = np.linspace(-6.0, 6.0, 60001)
        for T in (0.25, 0.5, 1.0):
            with self.subTest(T=T):
                mass = float(np.trapezoid(surf.density(k, T), k))
                self.assertAlmostEqual(mass, 1.0, places=3)

    def test_total_variance_is_vectorized_consistently(self):
        chain = market.generate_demo_chain(n_strikes=13)
        quotes = quotes_from_chain(market.chain_liquidity_filter(chain),
                                   chain["spot"], chain["r"], chain["q"])
        for method in ("svi", "ssvi", "cubic", "rbf"):
            surf = VolSurface(quotes, method)
            k = np.linspace(-0.3, 0.3, 11)
            T = np.full(11, 0.4)
            grid = np.asarray(surf.total_variance(k, T))
            with self.subTest(method=method):
                for i in range(11):
                    one = float(np.asarray(surf.total_variance(k[i], 0.4)).reshape(()))
                    self.assertAlmostEqual(grid[i], one, places=10)

    def test_grid_coverage_mask(self):
        chain = market.generate_demo_chain(n_strikes=15)
        quotes = quotes_from_chain(market.chain_liquidity_filter(chain),
                                   chain["spot"], chain["r"], chain["q"])
        g = VolSurface(quotes, "ssvi").grid(24, 18)
        cov = np.array(g["covered"])
        self.assertEqual(cov.shape, (24, 18))
        self.assertTrue(cov.any() and not cov.all(), "mask should be a strict subset")

    def test_liquidity_filter_rejects_wide_spreads(self):
        chain = market.generate_demo_chain(n_strikes=13)
        loose = market.chain_liquidity_filter(chain, max_spread_frac=10.0)
        tight = market.chain_liquidity_filter(chain, max_spread_frac=0.02)
        self.assertGreater(len(loose), len(tight))


class TestStrategy(unittest.TestCase):
    def test_long_call_payoff_and_breakeven(self):
        legs = [strategy.Leg("call", 1.0, 100.0, 0.25)]
        res = strategy.evaluate(legs, 100.0, 0.25, 0.04, 0.0)
        prem = res["entry_cost"]
        self.assertGreater(prem, 0.0)
        self.assertEqual(len(res["breakevens"]), 1)
        self.assertAlmostEqual(res["breakevens"][0], 100.0 + prem, delta=0.3)
        # A long call has unbounded upside, so max profit must be reported as None.
        self.assertIsNone(res["max_profit"])
        self.assertAlmostEqual(res["max_loss"], -prem, delta=0.05)

    def test_iron_condor_is_a_credit_with_bounded_risk(self):
        legs = [strategy.Leg(**l) for l in
                [dict(kind="put", quantity=1, strike=90, sigma=0.25),
                 dict(kind="put", quantity=-1, strike=95, sigma=0.25),
                 dict(kind="call", quantity=-1, strike=105, sigma=0.25),
                 dict(kind="call", quantity=1, strike=110, sigma=0.25)]]
        res = strategy.evaluate(legs, 100.0, 0.25, 0.04, 0.0)
        self.assertLess(res["entry_cost"], 0.0)          # a credit
        self.assertEqual(res["position"], "credit")
        self.assertIsNotNone(res["max_profit"])
        self.assertIsNotNone(res["max_loss"])
        self.assertEqual(len(res["breakevens"]), 2)

    def test_short_strangle_reports_unbounded_loss(self):
        legs = [strategy.Leg("call", -1.0, 110.0, 0.25), strategy.Leg("put", -1.0, 90.0, 0.25)]
        res = strategy.evaluate(legs, 100.0, 0.25, 0.04, 0.0)
        self.assertIsNone(res["max_loss"])

    def test_aggregate_greeks_sum_the_legs(self):
        legs = [strategy.Leg("call", 2.0, 100.0, 0.25), strategy.Leg("put", -1.0, 95.0, 0.30)]
        res = strategy.evaluate(legs, 100.0, 0.5, 0.04, 0.01)
        expected = (2.0 * float(bs.delta(100, 100, 0.5, 0.04, 0.01, 0.25, True))
                    - 1.0 * float(bs.delta(100, 95, 0.5, 0.04, 0.01, 0.30, False)))
        self.assertAlmostEqual(res["net_greeks"]["delta"], expected, places=6)

    def test_covered_call_caps_upside(self):
        legs = [strategy.Leg("underlying", 1.0), strategy.Leg("call", -1.0, 105.0, 0.25)]
        res = strategy.evaluate(legs, 100.0, 0.25, 0.04, 0.0)
        self.assertIsNotNone(res["max_profit"])
        self.assertAlmostEqual(res["net_greeks"]["delta"],
                               1.0 - float(bs.delta(100, 105, 0.25, 0.04, 0.0, 0.25, True)),
                               places=6)

    def test_presets_all_evaluate(self):
        for name, legs in strategy.presets(100.0, 0.25, 0.25).items():
            with self.subTest(preset=name):
                res = strategy.evaluate([strategy.Leg(**l) for l in legs], 100.0, 0.25, 0.04, 0.0)
                self.assertEqual(len(res["spots"]), len(res["curves"][0]["pnl"]))

    def test_invalid_legs_rejected(self):
        with self.assertRaises(ValueError):
            strategy.Leg("swaption", 1.0, 100.0).validate()
        with self.assertRaises(ValueError):
            strategy.Leg("call", 1.0, 0.0).validate()


class TestConvergence(unittest.TestCase):
    def test_measured_orders_match_theory(self):
        args = (100.0, 100.0, 1.0, 0.05, 0.02, 0.25)
        res = conv.lattice_convergence(*args, engines=("crr", "leisen_reimer", "trinomial"),
                                       n_max=600, n_points=18)
        by_key = {s["key"]: s for s in res["series"]}
        # CRR is first order; Leisen-Reimer is second order.
        self.assertAlmostEqual(by_key["crr"]["fit"]["order"], 1.0, delta=0.15)
        self.assertGreater(by_key["leisen_reimer"]["fit"]["order"], 1.6)
        for s in res["series"]:
            self.assertGreater(s["fit"]["r_squared"], 0.5, s["label"])

    def test_mc_order_is_root_n(self):
        args = (100.0, 100.0, 1.0, 0.05, 0.02, 0.25)
        res = conv.mc_convergence(*args, engines=("plain", "qmc"), n_max=131_072, n_points=8)
        by_key = {s["key"]: s for s in res["series"]}
        # The standard-error slope is the stable measurement; the raw error slope
        # is one draw per point and is noisy by nature.
        self.assertAlmostEqual(by_key["plain"]["se_fit"]["order"], 0.5, delta=0.08)
        self.assertGreater(by_key["qmc"]["se_fit"]["order"], 0.7)

    def test_shootout_reaches_target(self):
        res = conv.engine_shootout(100.0, 100.0, 1.0, 0.05, 0.02, 0.25, target_bp=10.0)
        reached = [r for r in res["rows"] if r["reached"]]
        self.assertGreater(len(reached), 4)
        for r in reached:
            self.assertLessEqual(abs(r["error"]), res["tolerance"] * 1.000001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
