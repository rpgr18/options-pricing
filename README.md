# Options Pricing & Greeks Workbench

An interactive desktop workbench for equity option pricing: seven pricing engines
side by side, a full analytic Greek set as rotatable 3-D surfaces, four implied
volatility surface interpolators scored against each other, and a convergence lab
that *measures* error rates rather than quoting them.

```bash
python3 run.py
```

That is the whole setup. NumPy is the only dependency, there is no build step, and
the server is Python's own `http.server` bound to loopback.

---

## Why this exists

The Black-Scholes formula is twenty lines. The interesting questions are the ones
that come after it:

- **When does each numerical method actually pay for itself?** Cox-Ross-Rubinstein
  needs 4096 steps to price a one-year at-the-money call to within a basis point.
  The same tree with payoff smoothing and Richardson extrapolation needs **32**;
  Leisen-Reimer needs 33 with no refinements at all. Plain Monte Carlo does not get
  there inside a million paths — randomized-Halton QMC does it in 65k. The
  convergence lab measures all of it, fits the observed order, and reports the wall
  clock cost of hitting a target accuracy.
- **What breaks when you fit a volatility surface to real quotes?** Raw SVI fits
  each expiry beautifully and independently — and on a sparse chain that
  independence produces *calendar arbitrage*, which this app detects and shows you
  rather than quietly smoothing over. SSVI, with three global shape parameters
  instead of five per slice, fits each smile worse and generalizes better.
- **How much of a "good fit" is just fitting noise?** A cubic spline through the
  quoted smile scores an in-sample RMSE of 0.01 volatility points and a held-out
  RMSE of 1.01 — the worst of the four interpolators. In-sample error is not a
  measure of quality, and the app reports both so you can see that.

## Screenshot tour

Six tabs, all driven from one shared contract panel in the left rail:

| Tab | What it does |
|---|---|
| **Pricer** | Hero premium, all 15 analytic Greeks in trading units, and every engine side by side with error in basis points, 95% intervals, variance-reduction efficiency and wall time. |
| **Greeks Lab** | Any Greek as a rotatable 3-D surface or heatmap over (strike × tenor), plus term slices you can read numbers off. |
| **Vol Surface** | Calibrate SVI / SSVI / cubic / RBF to a chain. Butterfly and calendar arbitrage scans, the Breeden-Litzenberger risk-neutral density, per-slice SVI parameters, and a held-out-error shootout. |
| **Convergence** | Log-log error against discretization size for every engine, fitted convergence orders with R², and cost-to-target-accuracy. |
| **Option Chain** | Load a synthetic SSVI chain (with known ground truth) or fetch a live chain. Every mid price is inverted to implied vol and compared against the feed's own number. Click a row to load it into the pricer. |
| **Strategy** | Multi-leg payoff and mark-to-market P&L at intermediate dates, breakevens, and aggregate position Greeks. |

---

## The engines

### Analytic — `optlib/blackscholes.py`

Generalized Black-Scholes-Merton with a continuous dividend yield `q`. Setting
`q = r` gives Black-76 on a forward; `q = r_foreign` gives Garman-Kohlhagen for FX.

Fifteen Greeks, all vectorized over any broadcastable combination of inputs so a
whole (strike × tenor) grid evaluates in one call:

| Order | Greeks |
|---|---|
| First | delta, vega, theta, rho, epsilon (dividend rho), dual delta |
| Second | gamma, vanna, volga, charm, zomma, dual gamma |
| Third | speed, color |

Every one is verified against a finite difference of the price it claims to
differentiate. Greeks are reported both raw (per unit) and in trading conventions
(vega per volatility point, theta per calendar day, rho per 1% of rate), so display
code never has to guess a scale factor.

The normal CDF uses Hart's rational/continued-fraction pair rather than a cheap
Abramowitz-Stegun approximation. That matters: implied-vol inversion differentiates
the CDF, so a 1e-7 approximation would put a visible floor on achievable IV
accuracy and pollute the convergence study. As implemented it is accurate to ~1e-16.

### Lattices — `optlib/lattice.py`

Four binomial parameterizations plus a trinomial, European or American:

- **Cox-Ross-Rubinstein** — `u = exp(σ√dt)`, the standard.
- **Jarrow-Rudd** — equal probability; drift carried by node spacing.
- **Tian** — matches the first three moments exactly.
- **Leisen-Reimer** — Peizer-Pratt inversion, tuned so terminal nodes straddle the
  strike. Second-order convergent, and the fastest route to a given accuracy of
  the plain schemes.
- **Boyle trinomial** — three branches with the `√(3/2)` optimal stretch.

Two accuracy refinements are available on every binomial engine, because they are
what makes the convergence chart interesting:

- **Payoff smoothing** replaces the penultimate time slice with the closed-form
  European value over the last step. The sawtooth in binomial convergence comes
  from the payoff kink falling between nodes; integrating the last step
  analytically removes it and turns oscillating O(1/n) into smooth O(1/n).
- **Richardson extrapolation** — `2·V(n) − V(n/2)`, valid once the error is a clean
  O(1/n) series, so it is paired with smoothing.

Delta and gamma are read off the first two time slices. **Theta is obtained by
inverting the Black-Scholes PDE** rather than differencing the step-2 centre node:
Jarrow-Rudd and Tian are drift-shifted trees where `u·d ≠ 1`, so the "centre" node
sits at a different spot than S and the naive difference quotient picks up a
spurious delta term. The PDE route is exact for European exercise, valid in the
continuation region for American, and independent of the tree geometry. A test
pins all five schemes to the same theta.

The **early-exercise boundary** is extracted from the lattice. Note which edge of
the exercise region it comes from: a call is exercised when spot is *high*, so its
region is `S ≥ S*(t)` and the boundary is the lowest exercised node; a put is the
mirror image. Getting this backwards produces a plausible-looking curve that is
completely wrong, so both directions are tested.

### Monte Carlo — `optlib/montecarlo.py`

Exact one-step GBM sampling of the terminal price, with three independent variance
reduction mechanisms. Measured on a one-year ATM call at 200k paths:

| Estimator | Efficiency vs plain MC |
|---|---|
| Plain | 1.0× |
| Antithetic | 1.7× |
| Regression-optimal control variate on the discounted spot | 6.1× |
| Antithetic + control | 24.7× |
| + randomized-Halton QMC | 5410× |

Efficiency is measured honestly: against plain i.i.d. Monte Carlo using the *same
number of payoff evaluations*, with antithetic pairs averaged before the variance
is taken (a pair is one draw from the estimator's point of view).

**QMC error bars are handled correctly.** The usual `√(variance/n)` standard error
is invalid for quasi-Monte Carlo — the points are deterministic and correlated by
construction, so the CLT does not apply. Instead the path budget is split across
independently shifted Halton sets (Cranley-Patterson rotation) and the error comes
from the spread of the replication means. That is the only defensible QMC error
bar, and it is what makes the QMC-vs-pseudorandom comparison in the convergence
lab an honest one.

Greeks use **pathwise** (infinitesimal perturbation) estimators for delta and vega,
which are unbiased and far tighter than bumping. Gamma uses the likelihood-ratio
weight `(Z² − σ√T·Z − 1)/(σ²TS²)`, because the pathwise method fails for a payoff
whose second derivative is a point mass.

American options use **Longstaff-Schwartz** least-squares Monte Carlo, regressing
the discounted continuation value on a polynomial basis over the in-the-money paths
only. LSMC is biased low by construction — the exercise rule is estimated from the
same paths it is applied to — and the UI says so rather than presenting it as a
competing estimate.

### Implied volatility — `optlib/implied_vol.py`

No-arbitrage bound check, then a Corrado-Miller seed, then **guarded** Newton that
maintains a bracket from every evaluation and bisects whenever a Newton step would
leave it. Newton alone is not safe here: vega collapses for deep out-of-the-money
or nearly expired options, and an unguarded step flies off to a negative or absurd
volatility. Round-trips to 1e-8 across volatilities from 1% to 500%.

---

## Volatility surfaces — `optlib/surface.py`

Everything is built in **total implied variance** `w = σ²T` over **log-moneyness**
`k = log(K/F(T))`. That is not cosmetic: the no-arbitrage conditions, the SVI
parameterizations and the Breeden-Litzenberger density all take their simple form
in `(k, w)` coordinates, and interpolating `w` linearly in `T` is exactly the
condition that keeps a surface calendar-arbitrage free between two slices.

### Four interpolators, scored on the same quotes

**`svi`** — raw SVI per expiry, calibrated by the **Zeliade quasi-explicit method**:
for a fixed `(m, σ)` the fit is a *convex* least-squares problem in the reduced
parameters `(a, d, c) = (a, ρbσ, bσ)`, leaving only a two-dimensional search for
the simplex. Fitting all five parameters directly is notoriously prone to local
minima; the reduction is what makes the fit repeatable. A short penalized polish
pass on all five parameters recovers the accuracy the domain clamps cost.

**`ssvi`** — Surface SVI (Gatheral-Jacquier 2014):

```
w(k, θ) = θ/2 · { 1 + ρ·φ(θ)·k + √((φ(θ)·k + ρ)² + 1 − ρ²) }
φ(θ)    = η / (θ^γ · (1+θ)^(1−γ))
```

One `(ρ, η, γ)` triple shared by the whole surface on top of a per-expiry ATM
variance curve, fitted in two stages because they are nearly separable. Three
global shape parameters cannot contort to fit noise, and the sufficient conditions
`θφ(1+|ρ|) ≤ 4` and `θφ²(1+|ρ|) ≤ 4` are enforced.

**`cubic`** — natural cubic spline through the quoted smile. Interpolates exactly,
which is the problem.

**`rbf`** — thin-plate spline over all scattered `(k, T)` at once with a ridge term.
Coordinates are normalized to the unit square first, or whichever axis has the
larger numeric range dominates the distance metric entirely.

### Diagnostics

- **Butterfly arbitrage** — Durrleman's `g(k) = (1 − k·w'/2w)² − (w'/4)(1/w + 1/4) + w''/2`,
  scanned across every slice. Where `g < 0` the surface implies a negative
  risk-neutral density. Viewable as its own 3-D surface on a diverging scale.
- **Calendar arbitrage** — total variance must be non-decreasing in `T` at fixed `k`.
- **Risk-neutral density** via Gatheral's form, `p(k) = g(k)/√(2πw) · exp(−d₋²/2)`.
  It integrates to 1.000 to three decimal places, which validates the whole
  `(k, w)` → density pipeline.
- **Held-out RMSE** by k-fold refitting on interleaved strikes, never at a slice
  endpoint (that would score extrapolation, not interpolation).

### What the shootout actually shows

Fitted to the same 152 synthetic quotes across 10 expiries, in volatility points:

| Interpolator | In-sample | Held out | vs known truth | Butterfly | Calendar |
|---|---|---|---|---|---|
| Raw SVI | 0.46 | 0.61 | 0.28 | ✕ | ✕ |
| SSVI | 1.02 | 0.80 | 0.84 | ✓ | ✓ |
| Cubic spline | **0.01** | **1.01** | 0.55 | ✕ | ✕ |
| Thin-plate RBF | 0.12 | 0.86 | 0.50 | ✕ | ✕ |

The cubic spline is a hundred times better in-sample than raw SVI and the worst of
the four out of sample. That inversion is the point of the panel.

---

## Market data — `optlib/market.py`

Two sources.

**Synthetic (default).** Not a toy. An SSVI surface — provably arbitrage-free under
conditions the code checks — is priced with Black-Scholes, rounded to real exchange
ticks (a penny under $3, a nickel above), and wrapped in a bid/ask spread. That
gives two things a live feed cannot: a surface *known* to be admissible, and a
ground-truth volatility for every quote. Inverting the mids back reproduces the
real difficulty of the task, including the discretization noise that blows up in
the wings where vega is small.

Round-tripping the default chain: **306 of 306 quotes invert successfully**, with
an RMSE of 1.02 volatility points against the quoted IV — that gap is the tick
rounding, and it is why the wings are hard.

**Live.** Yahoo Finance delayed quotes, including the cookie/crumb handshake its
JSON endpoints require. The app computes its own implied vol from the mid and shows
it beside Yahoo's published figure. Yahoo gates and rate-limits these endpoints
aggressively, so the fetch can fail for a perfectly valid ticker; the UI says so
plainly and the synthetic chain is always available.

Quote selection defaults to **out-of-the-money only**. That is standard practice:
OTM quotes carry nearly all the volatility information, while deep-in-the-money
options are mostly intrinsic value, so a tick of price error there maps to a huge
vol error and would dominate the fit.

---

## The interface

Vanilla ES modules, no framework, no build step. Charts are hand-rolled canvas
primitives; the 3-D surface is a depth-sorted quad mesh with diffuse shading, no
WebGL.

Some deliberate choices:

- **The Greek profile panel is faceted, not overlaid.** Delta spans 0 to 1 while
  gamma is order 0.01. One shared axis flattens gamma into the baseline; a second
  y-axis invents a correlation that is not in the data. Small multiples with a
  shared x-axis is the honest answer.
- **The volatility surface is drawn only where strikes are quoted.** Quoted ranges
  widen with maturity, so the fitted region is a cone, not a rectangle. Rendering
  the rectangle gives a 4-day option 70% out of the money the same visual weight
  as the fitted core, and its extreme values dominate the colour scale and the eye.
  Extrapolation is one click away, dimmed.
- **The tenor axis on the Greeks surface starts at one week.** Gamma and theta
  diverge as `T → 0`; a single one-day cell would set the colour scale for the
  entire surface. The term-slice chart below it goes right down to a week.
- **Every chart has a table view.** No value is reachable only by hovering.
- **Categorical colours are assigned in a fixed order and never cycled.** The
  palette is capped at five series and validated for colour-vision deficiency
  separation against both the dark and light chart surfaces — worst adjacent pair
  ΔE 8.4 in OKLab×100, worst normal-vision pair 19.3, all steps ≥ 3:1 contrast.
  Series six is refused rather than invented.
- Light and dark themes are both *selected*, not flipped: each mode's series steps
  were chosen for its own surface.

---

## Layout

```
run.py                    launcher; serves loopback IPv4 + IPv6, opens a browser
optlib/
  normal.py               Hart normal CDF/PDF, Acklam+Halley inverse
  blackscholes.py         BSM price and 15 analytic Greeks, vectorized
  lattice.py              4 binomial schemes + Boyle trinomial, EU/US, boundary
  montecarlo.py           European MC, variance reduction, pathwise/LR Greeks, LSMC
  implied_vol.py          bracketed Newton IV inversion
  surface.py              SVI, SSVI, cubic, RBF + arbitrage diagnostics
  optimize.py             Nelder-Mead and projected-gradient least squares
  convergence.py          error-rate studies and cost-to-accuracy shootout
  market.py               SSVI synthetic chain generator; Yahoo Finance fetch
  strategy.py             multi-leg payoff, P&L and aggregate Greeks
server/app.py             stdlib JSON API + static file server
web/                      ES modules, hand-rolled canvas charts, 3-D surface
tests/test_pricing.py     54 tests
```

No SciPy. The optimizers, the normal distribution, the spline and the RBF solve are
all written against NumPy directly — partly because the dependency is not worth it
for this much code, and partly because SciPy and NumPy ABI mismatches are a common
way for a project like this to stop running on someone else's machine.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

54 tests, ~21 seconds. They are mostly *relationships* rather than golden numbers,
because that is what catches bugs: put-call parity, every analytic Greek against a
finite difference of the price it differentiates, independent engines against each
other, densities integrating to one, and published benchmark values where they
exist. A golden number locks in whatever the code did on the day it was written,
including its bugs.

A few worth calling out:

- The American put at `S=K=100, T=1, r=5%, σ=20%` must equal the literature value
  of **6.0903**.
- With `q = 0`, an American call must equal its European counterpart — early
  exercise is never optimal.
- American ≥ European is asserted **on the same lattice**. The early-exercise
  premium can be far smaller than the lattice's own O(1/n) discretization error, so
  comparing against the closed form measures discretization noise, not the property
  of interest.
- All five lattice schemes must agree on theta, which is what pins the PDE-based
  derivation in place.
- Monte Carlo assertions are stated in standard errors, not absolute tolerances,
  and cases where the payoff is essentially never in the money assert that MC
  correctly returns nothing rather than pretending to a tolerance no finite sample
  can meet.

## Notes and limitations

- Volatility is a single input per contract; there is no local-vol or stochastic-vol
  model. The surface tools calibrate to quotes but do not feed back into a pricing
  PDE.
- The Leisen-Reimer scheme gets *worse* with payoff smoothing, not better. Its whole
  mechanism is tuning the tree so terminal nodes straddle the strike; smoothing the
  penultimate layer destroys that alignment. This is visible in the convergence tab
  and is a real property, not a bug.
- Yahoo Finance is an undocumented endpoint that changes without notice. Treat the
  live fetch as a demonstration, not a data pipeline.
- American exercise assumes a continuous dividend yield. Discrete dividends would
  need a different lattice construction.
