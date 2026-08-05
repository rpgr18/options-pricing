"""Multi-leg option strategy valuation: payoff, P&L and aggregate Greeks.

A leg is a signed quantity of a call, a put, or the underlying. The strategy is
evaluated two ways, and the distinction is the whole point of the view that uses
this module:

* **Payoff at expiry** -- intrinsic value only, the textbook kinked diagram.
* **Mark-to-market P&L** -- the position revalued at an arbitrary future date
  and spot with Black-Scholes, which is what the position actually does. A short
  strangle looks safe at expiry and terrifying a week in; only the second curve
  shows that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import blackscholes as bs

LEG_KINDS = ("call", "put", "underlying")
AGGREGATE_GREEKS = ("delta", "gamma", "vega", "theta", "rho")


@dataclass
class Leg:
    kind: str          # "call" | "put" | "underlying"
    quantity: float    # signed; +1 long one contract-share, -1 short
    strike: float = 0.0
    sigma: float = 0.25
    premium: float | None = None   # entry price; theoretical if omitted
    multiplier: float = 1.0

    def validate(self) -> None:
        if self.kind not in LEG_KINDS:
            raise ValueError(f"leg kind must be one of {LEG_KINDS}, got {self.kind!r}")
        if self.kind != "underlying" and not (self.strike > 0):
            raise ValueError(f"{self.kind} leg needs a positive strike")
        if not np.isfinite(self.quantity):
            raise ValueError("leg quantity must be finite")


def _leg_value(leg: Leg, S, T: float, r: float, q: float):
    """Present value of one unit of the leg at spot S with T years remaining."""
    S = np.asarray(S, dtype=float)
    if leg.kind == "underlying":
        return S
    if T <= 0.0:
        return np.maximum(S - leg.strike, 0.0) if leg.kind == "call" else np.maximum(leg.strike - S, 0.0)
    return np.asarray(bs.price(S, leg.strike, T, r, q, leg.sigma, leg.kind == "call"), dtype=float)


def _leg_greek(leg: Leg, name: str, S, T: float, r: float, q: float):
    S = np.asarray(S, dtype=float)
    if leg.kind == "underlying":
        # The underlying has unit delta and no other sensitivity.
        return np.ones_like(S) if name == "delta" else np.zeros_like(S)
    if T <= 0.0:
        if name == "delta":
            itm = (S > leg.strike) if leg.kind == "call" else (S < leg.strike)
            return np.where(itm, 1.0 if leg.kind == "call" else -1.0, 0.0)
        return np.zeros_like(S)
    return np.asarray(bs.evaluate(name, S, leg.strike, T, r, q, leg.sigma, leg.kind == "call"), dtype=float)


def entry_cost(legs: list[Leg], S0: float, T0: float, r: float, q: float) -> float:
    """Net debit (positive) or credit (negative) to open the position."""
    total = 0.0
    for leg in legs:
        prem = leg.premium
        if prem is None:
            prem = float(np.asarray(_leg_value(leg, S0, T0, r, q)).reshape(()))
        total += leg.quantity * leg.multiplier * float(prem)
    return total


def evaluate(
    legs: list[Leg],
    S0: float,
    T0: float,
    r: float,
    q: float,
    spots=None,
    horizons=None,
    n_spots: int = 161,
    spot_span: float = 0.45,
) -> dict:
    """Payoff, mark-to-market P&L curves and aggregate Greeks for a strategy.

    `horizons` are years remaining at each revaluation date, largest first; 0.0 is
    expiry. Defaults walk from today to expiry so time decay is visible.
    """
    for leg in legs:
        leg.validate()
    if not legs:
        raise ValueError("a strategy needs at least one leg")

    if spots is None:
        spots = np.linspace(S0 * (1.0 - spot_span), S0 * (1.0 + spot_span), n_spots)
    spots = np.asarray(spots, dtype=float)

    if horizons is None:
        horizons = [T0, T0 * 0.5, T0 * 0.25, 0.0]
    horizons = [float(h) for h in horizons]

    cost = entry_cost(legs, S0, T0, r, q)

    curves = []
    for T in horizons:
        value = np.zeros_like(spots)
        for leg in legs:
            value += leg.quantity * leg.multiplier * _leg_value(leg, spots, T, r, q)
        curves.append({
            "T": T,
            "days_out": round((T0 - T) * 365.0, 1),
            "label": "At expiry" if T <= 0 else f"{(T0 - T) * 365.0:.0f}d from now",
            "value": value.tolist(),
            "pnl": (value - cost).tolist(),
        })

    greeks: dict[str, list[float]] = {}
    for name in AGGREGATE_GREEKS:
        agg = np.zeros_like(spots)
        for leg in legs:
            agg += leg.quantity * leg.multiplier * _leg_greek(leg, name, spots, T0, r, q)
        scale, _ = bs.GREEK_DISPLAY[name]
        greeks[name] = (agg * scale).tolist()

    payoff = np.asarray(curves[-1]["pnl"], dtype=float)
    # Breakeven spots: sign changes of the expiry P&L, refined by linear interpolation.
    sign = np.sign(payoff)
    idx = np.flatnonzero(sign[:-1] * sign[1:] < 0)
    breakevens = [
        float(spots[i] + (spots[i + 1] - spots[i]) * (0.0 - payoff[i]) / (payoff[i + 1] - payoff[i]))
        for i in idx
    ]

    # An unbounded tail is reported as such rather than as the grid edge, which
    # would silently understate the risk of a naked short leg.
    edge_slope_lo = payoff[1] - payoff[0]
    edge_slope_hi = payoff[-1] - payoff[-2]
    return {
        "spots": spots.tolist(),
        "spot0": S0,
        "entry_cost": cost,
        "position": "debit" if cost > 0 else ("credit" if cost < 0 else "flat"),
        "curves": curves,
        "greeks": greeks,
        "breakevens": breakevens,
        "max_profit": None if edge_slope_hi > 1e-9 or edge_slope_lo < -1e-9 else float(payoff.max()),
        "max_loss": None if edge_slope_hi < -1e-9 or edge_slope_lo > 1e-9 else float(payoff.min()),
        "payoff_at_spot0": float(np.interp(S0, spots, payoff)),
        "net_greeks": {name: float(np.interp(S0, spots, np.asarray(vals))) for name, vals in greeks.items()},
    }


# Preset strategies, parameterized off spot so they stay sensible at any level.
def presets(S0: float, T: float, sigma: float) -> dict[str, list[dict]]:
    lo, hi = round(S0 * 0.95, 2), round(S0 * 1.05, 2)
    far_lo, far_hi = round(S0 * 0.9, 2), round(S0 * 1.1, 2)
    atm = round(S0, 2)
    return {
        "long_call": [{"kind": "call", "quantity": 1, "strike": atm, "sigma": sigma}],
        "long_put": [{"kind": "put", "quantity": 1, "strike": atm, "sigma": sigma}],
        "covered_call": [
            {"kind": "underlying", "quantity": 1},
            {"kind": "call", "quantity": -1, "strike": hi, "sigma": sigma},
        ],
        "bull_call_spread": [
            {"kind": "call", "quantity": 1, "strike": atm, "sigma": sigma},
            {"kind": "call", "quantity": -1, "strike": hi, "sigma": sigma},
        ],
        "bear_put_spread": [
            {"kind": "put", "quantity": 1, "strike": atm, "sigma": sigma},
            {"kind": "put", "quantity": -1, "strike": lo, "sigma": sigma},
        ],
        "straddle": [
            {"kind": "call", "quantity": 1, "strike": atm, "sigma": sigma},
            {"kind": "put", "quantity": 1, "strike": atm, "sigma": sigma},
        ],
        "short_strangle": [
            {"kind": "call", "quantity": -1, "strike": hi, "sigma": sigma},
            {"kind": "put", "quantity": -1, "strike": lo, "sigma": sigma},
        ],
        "iron_condor": [
            {"kind": "put", "quantity": 1, "strike": far_lo, "sigma": sigma},
            {"kind": "put", "quantity": -1, "strike": lo, "sigma": sigma},
            {"kind": "call", "quantity": -1, "strike": hi, "sigma": sigma},
            {"kind": "call", "quantity": 1, "strike": far_hi, "sigma": sigma},
        ],
        "butterfly": [
            {"kind": "call", "quantity": 1, "strike": lo, "sigma": sigma},
            {"kind": "call", "quantity": -2, "strike": atm, "sigma": sigma},
            {"kind": "call", "quantity": 1, "strike": hi, "sigma": sigma},
        ],
        "calendar_ratio": [
            {"kind": "call", "quantity": -1, "strike": atm, "sigma": sigma},
            {"kind": "call", "quantity": 2, "strike": hi, "sigma": sigma},
        ],
    }
