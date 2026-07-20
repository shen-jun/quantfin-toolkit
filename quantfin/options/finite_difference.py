"""
Finite difference pricer for European options, solving the Black-Scholes
PDE directly on a grid over (underlying price, time) instead of using the
closed-form formula or a tree.

    dV/dt + 0.5*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V = 0

All three classic schemes are here: explicit, implicit, and Crank-Nicolson
(which is just the average of the two). Explicit is included mainly because
it's the standard teaching example and it's a good illustration of *why*
implicit/Crank-Nicolson are preferred in practice - it's numerically
unstable unless the time step is small relative to the price step, and
`price()` will warn you if you ask for it with an unstable combination
instead of silently returning garbage.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np


class FiniteDifferenceOptionPricer:
    """
    Grid-based European option pricer.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_maturity: Time to expiry, in years.
        risk_free_rate: Continuously compounded risk-free rate.
        volatility: Annualized volatility.
        option_type: "call" or "put".
        s_max_multiplier: The price grid runs from 0 to
            s_max_multiplier * strike. 3-4x is usually plenty for a vanilla
            option.
        n_price_steps: Number of steps in the underlying-price direction.
        n_time_steps: Number of steps in the time direction.
    """

    def __init__(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        option_type: str = "call",
        s_max_multiplier: float = 3.0,
        n_price_steps: int = 100,
        n_time_steps: int = 100,
    ) -> None:
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")
        if strike <= 0:
            raise ValueError(f"strike must be positive, got {strike}")
        if time_to_maturity <= 0:
            raise ValueError(f"time_to_maturity must be positive, got {time_to_maturity}")
        if volatility <= 0:
            raise ValueError(f"volatility must be positive, got {volatility}")
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
        if n_price_steps < 10 or n_time_steps < 10:
            raise ValueError("n_price_steps and n_time_steps should be at least 10 for a sane grid")

        self.spot = spot
        self.strike = strike
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.option_type = option_type
        self.n_price_steps = n_price_steps
        self.n_time_steps = n_time_steps

        self.s_max = s_max_multiplier * strike
        self.ds = self.s_max / n_price_steps
        self.dt = time_to_maturity / n_time_steps
        self.price_grid = np.linspace(0.0, self.s_max, n_price_steps + 1)

    def _terminal_payoff(self) -> np.ndarray:
        if self.option_type == "call":
            return np.maximum(self.price_grid - self.strike, 0.0)
        return np.maximum(self.strike - self.price_grid, 0.0)

    def _boundary_values(self, time_to_expiry: float) -> Tuple[float, float]:
        """Value at S=0 and S=s_max, given the remaining time to expiry."""
        discount = np.exp(-self.risk_free_rate * time_to_expiry)
        if self.option_type == "call":
            return 0.0, self.s_max - self.strike * discount
        return self.strike * discount, 0.0

    def _check_explicit_stability(self) -> None:
        # Standard stability guideline for the explicit scheme on this grid
        # (see e.g. Wilmott, Howison & Dewynne): dt should stay below
        # roughly 1 / (sigma^2 * M^2), where M is the number of price steps.
        max_stable_dt = 1.0 / (self.volatility ** 2 * self.n_price_steps ** 2)
        if self.dt > max_stable_dt:
            warnings.warn(
                "explicit finite difference scheme is likely unstable for this grid "
                f"(dt={self.dt:.2e} > ~{max_stable_dt:.2e}); increase n_time_steps, "
                "reduce n_price_steps, or use method='implicit'/'crank_nicolson' instead",
                stacklevel=2,
            )

    def _price_explicit(self) -> np.ndarray:
        self._check_explicit_stability()
        j = np.arange(1, self.n_price_steps)  # interior nodes only
        a = 0.5 * self.dt * (self.volatility ** 2 * j ** 2 - self.risk_free_rate * j)
        b = 1.0 - self.dt * (self.volatility ** 2 * j ** 2 + self.risk_free_rate)
        c = 0.5 * self.dt * (self.volatility ** 2 * j ** 2 + self.risk_free_rate * j)

        values = self._terminal_payoff()

        for step in range(self.n_time_steps - 1, -1, -1):
            time_to_expiry = self.time_to_maturity - step * self.dt
            lower_bound, upper_bound = self._boundary_values(time_to_expiry)

            new_values = np.empty_like(values)
            new_values[0] = lower_bound
            new_values[-1] = upper_bound
            new_values[1:-1] = a * values[:-2] + b * values[1:-1] + c * values[2:]
            values = new_values

        return values

    def _build_implicit_matrix(self) -> np.ndarray:
        j = np.arange(1, self.n_price_steps)
        alpha = 0.5 * self.dt * (self.risk_free_rate * j - self.volatility ** 2 * j ** 2)
        beta = 1.0 + self.dt * (self.volatility ** 2 * j ** 2 + self.risk_free_rate)
        gamma = -0.5 * self.dt * (self.volatility ** 2 * j ** 2 + self.risk_free_rate * j)

        n_interior = self.n_price_steps - 1
        matrix = np.zeros((n_interior, n_interior))
        for row in range(n_interior):
            matrix[row, row] = beta[row]
            if row > 0:
                matrix[row, row - 1] = alpha[row]
            if row < n_interior - 1:
                matrix[row, row + 1] = gamma[row]
        return matrix

    def _price_implicit(self) -> np.ndarray:
        matrix = self._build_implicit_matrix()
        j = np.arange(1, self.n_price_steps)
        alpha = 0.5 * self.dt * (self.risk_free_rate * j - self.volatility ** 2 * j ** 2)
        gamma = -0.5 * self.dt * (self.volatility ** 2 * j ** 2 + self.risk_free_rate * j)

        values = self._terminal_payoff()

        for step in range(self.n_time_steps - 1, -1, -1):
            time_to_expiry = self.time_to_maturity - step * self.dt
            lower_bound, upper_bound = self._boundary_values(time_to_expiry)

            rhs = values[1:-1].copy()
            rhs[0] -= alpha[0] * lower_bound
            rhs[-1] -= gamma[-1] * upper_bound

            interior_values = np.linalg.solve(matrix, rhs)

            new_values = np.empty_like(values)
            new_values[0] = lower_bound
            new_values[-1] = upper_bound
            new_values[1:-1] = interior_values
            values = new_values

        return values

    def _price_crank_nicolson(self) -> np.ndarray:
        # Crank-Nicolson is just the average of one explicit half-step and
        # one implicit half-step, which is the standard way to describe it
        # even though most implementations (this one included) build it
        # directly as a single tridiagonal solve per time step.
        j = np.arange(1, self.n_price_steps)
        a = 0.25 * self.dt * (self.volatility ** 2 * j ** 2 - self.risk_free_rate * j)
        b = -0.5 * self.dt * (self.volatility ** 2 * j ** 2 + self.risk_free_rate)
        c = 0.25 * self.dt * (self.volatility ** 2 * j ** 2 + self.risk_free_rate * j)

        n_interior = self.n_price_steps - 1
        lhs = np.zeros((n_interior, n_interior))
        for row in range(n_interior):
            lhs[row, row] = 1.0 - b[row]
            if row > 0:
                lhs[row, row - 1] = -a[row]
            if row < n_interior - 1:
                lhs[row, row + 1] = -c[row]

        values = self._terminal_payoff()

        for step in range(self.n_time_steps - 1, -1, -1):
            time_to_expiry_new = self.time_to_maturity - step * self.dt
            time_to_expiry_old = self.time_to_maturity - (step + 1) * self.dt
            lower_new, upper_new = self._boundary_values(time_to_expiry_new)
            lower_old, upper_old = self._boundary_values(time_to_expiry_old)

            interior_old = values[1:-1]
            rhs = a * values[:-2] + (1.0 + b) * interior_old + c * values[2:]
            rhs[0] += a[0] * lower_old - a[0] * lower_new
            rhs[-1] += c[-1] * upper_old - c[-1] * upper_new

            interior_new = np.linalg.solve(lhs, rhs)

            new_values = np.empty_like(values)
            new_values[0] = lower_new
            new_values[-1] = upper_new
            new_values[1:-1] = interior_new
            values = new_values

        return values

    def price(self, method: str = "crank_nicolson") -> float:
        """
        Solve the PDE and return the option price at `spot`, interpolating
        between grid points since `spot` generally won't land exactly on
        one.

        Args:
            method: "explicit", "implicit", or "crank_nicolson".
        """
        if method == "explicit":
            values = self._price_explicit()
        elif method == "implicit":
            values = self._price_implicit()
        elif method == "crank_nicolson":
            values = self._price_crank_nicolson()
        else:
            raise ValueError(f"method must be 'explicit', 'implicit', or 'crank_nicolson', got {method!r}")

        return float(np.interp(self.spot, self.price_grid, values))
