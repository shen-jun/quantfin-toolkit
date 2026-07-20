"""
Simulators for the stochastic processes that show up over and over in
quant finance: the Wiener process itself, geometric Brownian motion (the
process behind Black-Scholes), the Ornstein-Uhlenbeck process, and Vasicek
short-rate paths (which is really just OU with a different name once you've
seen both).

Note on randomness: every function here takes an optional `seed` so results
are reproducible in tests and examples. If you don't pass one, NumPy's
default global random state is used.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def simulate_wiener_process(
    n_steps: int,
    dt: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single path of a standard Wiener process W(t), with W(0) = 0.

    Args:
        n_steps: Number of time steps to simulate (the path will have
            n_steps + 1 points, including t=0).
        dt: Size of each time step.
        seed: Optional seed for reproducibility.

    Returns:
        A tuple (t, w) of NumPy arrays, both of length n_steps + 1.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    rng = np.random.default_rng(seed)

    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    increments = rng.normal(loc=0.0, scale=np.sqrt(dt), size=n_steps)

    w = np.zeros(n_steps + 1)
    w[1:] = np.cumsum(increments)

    return t, w


def simulate_geometric_brownian_motion(
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_steps: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single path of geometric Brownian motion:
        dS = mu * S * dt + sigma * S * dW

    This is solved in closed form (no discretization error) via
        S(t) = S0 * exp((mu - 0.5*sigma^2)*t + sigma*W(t))

    Args:
        s0: Starting value (e.g. today's stock price). Must be positive.
        mu: Drift.
        sigma: Volatility. Must be non-negative.
        t: Total time horizon.
        n_steps: Number of steps between 0 and t.
        seed: Optional seed for reproducibility.

    Returns:
        (time_grid, path) arrays of length n_steps + 1.
    """
    if s0 <= 0:
        raise ValueError(f"s0 must be positive, got {s0}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if t <= 0:
        raise ValueError(f"t must be positive, got {t}")

    dt = t / n_steps
    time_grid, w = simulate_wiener_process(n_steps, dt=dt, seed=seed)

    path = s0 * np.exp((mu - 0.5 * sigma ** 2) * time_grid + sigma * w)
    return time_grid, path


def simulate_ornstein_uhlenbeck(
    x0: float,
    theta: float,
    mu: float,
    sigma: float,
    n_steps: int,
    dt: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate an Ornstein-Uhlenbeck (mean-reverting) process using a simple
    Euler-Maruyama discretization:
        x(t + dt) = x(t) + theta * (mu - x(t)) * dt + sigma * sqrt(dt) * Z

    Args:
        x0: Starting value.
        theta: Speed of mean reversion. Larger = pulls back to mu faster.
        mu: Long-run mean the process reverts to.
        sigma: Volatility of the process.
        n_steps: Number of steps to simulate.
        dt: Size of each time step.
        seed: Optional seed for reproducibility.

    Returns:
        Array of length n_steps + 1 (including the starting value x0).
    """
    if theta < 0:
        raise ValueError(f"theta must be non-negative, got {theta}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    rng = np.random.default_rng(seed)
    x = np.zeros(n_steps + 1)
    x[0] = x0

    shocks = rng.normal(loc=0.0, scale=np.sqrt(dt), size=n_steps)
    for i in range(1, n_steps + 1):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) * dt + sigma * shocks[i - 1]

    return x


def simulate_vasicek(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    t: float,
    n_steps: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a short-rate path under the Vasicek model, which is an
    Ornstein-Uhlenbeck process applied to interest rates:
        dr = kappa * (theta - r) * dt + sigma * dW

    We just delegate to simulate_ornstein_uhlenbeck under the hood - Vasicek
    is that same process with different variable names (r0/kappa/theta
    instead of x0/theta/mu), which is exactly why the two tutorial scripts
    this was based on used to duplicate almost the same code twice.

    Args:
        r0: Starting short rate.
        kappa: Speed of mean reversion.
        theta: Long-run mean short rate.
        sigma: Volatility of the short rate.
        t: Total time horizon, in years.
        n_steps: Number of steps between 0 and t.
        seed: Optional seed for reproducibility.

    Returns:
        (time_grid, rate_path) arrays of length n_steps + 1.
    """
    if t <= 0:
        raise ValueError(f"t must be positive, got {t}")

    dt = t / n_steps
    time_grid = np.linspace(0.0, t, n_steps + 1)
    rate_path = simulate_ornstein_uhlenbeck(
        x0=r0, theta=kappa, mu=theta, sigma=sigma, n_steps=n_steps, dt=dt, seed=seed
    )
    return time_grid, rate_path
