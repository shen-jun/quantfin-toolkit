"""
Monte Carlo pricer for vanilla European options, simulating terminal prices
directly under risk-neutral GBM (no need to walk the whole path since we
only care about the value at expiry).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def price_option_monte_carlo(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
    n_simulations: int = 100_000,
    antithetic: bool = True,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Price a European call or put via Monte Carlo simulation.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_maturity: Time to expiry, in years.
        risk_free_rate: Continuously compounded risk-free rate.
        volatility: Annualized volatility.
        option_type: "call" or "put".
        n_simulations: Number of simulated terminal prices. If antithetic
            variates are used, this is the number of *pairs*, so the
            effective sample size is 2x this.
        antithetic: Whether to use antithetic variates (pairing each random
            draw Z with -Z) to cut down simulation noise for roughly the
            same amount of work.
        seed: Optional seed for reproducibility.

    Returns:
        A tuple (price, standard_error). standard_error gives you a sense
        of how much simulation noise is left - the price should be
        accurate to roughly +/- 2*standard_error at a 95% confidence level.
    """
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
    if n_simulations <= 0:
        raise ValueError(f"n_simulations must be positive, got {n_simulations}")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_simulations)
    if antithetic:
        z = np.concatenate([z, -z])

    drift = (risk_free_rate - 0.5 * volatility ** 2) * time_to_maturity
    diffusion = volatility * math.sqrt(time_to_maturity) * z
    terminal_prices = spot * np.exp(drift + diffusion)

    if option_type == "call":
        payoffs = np.maximum(terminal_prices - strike, 0.0)
    else:
        payoffs = np.maximum(strike - terminal_prices, 0.0)

    discount = math.exp(-risk_free_rate * time_to_maturity)
    discounted_payoffs = discount * payoffs

    price = float(np.mean(discounted_payoffs))
    standard_error = float(np.std(discounted_payoffs, ddof=1) / math.sqrt(len(discounted_payoffs)))

    return price, standard_error
