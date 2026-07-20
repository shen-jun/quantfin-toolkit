"""
Cox-Ross-Rubinstein binomial tree option pricer. Supports both European and
American exercise - the whole point of reaching for a tree instead of the
closed-form Black-Scholes formula is usually early exercise, so it felt
wrong to leave that out.
"""

from __future__ import annotations

import math

import numpy as np


def price_binomial_option(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    n_steps: int,
    option_type: str = "call",
    exercise: str = "european",
) -> float:
    """
    Price a vanilla option using a recombining CRR binomial tree.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_maturity: Time to expiry, in years.
        risk_free_rate: Continuously compounded risk-free rate.
        volatility: Annualized volatility of the underlying.
        n_steps: Number of steps in the tree. More steps = closer to the
            Black-Scholes price for European options, at the cost of
            O(n_steps^2) work.
        option_type: "call" or "put".
        exercise: "european" (exercise only at maturity) or "american"
            (exercise allowed at any node).

    Returns:
        The option price at time 0.
    """
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if time_to_maturity <= 0:
        raise ValueError(f"time_to_maturity must be positive, got {time_to_maturity}")
    if volatility <= 0:
        raise ValueError(f"volatility must be positive, got {volatility}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if exercise not in ("european", "american"):
        raise ValueError(f"exercise must be 'european' or 'american', got {exercise!r}")

    dt = time_to_maturity / n_steps
    up = math.exp(volatility * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp(risk_free_rate * dt)

    risk_neutral_prob = (growth - down) / (up - down)
    if not 0.0 < risk_neutral_prob < 1.0:
        raise ValueError(
            "risk-neutral probability fell outside (0, 1); this usually means dt is too "
            "large relative to the volatility - try more steps"
        )

    discount = math.exp(-risk_free_rate * dt)

    # Terminal stock prices: spot * up^j * down^(n_steps - j) for j = 0..n_steps.
    j = np.arange(n_steps + 1)
    terminal_prices = spot * (up ** j) * (down ** (n_steps - j))

    if option_type == "call":
        values = np.maximum(terminal_prices - strike, 0.0)
    else:
        values = np.maximum(strike - terminal_prices, 0.0)

    # Walk backward through the tree, discounting expected value at each node.
    for step in range(n_steps - 1, -1, -1):
        values = discount * (risk_neutral_prob * values[1:] + (1 - risk_neutral_prob) * values[:-1])

        if exercise == "american":
            j = np.arange(step + 1)
            stock_prices_at_step = spot * (up ** j) * (down ** (step - j))
            if option_type == "call":
                intrinsic = np.maximum(stock_prices_at_step - strike, 0.0)
            else:
                intrinsic = np.maximum(strike - stock_prices_at_step, 0.0)
            values = np.maximum(values, intrinsic)

    return float(values[0])
