"""
Closed-form Black-Scholes-Merton pricing and Greeks for European options,
with an optional continuous dividend yield.
"""

from __future__ import annotations

import math
from typing import Tuple

from quantfin._numerics import normal_cdf, normal_pdf


def _validate_inputs(spot: float, strike: float, time_to_maturity: float, volatility: float) -> None:
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if time_to_maturity <= 0:
        raise ValueError(f"time_to_maturity must be positive, got {time_to_maturity}")
    if volatility <= 0:
        raise ValueError(f"volatility must be positive, got {volatility}")


def _d1_d2(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> Tuple[float, float]:
    """The d1 and d2 terms shared by every closed-form formula below."""
    sqrt_t = math.sqrt(time_to_maturity)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    return d1, d2


def call_price(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """European call price under Black-Scholes-Merton."""
    _validate_inputs(spot, strike, time_to_maturity, volatility)
    d1, d2 = _d1_d2(spot, strike, time_to_maturity, risk_free_rate, volatility, dividend_yield)
    return (
        spot * math.exp(-dividend_yield * time_to_maturity) * normal_cdf(d1)
        - strike * math.exp(-risk_free_rate * time_to_maturity) * normal_cdf(d2)
    )


def put_price(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """European put price under Black-Scholes-Merton."""
    _validate_inputs(spot, strike, time_to_maturity, volatility)
    d1, d2 = _d1_d2(spot, strike, time_to_maturity, risk_free_rate, volatility, dividend_yield)
    return (
        strike * math.exp(-risk_free_rate * time_to_maturity) * normal_cdf(-d2)
        - spot * math.exp(-dividend_yield * time_to_maturity) * normal_cdf(-d1)
    )


def delta(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> float:
    """Delta: sensitivity of the option price to a $1 move in the underlying."""
    _validate_inputs(spot, strike, time_to_maturity, volatility)
    d1, _ = _d1_d2(spot, strike, time_to_maturity, risk_free_rate, volatility, dividend_yield)
    discount = math.exp(-dividend_yield * time_to_maturity)

    if option_type == "call":
        return discount * normal_cdf(d1)
    if option_type == "put":
        return discount * (normal_cdf(d1) - 1.0)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def gamma(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Gamma: rate of change of delta with respect to the underlying. Same
    value for calls and puts."""
    _validate_inputs(spot, strike, time_to_maturity, volatility)
    d1, _ = _d1_d2(spot, strike, time_to_maturity, risk_free_rate, volatility, dividend_yield)
    discount = math.exp(-dividend_yield * time_to_maturity)
    return discount * normal_pdf(d1) / (spot * volatility * math.sqrt(time_to_maturity))


def vega(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Vega: sensitivity of the option price to a 1.0 (i.e. 100 percentage
    point) move in volatility. Same value for calls and puts. Divide by 100
    if you want the more commonly quoted "per 1 vol point" number."""
    _validate_inputs(spot, strike, time_to_maturity, volatility)
    d1, _ = _d1_d2(spot, strike, time_to_maturity, risk_free_rate, volatility, dividend_yield)
    discount = math.exp(-dividend_yield * time_to_maturity)
    return spot * discount * normal_pdf(d1) * math.sqrt(time_to_maturity)


def theta(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> float:
    """Theta: sensitivity of the option price to the passage of time,
    expressed per year (divide by 365 for a per-calendar-day number)."""
    _validate_inputs(spot, strike, time_to_maturity, volatility)
    d1, d2 = _d1_d2(spot, strike, time_to_maturity, risk_free_rate, volatility, dividend_yield)
    sqrt_t = math.sqrt(time_to_maturity)
    div_discount = math.exp(-dividend_yield * time_to_maturity)
    rate_discount = math.exp(-risk_free_rate * time_to_maturity)

    decay_term = -(spot * div_discount * normal_pdf(d1) * volatility) / (2 * sqrt_t)

    if option_type == "call":
        return (
            decay_term
            - risk_free_rate * strike * rate_discount * normal_cdf(d2)
            + dividend_yield * spot * div_discount * normal_cdf(d1)
        )
    if option_type == "put":
        return (
            decay_term
            + risk_free_rate * strike * rate_discount * normal_cdf(-d2)
            - dividend_yield * spot * div_discount * normal_cdf(-d1)
        )
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
