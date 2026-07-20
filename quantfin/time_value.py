"""
Time value of money - the four basic building blocks (discrete and
continuous compounding, present and future value) that pretty much
everything else in this package sits on top of.
"""

from __future__ import annotations

import math


def future_value_discrete(present_value: float, rate: float, periods: float) -> float:
    """
    Future value of a lump sum under discrete (annual) compounding.

    Args:
        present_value: Amount invested today.
        rate: Periodic interest rate, expressed as a decimal (5% -> 0.05).
        periods: Number of compounding periods.

    Returns:
        The value of the investment after `periods` periods.
    """
    return present_value * (1.0 + rate) ** periods


def present_value_discrete(future_value: float, rate: float, periods: float) -> float:
    """
    Present value of a future lump sum under discrete compounding.

    Args:
        future_value: Amount to be received in the future.
        rate: Periodic discount rate, expressed as a decimal.
        periods: Number of compounding periods until the cash flow.

    Returns:
        The value of that future amount in today's money.
    """
    return future_value * (1.0 + rate) ** (-periods)


def future_value_continuous(present_value: float, rate: float, time: float) -> float:
    """
    Future value under continuous compounding: PV * e^(r*t).

    Args:
        present_value: Amount invested today.
        rate: Continuously compounded rate, as a decimal.
        time: Time horizon in years.
    """
    return present_value * math.exp(rate * time)


def present_value_continuous(future_value: float, rate: float, time: float) -> float:
    """
    Present value under continuous compounding: FV * e^(-r*t).

    Args:
        future_value: Amount to be received at time `time`.
        rate: Continuously compounded discount rate, as a decimal.
        time: Time horizon in years.
    """
    return future_value * math.exp(-rate * time)
