"""
Value at Risk and Expected Shortfall, computed three ways: from the
empirical (historical) distribution of returns, from a fitted normal
distribution (parametric), and via Monte Carlo simulation.

Sign convention: every function here returns VaR/ES as a *positive* number
representing the size of the loss (e.g. 0.05 means "a 5% loss"), which is
the convention most desks actually use day to day, even though it means the
formulas below have their signs flipped relative to how you'd write them if
you were reporting the P&L level directly. It's worth being explicit about
this - it's a very easy place to introduce a sign error, and a version of
exactly that sign confusion was in the tutorial script this module started
from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from quantfin._numerics import normal_pdf, normal_ppf


def historical_var(returns: np.ndarray, confidence_level: float = 0.99) -> float:
    """
    Historical (empirical) VaR: the loss at the given percentile of the
    observed return distribution, with no distributional assumption.

    Args:
        returns: Array of historical returns (as decimals, e.g. -0.02 for -2%).
        confidence_level: e.g. 0.99 for 99% VaR.

    Returns:
        VaR as a positive number (size of the loss).
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    returns = np.asarray(returns)
    if len(returns) < 10:
        raise ValueError("need a reasonable number of observations (at least 10) for historical VaR")

    loss_quantile = np.percentile(returns, (1.0 - confidence_level) * 100.0)
    return float(-loss_quantile)


def parametric_var(mean: float, std: float, confidence_level: float = 0.99, horizon_days: int = 1) -> float:
    """
    Parametric (variance-covariance) VaR, assuming returns are normally
    distributed.

    Args:
        mean: Expected daily return.
        std: Daily return standard deviation.
        confidence_level: e.g. 0.99 for 99% VaR.
        horizon_days: Number of days to scale to (using the standard
            square-root-of-time rule for the volatility).

    Returns:
        VaR as a positive number (size of the loss).
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    if std < 0:
        raise ValueError(f"std must be non-negative, got {std}")
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")

    z = normal_ppf(confidence_level)
    return std * math.sqrt(horizon_days) * z - mean * horizon_days


def expected_shortfall(returns: np.ndarray, confidence_level: float = 0.99, method: str = "historical") -> float:
    """
    Expected Shortfall / CVaR: the average loss given that the loss
    exceeds VaR. Unlike VaR, this actually tells you something about how
    bad the tail is, not just where it starts.

    Args:
        returns: Array of historical returns.
        confidence_level: e.g. 0.99.
        method: "historical" (average of the observed losses beyond the
            empirical VaR threshold) or "parametric" (closed-form, assuming
            returns are normally distributed with the sample mean and
            standard deviation).

    Returns:
        Expected shortfall as a positive number.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    returns = np.asarray(returns)
    if len(returns) < 10:
        raise ValueError("need a reasonable number of observations (at least 10) for expected shortfall")

    if method == "historical":
        var_threshold = historical_var(returns, confidence_level)
        losses = -returns
        tail_losses = losses[losses >= var_threshold]
        if len(tail_losses) == 0:
            # can happen with a small sample and a very high confidence level
            return var_threshold
        return float(tail_losses.mean())

    if method == "parametric":
        mean = float(returns.mean())
        std = float(returns.std(ddof=1))
        z = normal_ppf(confidence_level)
        return std * normal_pdf(z) / (1.0 - confidence_level) - mean

    raise ValueError(f"method must be 'historical' or 'parametric', got {method!r}")


class MonteCarloVaR:
    """
    Monte Carlo VaR: simulate a large number of possible returns under a
    normal model and take the empirical percentile of the simulated
    losses. For a genuinely normal return model this should converge to
    the same answer as `parametric_var` - the point of doing it this way
    is that it's easy to swap in a richer return model later (fat tails,
    jumps, whatever) without changing how VaR is read off the simulation.
    """

    def __init__(self, mu: float, sigma: float, n_simulations: int = 100_000, seed: Optional[int] = None) -> None:
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        if n_simulations <= 0:
            raise ValueError(f"n_simulations must be positive, got {n_simulations}")

        self.mu = mu
        self.sigma = sigma
        self.n_simulations = n_simulations
        self.seed = seed

    def estimate(self, position_value: float, confidence_level: float = 0.99, horizon_days: int = 1) -> float:
        """
        Args:
            position_value: Dollar (or whatever currency) value of the position.
            confidence_level: e.g. 0.99.
            horizon_days: Number of days to scale to.

        Returns:
            VaR in the same currency as position_value, as a positive number.
        """
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
        if horizon_days <= 0:
            raise ValueError(f"horizon_days must be positive, got {horizon_days}")

        rng = np.random.default_rng(self.seed)
        z = rng.standard_normal(self.n_simulations)
        simulated_returns = self.mu * horizon_days + self.sigma * math.sqrt(horizon_days) * z
        simulated_losses = -position_value * simulated_returns

        return float(np.percentile(simulated_losses, confidence_level * 100.0))
