"""
Capital Asset Pricing Model: beta estimation (two ways - the covariance
formula and OLS regression, which should agree) and expected return /
risk premium.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class CAPMModel:
    """
    Args:
        asset_returns: Historical returns of the asset being priced.
        market_returns: Historical returns of the market portfolio /
            benchmark, same length and same periods as asset_returns.
        risk_free_rate: Risk-free rate, in the same units/frequency as the
            return series (e.g. if returns are monthly, this should be a
            monthly risk-free rate).
    """

    def __init__(self, asset_returns: np.ndarray, market_returns: np.ndarray, risk_free_rate: float) -> None:
        asset_returns = np.asarray(asset_returns, dtype=float)
        market_returns = np.asarray(market_returns, dtype=float)

        if asset_returns.shape != market_returns.shape:
            raise ValueError("asset_returns and market_returns must be the same length")
        if len(asset_returns) < 2:
            raise ValueError("need at least 2 observations to estimate beta")

        self.asset_returns = asset_returns
        self.market_returns = market_returns
        self.risk_free_rate = risk_free_rate

    def beta_from_covariance(self) -> float:
        """Beta = Cov(asset, market) / Var(market)."""
        covariance_matrix = np.cov(self.asset_returns, self.market_returns, ddof=1)
        market_variance = covariance_matrix[1, 1]
        if market_variance == 0:
            raise ValueError("market_returns has zero variance; beta is undefined")
        return float(covariance_matrix[0, 1] / market_variance)

    def beta_from_regression(self) -> Tuple[float, float]:
        """Beta and alpha from an OLS regression of asset returns on market
        returns. Should give (numerically) the same beta as
        beta_from_covariance - having both is mostly useful as a sanity
        check on each other."""
        beta, alpha = np.polyfit(self.market_returns, self.asset_returns, deg=1)
        return float(beta), float(alpha)

    def expected_return(self) -> float:
        """CAPM expected return: rf + beta * (market return - rf)."""
        beta = self.beta_from_covariance()
        market_mean_return = float(self.market_returns.mean())
        return self.risk_free_rate + beta * (market_mean_return - self.risk_free_rate)

    def market_risk_premium(self) -> float:
        """Average market return in excess of the risk-free rate."""
        return float(self.market_returns.mean()) - self.risk_free_rate


def required_risk_premium(risk_aversion_coefficient: float, variance: float) -> float:
    """
    The risk premium an investor with a given (absolute) risk aversion
    coefficient would need to be compensated for holding an asset or
    portfolio with the given return variance, under the standard
    mean-variance / quadratic-utility approximation:

        risk premium = 0.5 * risk_aversion_coefficient * variance

    This is the same risk-aversion parameter that shows up in
    `quantfin.portfolio.optimize_portfolio`'s "mean_variance" objective -
    it's what ties an investor's risk tolerance to how much expected return
    they need to be paid for taking on a given amount of risk.
    """
    if risk_aversion_coefficient <= 0:
        raise ValueError(f"risk_aversion_coefficient must be positive, got {risk_aversion_coefficient}")
    if variance < 0:
        raise ValueError(f"variance must be non-negative, got {variance}")

    return 0.5 * risk_aversion_coefficient * variance
