"""
Mean-variance portfolio construction: efficient frontier, optimal weights
(with optional quadratic transaction costs), and risk decomposition
(marginal / component contribution to risk).

A note on the optimizer: rather than reaching for a general-purpose
numerical optimizer, everything here is solved as an equality-constrained
quadratic program via its closed-form (KKT) solution - a couple of linear
solves, no iterative solver, no convergence tolerances to worry about. That
only works because we're only enforcing the budget constraint (weights sum
to 1), not inequality constraints like "no short selling" - if you need
long-only or position-limit constraints, you'd swap this out for a proper
QP solver (e.g. cvxpy). Modeling transaction costs as a *quadratic* penalty
on trade size (rather than the more realistic linear-plus-impact cost) is
what keeps the whole thing quadratic and solvable this way; it's a standard
simplification, not a full transaction cost model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class OptimizationResult:
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float


def _solve_equality_constrained_qp(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve: minimize (1/2) w'Aw - w'b, subject to 1'w = 1.

    Via the Lagrangian, the solution has the form w = A^-1(b + gamma*1),
    with gamma chosen so the budget constraint holds. This one function is
    what all three optimization objectives below reduce to, once you write
    out their KKT conditions.
    """
    n = A.shape[0]
    ones = np.ones(n)
    A_inv_b = np.linalg.solve(A, b)
    A_inv_ones = np.linalg.solve(A, ones)
    gamma = (1.0 - ones @ A_inv_b) / (ones @ A_inv_ones)
    return A_inv_b + gamma * A_inv_ones


def optimize_portfolio(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    objective: str = "max_sharpe",
    risk_free_rate: float = 0.0,
    risk_aversion: float = 1.0,
    transaction_cost_coefficient: float = 0.0,
    current_weights: Optional[np.ndarray] = None,
) -> OptimizationResult:
    """
    Solve for optimal portfolio weights, fully invested (weights sum to 1,
    short selling allowed - there's no long-only constraint here).

    Args:
        expected_returns: Length-n array of expected returns.
        covariance: n x n covariance matrix.
        objective:
            - "max_sharpe": the tangency portfolio that maximizes the
              Sharpe ratio. Closed form; does not support transaction costs.
            - "min_variance": minimum-variance portfolio, ignoring expected
              returns entirely.
            - "mean_variance": maximize w'mu - (risk_aversion/2)*w'*Sigma*w,
              i.e. classic mean-variance utility with a risk aversion
              coefficient. This is the objective to use if you want
              transaction costs.
        risk_free_rate: Used for the Sharpe ratio and the "max_sharpe" objective.
        risk_aversion: Only used for objective="mean_variance". Higher = more
            risk-averse (closer to the minimum-variance portfolio).
        transaction_cost_coefficient: Coefficient kappa on a quadratic
            transaction cost penalty, (kappa/2) * ||w - current_weights||^2.
            0 means no transaction costs. Only supported for "min_variance"
            and "mean_variance".
        current_weights: Starting weights, required if
            transaction_cost_coefficient > 0.

    Returns:
        An OptimizationResult with the optimal weights and their resulting
        expected return, volatility, and Sharpe ratio.
    """
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    n = len(mu)

    if sigma.shape != (n, n):
        raise ValueError(f"covariance must be {n}x{n} to match expected_returns, got {sigma.shape}")
    if transaction_cost_coefficient < 0:
        raise ValueError(f"transaction_cost_coefficient must be non-negative, got {transaction_cost_coefficient}")
    if transaction_cost_coefficient > 0 and current_weights is None:
        raise ValueError("current_weights is required when transaction_cost_coefficient > 0")

    if current_weights is not None:
        current_weights = np.asarray(current_weights, dtype=float)
        if current_weights.shape != (n,):
            raise ValueError(f"current_weights must have length {n}, got {current_weights.shape}")

    if objective == "max_sharpe":
        if transaction_cost_coefficient > 0:
            raise ValueError(
                "transaction costs aren't supported for objective='max_sharpe' - the tangency "
                "portfolio's closed form assumes no trading friction. Use objective='mean_variance' instead."
            )
        excess_returns = mu - risk_free_rate
        inv_sigma_excess = np.linalg.solve(sigma, excess_returns)
        denominator = np.ones(n) @ inv_sigma_excess
        if abs(denominator) < 1e-12:
            raise ValueError("cannot form a tangency portfolio: 1' * inv(Sigma) * (mu - rf) is approximately 0")
        weights = inv_sigma_excess / denominator

    elif objective == "min_variance":
        a_matrix = sigma + transaction_cost_coefficient * np.eye(n)
        b_vector = transaction_cost_coefficient * current_weights if transaction_cost_coefficient > 0 else np.zeros(n)
        weights = _solve_equality_constrained_qp(a_matrix, b_vector)

    elif objective == "mean_variance":
        if risk_aversion <= 0:
            raise ValueError(f"risk_aversion must be positive, got {risk_aversion}")
        a_matrix = risk_aversion * sigma + transaction_cost_coefficient * np.eye(n)
        b_vector = mu + (transaction_cost_coefficient * current_weights if transaction_cost_coefficient > 0 else 0.0)
        weights = _solve_equality_constrained_qp(a_matrix, b_vector)

    else:
        raise ValueError(f"objective must be 'max_sharpe', 'min_variance', or 'mean_variance', got {objective!r}")

    expected_return = float(weights @ mu)
    volatility = float(np.sqrt(weights @ sigma @ weights))
    sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else float("nan")

    return OptimizationResult(
        weights=weights, expected_return=expected_return, volatility=volatility, sharpe_ratio=sharpe_ratio
    )


def efficient_frontier(expected_returns: np.ndarray, covariance: np.ndarray, n_points: int = 50) -> pd.DataFrame:
    """
    Trace out the mean-variance efficient frontier: for a range of target
    returns spanning the individual assets' expected returns, find the
    minimum-variance portfolio that achieves each target.

    Args:
        expected_returns: Length-n array of expected returns.
        covariance: n x n covariance matrix.
        n_points: Number of points to compute along the frontier.

    Returns:
        A DataFrame with columns "target_return", "volatility", and
        "weights" (each entry an array of length n).
    """
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    n = len(mu)

    if sigma.shape != (n, n):
        raise ValueError(f"covariance must be {n}x{n} to match expected_returns, got {sigma.shape}")
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points}")

    ones = np.ones(n)
    inv_sigma_ones = np.linalg.solve(sigma, ones)
    inv_sigma_mu = np.linalg.solve(sigma, mu)

    a_coef = ones @ inv_sigma_ones
    b_coef = ones @ inv_sigma_mu
    c_coef = mu @ inv_sigma_mu
    determinant = a_coef * c_coef - b_coef ** 2

    if abs(determinant) < 1e-12:
        raise ValueError(
            "efficient frontier is degenerate (A*C - B^2 is ~0) - this usually happens when "
            "expected_returns are identical across assets or nearly collinear with the covariance structure"
        )

    target_returns = np.linspace(mu.min(), mu.max(), n_points)
    rows = []
    for target in target_returns:
        gamma_1 = (c_coef - b_coef * target) / determinant
        gamma_2 = (a_coef * target - b_coef) / determinant
        weights = np.linalg.solve(sigma, gamma_1 * ones + gamma_2 * mu)
        volatility = float(np.sqrt(weights @ sigma @ weights))
        rows.append({"target_return": float(target), "volatility": volatility, "weights": weights})

    return pd.DataFrame(rows)


def marginal_contribution_to_risk(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """
    Marginal Contribution to Risk (MCTR): how much portfolio volatility
    changes per unit change in each asset's weight, i.e.
    d(sigma_p) / d(w_i) = (Sigma @ w)_i / sigma_p.
    """
    w = np.asarray(weights, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    portfolio_volatility = np.sqrt(w @ sigma @ w)

    if portfolio_volatility == 0:
        raise ValueError("portfolio volatility is zero; marginal contribution to risk is undefined")

    return (sigma @ w) / portfolio_volatility


def component_contribution_to_risk(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """
    Component Contribution to Risk (CCTR): each asset's weight times its
    MCTR. Because portfolio volatility is homogeneous of degree 1 in the
    weights, these sum exactly to total portfolio volatility (Euler's
    theorem) - useful for a risk attribution report where you want each
    position's contribution to add up to the whole.
    """
    w = np.asarray(weights, dtype=float)
    mctr = marginal_contribution_to_risk(w, covariance)
    return w * mctr
