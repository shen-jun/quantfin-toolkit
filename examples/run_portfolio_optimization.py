"""
Portfolio construction: trace out an efficient frontier, solve for a few
different optimal portfolios (min variance, max Sharpe, mean-variance with
transaction costs), and break one of them down into per-asset risk
contributions.

Run with: python examples/run_portfolio_optimization.py
"""

import numpy as np

from quantfin.portfolio import (
    component_contribution_to_risk,
    efficient_frontier,
    marginal_contribution_to_risk,
    optimize_portfolio,
)

ASSET_NAMES = ["equities", "bonds", "commodities", "credit"]


def main() -> None:
    expected_returns = np.array([0.08, 0.03, 0.05, 0.045])
    covariance = np.array(
        [
            [0.0400, 0.0020, 0.0180, 0.0090],
            [0.0020, 0.0025, -0.0010, 0.0015],
            [0.0180, -0.0010, 0.0625, 0.0080],
            [0.0090, 0.0015, 0.0080, 0.0100],
        ]
    )
    risk_free_rate = 0.02

    print("=== Efficient frontier (5 sample points) ===")
    frontier = efficient_frontier(expected_returns, covariance, n_points=5)
    for _, row in frontier.iterrows():
        print(f"  target return {row['target_return']:.2%} -> volatility {row['volatility']:.2%}")

    print("\n=== Minimum variance portfolio ===")
    min_var = optimize_portfolio(expected_returns, covariance, objective="min_variance")
    _print_portfolio(min_var.weights, min_var)

    print("\n=== Max Sharpe (tangency) portfolio ===")
    max_sharpe = optimize_portfolio(expected_returns, covariance, objective="max_sharpe", risk_free_rate=risk_free_rate)
    _print_portfolio(max_sharpe.weights, max_sharpe)

    print("\n=== Mean-variance portfolio with transaction costs ===")
    current_weights = np.array([0.4, 0.4, 0.1, 0.1])  # where the portfolio starts out
    with_costs = optimize_portfolio(
        expected_returns,
        covariance,
        objective="mean_variance",
        risk_aversion=3.0,
        transaction_cost_coefficient=0.5,
        current_weights=current_weights,
    )
    without_costs = optimize_portfolio(expected_returns, covariance, objective="mean_variance", risk_aversion=3.0)
    print("  Starting weights:", dict(zip(ASSET_NAMES, current_weights.round(3))))
    print("  Target with no transaction costs:")
    _print_portfolio(without_costs.weights, without_costs, indent=4)
    print("  Target with transaction costs (should sit closer to the starting weights):")
    _print_portfolio(with_costs.weights, with_costs, indent=4)

    print("\n=== Risk decomposition of the max-Sharpe portfolio ===")
    mctr = marginal_contribution_to_risk(max_sharpe.weights, covariance)
    cctr = component_contribution_to_risk(max_sharpe.weights, covariance)
    for name, w, contribution in zip(ASSET_NAMES, max_sharpe.weights, cctr):
        print(f"  {name:<12} weight={w:6.2%}   contribution to portfolio vol={contribution:6.2%}")
    print(f"  sum of contributions = {cctr.sum():.4%}  (should equal portfolio volatility {max_sharpe.volatility:.4%})")


def _print_portfolio(weights: np.ndarray, result, indent: int = 2) -> None:
    pad = " " * indent
    for name, w in zip(ASSET_NAMES, weights):
        print(f"{pad}{name:<12} {w:7.2%}")
    print(f"{pad}expected return={result.expected_return:.2%}  volatility={result.volatility:.2%}  "
          f"sharpe={result.sharpe_ratio:.3f}")


if __name__ == "__main__":
    main()
