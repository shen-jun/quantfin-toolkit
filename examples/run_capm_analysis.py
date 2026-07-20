"""
CAPM beta estimation two ways (from covariance and from a regression), the
resulting expected return, and the risk premium a risk-averse investor would
require to hold the market portfolio.

Run with: python examples/run_capm_analysis.py
"""

import numpy as np

from quantfin.risk.capm import CAPMModel, required_risk_premium


def main() -> None:
    rng = np.random.default_rng(3)
    n_obs = 500
    true_beta = 1.35
    true_alpha = 0.0001  # small daily alpha

    market_returns = rng.normal(0.0004, 0.011, size=n_obs)
    idiosyncratic = rng.normal(0.0, 0.006, size=n_obs)
    asset_returns = true_alpha + true_beta * market_returns + idiosyncratic

    print(f"Simulated data: true beta = {true_beta}, {n_obs} daily observations\n")

    # ~5% annualised, expressed as a daily rate here
    model = CAPMModel(asset_returns=asset_returns, market_returns=market_returns, risk_free_rate=0.0002)

    beta_cov = model.beta_from_covariance()
    beta_reg, alpha_reg = model.beta_from_regression()
    print(f"Beta from covariance ratio: {beta_cov:.4f}")
    print(f"Beta from OLS regression:   {beta_reg:.4f}  (intercept/alpha: {alpha_reg:.6f})")
    print("(these should agree closely - covariance ratio IS the OLS slope for a single regressor)")

    market_premium = model.market_risk_premium()
    expected_return = model.expected_return()
    print(f"\nMarket risk premium (annualised, x252): {market_premium * 252:.2%}")
    print(f"CAPM expected return (annualised, x252): {expected_return * 252:.2%}")

    print("\n=== Required risk premium for a risk-averse investor ===")
    market_variance = market_returns.var(ddof=1)
    for risk_aversion in (2.0, 5.0, 10.0):
        premium = required_risk_premium(risk_aversion, market_variance)
        print(f"  risk aversion A={risk_aversion:>4.1f}:  required premium = {premium * 252:.2%} annualised")
    print("  (higher risk aversion -> investor demands more compensation per unit of variance)")


if __name__ == "__main__":
    main()
