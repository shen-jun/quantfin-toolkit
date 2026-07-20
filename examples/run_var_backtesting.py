"""
Estimate VaR and Expected Shortfall a few different ways, then run the
backtests a model validator would actually ask for: Kupiec's proportion-of-
failures test, Christoffersen's independence/conditional coverage tests, and
the bias statistic.

Run with: python examples/run_var_backtesting.py
"""

import numpy as np

from quantfin.risk.backtesting import (
    bias_statistic,
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
    kupiec_pof_test,
)
from quantfin.risk.value_at_risk import MonteCarloVaR, expected_shortfall, historical_var, parametric_var


def main() -> None:
    rng = np.random.default_rng(7)
    n_obs = 1000
    confidence_level = 0.99

    # Simulate daily returns with a bit of volatility clustering (rough GARCH-like
    # behaviour) so the exception counts aren't just IID noise.
    returns = np.empty(n_obs)
    vol = 0.01
    for t in range(n_obs):
        vol = 0.02 * 0.01 + 0.90 * vol + 0.08 * vol  # mean-reverting volatility level
        returns[t] = rng.normal(0.0003, vol)

    # Note on sign convention: every VaR/ES function here returns a *positive*
    # number representing the size of the loss (e.g. 0.02 means "a 2% loss"),
    # not the P&L level itself.
    print("=== Point-in-time VaR / ES estimates (using the full sample) ===")
    hvar = historical_var(returns, confidence_level=confidence_level)
    pvar = parametric_var(mean=returns.mean(), std=returns.std(ddof=1), confidence_level=confidence_level)
    mc = MonteCarloVaR(mu=returns.mean(), sigma=returns.std(ddof=1), seed=42)
    mcvar = mc.estimate(position_value=1.0, confidence_level=confidence_level)
    es = expected_shortfall(returns, confidence_level=confidence_level, method="historical")
    print(f"  Historical VaR (99%):  {hvar:.4%}")
    print(f"  Parametric VaR (99%):  {pvar:.4%}")
    print(f"  Monte Carlo VaR (99%): {mcvar:.4%}")
    print(f"  Historical ES (99%):   {es:.4%}  (should exceed the VaR - it's the average loss beyond it)")

    print("\n=== Rolling 250-day parametric VaR backtest ===")
    window = 250
    exceptions = []
    predicted_vols = []
    for t in range(window, n_obs):
        history = returns[t - window : t]
        var_estimate = parametric_var(mean=history.mean(), std=history.std(ddof=1), confidence_level=confidence_level)
        # An exception is a loss bigger than the VaR estimate, i.e. a return
        # worse than -var_estimate (both VaR and the loss are positive numbers here).
        exceptions.append(1 if returns[t] < -var_estimate else 0)
        predicted_vols.append(history.std(ddof=1))
    exceptions = np.array(exceptions)
    predicted_vols = np.array(predicted_vols)
    realized = returns[window:]

    print(f"  Observations backtested: {len(exceptions)}")
    print(f"  Exceptions observed:     {exceptions.sum()} "
          f"(expected under a well-calibrated 99% VaR: ~{len(exceptions) * 0.01:.1f})")

    kupiec = kupiec_pof_test(exceptions.sum(), len(exceptions), confidence_level=confidence_level)
    print("\nKupiec proportion-of-failures test:")
    print(f"  LR statistic: {kupiec.statistic:.4f}   p-value: {kupiec.p_value:.4f}   "
          f"reject at 5%: {kupiec.reject_null}")

    independence = christoffersen_independence_test(exceptions)
    print("\nChristoffersen independence test (are exceptions clustered in time?):")
    print(f"  LR statistic: {independence.statistic:.4f}   p-value: {independence.p_value:.4f}   "
          f"reject at 5%: {independence.reject_null}")

    conditional_coverage = christoffersen_conditional_coverage_test(exceptions, confidence_level=confidence_level)
    print("\nChristoffersen conditional coverage test (correct rate AND independent?):")
    print(f"  LR statistic: {conditional_coverage.statistic:.4f}   p-value: {conditional_coverage.p_value:.4f}   "
          f"reject at 5%: {conditional_coverage.reject_null}")

    # bias_statistic returns one value per period (NaN until the rolling window
    # fills up); a well-calibrated volatility forecast keeps this close to 1.0.
    bias_series = bias_statistic(realized, predicted_vols, window=12)
    bias_valid = bias_series[~np.isnan(bias_series)]
    print(f"\nBias statistic (rolling 12-period, should sit close to 1.0 if volatility forecasts are "
          f"well-calibrated): mean={bias_valid.mean():.4f}, last value={bias_valid[-1]:.4f}")


if __name__ == "__main__":
    main()
