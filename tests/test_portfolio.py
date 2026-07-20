import unittest

import numpy as np

from quantfin.portfolio import (
    component_contribution_to_risk,
    efficient_frontier,
    marginal_contribution_to_risk,
    optimize_portfolio,
)


class TestMinVariance(unittest.TestCase):
    def test_weights_sum_to_one(self):
        cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]])
        mu = np.array([0.05, 0.07, 0.10])
        result = optimize_portfolio(mu, cov, objective="min_variance")
        self.assertAlmostEqual(result.weights.sum(), 1.0, places=8)

    def test_matches_hand_calculation_for_uncorrelated_assets(self):
        # For uncorrelated assets, minimum-variance weights are inversely
        # proportional to variance - easy to check by hand.
        cov = np.diag([0.04, 0.09])
        mu = np.array([0.05, 0.05])  # doesn't matter for min_variance
        result = optimize_portfolio(mu, cov, objective="min_variance")
        expected_ratio = 0.09 / 0.04  # w1/w2
        self.assertAlmostEqual(result.weights[0] / result.weights[1], expected_ratio, places=6)

    def test_large_transaction_cost_pulls_weights_toward_current(self):
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        mu = np.array([0.05, 0.08])
        current = np.array([0.2, 0.8])
        result = optimize_portfolio(
            mu, cov, objective="min_variance", transaction_cost_coefficient=1e6, current_weights=current
        )
        np.testing.assert_allclose(result.weights, current, atol=1e-3)

    def test_requires_current_weights_when_costs_specified(self):
        cov = np.eye(2) * 0.04
        mu = np.array([0.05, 0.05])
        with self.assertRaises(ValueError):
            optimize_portfolio(mu, cov, objective="min_variance", transaction_cost_coefficient=0.5)


class TestMaxSharpe(unittest.TestCase):
    def test_weights_sum_to_one(self):
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        mu = np.array([0.08, 0.12])
        result = optimize_portfolio(mu, cov, objective="max_sharpe", risk_free_rate=0.02)
        self.assertAlmostEqual(result.weights.sum(), 1.0, places=8)

    def test_first_order_condition_holds(self):
        # At the tangency portfolio, Sigma @ w should be proportional to
        # (mu - rf) - that's exactly the condition the closed form solves.
        cov = np.array([[0.05, 0.01, 0.0], [0.01, 0.08, 0.02], [0.0, 0.02, 0.12]])
        mu = np.array([0.06, 0.09, 0.11])
        rf = 0.01
        result = optimize_portfolio(mu, cov, objective="max_sharpe", risk_free_rate=rf)

        lhs = cov @ result.weights
        rhs = mu - rf
        ratios = lhs / rhs
        self.assertAlmostEqual(ratios.std() / ratios.mean(), 0.0, places=6)

    def test_rejects_transaction_costs(self):
        cov = np.eye(2) * 0.04
        mu = np.array([0.05, 0.08])
        with self.assertRaises(ValueError):
            optimize_portfolio(
                mu, cov, objective="max_sharpe", transaction_cost_coefficient=0.1, current_weights=np.array([0.5, 0.5])
            )


class TestMeanVariance(unittest.TestCase):
    def test_higher_risk_aversion_moves_toward_min_variance_portfolio(self):
        cov = np.array([[0.04, 0.0], [0.0, 0.09]])
        mu = np.array([0.05, 0.15])

        low_aversion = optimize_portfolio(mu, cov, objective="mean_variance", risk_aversion=0.1)
        high_aversion = optimize_portfolio(mu, cov, objective="mean_variance", risk_aversion=1000.0)
        min_var = optimize_portfolio(mu, cov, objective="min_variance")

        # Very high risk aversion should look almost exactly like the pure
        # minimum-variance portfolio; low risk aversion should not.
        np.testing.assert_allclose(high_aversion.weights, min_var.weights, atol=1e-2)
        self.assertGreater(np.abs(low_aversion.weights - min_var.weights).max(), 0.05)

    def test_rejects_non_positive_risk_aversion(self):
        cov = np.eye(2) * 0.04
        mu = np.array([0.05, 0.08])
        with self.assertRaises(ValueError):
            optimize_portfolio(mu, cov, objective="mean_variance", risk_aversion=0.0)


class TestOptimizeInputValidation(unittest.TestCase):
    def test_rejects_mismatched_covariance_shape(self):
        mu = np.array([0.05, 0.08, 0.1])
        cov = np.eye(2)
        with self.assertRaises(ValueError):
            optimize_portfolio(mu, cov)

    def test_rejects_unknown_objective(self):
        mu = np.array([0.05, 0.08])
        cov = np.eye(2) * 0.04
        with self.assertRaises(ValueError):
            optimize_portfolio(mu, cov, objective="max_alpha")


class TestEfficientFrontier(unittest.TestCase):
    def test_returns_expected_columns_and_row_count(self):
        cov = np.array([[0.04, 0.0], [0.0, 0.09]])
        mu = np.array([0.05, 0.10])
        frontier = efficient_frontier(mu, cov, n_points=20)
        self.assertEqual(len(frontier), 20)
        self.assertListEqual(list(frontier.columns), ["target_return", "volatility", "weights"])

    def test_frontier_is_bullet_shaped(self):
        # Volatility should be lowest somewhere in the middle of the target
        # return range, not at either end.
        cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]])
        mu = np.array([0.05, 0.08, 0.11])
        frontier = efficient_frontier(mu, cov, n_points=30)
        min_vol_index = frontier["volatility"].idxmin()
        self.assertGreater(min_vol_index, 0)
        self.assertLess(min_vol_index, len(frontier) - 1)


class TestRiskDecomposition(unittest.TestCase):
    def test_component_contributions_sum_to_portfolio_volatility(self):
        cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]])
        weights = np.array([0.5, 0.3, 0.2])
        cctr = component_contribution_to_risk(weights, cov)
        portfolio_vol = float(np.sqrt(weights @ cov @ weights))
        self.assertAlmostEqual(cctr.sum(), portfolio_vol, places=8)

    def test_mctr_matches_finite_difference_approximation(self):
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        weights = np.array([0.6, 0.4])
        mctr = marginal_contribution_to_risk(weights, cov)

        epsilon = 1e-6
        base_vol = np.sqrt(weights @ cov @ weights)
        bumped = weights.copy()
        bumped[0] += epsilon
        bumped_vol = np.sqrt(bumped @ cov @ bumped)
        numerical_mctr_0 = (bumped_vol - base_vol) / epsilon

        self.assertAlmostEqual(mctr[0], numerical_mctr_0, places=4)

    def test_rejects_zero_volatility_portfolio(self):
        cov = np.zeros((2, 2))
        weights = np.array([0.5, 0.5])
        with self.assertRaises(ValueError):
            marginal_contribution_to_risk(weights, cov)


if __name__ == "__main__":
    unittest.main()
