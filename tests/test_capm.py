import unittest

import numpy as np

from quantfin.risk.capm import CAPMModel, required_risk_premium


class TestCAPMModel(unittest.TestCase):
    def test_beta_from_covariance_and_regression_agree(self):
        rng = np.random.default_rng(1)
        market = rng.normal(0.0, 0.02, size=250)
        true_beta = 1.4
        asset = 0.0005 + true_beta * market + rng.normal(0.0, 0.005, size=250)

        model = CAPMModel(asset, market, risk_free_rate=0.0001)
        beta_cov = model.beta_from_covariance()
        beta_reg, _ = model.beta_from_regression()

        self.assertAlmostEqual(beta_cov, beta_reg, places=8)
        self.assertAlmostEqual(beta_cov, true_beta, delta=0.15)

    def test_beta_of_one_for_market_against_itself(self):
        rng = np.random.default_rng(2)
        market = rng.normal(0.0, 0.02, size=100)
        model = CAPMModel(market, market, risk_free_rate=0.0)
        self.assertAlmostEqual(model.beta_from_covariance(), 1.0, places=8)

    def test_expected_return_matches_capm_formula(self):
        market = np.array([0.02, -0.01, 0.03, 0.00, 0.015])
        asset = 1.5 * market
        model = CAPMModel(asset, market, risk_free_rate=0.001)
        expected = model.expected_return()
        beta = model.beta_from_covariance()
        hand_calc = 0.001 + beta * (market.mean() - 0.001)
        self.assertAlmostEqual(expected, hand_calc, places=10)

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            CAPMModel(np.array([0.01, 0.02, 0.03]), np.array([0.01, 0.02]), risk_free_rate=0.0)


class TestRequiredRiskPremium(unittest.TestCase):
    def test_matches_hand_calculation(self):
        self.assertAlmostEqual(required_risk_premium(risk_aversion_coefficient=4.0, variance=0.04), 0.08, places=10)

    def test_higher_risk_aversion_requires_higher_premium(self):
        low = required_risk_premium(risk_aversion_coefficient=1.0, variance=0.04)
        high = required_risk_premium(risk_aversion_coefficient=5.0, variance=0.04)
        self.assertGreater(high, low)

    def test_rejects_non_positive_risk_aversion(self):
        with self.assertRaises(ValueError):
            required_risk_premium(risk_aversion_coefficient=0.0, variance=0.04)


if __name__ == "__main__":
    unittest.main()
