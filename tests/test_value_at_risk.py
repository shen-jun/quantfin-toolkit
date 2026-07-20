import unittest

import numpy as np

from quantfin.risk.value_at_risk import MonteCarloVaR, expected_shortfall, historical_var, parametric_var


class TestHistoricalVar(unittest.TestCase):
    def test_positive_value_for_typical_returns(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(loc=0.0005, scale=0.02, size=1000)
        var = historical_var(returns, confidence_level=0.99)
        self.assertGreater(var, 0.0)

    def test_higher_confidence_gives_larger_var(self):
        rng = np.random.default_rng(2)
        returns = rng.normal(loc=0.0, scale=0.02, size=2000)
        var_95 = historical_var(returns, confidence_level=0.95)
        var_99 = historical_var(returns, confidence_level=0.99)
        self.assertGreater(var_99, var_95)

    def test_rejects_too_few_observations(self):
        with self.assertRaises(ValueError):
            historical_var(np.array([0.01, -0.02]))


class TestParametricVar(unittest.TestCase):
    def test_matches_hand_calculation(self):
        # z_0.99 ~ 2.3263
        var = parametric_var(mean=0.0, std=0.02, confidence_level=0.99, horizon_days=1)
        self.assertAlmostEqual(var, 0.02 * 2.3263478740, places=4)

    def test_scales_with_sqrt_of_horizon(self):
        var_1d = parametric_var(mean=0.0, std=0.02, confidence_level=0.99, horizon_days=1)
        var_4d = parametric_var(mean=0.0, std=0.02, confidence_level=0.99, horizon_days=4)
        self.assertAlmostEqual(var_4d, var_1d * 2.0, places=6)  # sqrt(4) = 2

    def test_rejects_negative_std(self):
        with self.assertRaises(ValueError):
            parametric_var(mean=0.0, std=-0.01, confidence_level=0.99)


class TestExpectedShortfall(unittest.TestCase):
    def test_es_is_at_least_as_large_as_var(self):
        rng = np.random.default_rng(3)
        returns = rng.normal(loc=0.0, scale=0.02, size=3000)
        var = historical_var(returns, confidence_level=0.99)
        es = expected_shortfall(returns, confidence_level=0.99, method="historical")
        self.assertGreaterEqual(es, var)

    def test_parametric_and_historical_agree_for_normal_data(self):
        rng = np.random.default_rng(4)
        returns = rng.normal(loc=0.0, scale=0.02, size=200_000)
        es_historical = expected_shortfall(returns, confidence_level=0.99, method="historical")
        es_parametric = expected_shortfall(returns, confidence_level=0.99, method="parametric")
        self.assertAlmostEqual(es_historical, es_parametric, delta=0.002)

    def test_rejects_unknown_method(self):
        rng = np.random.default_rng(5)
        returns = rng.normal(size=100)
        with self.assertRaises(ValueError):
            expected_shortfall(returns, method="bootstrap")


class TestMonteCarloVar(unittest.TestCase):
    def test_close_to_parametric_var_for_normal_model(self):
        mc = MonteCarloVaR(mu=0.0, sigma=0.02, n_simulations=500_000, seed=7)
        mc_var = mc.estimate(position_value=1.0, confidence_level=0.99)
        analytical_var = parametric_var(mean=0.0, std=0.02, confidence_level=0.99)
        self.assertAlmostEqual(mc_var, analytical_var, delta=0.002)

    def test_same_seed_gives_same_result(self):
        mc = MonteCarloVaR(mu=0.0, sigma=0.02, n_simulations=1000, seed=42)
        result_a = mc.estimate(position_value=1_000_000)
        result_b = mc.estimate(position_value=1_000_000)
        self.assertEqual(result_a, result_b)

    def test_var_scales_with_position_value(self):
        mc = MonteCarloVaR(mu=0.0, sigma=0.02, n_simulations=200_000, seed=8)
        var_small = mc.estimate(position_value=1.0)
        var_large = mc.estimate(position_value=10.0)
        self.assertAlmostEqual(var_large, var_small * 10.0, places=6)

    def test_rejects_negative_sigma(self):
        with self.assertRaises(ValueError):
            MonteCarloVaR(mu=0.0, sigma=-0.1)


if __name__ == "__main__":
    unittest.main()
