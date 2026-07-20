import unittest

import numpy as np

from quantfin.risk.backtesting import (
    bias_statistic,
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
    kupiec_pof_test,
)


class TestKupiecPOFTest(unittest.TestCase):
    def test_exact_expected_exception_rate_does_not_reject(self):
        # 1% expected exception rate, 1000 obs -> exactly 10 exceptions expected.
        result = kupiec_pof_test(exceptions=10, n_obs=1000, confidence_level=0.99)
        self.assertFalse(result.reject_null)
        self.assertLess(result.statistic, 0.5)

    def test_far_too_many_exceptions_rejects(self):
        result = kupiec_pof_test(exceptions=100, n_obs=1000, confidence_level=0.99)
        self.assertTrue(result.reject_null)

    def test_zero_exceptions_is_a_valid_edge_case(self):
        result = kupiec_pof_test(exceptions=0, n_obs=200, confidence_level=0.99)
        self.assertGreaterEqual(result.statistic, 0.0)
        self.assertFalse(np.isnan(result.statistic))

    def test_all_exceptions_is_a_valid_edge_case(self):
        result = kupiec_pof_test(exceptions=200, n_obs=200, confidence_level=0.99)
        self.assertTrue(result.reject_null)

    def test_rejects_invalid_exception_count(self):
        with self.assertRaises(ValueError):
            kupiec_pof_test(exceptions=-1, n_obs=100, confidence_level=0.99)
        with self.assertRaises(ValueError):
            kupiec_pof_test(exceptions=101, n_obs=100, confidence_level=0.99)


class TestChristoffersenIndependenceTest(unittest.TestCase):
    def test_clustered_exceptions_score_higher_than_spread_out_ones(self):
        # Same total number of exceptions (10 out of 100) in both cases,
        # but clustered together vs. spaced apart with no two adjacent.
        clustered = np.array([0] * 10 + [1] * 10 + [0] * 80)
        spread = np.zeros(100, dtype=int)
        spread[np.arange(0, 100, 10)] = 1  # positions 0, 10, 20, ... - never adjacent

        clustered_result = christoffersen_independence_test(clustered)
        spread_result = christoffersen_independence_test(spread)

        self.assertGreater(clustered_result.statistic, spread_result.statistic)
        self.assertTrue(clustered_result.reject_null)
        self.assertFalse(spread_result.reject_null)

    def test_rejects_non_binary_input(self):
        with self.assertRaises(ValueError):
            christoffersen_independence_test(np.array([0, 1, 2, 0, 1]))

    def test_rejects_too_short_series(self):
        with self.assertRaises(ValueError):
            christoffersen_independence_test(np.array([1]))


class TestConditionalCoverageTest(unittest.TestCase):
    def test_statistic_is_sum_of_components(self):
        indicator = np.array([0] * 10 + [1] * 10 + [0] * 80)
        combined = christoffersen_conditional_coverage_test(indicator, confidence_level=0.99)
        exceptions = int(indicator.sum())
        pof = kupiec_pof_test(exceptions, len(indicator), 0.99)
        independence = christoffersen_independence_test(indicator)
        self.assertAlmostEqual(combined.statistic, pof.statistic + independence.statistic, places=8)


class TestBiasStatistic(unittest.TestCase):
    def test_under_forecasting_model_shows_bias_above_one(self):
        rng = np.random.default_rng(1)
        true_sigma = 0.02
        realized = rng.normal(loc=0.0, scale=true_sigma, size=500)
        predicted = np.full(500, true_sigma / 2.0)  # model predicts half the true volatility

        result = bias_statistic(realized, predicted, window=60)
        tail = result[~np.isnan(result)]
        self.assertGreater(tail.mean(), 1.5)  # should be close to 2.0

    def test_over_forecasting_model_shows_bias_below_one(self):
        rng = np.random.default_rng(2)
        true_sigma = 0.02
        realized = rng.normal(loc=0.0, scale=true_sigma, size=500)
        predicted = np.full(500, true_sigma * 2.0)  # model predicts double the true volatility

        result = bias_statistic(realized, predicted, window=60)
        tail = result[~np.isnan(result)]
        self.assertLess(tail.mean(), 0.7)  # should be close to 0.5

    def test_first_window_minus_one_entries_are_nan(self):
        realized = np.random.default_rng(3).normal(size=100)
        predicted = np.full(100, 0.02)
        result = bias_statistic(realized, predicted, window=20)
        self.assertTrue(np.all(np.isnan(result[:19])))
        self.assertFalse(np.isnan(result[19]))

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            bias_statistic(np.zeros(10), np.ones(9))

    def test_rejects_non_positive_predicted_volatility(self):
        with self.assertRaises(ValueError):
            bias_statistic(np.zeros(10), np.zeros(10))


if __name__ == "__main__":
    unittest.main()
