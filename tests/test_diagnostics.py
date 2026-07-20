import matplotlib
matplotlib.use("Agg")  # headless backend, so plotting tests don't need a display

import unittest

import numpy as np

from quantfin.diagnostics import empirical_cdf, normality_test_summary, plot_ecdf, qq_plot


class TestEmpiricalCDF(unittest.TestCase):
    def test_output_is_monotonic_and_bounded(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=200)
        values, probs = empirical_cdf(data)
        self.assertTrue(np.all(np.diff(values) >= 0))
        self.assertAlmostEqual(probs[-1], 1.0)
        self.assertAlmostEqual(probs[0], 1.0 / len(data))

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            empirical_cdf(np.array([]))


class TestPlots(unittest.TestCase):
    def test_plot_ecdf_returns_axes(self):
        rng = np.random.default_rng(2)
        ax = plot_ecdf(rng.normal(size=100))
        self.assertIsNotNone(ax)

    def test_qq_plot_returns_axes(self):
        rng = np.random.default_rng(3)
        ax = qq_plot(rng.normal(size=100))
        self.assertIsNotNone(ax)


class TestNormalityTestSummary(unittest.TestCase):
    def test_normal_data_is_not_strongly_rejected(self):
        rng = np.random.default_rng(42)
        data = rng.normal(loc=0.0, scale=1.0, size=5000)
        result = normality_test_summary(data)
        # With a large, genuinely normal sample the JB p-value should
        # usually be comfortably above the typical 5% cutoff.
        self.assertGreater(result["jarque_bera_p_value"], 0.05)

    def test_skewed_data_is_flagged(self):
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=1.0, size=5000)
        result = normality_test_summary(data)
        self.assertGreater(result["skewness"], 0.5)
        self.assertLess(result["jarque_bera_p_value"], 0.01)

    def test_rejects_too_small_sample(self):
        with self.assertRaises(ValueError):
            normality_test_summary(np.array([1.0, 2.0, 3.0]))

    def test_rejects_zero_variance_data(self):
        with self.assertRaises(ValueError):
            normality_test_summary(np.ones(20))


if __name__ == "__main__":
    unittest.main()
