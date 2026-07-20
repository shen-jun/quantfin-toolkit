import unittest

import numpy as np
import pandas as pd

from quantfin.covariance import (
    eigenvalue_diagnostics,
    ledoit_wolf_shrinkage,
    pca_factor_covariance,
    sample_covariance,
)


def _simulate_returns(n_obs: int, n_assets: int, true_cov: np.ndarray, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.multivariate_normal(mean=np.zeros(n_assets), cov=true_cov, size=n_obs)
    columns = [f"asset_{i}" for i in range(n_assets)]
    return pd.DataFrame(data, columns=columns)


class TestSampleCovariance(unittest.TestCase):
    def test_matches_pandas_cov(self):
        returns = _simulate_returns(100, 5, np.eye(5), seed=1)
        result = sample_covariance(returns)
        pd.testing.assert_frame_equal(result, returns.cov())

    def test_rejects_too_few_observations(self):
        returns = pd.DataFrame({"a": [0.01]})
        with self.assertRaises(ValueError):
            sample_covariance(returns)


class TestLedoitWolfShrinkage(unittest.TestCase):
    def test_shrinkage_intensity_is_between_zero_and_one(self):
        returns = _simulate_returns(60, 20, np.eye(20), seed=2)
        result = ledoit_wolf_shrinkage(returns)
        self.assertGreaterEqual(result.shrinkage_intensity, 0.0)
        self.assertLessEqual(result.shrinkage_intensity, 1.0)

    def test_returns_symmetric_matrix(self):
        returns = _simulate_returns(60, 15, np.eye(15), seed=3)
        result = ledoit_wolf_shrinkage(returns)
        np.testing.assert_allclose(result.covariance.values, result.covariance.values.T, atol=1e-10)

    def test_shrinkage_moves_estimate_closer_to_true_covariance_in_small_sample(self):
        # Classic use case: few observations, many assets - the sample
        # covariance is noisy, so shrinking toward a stable target should
        # get us closer to the true (known, because we simulated it)
        # covariance matrix.
        n_assets = 30
        true_cov = np.eye(n_assets) * 0.04
        returns = _simulate_returns(n_obs=35, n_assets=n_assets, true_cov=true_cov, seed=4)

        sample_cov = sample_covariance(returns).values
        shrunk = ledoit_wolf_shrinkage(returns, target="identity").covariance.values

        sample_error = np.linalg.norm(sample_cov - true_cov)
        shrunk_error = np.linalg.norm(shrunk - true_cov)
        self.assertLess(shrunk_error, sample_error)

    def test_both_targets_run_without_error(self):
        returns = _simulate_returns(50, 10, np.eye(10), seed=5)
        for target in ("identity", "constant_correlation"):
            result = ledoit_wolf_shrinkage(returns, target=target)
            self.assertEqual(result.target_used, target)

    def test_rejects_unknown_target(self):
        returns = _simulate_returns(50, 10, np.eye(10), seed=6)
        with self.assertRaises(ValueError):
            ledoit_wolf_shrinkage(returns, target="something_else")


class TestPCAFactorCovariance(unittest.TestCase):
    def test_explained_variance_ratio_is_between_zero_and_one(self):
        returns = _simulate_returns(200, 15, np.eye(15), seed=7)
        result = pca_factor_covariance(returns, n_factors=3)
        self.assertGreater(result.explained_variance_ratio, 0.0)
        self.assertLessEqual(result.explained_variance_ratio, 1.0)

    def test_more_factors_explain_at_least_as_much_variance(self):
        returns = _simulate_returns(200, 15, np.eye(15), seed=8)
        few_factors = pca_factor_covariance(returns, n_factors=2)
        more_factors = pca_factor_covariance(returns, n_factors=8)
        self.assertGreaterEqual(more_factors.explained_variance_ratio, few_factors.explained_variance_ratio)

    def test_loadings_shape(self):
        returns = _simulate_returns(200, 15, np.eye(15), seed=9)
        result = pca_factor_covariance(returns, n_factors=4)
        self.assertEqual(result.factor_loadings.shape, (15, 4))

    def test_rejects_n_factors_out_of_range(self):
        returns = _simulate_returns(200, 15, np.eye(15), seed=10)
        with self.assertRaises(ValueError):
            pca_factor_covariance(returns, n_factors=0)
        with self.assertRaises(ValueError):
            pca_factor_covariance(returns, n_factors=15)


class TestEigenvalueDiagnostics(unittest.TestCase):
    def test_well_conditioned_matrix_has_reasonable_condition_number(self):
        cov = pd.DataFrame(np.eye(5))
        diagnostics = eigenvalue_diagnostics(cov)
        self.assertAlmostEqual(diagnostics["condition_number"], 1.0, places=8)

    def test_singular_matrix_has_infinite_condition_number(self):
        # A rank-deficient covariance (duplicate columns) has a zero eigenvalue.
        values = np.array([[1.0, 1.0], [1.0, 1.0]])
        cov = pd.DataFrame(values)
        diagnostics = eigenvalue_diagnostics(cov)
        self.assertEqual(diagnostics["condition_number"], float("inf"))


if __name__ == "__main__":
    unittest.main()
