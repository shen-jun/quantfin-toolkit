import unittest

import numpy as np

from quantfin.stochastic_processes import (
    simulate_geometric_brownian_motion,
    simulate_ornstein_uhlenbeck,
    simulate_vasicek,
    simulate_wiener_process,
)


class TestStochasticProcesses(unittest.TestCase):
    def test_wiener_process_starts_at_zero(self):
        t, w = simulate_wiener_process(n_steps=500, dt=0.01, seed=1)
        self.assertEqual(w[0], 0.0)
        self.assertEqual(len(t), 501)
        self.assertEqual(len(w), 501)

    def test_wiener_process_variance_grows_like_t(self):
        # Var(W(t)) = t. Not exact with one path, so we average many paths
        # and check we're in the right ballpark rather than pinning an exact
        # number (this is inherently a statistical test).
        n_steps, dt = 250, 1.0
        final_values = [simulate_wiener_process(n_steps, dt, seed=i)[1][-1] for i in range(500)]
        sample_variance = np.var(final_values)
        expected_variance = n_steps * dt
        self.assertAlmostEqual(sample_variance / expected_variance, 1.0, delta=0.15)

    def test_gbm_stays_positive_and_reproducible(self):
        _, path_a = simulate_geometric_brownian_motion(s0=100, mu=0.05, sigma=0.2, t=1.0, n_steps=252, seed=42)
        _, path_b = simulate_geometric_brownian_motion(s0=100, mu=0.05, sigma=0.2, t=1.0, n_steps=252, seed=42)
        self.assertTrue(np.all(path_a > 0))
        np.testing.assert_array_equal(path_a, path_b)

    def test_gbm_rejects_non_positive_start(self):
        with self.assertRaises(ValueError):
            simulate_geometric_brownian_motion(s0=0, mu=0.05, sigma=0.2, t=1.0, n_steps=10)

    def test_ou_process_reverts_toward_mean(self):
        path = simulate_ornstein_uhlenbeck(x0=10.0, theta=5.0, mu=0.0, sigma=0.01, n_steps=2000, dt=0.01, seed=7)
        # Started far from the mean with high reversion speed and low noise,
        # so it should end up much closer to mu=0 than where it started.
        self.assertLess(abs(path[-1]), abs(path[0]))

    def test_vasicek_matches_ou_with_renamed_arguments(self):
        _, rates = simulate_vasicek(r0=0.03, kappa=0.8, theta=0.05, sigma=0.01, t=5.0, n_steps=500, seed=3)
        ou_path = simulate_ornstein_uhlenbeck(x0=0.03, theta=0.8, mu=0.05, sigma=0.01, n_steps=500, dt=5.0 / 500, seed=3)
        np.testing.assert_allclose(rates, ou_path)


if __name__ == "__main__":
    unittest.main()
