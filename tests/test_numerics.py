import unittest

from quantfin._numerics import (
    bisection_solve,
    chi2_cdf_df1,
    chi2_cdf_df2,
    normal_cdf,
    normal_pdf,
    normal_ppf,
)


class TestNumerics(unittest.TestCase):
    def test_normal_cdf_known_values(self):
        self.assertAlmostEqual(normal_cdf(0.0), 0.5, places=8)
        self.assertAlmostEqual(normal_cdf(1.959963985), 0.975, places=6)
        self.assertAlmostEqual(normal_cdf(-1.959963985), 0.025, places=6)

    def test_normal_ppf_is_inverse_of_cdf(self):
        for p in (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999):
            z = normal_ppf(p)
            self.assertAlmostEqual(normal_cdf(z), p, places=8)

    def test_normal_ppf_matches_well_known_z_scores(self):
        self.assertAlmostEqual(normal_ppf(0.95), 1.644853627, places=5)
        self.assertAlmostEqual(normal_ppf(0.99), 2.326347874, places=5)

    def test_normal_ppf_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            normal_ppf(0.0)
        with self.assertRaises(ValueError):
            normal_ppf(1.0)

    def test_normal_pdf_peak_at_zero(self):
        self.assertAlmostEqual(normal_pdf(0.0), 1.0 / (2 * 3.141592653589793) ** 0.5, places=8)

    def test_chi2_cdf_df1_matches_normal_relationship(self):
        # For df=1, chi2_cdf(x) = P(Z^2 <= x) = 2*Phi(sqrt(x)) - 1.
        self.assertAlmostEqual(chi2_cdf_df1(3.841459), 0.95, places=4)

    def test_chi2_cdf_df2_is_exponential(self):
        self.assertAlmostEqual(chi2_cdf_df2(5.991465), 0.95, places=4)

    def test_bisection_solve_finds_root_of_simple_function(self):
        root = bisection_solve(lambda x: x ** 2 - 4, lower=0, upper=10)
        self.assertAlmostEqual(root, 2.0, places=6)

    def test_bisection_solve_rejects_same_sign_bracket(self):
        with self.assertRaises(ValueError):
            bisection_solve(lambda x: x ** 2 + 1, lower=0, upper=10)


if __name__ == "__main__":
    unittest.main()
