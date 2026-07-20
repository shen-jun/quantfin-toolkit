import unittest

from quantfin.options import black_scholes as bs
from quantfin.options.binomial_tree import price_binomial_option


class TestBinomialTree(unittest.TestCase):
    def setUp(self):
        self.params = dict(spot=100.0, strike=100.0, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.2)

    def test_european_call_converges_to_black_scholes(self):
        tree_price = price_binomial_option(**self.params, n_steps=500, option_type="call", exercise="european")
        bs_price = bs.call_price(**self.params)
        self.assertAlmostEqual(tree_price, bs_price, places=2)

    def test_european_put_converges_to_black_scholes(self):
        tree_price = price_binomial_option(**self.params, n_steps=500, option_type="put", exercise="european")
        bs_price = bs.put_price(**self.params)
        self.assertAlmostEqual(tree_price, bs_price, places=2)

    def test_american_call_equals_european_when_no_dividends(self):
        # Well-known result: with no dividends, early exercise of an
        # American call is never optimal, so American == European.
        european = price_binomial_option(**self.params, n_steps=300, option_type="call", exercise="european")
        american = price_binomial_option(**self.params, n_steps=300, option_type="call", exercise="american")
        self.assertAlmostEqual(european, american, places=4)

    def test_american_put_worth_more_than_european_put(self):
        # Early exercise can be optimal for puts, so American >= European.
        european = price_binomial_option(**self.params, n_steps=300, option_type="put", exercise="european")
        american = price_binomial_option(**self.params, n_steps=300, option_type="put", exercise="american")
        self.assertGreaterEqual(american, european - 1e-9)

    def test_rejects_bad_option_type(self):
        with self.assertRaises(ValueError):
            price_binomial_option(**self.params, n_steps=50, option_type="straddle")

    def test_rejects_bad_exercise_style(self):
        with self.assertRaises(ValueError):
            price_binomial_option(**self.params, n_steps=50, exercise="bermudan")


if __name__ == "__main__":
    unittest.main()
