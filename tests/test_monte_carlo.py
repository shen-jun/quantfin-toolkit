import unittest

from quantfin.options import black_scholes as bs
from quantfin.options.monte_carlo import price_option_monte_carlo


class TestMonteCarloOptionPricing(unittest.TestCase):
    def setUp(self):
        self.params = dict(spot=100.0, strike=100.0, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.2)

    def test_call_price_within_a_few_standard_errors_of_black_scholes(self):
        price, se = price_option_monte_carlo(**self.params, option_type="call", n_simulations=200_000, seed=1)
        bs_price = bs.call_price(**self.params)
        self.assertLess(abs(price - bs_price), 4 * se)

    def test_put_price_within_a_few_standard_errors_of_black_scholes(self):
        price, se = price_option_monte_carlo(**self.params, option_type="put", n_simulations=200_000, seed=1)
        bs_price = bs.put_price(**self.params)
        self.assertLess(abs(price - bs_price), 4 * se)

    def test_same_seed_gives_same_result(self):
        price_a, _ = price_option_monte_carlo(**self.params, seed=123, n_simulations=1000)
        price_b, _ = price_option_monte_carlo(**self.params, seed=123, n_simulations=1000)
        self.assertEqual(price_a, price_b)

    def test_antithetic_variates_reduce_standard_error(self):
        _, se_plain = price_option_monte_carlo(**self.params, n_simulations=20_000, antithetic=False, seed=1)
        _, se_antithetic = price_option_monte_carlo(**self.params, n_simulations=20_000, antithetic=True, seed=1)
        self.assertLess(se_antithetic, se_plain)

    def test_rejects_non_positive_simulations(self):
        with self.assertRaises(ValueError):
            price_option_monte_carlo(**self.params, n_simulations=0)


if __name__ == "__main__":
    unittest.main()
