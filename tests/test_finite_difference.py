import unittest
import warnings

from quantfin.options import black_scholes as bs
from quantfin.options.finite_difference import FiniteDifferenceOptionPricer


class TestFiniteDifference(unittest.TestCase):
    def setUp(self):
        self.params = dict(spot=100.0, strike=100.0, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.2)
        self.bs_call = bs.call_price(**self.params)
        self.bs_put = bs.put_price(**self.params)

    def test_implicit_call_close_to_black_scholes(self):
        pricer = FiniteDifferenceOptionPricer(**self.params, option_type="call", n_price_steps=160, n_time_steps=160)
        price = pricer.price(method="implicit")
        self.assertAlmostEqual(price, self.bs_call, delta=0.15)

    def test_crank_nicolson_call_close_to_black_scholes(self):
        pricer = FiniteDifferenceOptionPricer(**self.params, option_type="call", n_price_steps=160, n_time_steps=160)
        price = pricer.price(method="crank_nicolson")
        self.assertAlmostEqual(price, self.bs_call, delta=0.15)

    def test_crank_nicolson_put_close_to_black_scholes(self):
        pricer = FiniteDifferenceOptionPricer(**self.params, option_type="put", n_price_steps=160, n_time_steps=160)
        price = pricer.price(method="crank_nicolson")
        self.assertAlmostEqual(price, self.bs_put, delta=0.15)

    def test_explicit_scheme_close_to_black_scholes_when_grid_is_stable(self):
        # Deliberately chosen so the CFL-style stability condition holds:
        # few price steps, lots of time steps.
        pricer = FiniteDifferenceOptionPricer(**self.params, option_type="call", n_price_steps=40, n_time_steps=4000)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # fail the test if we get a stability warning we didn't expect
            price = pricer.price(method="explicit")
        self.assertAlmostEqual(price, self.bs_call, delta=0.3)

    def test_explicit_scheme_warns_when_grid_is_unstable(self):
        pricer = FiniteDifferenceOptionPricer(**self.params, option_type="call", n_price_steps=200, n_time_steps=50)
        with self.assertWarns(UserWarning):
            pricer.price(method="explicit")

    def test_rejects_bad_method_name(self):
        pricer = FiniteDifferenceOptionPricer(**self.params, n_price_steps=50, n_time_steps=50)
        with self.assertRaises(ValueError):
            pricer.price(method="spectral")

    def test_rejects_too_coarse_a_grid(self):
        with self.assertRaises(ValueError):
            FiniteDifferenceOptionPricer(**self.params, n_price_steps=3, n_time_steps=50)


if __name__ == "__main__":
    unittest.main()
