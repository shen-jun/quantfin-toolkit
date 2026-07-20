import unittest

from quantfin.options import black_scholes as bs


class TestBlackScholes(unittest.TestCase):
    def setUp(self):
        self.params = dict(spot=100.0, strike=100.0, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.2)

    def test_call_price_matches_known_textbook_value(self):
        # Standard textbook example (Hull): S=100, K=100, T=1, r=5%, sigma=20%.
        price = bs.call_price(**self.params)
        self.assertAlmostEqual(price, 10.4506, places=3)

    def test_put_price_matches_known_textbook_value(self):
        price = bs.put_price(**self.params)
        self.assertAlmostEqual(price, 5.5735, places=3)

    def test_put_call_parity_holds(self):
        c = bs.call_price(**self.params)
        p = bs.put_price(**self.params)
        s, k, t, r = self.params["spot"], self.params["strike"], self.params["time_to_maturity"], self.params["risk_free_rate"]
        import math
        self.assertAlmostEqual(c - p, s - k * math.exp(-r * t), places=6)

    def test_call_delta_between_zero_and_one(self):
        d = bs.delta(**self.params, option_type="call")
        self.assertTrue(0.0 < d < 1.0)

    def test_put_delta_between_minus_one_and_zero(self):
        d = bs.delta(**self.params, option_type="put")
        self.assertTrue(-1.0 < d < 0.0)

    def test_gamma_is_positive(self):
        self.assertGreater(bs.gamma(**self.params), 0.0)

    def test_vega_is_positive(self):
        self.assertGreater(bs.vega(**self.params), 0.0)

    def test_deep_itm_call_delta_approaches_one(self):
        params = dict(self.params)
        params["spot"] = 300.0
        self.assertGreater(bs.delta(**params, option_type="call"), 0.99)

    def test_rejects_non_positive_volatility(self):
        with self.assertRaises(ValueError):
            bs.call_price(spot=100, strike=100, time_to_maturity=1, risk_free_rate=0.05, volatility=0.0)

    def test_dividend_yield_reduces_call_price(self):
        no_div = bs.call_price(**self.params, dividend_yield=0.0)
        with_div = bs.call_price(**self.params, dividend_yield=0.03)
        self.assertLess(with_div, no_div)


if __name__ == "__main__":
    unittest.main()
