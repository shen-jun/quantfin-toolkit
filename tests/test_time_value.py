import math
import unittest

from quantfin.time_value import (
    future_value_continuous,
    future_value_discrete,
    present_value_continuous,
    present_value_discrete,
)


class TestTimeValue(unittest.TestCase):
    def test_future_value_discrete_matches_hand_calculation(self):
        # $100 at 5% for 5 years, compounded annually.
        result = future_value_discrete(100, 0.05, 5)
        self.assertAlmostEqual(result, 127.62815625, places=6)

    def test_present_value_discrete_is_inverse_of_future_value(self):
        fv = future_value_discrete(100, 0.05, 5)
        pv = present_value_discrete(fv, 0.05, 5)
        self.assertAlmostEqual(pv, 100.0, places=8)

    def test_future_value_continuous_matches_exp_formula(self):
        result = future_value_continuous(100, 0.05, 5)
        self.assertAlmostEqual(result, 100 * math.exp(0.25), places=8)

    def test_present_value_continuous_is_inverse_of_future_value(self):
        fv = future_value_continuous(100, 0.05, 5)
        pv = present_value_continuous(fv, 0.05, 5)
        self.assertAlmostEqual(pv, 100.0, places=8)

    def test_zero_rate_leaves_value_unchanged(self):
        self.assertAlmostEqual(future_value_discrete(100, 0.0, 10), 100.0)
        self.assertAlmostEqual(future_value_continuous(100, 0.0, 10), 100.0)


if __name__ == "__main__":
    unittest.main()
