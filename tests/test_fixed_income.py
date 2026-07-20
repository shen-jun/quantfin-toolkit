import math
import unittest
from datetime import date

from quantfin.fixed_income import (
    BinomialShortRateLattice,
    Bond,
    CallableBond,
    CouponBond,
    DayCountConvention,
    PutableBond,
    YieldCurve,
    ZeroCouponBond,
    year_fraction,
)


class TestYearFraction(unittest.TestCase):
    def test_act_365_one_year(self):
        frac = year_fraction(date(2024, 1, 1), date(2025, 1, 1), DayCountConvention.ACT_365)
        self.assertAlmostEqual(frac, 366 / 365, places=6)  # 2024 is a leap year

    def test_act_360(self):
        frac = year_fraction(date(2024, 1, 1), date(2024, 7, 1), DayCountConvention.ACT_360)
        self.assertAlmostEqual(frac, 182 / 360, places=6)

    def test_thirty_360_half_year(self):
        frac = year_fraction(date(2024, 1, 1), date(2024, 7, 1), DayCountConvention.THIRTY_360)
        self.assertAlmostEqual(frac, 0.5, places=8)

    def test_end_before_start_raises(self):
        with self.assertRaises(ValueError):
            year_fraction(date(2024, 6, 1), date(2024, 1, 1), DayCountConvention.ACT_365)


class TestYieldCurve(unittest.TestCase):
    def setUp(self):
        self.curve = YieldCurve(tenors=[0.5, 1, 2, 5, 10], rates=[0.03, 0.032, 0.035, 0.04, 0.042])

    def test_discount_factor_at_zero_is_one(self):
        self.assertEqual(self.curve.discount_factor(0), 1.0)

    def test_discount_factor_matches_exp_formula(self):
        t = 2.0
        expected = math.exp(-self.curve.spot_rate(t) * t)
        self.assertAlmostEqual(self.curve.discount_factor(t), expected, places=10)

    def test_flat_extrapolation_outside_range(self):
        self.assertEqual(self.curve.spot_rate(0.1), self.curve.spot_rate(0.5))
        self.assertEqual(self.curve.spot_rate(20), self.curve.spot_rate(10))

    def test_forward_rate_telescopes_to_spot_rate(self):
        # Splitting [0, 5] into [0, 2] and [2, 5] and compounding the two
        # forward rates should reproduce the 5-year spot rate exactly.
        f1 = self.curve.forward_rate(0, 2)
        f2 = self.curve.forward_rate(2, 5)
        reconstructed = (f1 * 2 + f2 * 3) / 5
        self.assertAlmostEqual(reconstructed, self.curve.spot_rate(5), places=8)

    def test_rejects_unsorted_tenors(self):
        with self.assertRaises(ValueError):
            YieldCurve(tenors=[1, 0.5, 2], rates=[0.03, 0.03, 0.03])

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            YieldCurve(tenors=[1, 2], rates=[0.03])


class TestBondPricing(unittest.TestCase):
    def setUp(self):
        # A flat curve makes it easy to hand-check present values.
        self.flat_curve = YieldCurve(tenors=[0.5, 30], rates=[0.05, 0.05])
        self.valuation_date = date(2024, 1, 1)

    def test_zero_coupon_bond_price_matches_discount_factor(self):
        bond = ZeroCouponBond(face_value=1000, issue_date=date(2020, 1, 1), maturity_date=date(2029, 1, 1))
        t = year_fraction(self.valuation_date, date(2029, 1, 1), DayCountConvention.ACT_365)
        expected = 1000 * self.flat_curve.discount_factor(t)
        self.assertAlmostEqual(bond.dirty_price(self.flat_curve, self.valuation_date), expected, places=4)

    def test_zero_coupon_bond_has_no_accrued_interest(self):
        bond = ZeroCouponBond(face_value=1000, issue_date=date(2020, 1, 1), maturity_date=date(2029, 1, 1))
        self.assertEqual(bond.accrued_interest(self.valuation_date), 0.0)
        self.assertAlmostEqual(
            bond.clean_price(self.flat_curve, self.valuation_date),
            bond.dirty_price(self.flat_curve, self.valuation_date),
            places=8,
        )

    def test_dirty_price_exceeds_clean_price_between_coupons(self):
        bond = CouponBond(
            face_value=1000,
            coupon_rate=0.05,
            issue_date=date(2022, 1, 1),
            maturity_date=date(2032, 1, 1),
            frequency=2,
        )
        # A couple of months after a coupon date, there should be some
        # accrued interest sitting between clean and dirty price.
        settlement = date(2024, 3, 1)
        dirty = bond.dirty_price(self.flat_curve, settlement)
        clean = bond.clean_price(self.flat_curve, settlement)
        self.assertGreater(dirty, clean)
        self.assertGreater(bond.accrued_interest(settlement), 0.0)

    def test_accrued_interest_is_zero_right_at_a_coupon_date(self):
        bond = CouponBond(
            face_value=1000, coupon_rate=0.05, issue_date=date(2022, 1, 1),
            maturity_date=date(2032, 1, 1), frequency=2,
        )
        coupon_dates = bond.coupon_dates()
        # Settling exactly on a coupon date should mean the previous coupon
        # period just finished, so nothing has accrued for the *next* one yet.
        self.assertAlmostEqual(bond.accrued_interest(coupon_dates[2]), 0.0, places=6)

    def test_yield_to_maturity_round_trips_through_price(self):
        bond = CouponBond(
            face_value=1000, coupon_rate=0.04, issue_date=date(2022, 1, 1),
            maturity_date=date(2032, 1, 1), frequency=2,
        )
        settlement = date(2024, 1, 1)
        observed_price = 950.0
        ytm = bond.yield_to_maturity(observed_price, settlement)

        flows = bond._remaining_cash_flows(settlement)
        repriced = sum(cf / (1 + ytm / bond.frequency) ** (t * bond.frequency) for t, cf in flows)
        self.assertAlmostEqual(repriced, observed_price, places=4)

    def test_zero_coupon_bond_macaulay_duration_equals_maturity(self):
        bond = ZeroCouponBond(face_value=1000, issue_date=date(2024, 1, 1), maturity_date=date(2034, 1, 1))
        settlement = date(2024, 1, 1)
        price = bond.dirty_price(self.flat_curve, settlement)
        duration = bond.macaulay_duration(price, settlement)
        expected_t = year_fraction(settlement, date(2034, 1, 1), DayCountConvention.ACT_365)
        # A zero has exactly one cash flow, so duration must equal time to
        # that cash flow.
        self.assertAlmostEqual(duration, expected_t, places=3)

    def test_modified_duration_is_smaller_than_macaulay_duration(self):
        bond = CouponBond(
            face_value=1000, coupon_rate=0.05, issue_date=date(2022, 1, 1),
            maturity_date=date(2032, 1, 1), frequency=2,
        )
        settlement = date(2024, 1, 1)
        price = bond.dirty_price(self.flat_curve, settlement)
        mac = bond.macaulay_duration(price, settlement)
        mod = bond.modified_duration(price, settlement)
        self.assertLess(mod, mac)

    def test_bond_rejects_invalid_construction(self):
        with self.assertRaises(ValueError):
            Bond(face_value=-100, coupon_rate=0.05, issue_date=date(2020, 1, 1), maturity_date=date(2025, 1, 1))
        with self.assertRaises(ValueError):
            Bond(face_value=100, coupon_rate=0.05, issue_date=date(2025, 1, 1), maturity_date=date(2020, 1, 1))
        with self.assertRaises(ValueError):
            Bond(face_value=100, coupon_rate=0.05, issue_date=date(2020, 1, 1), maturity_date=date(2025, 1, 1), frequency=3)


class TestBinomialShortRateLattice(unittest.TestCase):
    def setUp(self):
        self.curve = YieldCurve(tenors=[0.5, 30], rates=[0.05, 0.05])
        self.valuation_date = date(2024, 1, 1)

    def test_tree_shape(self):
        lattice = BinomialShortRateLattice(self.curve, volatility=0.01, horizon=5, n_steps=10)
        rates = lattice.build()
        self.assertEqual(len(rates), 10)
        for i, level in enumerate(rates):
            self.assertEqual(len(level), i + 1)

    def test_zero_volatility_tree_matches_flat_curve_discounting_exactly(self):
        # With volatility=0 every node at a given step has the same rate,
        # so the tree collapses to plain discounting - and because forward
        # rates telescope exactly, this should match the curve's discount
        # factor to machine precision, regardless of how many steps we use.
        lattice = BinomialShortRateLattice(self.curve, volatility=0.0, horizon=5, n_steps=20)
        price = lattice.price_cash_flows(cash_flows={}, terminal_value=1000.0)
        expected = 1000.0 * self.curve.discount_factor(5)
        self.assertAlmostEqual(price, expected, places=6)

    def test_callable_bond_price_never_exceeds_call_price_once_callable(self):
        bond = CallableBond(
            face_value=1000, coupon_rate=0.06, issue_date=date(2020, 1, 1),
            maturity_date=date(2030, 1, 1), frequency=2,
        )
        lattice = BinomialShortRateLattice(self.curve, volatility=0.01, horizon=6, n_steps=60)
        price = bond.price(
            lattice, valuation_date=self.valuation_date, call_price=1020.0, call_start_date=date(2024, 1, 1)
        )
        self.assertLessEqual(price, 1020.0 + 1e-6)

    def test_call_clamp_applied_only_on_last_step_matches_closed_form(self):
        # With volatility=0 the tree is deterministic, and if the call
        # constraint only binds on the very last step before maturity, the
        # value at t=0 is just that clamped amount discounted back through
        # the (unclamped) earlier steps - which we can check in closed form
        # against the curve's own discount factor.
        n_steps = 24
        horizon = 6.0
        lattice = BinomialShortRateLattice(self.curve, volatility=0.0, horizon=horizon, n_steps=n_steps)
        call_price = 1.0
        price = lattice.price_cash_flows(
            cash_flows={},
            terminal_value=1000.0,
            exercise="call",
            exercise_price=call_price,
            exercise_start_step=n_steps - 1,
        )
        dt = horizon / n_steps
        expected = call_price * self.curve.discount_factor((n_steps - 1) * dt)
        self.assertAlmostEqual(price, expected, places=6)

    def test_putable_bond_price_never_below_put_price_once_putable(self):
        bond = PutableBond(
            face_value=1000, coupon_rate=0.03, issue_date=date(2020, 1, 1),
            maturity_date=date(2030, 1, 1), frequency=2,
        )
        lattice = BinomialShortRateLattice(self.curve, volatility=0.01, horizon=6, n_steps=60)
        price = bond.price(
            lattice, valuation_date=self.valuation_date, put_price=980.0, put_start_date=date(2024, 1, 1)
        )
        self.assertGreaterEqual(price, 980.0 - 1e-6)


if __name__ == "__main__":
    unittest.main()
