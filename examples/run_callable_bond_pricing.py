"""
Callable/putable bond pricing off a short-rate binomial lattice, compared
against the price of a plain (non-callable) bond with the same cash flows.

Run with: python examples/run_callable_bond_pricing.py
"""

from datetime import date

from quantfin.fixed_income import (
    BinomialShortRateLattice,
    CallableBond,
    CouponBond,
    PutableBond,
    YieldCurve,
    map_coupons_to_steps,
)


def main() -> None:
    curve = YieldCurve(tenors=[0.5, 1, 2, 5, 10], rates=[0.04, 0.042, 0.044, 0.045, 0.046])
    valuation_date = date(2024, 1, 1)
    lattice = BinomialShortRateLattice(curve, volatility=0.012, horizon=10, n_steps=120)

    plain = CouponBond(
        face_value=1000, coupon_rate=0.055, issue_date=date(2020, 1, 1), maturity_date=date(2034, 1, 1), frequency=2
    )
    plain_price_curve = plain.dirty_price(curve, valuation_date)

    # For a fair comparison against the callable/putable prices below (which
    # go through the lattice), also price the same plain bond through the
    # lattice with no early exercise. These won't match plain_price_curve
    # exactly - discounting through a tree with up/down noise is not quite
    # the same as discounting deterministically off the curve, by a small
    # convexity effect - but they should be close.
    plain_cash_flows = map_coupons_to_steps(plain, valuation_date, lattice)
    plain_price_lattice = lattice.price_cash_flows(cash_flows=plain_cash_flows, terminal_value=plain.face_value)

    print(f"Plain bond price (curve discounting):        {plain_price_curve:.2f}")
    print(f"Plain bond price (same lattice, no options): {plain_price_lattice:.2f}  <-- fair baseline for below")

    callable_bond = CallableBond(
        face_value=1000, coupon_rate=0.055, issue_date=date(2020, 1, 1), maturity_date=date(2034, 1, 1), frequency=2
    )
    callable_price = callable_bond.price(
        lattice, valuation_date=valuation_date, call_price=1020.0, call_start_date=date(2027, 1, 1)
    )
    print(f"Callable bond price (callable from 2027):    {callable_price:.2f}  <-- worth less: issuer can call it away")

    putable_bond = PutableBond(
        face_value=1000, coupon_rate=0.055, issue_date=date(2020, 1, 1), maturity_date=date(2034, 1, 1), frequency=2
    )
    putable_price = putable_bond.price(
        lattice, valuation_date=valuation_date, put_price=980.0, put_start_date=date(2027, 1, 1)
    )
    print(f"Putable bond price (putable from 2027):      {putable_price:.2f}  <-- worth more: holder has a floor")


if __name__ == "__main__":
    main()
