"""
Bond pricing walkthrough: build a yield curve, price a zero-coupon bond and
a coupon bond off it, and look at clean vs. dirty price and a few risk
measures (duration, convexity).

Run with: python examples/run_bond_pricing.py
"""

from datetime import date

from quantfin.fixed_income import CouponBond, DayCountConvention, YieldCurve, ZeroCouponBond


def main() -> None:
    curve = YieldCurve(tenors=[0.5, 1, 2, 5, 10, 30], rates=[0.045, 0.047, 0.048, 0.045, 0.043, 0.042])
    valuation_date = date(2024, 6, 1)

    print("=== Zero-coupon bond ===")
    zero = ZeroCouponBond(face_value=1000, issue_date=date(2022, 1, 1), maturity_date=date(2034, 1, 1))
    zero_price = zero.dirty_price(curve, valuation_date)
    print(f"Price: {zero_price:.2f}")
    print(f"YTM:   {zero.yield_to_maturity(zero_price, valuation_date):.4%}")

    print("\n=== 5% semi-annual coupon bond ===")
    bond = CouponBond(
        face_value=1000,
        coupon_rate=0.05,
        issue_date=date(2020, 3, 15),
        maturity_date=date(2030, 3, 15),
        frequency=2,
        day_count=DayCountConvention.THIRTY_360,
    )
    dirty = bond.dirty_price(curve, valuation_date)
    clean = bond.clean_price(curve, valuation_date)
    accrued = bond.accrued_interest(valuation_date)
    print(f"Dirty price:  {dirty:.2f}")
    print(f"Accrued int.: {accrued:.2f}")
    print(f"Clean price:  {clean:.2f}")

    ytm = bond.yield_to_maturity(dirty, valuation_date)
    print(f"YTM:               {ytm:.4%}")
    print(f"Macaulay duration: {bond.macaulay_duration(dirty, valuation_date):.3f} years")
    print(f"Modified duration: {bond.modified_duration(dirty, valuation_date):.3f}")
    print(f"Convexity:         {bond.convexity(dirty, valuation_date):.3f}")


if __name__ == "__main__":
    main()
