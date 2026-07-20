"""
Fixed income pricing: a term-structure-aware bond pricer that separates
clean and dirty price properly, supports a couple of standard day count
conventions, and can price simple callable/putable bonds off a short-rate
binomial tree.

Scope note: the callable/putable bond pricer here uses a short-rate tree
whose drift at each step is read straight off the curve's forward rates,
spread up/down by the given volatility. That's a simplified, uncalibrated
tree - it prices embedded optionality using the right mechanics (backward
induction with early exercise), but a production system would calibrate the
tree's drift term by term so that it reprices the whole input curve exactly
(the classic Ho-Lee / Black-Derman-Toy style calibration). Doing that
properly is a project on its own, and it isn't needed to demonstrate the
pricing logic, so it's left out on purpose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np
from dateutil.relativedelta import relativedelta

from quantfin._numerics import bisection_solve


class DayCountConvention(Enum):
    """The handful of day count conventions this library knows about."""

    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"
    THIRTY_360 = "30/360"


def year_fraction(start: date, end: date, convention: DayCountConvention) -> float:
    """
    Fraction of a year between two dates under a given day count convention.

    Args:
        start: Start date.
        end: End date (must not be earlier than start).
        convention: Which day count convention to apply.

    Returns:
        Year fraction as a float (e.g. roughly 0.5 for six months).
    """
    if end < start:
        raise ValueError(f"end date {end} is earlier than start date {start}")

    if convention == DayCountConvention.ACT_360:
        return (end - start).days / 360.0

    if convention == DayCountConvention.ACT_365:
        return (end - start).days / 365.0

    if convention == DayCountConvention.THIRTY_360:
        d1, m1, y1 = start.day, start.month, start.year
        d2, m2, y2 = end.day, end.month, end.year
        # Standard 30/360 (Bond Basis) day-rolling rule.
        d1 = min(d1, 30)
        if d1 == 30:
            d2 = min(d2, 30)
        days = (y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)
        return days / 360.0

    raise ValueError(f"unsupported day count convention: {convention}")


@dataclass
class YieldCurve:
    """
    A simple zero curve: continuously compounded zero rates at a set of
    tenors, linearly interpolated in between and held flat outside the
    given range.

    Args:
        tenors: Tenor points in years, e.g. [0.25, 1, 2, 5, 10]. Must be
            sorted in ascending order.
        rates: Continuously compounded zero rates at each tenor, as
            decimals (5% -> 0.05). Must be the same length as `tenors`.
    """

    tenors: Sequence[float]
    rates: Sequence[float]

    def __post_init__(self) -> None:
        self.tenors = np.asarray(self.tenors, dtype=float)
        self.rates = np.asarray(self.rates, dtype=float)

        if len(self.tenors) == 0:
            raise ValueError("a yield curve needs at least one tenor/rate pair")
        if len(self.tenors) != len(self.rates):
            raise ValueError("tenors and rates must be the same length")
        if np.any(np.diff(self.tenors) <= 0):
            raise ValueError("tenors must be strictly increasing")

    def spot_rate(self, t: float) -> float:
        """Continuously compounded zero rate at time t (flat-extrapolated
        outside the curve's range)."""
        if t < 0:
            raise ValueError(f"t must be non-negative, got {t}")
        return float(np.interp(t, self.tenors, self.rates))

    def discount_factor(self, t: float) -> float:
        """Discount factor for a cash flow at time t."""
        if t < 0:
            raise ValueError(f"t must be non-negative, got {t}")
        if t == 0:
            return 1.0
        return math.exp(-self.spot_rate(t) * t)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Continuously compounded forward rate between t1 and t2 (t2 > t1),
        implied by the zero curve."""
        if t2 <= t1:
            raise ValueError(f"t2 ({t2}) must be greater than t1 ({t1})")
        return (self.spot_rate(t2) * t2 - self.spot_rate(t1) * t1) / (t2 - t1)


@dataclass
class Bond:
    """
    A fixed-rate coupon bond. `ZeroCouponBond` and `CouponBond` below are
    thin, more explicitly named wrappers around this same class.

    Args:
        face_value: Redemption / par amount.
        coupon_rate: Annual coupon rate as a decimal (0.05 for 5%). Use 0
            for a zero-coupon bond.
        issue_date: Date the bond was issued.
        maturity_date: Date the bond redeems.
        frequency: Number of coupon payments per year (1, 2, 4, or 12).
        day_count: Day count convention used for accrued interest.
    """

    face_value: float
    coupon_rate: float
    issue_date: date
    maturity_date: date
    frequency: int = 2
    day_count: DayCountConvention = DayCountConvention.THIRTY_360

    def __post_init__(self) -> None:
        if self.face_value <= 0:
            raise ValueError(f"face_value must be positive, got {self.face_value}")
        if self.coupon_rate < 0:
            raise ValueError(f"coupon_rate must be non-negative, got {self.coupon_rate}")
        if self.frequency not in (1, 2, 4, 12):
            raise ValueError(f"frequency must be one of 1, 2, 4, 12, got {self.frequency}")
        if self.maturity_date <= self.issue_date:
            raise ValueError("maturity_date must be after issue_date")

    def coupon_dates(self) -> List[date]:
        """Coupon payment dates, generated by stepping backward from
        maturity in increments of 12/frequency months."""
        months_per_period = 12 // self.frequency
        dates: List[date] = []
        current = self.maturity_date
        while current > self.issue_date:
            dates.append(current)
            current = current - relativedelta(months=months_per_period)
        dates.reverse()
        return dates

    def _coupon_amount(self) -> float:
        return self.face_value * self.coupon_rate / self.frequency

    def accrued_interest(self, settlement_date: date) -> float:
        """Interest accrued since the last coupon date, as of settlement_date."""
        if settlement_date >= self.maturity_date:
            return 0.0

        # Note the strict ">" here: if settlement_date lands exactly on a
        # coupon date, that coupon has just been paid, so accrual for the
        # *next* period hasn't started yet - accrued interest should be 0,
        # not a full period's worth.
        prev_date = self.issue_date
        next_date: Optional[date] = None
        for coupon_date in self.coupon_dates():
            if coupon_date > settlement_date:
                next_date = coupon_date
                break
            prev_date = coupon_date

        if next_date is None:
            return 0.0

        period_elapsed = year_fraction(prev_date, settlement_date, self.day_count)
        full_period = year_fraction(prev_date, next_date, self.day_count)
        if full_period <= 0:
            return 0.0

        return self._coupon_amount() * (period_elapsed / full_period)

    def dirty_price(self, curve: YieldCurve, settlement_date: date) -> float:
        """Present value of all remaining cash flows, discounted off `curve`."""
        coupon = self._coupon_amount()
        remaining_dates = [d for d in self.coupon_dates() if d > settlement_date]

        price = 0.0
        for i, coupon_date in enumerate(remaining_dates):
            t = year_fraction(settlement_date, coupon_date, DayCountConvention.ACT_365)
            cash_flow = coupon
            if i == len(remaining_dates) - 1:
                cash_flow += self.face_value
            price += cash_flow * curve.discount_factor(t)

        return price

    def clean_price(self, curve: YieldCurve, settlement_date: date) -> float:
        """Dirty price minus accrued interest - the price you'd see quoted."""
        return self.dirty_price(curve, settlement_date) - self.accrued_interest(settlement_date)

    def _remaining_cash_flows(self, settlement_date: date) -> List[tuple]:
        coupon = self._coupon_amount()
        remaining_dates = [d for d in self.coupon_dates() if d > settlement_date]
        flows = []
        for i, coupon_date in enumerate(remaining_dates):
            t = year_fraction(settlement_date, coupon_date, DayCountConvention.ACT_365)
            cash_flow = coupon
            if i == len(remaining_dates) - 1:
                cash_flow += self.face_value
            flows.append((t, cash_flow))
        return flows

    def yield_to_maturity(self, price: float, settlement_date: date) -> float:
        """
        Flat yield (compounded `frequency` times a year) that discounts the
        bond's remaining cash flows back to the given price.

        Args:
            price: Observed (dirty) price of the bond.
            settlement_date: Valuation / settlement date.

        Returns:
            Yield to maturity as a decimal.
        """
        flows = self._remaining_cash_flows(settlement_date)
        if not flows:
            raise ValueError("bond has no remaining cash flows after settlement_date")

        def price_error(y: float) -> float:
            total = sum(cf / (1.0 + y / self.frequency) ** (t * self.frequency) for t, cf in flows)
            return total - price

        return bisection_solve(price_error, lower=-0.5, upper=2.0)

    def macaulay_duration(self, price: float, settlement_date: date) -> float:
        """Macaulay duration: the cash-flow-weighted average time to
        receipt, discounting at the bond's own yield to maturity."""
        ytm = self.yield_to_maturity(price, settlement_date)
        flows = self._remaining_cash_flows(settlement_date)

        weighted_time = 0.0
        total_pv = 0.0
        for t, cf in flows:
            pv = cf / (1.0 + ytm / self.frequency) ** (t * self.frequency)
            weighted_time += t * pv
            total_pv += pv

        if total_pv == 0:
            raise ValueError("present value of remaining cash flows is zero")

        return weighted_time / total_pv

    def modified_duration(self, price: float, settlement_date: date) -> float:
        """Modified duration = Macaulay duration / (1 + y/frequency)."""
        ytm = self.yield_to_maturity(price, settlement_date)
        mac_duration = self.macaulay_duration(price, settlement_date)
        return mac_duration / (1.0 + ytm / self.frequency)

    def convexity(self, price: float, settlement_date: date) -> float:
        """Bond convexity, discounting at the bond's own yield to maturity."""
        ytm = self.yield_to_maturity(price, settlement_date)
        flows = self._remaining_cash_flows(settlement_date)

        weighted_convexity = 0.0
        total_pv = 0.0
        for t, cf in flows:
            pv = cf / (1.0 + ytm / self.frequency) ** (t * self.frequency)
            weighted_convexity += pv * t * (t + 1.0 / self.frequency)
            total_pv += pv

        if total_pv == 0:
            raise ValueError("present value of remaining cash flows is zero")

        return weighted_convexity / (total_pv * (1.0 + ytm / self.frequency) ** 2)


class ZeroCouponBond(Bond):
    """A Bond with coupon_rate fixed at 0 - just a named convenience so
    call sites don't have to remember to pass coupon_rate=0."""

    def __init__(
        self,
        face_value: float,
        issue_date: date,
        maturity_date: date,
        day_count: DayCountConvention = DayCountConvention.THIRTY_360,
    ) -> None:
        super().__init__(
            face_value=face_value,
            coupon_rate=0.0,
            issue_date=issue_date,
            maturity_date=maturity_date,
            frequency=1,
            day_count=day_count,
        )


class CouponBond(Bond):
    """Alias for Bond with a more descriptive name - a regular
    fixed-coupon bond, as opposed to ZeroCouponBond."""

    pass


class BinomialShortRateLattice:
    """
    A simple recombining binomial short-rate tree, used to price bonds with
    embedded call/put optionality. See the module docstring for the
    (deliberate) simplification this makes versus a fully calibrated tree.
    """

    def __init__(self, curve: YieldCurve, volatility: float, horizon: float, n_steps: int) -> None:
        if volatility < 0:
            raise ValueError(f"volatility must be non-negative, got {volatility}")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")

        self.curve = curve
        self.volatility = volatility
        self.horizon = horizon
        self.n_steps = n_steps
        self.dt = horizon / n_steps

    def build(self) -> List[List[float]]:
        """
        Build the short-rate tree. rates[i][j] is the short rate that
        applies over step i (from time i*dt to (i+1)*dt) at node j
        (j = 0 .. i, with j=0 the lowest rate at that step).
        """
        rates: List[List[float]] = []
        for i in range(self.n_steps):
            t1 = i * self.dt
            t2 = (i + 1) * self.dt
            drift = self.curve.spot_rate(t2) if t1 == 0 else self.curve.forward_rate(t1, t2)
            spread = self.volatility * math.sqrt(self.dt)
            level = [drift + (2 * j - i) * spread for j in range(i + 1)]
            rates.append(level)
        return rates

    def price_cash_flows(
        self,
        cash_flows: Dict[int, float],
        terminal_value: float,
        exercise: Optional[str] = None,
        exercise_price: Optional[float] = None,
        exercise_start_step: int = 0,
    ) -> float:
        """
        Backward-induct a schedule of cash flows through the tree.

        Args:
            cash_flows: Mapping of step index -> cash flow paid at that step
                (e.g. coupon payments, keyed by the tree step nearest to
                each coupon date).
            terminal_value: Extra amount paid at the final step (e.g. bond
                redemption), on top of anything in cash_flows[n_steps].
            exercise: None, "call", or "put". If set, the holder's payoff at
                each node from exercise_start_step onward is capped at
                (for "call") or floored at (for "put") exercise_price.
            exercise_price: Strike / call or put price.
            exercise_start_step: First step at which early exercise is
                allowed.

        Returns:
            Present value at step 0.
        """
        if exercise not in (None, "call", "put"):
            raise ValueError(f"exercise must be None, 'call', or 'put', got {exercise!r}")
        if exercise is not None and exercise_price is None:
            raise ValueError("exercise_price is required when exercise is set")

        rates = self.build()
        n = self.n_steps

        values = [terminal_value + cash_flows.get(n, 0.0)] * (n + 1)

        for i in range(n - 1, -1, -1):
            next_values = values
            values = []
            for j in range(i + 1):
                r = rates[i][j]
                continuation = 0.5 * (next_values[j] + next_values[j + 1]) * math.exp(-r * self.dt)
                node_value = continuation + cash_flows.get(i, 0.0)

                if exercise == "call" and i >= exercise_start_step:
                    node_value = min(node_value, exercise_price)
                elif exercise == "put" and i >= exercise_start_step:
                    node_value = max(node_value, exercise_price)

                values.append(node_value)

        return values[0]


def map_coupons_to_steps(
    bond: Bond, valuation_date: date, lattice: BinomialShortRateLattice
) -> Dict[int, float]:
    """Map each of a bond's remaining coupon dates onto the nearest step of
    a binomial short-rate lattice."""
    coupon = bond._coupon_amount()
    cash_flows: Dict[int, float] = {}
    for coupon_date in bond.coupon_dates():
        if coupon_date <= valuation_date:
            continue
        t = year_fraction(valuation_date, coupon_date, DayCountConvention.ACT_365)
        step = int(round(t / lattice.dt))
        step = max(0, min(step, lattice.n_steps))
        cash_flows[step] = cash_flows.get(step, 0.0) + coupon
    return cash_flows


class CallableBond(Bond):
    """A coupon bond the issuer can redeem early at `call_price` from
    `call_start_date` onward."""

    def price(
        self,
        lattice: BinomialShortRateLattice,
        valuation_date: date,
        call_price: float,
        call_start_date: date,
    ) -> float:
        cash_flows = map_coupons_to_steps(self, valuation_date, lattice)
        call_start_step = int(round(
            year_fraction(valuation_date, call_start_date, DayCountConvention.ACT_365) / lattice.dt
        ))
        return lattice.price_cash_flows(
            cash_flows=cash_flows,
            terminal_value=self.face_value,
            exercise="call",
            exercise_price=call_price,
            exercise_start_step=call_start_step,
        )


class PutableBond(Bond):
    """A coupon bond the holder can put back to the issuer at `put_price`
    from `put_start_date` onward."""

    def price(
        self,
        lattice: BinomialShortRateLattice,
        valuation_date: date,
        put_price: float,
        put_start_date: date,
    ) -> float:
        cash_flows = map_coupons_to_steps(self, valuation_date, lattice)
        put_start_step = int(round(
            year_fraction(valuation_date, put_start_date, DayCountConvention.ACT_365) / lattice.dt
        ))
        return lattice.price_cash_flows(
            cash_flows=cash_flows,
            terminal_value=self.face_value,
            exercise="put",
            exercise_price=put_price,
            exercise_start_step=put_start_step,
        )
