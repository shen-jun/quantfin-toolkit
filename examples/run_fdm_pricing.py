"""
Price the same option with all three finite difference schemes (explicit,
implicit, Crank-Nicolson) and compare against Black-Scholes - including a
deliberately unstable explicit setup, to show the warning it triggers.

Run with: python examples/run_fdm_pricing.py
"""

import warnings

from quantfin.options import black_scholes as bs
from quantfin.options.finite_difference import FiniteDifferenceOptionPricer


def main() -> None:
    params = dict(spot=100.0, strike=100.0, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.2)
    bs_price = bs.call_price(**params)
    print(f"Black-Scholes closed form: {bs_price:.4f}\n")

    # This grid is stable for all three schemes (dt is comfortably below the
    # explicit scheme's stability threshold given n_price_steps=60).
    pricer = FiniteDifferenceOptionPricer(**params, option_type="call", n_price_steps=60, n_time_steps=400)
    for method in ("explicit", "implicit", "crank_nicolson"):
        price = pricer.price(method=method)
        if abs(price) < 1e6:
            print(f"{method:>15}: {price:.4f}  (error vs. Black-Scholes: {price - bs_price:+.4f})")
        else:
            # This grid is stable for implicit/Crank-Nicolson but not for explicit;
            # a blown-up explicit result can be astronomically large.
            print(f"{method:>15}: {price:.4e}  (diverged - see the stability warning above)")

    print("\nNow with a grid that violates the explicit scheme's stability condition")
    print("(too many price steps relative to time steps):")
    unstable_pricer = FiniteDifferenceOptionPricer(**params, option_type="call", n_price_steps=300, n_time_steps=50)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unstable_price = unstable_pricer.price(method="explicit")
        for w in caught:
            print(f"  Warning: {w.message}")
    # The blown-up result can be astronomically large once the scheme diverges,
    # so use scientific notation here rather than fixed-point.
    print(f"  Result: {unstable_price:.4e} (should be nowhere near {bs_price:.4f} - this explosion is exactly "
          "the failure mode implicit/Crank-Nicolson exist to avoid)")


if __name__ == "__main__":
    main()
