"""
Compare option pricing methods against each other: closed-form
Black-Scholes, a binomial tree, and Monte Carlo, for the same option.

Run with: python examples/run_option_pricing.py
"""

from quantfin.options import black_scholes as bs
from quantfin.options.binomial_tree import price_binomial_option
from quantfin.options.monte_carlo import price_option_monte_carlo


def main() -> None:
    params = dict(spot=100.0, strike=105.0, time_to_maturity=0.5, risk_free_rate=0.04, volatility=0.25)

    print("European call, S=100, K=105, T=0.5y, r=4%, sigma=25%\n")

    bs_price = bs.call_price(**params)
    print(f"Black-Scholes (closed form):     {bs_price:.4f}")

    tree_price = price_binomial_option(**params, n_steps=500, option_type="call", exercise="european")
    print(f"Binomial tree (500 steps):       {tree_price:.4f}")

    mc_price, mc_se = price_option_monte_carlo(**params, option_type="call", n_simulations=500_000, seed=1)
    print(f"Monte Carlo (500k paths):        {mc_price:.4f}  (+/- {2 * mc_se:.4f} at ~95% confidence)")

    print("\nGreeks (from the closed-form formulas):")
    print(f"  delta: {bs.delta(**params, option_type='call'):.4f}")
    print(f"  gamma: {bs.gamma(**params):.4f}")
    print(f"  vega:  {bs.vega(**params):.4f}")
    print(f"  theta: {bs.theta(**params, option_type='call'):.4f} per year "
          f"({bs.theta(**params, option_type='call') / 365:.4f} per day)")

    print("\nAmerican vs. European put (early exercise can matter for puts):")
    european_put = price_binomial_option(**params, n_steps=500, option_type="put", exercise="european")
    american_put = price_binomial_option(**params, n_steps=500, option_type="put", exercise="american")
    print(f"  European put: {european_put:.4f}")
    print(f"  American put: {american_put:.4f}")


if __name__ == "__main__":
    main()
