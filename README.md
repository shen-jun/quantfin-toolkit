# quantfin-toolkit

A small Python library covering the quantitative finance methods I worked through during my CQF and PhD coursework: time value of money, fixed income pricing, option pricing, portfolio construction, and risk model validation.

I started this as a way to consolidate a pile of one-off tutorial scripts into something I could actually reuse, test, and reason about, rather than re-deriving the same formulas in a new notebook every time. Along the way I rebuilt most of it around a proper model-validation lens (VaR backtesting, stability checks, closed-form derivations instead of black-box calls) because that's the part of the material I cared about getting right, not just the pricing formulas themselves.

It is not a production risk system and isn't trying to be one - there's no live market data feed, no database layer, and the numerical methods are chosen for clarity over raw performance. What it does aim to get right is the math: every formula is either derived in the docstring or backed by a closed-form identity that's checked in the test suite.

## Why no SciPy

Everything statistical here - the normal CDF/PDF, the inverse normal CDF, chi-square CDFs, portfolio optimization - is implemented from scratch in plain NumPy rather than imported from `scipy.stats` / `scipy.optimize`. That started out as a constraint (no SciPy available in the environment I first wrote this in) but I kept it that way on purpose: writing Acklam's rational approximation for the inverse normal CDF, or deriving the closed-form KKT solution to the Markowitz problem instead of calling a generic optimizer, forces you to actually understand the method rather than trust a library. The tradeoff is that this code is not as numerically robust or fast as SciPy for edge cases - if you need that, swap the calls in `quantfin/_numerics.py` and `quantfin/portfolio.py` for the SciPy equivalents.

## What's in here

```
quantfin/
    _numerics.py            Normal CDF/PDF/inverse, chi-square CDFs, bisection solver
                             (internal helpers used everywhere else - see "Why no SciPy")
    time_value.py            Discrete and continuous compounding: future/present value
    stochastic_processes.py  Wiener process, GBM, Ornstein-Uhlenbeck, Vasicek short-rate
    fixed_income.py          Day count conventions, yield curves, bond pricing (clean/
                             dirty price, YTM, duration, convexity), a binomial short-rate
                             lattice, and callable/putable bond pricing off that lattice
    covariance.py            Sample covariance, Ledoit-Wolf shrinkage, PCA factor-model
                             covariance, eigenvalue diagnostics for condition number
    diagnostics.py           ECDF, QQ-plot, and a Jarque-Bera normality test summary
    portfolio.py             Mean-variance optimization (closed-form, incl. quadratic
                             transaction costs), efficient frontier, MCTR/CCTR risk
                             decomposition
    options/
        black_scholes.py      Closed-form price and Greeks
        binomial_tree.py       CRR tree, European and American exercise
        finite_difference.py   Explicit/implicit/Crank-Nicolson PDE solvers, with an
                               explicit stability check
        monte_carlo.py          Antithetic-variate Monte Carlo pricing
    risk/
        value_at_risk.py       Historical, parametric, and Monte Carlo VaR and
                               Expected Shortfall
        backtesting.py          Kupiec POF test, Christoffersen independence and
                               conditional coverage tests, bias statistic
        capm.py                  CAPM beta (two ways), expected return, risk premium

tests/       One test file per module, using the standard library's unittest
examples/    One runnable script per topic area, e.g. run_option_pricing.py
```

## Installation

```bash
git clone https://github.com/<your-username>/quantfin-toolkit.git
cd quantfin-toolkit
pip install -r requirements.txt
```

No `scipy` or `pytest` required - the test suite runs on the standard library's `unittest`.

## Quick start

```python
from quantfin.options import black_scholes as bs

price = bs.call_price(spot=100, strike=105, time_to_maturity=0.5, risk_free_rate=0.04, volatility=0.25)
delta = bs.delta(spot=100, strike=105, time_to_maturity=0.5, risk_free_rate=0.04, volatility=0.25, option_type="call")
print(f"price={price:.4f}, delta={delta:.4f}")
```

```python
from quantfin.risk.value_at_risk import historical_var
from quantfin.risk.backtesting import kupiec_pof_test

var_99 = historical_var(daily_returns, confidence_level=0.99)
result = kupiec_pof_test(n_exceptions=7, n_obs=750, confidence_level=0.99)
print(result.reject_null)  # did the model fail the backtest at the 5% level?
```

More complete, runnable walkthroughs live in `examples/` - one script per topic, each printing its results to the console:

| Script | What it covers |
|---|---|
| `run_bond_pricing.py` | Zero-coupon and coupon bond pricing, YTM, duration, convexity |
| `run_callable_bond_pricing.py` | Callable/putable bonds via the short-rate lattice |
| `run_option_pricing.py` | Black-Scholes vs. binomial tree vs. Monte Carlo, Greeks |
| `run_fdm_pricing.py` | Explicit/implicit/Crank-Nicolson PDE solvers, incl. an unstable-grid demo |
| `run_covariance_estimation.py` | Sample covariance vs. Ledoit-Wolf vs. PCA factor model |
| `run_portfolio_optimization.py` | Efficient frontier, transaction-cost-aware optimization, MCTR/CCTR |
| `run_var_backtesting.py` | VaR/ES estimation and a full Kupiec/Christoffersen backtest |
| `run_capm_analysis.py` | CAPM beta, expected return, and risk-aversion-implied risk premium |

Run any of them with, e.g.:

```bash
PYTHONPATH=. python examples/run_option_pricing.py
```

## Running the tests

```bash
python -m unittest discover -s tests -v
```

The tests aren't just smoke tests - most modules are checked against known closed-form identities (put-call parity, Euler's theorem for risk decomposition, telescoping discount-factor identities for the bond lattice) or hand-calculated reference values, not just "does it run without crashing."

## A couple of known simplifications

- The short-rate lattice in `fixed_income.py` uses a simplified forward-rate-based drift rather than a fully calibrated Ho-Lee or Black-Derman-Toy tree. It reproduces the input yield curve correctly but isn't meant as a production term-structure model.
- `eigenvalue_diagnostics` in `covariance.py` reports condition number and extreme eigenvalues as a diagnostic, not a full eigenfactor/optimization-bias correction - it's there to flag when a covariance matrix is heading into ill-conditioned territory, not to fix it automatically.
- Transaction costs in `portfolio.py` are modeled as quadratic in trade size, which keeps the optimization closed-form but is a simplification of real (often piecewise-linear, with a fixed component) trading cost.