"""
Small internal numerics helpers used across the package.

This module exists mainly so we don't have to pull in SciPy as a hard
dependency just for a handful of things: the standard normal CDF/PDF/inverse
CDF, the chi-square CDF at 1 and 2 degrees of freedom (all we need for the
VaR backtests), and a plain bisection root finder for bond yields. Everything
here has a closed form or a well-known, well-tested approximation, so there's
no real accuracy tradeoff versus SciPy for what this library uses them for.

If you'd rather just depend on SciPy, swap these out for
`scipy.stats.norm` / `scipy.stats.chi2` / `scipy.optimize.brentq` - the
call signatures below were kept close to those on purpose.
"""

from __future__ import annotations

import math
from typing import Callable


def normal_cdf(x: float) -> float:
    """Standard normal CDF, computed exactly via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_ppf(p: float, tol: float = 1e-12, max_iter: int = 100) -> float:
    """
    Inverse standard normal CDF (quantile function), a.k.a. the z-score for
    a given probability. There's no closed form, so we start from a decent
    rational approximation (Acklam's algorithm) and polish it with a couple
    of Newton steps against the exact CDF/PDF above - that combination is
    accurate to essentially machine precision for our purposes.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in the open interval (0, 1), got {p}")

    # Acklam's rational approximation - gives a good starting guess.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    # Newton refinement against the exact CDF.
    for _ in range(max_iter):
        error = normal_cdf(x) - p
        if abs(error) < tol:
            break
        x -= error / normal_pdf(x)

    return x


def chi2_cdf_df1(x: float) -> float:
    """CDF of a chi-square distribution with 1 degree of freedom."""
    if x < 0:
        return 0.0
    return 2.0 * normal_cdf(math.sqrt(x)) - 1.0


def chi2_cdf_df2(x: float) -> float:
    """CDF of a chi-square distribution with 2 degrees of freedom (this one
    happens to have a very simple closed form: it's just an exponential)."""
    if x < 0:
        return 0.0
    return 1.0 - math.exp(-x / 2.0)


def bisection_solve(
    func: Callable[[float], float],
    lower: float,
    upper: float,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """
    Plain bisection root finder. We use this for things like bond yield to
    maturity, where the function is well-behaved (monotonic) and we'd rather
    have something boring and reliable than a fancier solver that can
    misbehave on a bad starting guess.
    """
    f_lower = func(lower)
    f_upper = func(upper)
    if f_lower == 0.0:
        return lower
    if f_upper == 0.0:
        return upper
    if f_lower * f_upper > 0.0:
        raise ValueError(
            "bisection_solve requires func(lower) and func(upper) to have "
            f"opposite signs; got f(lower)={f_lower:.6g}, f(upper)={f_upper:.6g}"
        )

    for _ in range(max_iter):
        midpoint = 0.5 * (lower + upper)
        f_mid = func(midpoint)

        if abs(f_mid) < tol or (upper - lower) / 2.0 < tol:
            return midpoint

        if f_lower * f_mid < 0.0:
            upper = midpoint
            f_upper = f_mid
        else:
            lower = midpoint
            f_lower = f_mid

    return 0.5 * (lower + upper)
