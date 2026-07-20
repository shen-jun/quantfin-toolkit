"""
VaR backtesting and model validation statistics: the Kupiec proportion-of-
failures test, the Christoffersen independence test (and their combination,
the conditional coverage test), and the bias statistic used to check
whether a risk model is systematically over- or under-forecasting
volatility.

This is the part of the package that turns "we computed a VaR number" into
"here's how we'd know if that VaR number is wrong" - which is really the
whole point of model validation as opposed to just model building.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantfin._numerics import chi2_cdf_df1, chi2_cdf_df2


@dataclass
class BacktestResult:
    statistic: float
    p_value: float
    reject_null: bool


def _safe_log(x: float, eps: float = 1e-12) -> float:
    return math.log(min(max(x, eps), 1.0 - eps))


def kupiec_pof_test(exceptions: int, n_obs: int, confidence_level: float) -> BacktestResult:
    """
    Kupiec's proportion-of-failures test: checks whether the number of VaR
    exceptions observed is consistent with the VaR model's stated
    confidence level, via a likelihood ratio test.

    Args:
        exceptions: Number of days the realized loss exceeded the VaR forecast.
        n_obs: Total number of days in the backtesting window.
        confidence_level: The VaR model's confidence level (e.g. 0.99).

    Returns:
        A BacktestResult. reject_null=True (at the 5% level) means the
        observed exception rate is statistically inconsistent with the
        model's stated confidence level - i.e. the model looks miscalibrated.
    """
    if n_obs <= 0:
        raise ValueError(f"n_obs must be positive, got {n_obs}")
    if not 0 <= exceptions <= n_obs:
        raise ValueError(f"exceptions must be between 0 and n_obs, got {exceptions} of {n_obs}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    expected_exception_rate = 1.0 - confidence_level
    observed_exception_rate = exceptions / n_obs

    log_likelihood_null = (n_obs - exceptions) * _safe_log(1 - expected_exception_rate) + exceptions * _safe_log(
        expected_exception_rate
    )
    log_likelihood_alt = (n_obs - exceptions) * _safe_log(1 - observed_exception_rate) + exceptions * _safe_log(
        observed_exception_rate
    )

    statistic = max(-2.0 * (log_likelihood_null - log_likelihood_alt), 0.0)
    p_value = 1.0 - chi2_cdf_df1(statistic)

    return BacktestResult(statistic=statistic, p_value=p_value, reject_null=p_value < 0.05)


def christoffersen_independence_test(exception_indicator: np.ndarray) -> BacktestResult:
    """
    Christoffersen's independence test: checks whether VaR exceptions are
    clustered in time (which would suggest the model is slow to react to
    changing volatility) rather than scattered independently, via a
    likelihood ratio test on a first-order Markov chain fitted to the
    exception sequence.

    Args:
        exception_indicator: Array of 0s and 1s, one per day, where 1 means
            the realized loss exceeded the VaR forecast that day.

    Returns:
        A BacktestResult. reject_null=True means exceptions are clustering
        rather than occurring independently over time.
    """
    indicator = np.asarray(exception_indicator).astype(int)
    if len(indicator) < 2:
        raise ValueError("need at least 2 observations to test for clustering")
    if not set(np.unique(indicator).tolist()).issubset({0, 1}):
        raise ValueError("exception_indicator must contain only 0s and 1s")

    n00 = n01 = n10 = n11 = 0
    for previous, current in zip(indicator[:-1], indicator[1:]):
        if previous == 0 and current == 0:
            n00 += 1
        elif previous == 0 and current == 1:
            n01 += 1
        elif previous == 1 and current == 0:
            n10 += 1
        else:
            n11 += 1

    n0 = n00 + n01
    n1 = n10 + n11
    total = n0 + n1

    prob_exception_after_no_exception = n01 / n0 if n0 > 0 else 0.0
    prob_exception_after_exception = n11 / n1 if n1 > 0 else 0.0
    pooled_exception_probability = (n01 + n11) / total if total > 0 else 0.0

    # Restricted model: exceptions are independent of yesterday's state
    # (same probability regardless of what happened the day before).
    log_likelihood_restricted = (n00 + n10) * _safe_log(1 - pooled_exception_probability) + (
        n01 + n11
    ) * _safe_log(pooled_exception_probability)

    # Unrestricted model: the exception probability is allowed to depend
    # on whether yesterday was an exception.
    log_likelihood_unrestricted = (
        n00 * _safe_log(1 - prob_exception_after_no_exception)
        + n01 * _safe_log(prob_exception_after_no_exception)
        + n10 * _safe_log(1 - prob_exception_after_exception)
        + n11 * _safe_log(prob_exception_after_exception)
    )

    statistic = max(-2.0 * (log_likelihood_restricted - log_likelihood_unrestricted), 0.0)
    p_value = 1.0 - chi2_cdf_df1(statistic)

    return BacktestResult(statistic=statistic, p_value=p_value, reject_null=p_value < 0.05)


def christoffersen_conditional_coverage_test(exception_indicator: np.ndarray, confidence_level: float) -> BacktestResult:
    """
    Combined test: adds the Kupiec (correct coverage) and Christoffersen
    (independence) likelihood ratio statistics together, which under the
    null is chi-square distributed with 2 degrees of freedom. This is the
    test to reach for if you want a single pass/fail check that covers both
    "is the exception rate right" and "are exceptions clustering".
    """
    indicator = np.asarray(exception_indicator).astype(int)
    exceptions = int(indicator.sum())
    n_obs = len(indicator)

    coverage_result = kupiec_pof_test(exceptions, n_obs, confidence_level)
    independence_result = christoffersen_independence_test(indicator)

    statistic = coverage_result.statistic + independence_result.statistic
    p_value = 1.0 - chi2_cdf_df2(statistic)

    return BacktestResult(statistic=statistic, p_value=p_value, reject_null=p_value < 0.05)


def bias_statistic(realized_returns: np.ndarray, predicted_volatility: np.ndarray, window: int = 12) -> np.ndarray:
    """
    Rolling bias statistic: the standard deviation of realized returns
    standardized by the model's predicted volatility, over a rolling
    window. If the risk model is well calibrated this should sit close to
    1.0 - persistently above 1 means the model is under-forecasting risk,
    persistently below 1 means it's over-forecasting.

    Args:
        realized_returns: Realized returns, one per period.
        predicted_volatility: The model's volatility forecast made at the
            start of each period, same length as realized_returns.
        window: Rolling window length, in periods.

    Returns:
        An array the same length as the inputs, with the first
        (window - 1) entries set to NaN (not enough history yet to compute
        a rolling statistic).
    """
    realized = np.asarray(realized_returns, dtype=float)
    predicted = np.asarray(predicted_volatility, dtype=float)

    if realized.shape != predicted.shape:
        raise ValueError("realized_returns and predicted_volatility must be the same length")
    if np.any(predicted <= 0):
        raise ValueError("predicted_volatility must be strictly positive everywhere")
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")
    if len(realized) < window:
        raise ValueError(f"series has {len(realized)} points, shorter than window={window}")

    standardized_returns = realized / predicted
    n = len(standardized_returns)
    result = np.full(n, np.nan)

    for t in range(window - 1, n):
        segment = standardized_returns[t - window + 1 : t + 1]
        result[t] = np.std(segment, ddof=1)

    return result
