"""
Simple statistical diagnostics for checking whether a returns series looks
normally distributed - which is an assumption a lot of the rest of this
package (parametric VaR, CAPM, Black-Scholes) leans on, and which real
returns data reliably violates to some degree (fat tails, skew).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from quantfin._numerics import chi2_cdf_df2, normal_ppf

try:
    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - matplotlib is a listed dependency, but we
    Axes = None       # don't want an import error here to break non-plotting usage.
    plt = None


def empirical_cdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical CDF of a 1-D sample.

    Returns:
        A tuple (sorted_values, cumulative_probabilities), suitable for
        plotting as a step function.
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("data must be a 1-D array")
    if len(data) == 0:
        raise ValueError("data must not be empty")

    sorted_values = np.sort(data)
    n = len(sorted_values)
    cumulative_probabilities = np.arange(1, n + 1) / n
    return sorted_values, cumulative_probabilities


def plot_ecdf(data: np.ndarray, ax: Optional["Axes"] = None) -> "Axes":
    """Plot the empirical CDF of `data` as a step function."""
    if plt is None:
        raise ImportError("matplotlib is required for plotting functions")

    x, y = empirical_cdf(data)
    if ax is None:
        _, ax = plt.subplots()

    ax.step(x, y, where="post")
    ax.set_xlabel("value")
    ax.set_ylabel("cumulative probability")
    ax.set_title("Empirical CDF")
    return ax


def qq_plot(data: np.ndarray, ax: Optional["Axes"] = None) -> "Axes":
    """
    Normal Q-Q plot: sample quantiles against the quantiles you'd expect
    from a normal distribution with the same mean and standard deviation.
    Points falling on the diagonal line mean "looks normal"; deviation at
    the ends is the usual sign of fat tails.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting functions")

    data = np.asarray(data)
    if len(data) < 2:
        raise ValueError("need at least 2 observations for a Q-Q plot")

    sorted_data = np.sort(data)
    n = len(sorted_data)
    mean, std = sorted_data.mean(), sorted_data.std(ddof=1)

    # Standard plotting-position convention (Blom-ish): (i - 0.5) / n.
    plotting_positions = (np.arange(1, n + 1) - 0.5) / n
    theoretical_quantiles = np.array([normal_ppf(p) for p in plotting_positions])
    theoretical_values = mean + std * theoretical_quantiles

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(theoretical_values, sorted_data, s=12)
    lo = min(theoretical_values.min(), sorted_data.min())
    hi = max(theoretical_values.max(), sorted_data.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="red", linewidth=1)
    ax.set_xlabel("theoretical normal quantiles")
    ax.set_ylabel("sample quantiles")
    ax.set_title("Normal Q-Q Plot")
    return ax


def normality_test_summary(data: np.ndarray) -> Dict[str, float]:
    """
    Skew, excess kurtosis, and a Jarque-Bera test for normality.

    Returns:
        A dict with "skewness", "excess_kurtosis", "jarque_bera_stat", and
        "jarque_bera_p_value". A small p-value is evidence against
        normality (the usual rule of thumb: reject at p < 0.05).
    """
    data = np.asarray(data)
    n = len(data)
    if n < 8:
        raise ValueError("Jarque-Bera needs a reasonable sample size (at least 8 points)")

    mean = data.mean()
    std = data.std(ddof=1)
    if std == 0:
        raise ValueError("data has zero variance; normality test is not meaningful")

    standardized = (data - mean) / std
    skewness = float(np.mean(standardized ** 3))
    excess_kurtosis = float(np.mean(standardized ** 4) - 3.0)

    jarque_bera_stat = (n / 6.0) * (skewness ** 2 + (excess_kurtosis ** 2) / 4.0)
    p_value = 1.0 - chi2_cdf_df2(jarque_bera_stat)

    return {
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "jarque_bera_stat": float(jarque_bera_stat),
        "jarque_bera_p_value": float(p_value),
    }
