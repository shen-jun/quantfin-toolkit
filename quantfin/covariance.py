"""
Covariance matrix estimation for a panel of asset (or factor) returns.

The sample covariance matrix is the obvious starting point, but it's a bad
estimate whenever you don't have a lot more time periods than assets -
which, for anything like a real equity universe, is basically always: you
might have a few thousand names and only a few years of daily data. Two
standard fixes are here: Ledoit-Wolf shrinkage (blend the noisy sample
covariance with a more stable, structured target) and a PCA-based
statistical factor model (explain the covariance structure with a handful
of factors instead of estimating every pairwise entry directly).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Plain sample covariance matrix of a returns panel (T rows of time
    periods, N columns of assets)."""
    if returns.shape[0] < 2:
        raise ValueError("need at least 2 observations to estimate a covariance matrix")
    return returns.cov()


@dataclass
class ShrinkageResult:
    covariance: pd.DataFrame
    shrinkage_intensity: float
    target_used: str


def _identity_target(sample_cov: np.ndarray) -> np.ndarray:
    """Shrinkage target: a scaled identity matrix, using the average
    sample variance as the scale."""
    n = sample_cov.shape[0]
    average_variance = np.trace(sample_cov) / n
    return average_variance * np.eye(n)


def _constant_correlation_target(sample_cov: np.ndarray) -> np.ndarray:
    """Shrinkage target: keep the sample variances on the diagonal, but
    replace every pairwise correlation with the average correlation across
    the whole matrix. This is the target used in Ledoit & Wolf's original
    2004 paper."""
    std_devs = np.sqrt(np.diag(sample_cov))
    correlation = sample_cov / np.outer(std_devs, std_devs)

    n = correlation.shape[0]
    off_diagonal_sum = correlation.sum() - np.trace(correlation)
    average_correlation = off_diagonal_sum / (n * (n - 1))

    target_correlation = np.full((n, n), average_correlation)
    np.fill_diagonal(target_correlation, 1.0)

    return target_correlation * np.outer(std_devs, std_devs)


def ledoit_wolf_shrinkage(returns: pd.DataFrame, target: str = "constant_correlation") -> ShrinkageResult:
    """
    Shrink the sample covariance matrix toward a structured target, using
    the Ledoit-Wolf (2004) formula for the optimal shrinkage intensity.

    The idea: the sample covariance S is unbiased but noisy, and a
    structured target F is biased but stable. We blend them as
    rho*F + (1-rho)*S, choosing rho to minimize the expected squared
    (Frobenius norm) distance from the true (unknown) covariance matrix.
    That optimal rho works out to be the ratio of "how much sampling noise
    is in S" to "how far S is from the target" - see Ledoit & Wolf (2004),
    "Honey, I Shrunk the Sample Covariance Matrix", for the derivation.

    Args:
        returns: T x N panel of returns (rows = time, columns = assets).
        target: "constant_correlation" or "identity".

    Returns:
        A ShrinkageResult with the shrunk covariance matrix, the estimated
        shrinkage intensity (0 = no shrinkage, 1 = fully the target), and
        which target was used.
    """
    if target not in ("constant_correlation", "identity"):
        raise ValueError(f"target must be 'constant_correlation' or 'identity', got {target!r}")

    values = returns.values
    n_obs, n_assets = values.shape
    if n_obs < 2:
        raise ValueError("need at least 2 observations to estimate a covariance matrix")

    centered = values - values.mean(axis=0)
    sample_cov = (centered.T @ centered) / n_obs

    target_matrix = _identity_target(sample_cov) if target == "identity" else _constant_correlation_target(sample_cov)

    # Distance between the sample covariance and the target.
    distance_sq = np.sum((sample_cov - target_matrix) ** 2) / n_assets

    # Estimate of the sampling noise in the sample covariance: how much,
    # on average, each period's outer product x_t @ x_t.T disagrees with S.
    outer_products = np.einsum("ti,tj->tij", centered, centered)
    noise_sq = np.sum((outer_products - sample_cov) ** 2) / (n_obs ** 2 * n_assets)
    noise_sq = min(noise_sq, distance_sq)

    shrinkage_intensity = 0.0 if distance_sq == 0 else noise_sq / distance_sq
    shrunk_cov = shrinkage_intensity * target_matrix + (1.0 - shrinkage_intensity) * sample_cov

    return ShrinkageResult(
        covariance=pd.DataFrame(shrunk_cov, index=returns.columns, columns=returns.columns),
        shrinkage_intensity=float(shrinkage_intensity),
        target_used=target,
    )


@dataclass
class FactorCovarianceResult:
    covariance: pd.DataFrame
    factor_loadings: np.ndarray
    explained_variance_ratio: float


def pca_factor_covariance(returns: pd.DataFrame, n_factors: int) -> FactorCovarianceResult:
    """
    Estimate a covariance matrix using a PCA-based statistical factor
    model: reconstruct it from the top `n_factors` principal components of
    the sample covariance, plus a diagonal "specific risk" term for
    whatever variance those factors don't explain. This is the same
    Sigma = X * Sigma_f * X' + Delta structure used for fundamental factor
    models, just with the factors coming from PCA instead of named style
    or industry exposures.

    Args:
        returns: T x N panel of returns.
        n_factors: Number of principal components to keep. Should be much
            smaller than the number of assets.

    Returns:
        A FactorCovarianceResult with the reconstructed covariance matrix,
        the factor loadings (eigenvectors), and the fraction of total
        variance the retained factors explain.
    """
    n_assets = returns.shape[1]
    if not 0 < n_factors < n_assets:
        raise ValueError(f"n_factors must be between 1 and {n_assets - 1}, got {n_factors}")

    values = returns.values
    n_obs = values.shape[0]
    if n_obs < 2:
        raise ValueError("need at least 2 observations to estimate a covariance matrix")

    centered = values - values.mean(axis=0)
    sample_cov = (centered.T @ centered) / (n_obs - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(sample_cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    loadings = eigenvectors[:, :n_factors]
    top_eigenvalues = eigenvalues[:n_factors]

    factor_covariance_part = loadings @ np.diag(top_eigenvalues) @ loadings.T
    specific_variance = np.diag(sample_cov) - np.diag(factor_covariance_part)
    specific_variance = np.clip(specific_variance, a_min=0.0, a_max=None)  # guard against tiny float noise

    reconstructed = factor_covariance_part + np.diag(specific_variance)
    explained_variance_ratio = float(top_eigenvalues.sum() / eigenvalues.sum())

    return FactorCovarianceResult(
        covariance=pd.DataFrame(reconstructed, index=returns.columns, columns=returns.columns),
        factor_loadings=loadings,
        explained_variance_ratio=explained_variance_ratio,
    )


def eigenvalue_diagnostics(covariance: pd.DataFrame) -> Dict[str, float]:
    """
    Report the smallest/largest eigenvalues and condition number of a
    covariance matrix.

    This is a diagnostic, not a fix: a very large condition number (or a
    smallest eigenvalue close to zero) is a warning sign that a portfolio
    optimizer run against this matrix could be exploiting estimation error
    in the low-variance directions rather than genuine diversification
    (the "optimization bias" problem). Actually correcting for it properly
    - e.g. Barra's eigenfactor risk adjustment - means re-simulating each
    eigenfactor's true variance via Monte Carlo, which is out of scope
    here; this function just tells you whether you should be worried.
    """
    eigenvalues = np.linalg.eigvalsh(covariance.values)
    eigenvalues = np.sort(eigenvalues)

    smallest = float(eigenvalues[0])
    largest = float(eigenvalues[-1])
    condition_number = float("inf") if smallest <= 0 else largest / smallest

    return {
        "smallest_eigenvalue": smallest,
        "largest_eigenvalue": largest,
        "condition_number": condition_number,
    }
