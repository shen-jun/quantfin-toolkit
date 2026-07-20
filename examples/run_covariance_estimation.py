"""
Covariance estimation on a simulated universe where the number of assets is
close to the number of observations - exactly the situation where the raw
sample covariance matrix starts to fall apart, and where shrinkage / factor
structure earns its keep.

Run with: python examples/run_covariance_estimation.py
"""

import numpy as np
import pandas as pd

from quantfin.covariance import eigenvalue_diagnostics, ledoit_wolf_shrinkage, pca_factor_covariance, sample_covariance


def main() -> None:
    rng = np.random.default_rng(0)
    n_assets = 60
    n_obs = 80  # deliberately not much more than n_assets

    # Simulate returns from a simple 3-factor structure plus idiosyncratic noise,
    # so we have a "true" covariance matrix to compare estimates against.
    n_true_factors = 3
    factor_returns = rng.normal(0.0, 0.015, size=(n_obs, n_true_factors))
    loadings = rng.normal(0.0, 1.0, size=(n_assets, n_true_factors))
    idiosyncratic = rng.normal(0.0, 0.01, size=(n_obs, n_assets))
    returns = factor_returns @ loadings.T + idiosyncratic

    columns = [f"asset_{i}" for i in range(n_assets)]
    returns_df = pd.DataFrame(returns, columns=columns)

    true_covariance = loadings @ loadings.T * (0.015 ** 2) + np.eye(n_assets) * (0.01 ** 2)

    print(f"{n_assets} assets, {n_obs} observations - close to the danger zone for sample covariance.\n")

    sample_cov = sample_covariance(returns_df)
    sample_diag = eigenvalue_diagnostics(sample_cov)
    print("Sample covariance:")
    print(f"  condition number:  {sample_diag['condition_number']:.1f}")
    print(f"  distance to true:  {np.linalg.norm(sample_cov.values - true_covariance):.4f}")

    shrunk = ledoit_wolf_shrinkage(returns_df, target="constant_correlation")
    shrunk_diag = eigenvalue_diagnostics(shrunk.covariance)
    print(f"\nLedoit-Wolf shrinkage (intensity={shrunk.shrinkage_intensity:.3f}):")
    print(f"  condition number:  {shrunk_diag['condition_number']:.1f}")
    print(f"  distance to true:  {np.linalg.norm(shrunk.covariance.values - true_covariance):.4f}")

    factor_model = pca_factor_covariance(returns_df, n_factors=3)
    factor_diag = eigenvalue_diagnostics(factor_model.covariance)
    print(f"\nPCA factor model (3 factors, explains {factor_model.explained_variance_ratio:.1%} of variance):")
    print(f"  condition number:  {factor_diag['condition_number']:.1f}")
    print(f"  distance to true:  {np.linalg.norm(factor_model.covariance.values - true_covariance):.4f}")

    print("\nBoth alternatives should have a much better-behaved condition number than the "
          "raw sample covariance, and (since we know the true covariance here) should usually "
          "land closer to it too.")


if __name__ == "__main__":
    main()
