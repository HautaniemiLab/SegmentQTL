"""
Tests for statistical_utils.py

This module tests:
- fit_ols_null: OLS null model fitting
- calculate_aic_full_ols: AIC computation
- fit_ols_and_test: OLS with partial F-test
- fit_multivariate_ols: Multivariate OLS
- fit_beta_mle: Beta distribution MLE fitting
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import f as f_dist

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from statistical_utils import (
    calculate_aic_full_ols,
    fit_beta_mle,
    fit_multivariate_ols,
    fit_ols_and_test,
    fit_ols_null,
    standardize_variants,
    standardize_variants_bootstrap,
)


class TestFitOlsNull:
    """Tests for fit_ols_null function."""

    def test_basic_regression(self):
        """Test basic OLS null model fitting."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        beta_true = np.array([2.0, 3.0])
        y = X @ beta_true + np.random.randn(n) * 0.5

        y_hat, residuals = fit_ols_null(y, X)

        assert y_hat.shape == (n,)
        assert residuals.shape == (n,)
        # Residuals should sum to approximately zero
        assert np.abs(np.sum(residuals)) < 1e-10
        # y_hat + residuals should equal y
        np.testing.assert_allclose(y_hat + residuals, y, rtol=1e-10)

    def test_intercept_only(self):
        """Test with intercept only (null model)."""
        np.random.seed(42)
        n = 50
        y = np.random.randn(n) + 5.0
        X = np.ones((n, 1))

        y_hat, residuals = fit_ols_null(y, X)

        # With intercept only, y_hat should be the mean
        np.testing.assert_allclose(y_hat, np.full(n, np.mean(y)), rtol=1e-10)

    def test_insufficient_samples(self):
        """Test with n <= p (should return NaNs)."""
        n = 5
        p = 10
        y = np.random.randn(n)
        X = np.random.randn(n, p)

        y_hat, residuals = fit_ols_null(y, X)

        assert np.all(np.isnan(y_hat))
        assert np.all(np.isnan(residuals))

    def test_perfect_fit(self):
        """Test case where data lies exactly on hyperplane."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        beta_true = np.array([1.0, 2.0])
        y = X @ beta_true  # No noise

        y_hat, residuals = fit_ols_null(y, X)

        np.testing.assert_allclose(y_hat, y, rtol=1e-10)
        np.testing.assert_allclose(residuals, np.zeros(n), atol=1e-10)


class TestCalculateAicFullOls:
    """Tests for calculate_aic_full_ols function."""

    def test_basic_aic(self):
        """Test basic AIC computation."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        y = X @ np.array([1.0, 2.0]) + np.random.randn(n) * 0.5

        aic = calculate_aic_full_ols(y, X)

        assert np.isfinite(aic)
        # AIC should be a reasonable value
        assert aic > -1000 and aic < 1000

    def test_empty_data(self):
        """Test with empty data."""
        y = np.array([])
        X = np.array([]).reshape(0, 2)

        aic = calculate_aic_full_ols(y, X)

        assert np.isnan(aic)

    def test_insufficient_samples(self):
        """Test with n <= k (should return NaN)."""
        n = 3
        X = np.random.randn(n, 5)  # More predictors than samples
        y = np.random.randn(n)

        aic = calculate_aic_full_ols(y, X)

        assert np.isnan(aic)

    def test_aic_comparison(self):
        """Test that better fit has lower AIC."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 * x + np.random.randn(n) * 0.1  # Strong linear relationship

        # Good model (includes the true predictor)
        aic_good = calculate_aic_full_ols(y, x.reshape(-1, 1))

        # Bad model (irrelevant predictor)
        z = np.random.randn(n)
        aic_bad = calculate_aic_full_ols(y, z.reshape(-1, 1))

        assert aic_good < aic_bad


class TestFitOlsAndTest:
    """Tests for fit_ols_and_test function."""

    def test_basic_f_test(self):
        """Test basic partial F-test."""
        np.random.seed(42)
        n = 100

        # Generate data with known effect
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.randn(n) * 0.5

        X_null = np.column_stack([np.ones(n), x1])
        X_alt = np.column_stack([np.ones(n), x1, x2])

        result = fit_ols_and_test(y, X_null, X_alt)

        assert "beta_alt" in result
        assert "se_alt" in result
        assert "rss_null" in result
        assert "rss_alt" in result
        assert "f_stat" in result
        assert "p_value" in result
        assert "r2_alt" in result

        # Alt model should have lower RSS
        assert result["rss_alt"] < result["rss_null"]
        # F-stat should be positive
        assert result["f_stat"] > 0
        # P-value should be small (x2 has true effect)
        assert result["p_value"] < 0.01

    def test_no_effect(self):
        """Test when additional predictors have no effect."""
        np.random.seed(42)
        n = 100

        x1 = np.random.randn(n)
        x2 = np.random.randn(n)  # Independent of y
        y = 1 + 2 * x1 + np.random.randn(n) * 2

        X_null = np.column_stack([np.ones(n), x1])
        X_alt = np.column_stack([np.ones(n), x1, x2])

        result = fit_ols_and_test(y, X_null, X_alt)

        # P-value should be large (x2 has no effect)
        assert result["p_value"] > 0.05

    def test_insufficient_samples(self):
        """Test with insufficient samples."""
        n = 5
        X_null = np.random.randn(n, 3)
        X_alt = np.random.randn(n, 6)
        y = np.random.randn(n)

        result = fit_ols_and_test(y, X_null, X_alt)

        assert np.isnan(result["f_stat"])
        assert np.isnan(result["p_value"])

    def test_r2_bounds(self):
        """Test that R² is between 0 and 1."""
        np.random.seed(42)
        n = 100
        X_null = np.ones((n, 1))
        X_alt = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)

        result = fit_ols_and_test(y, X_null, X_alt)

        assert 0 <= result["r2_alt"] <= 1

    def test_coefficient_recovery(self):
        """Test that coefficients are approximately recovered."""
        np.random.seed(42)
        n = 500
        x = np.random.randn(n)
        y = 2.0 + 3.0 * x + np.random.randn(n) * 0.1

        X_null = np.ones((n, 1))
        X_alt = np.column_stack([np.ones(n), x])

        result = fit_ols_and_test(y, X_null, X_alt)

        # Check intercept and slope
        np.testing.assert_allclose(result["beta_alt"][0], 2.0, atol=0.1)
        np.testing.assert_allclose(result["beta_alt"][1], 3.0, atol=0.1)


class TestFitMultivariateOls:
    """Tests for fit_multivariate_ols function."""

    def test_basic_fit(self):
        """Test basic multivariate OLS."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n), np.random.randn(n)])
        beta_true = np.array([1.0, 2.0, 3.0])
        y = X @ beta_true + np.random.randn(n) * 0.5

        result = fit_multivariate_ols(y, X)

        assert "beta" in result
        assert "se" in result
        assert "r2" in result
        assert "rss" in result

        # Coefficients should be close to true values
        np.testing.assert_allclose(result["beta"], beta_true, atol=0.3)

    def test_r2_bounds(self):
        """Test that R² is between 0 and 1."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)

        result = fit_multivariate_ols(y, X)

        assert 0 <= result["r2"] <= 1

    def test_insufficient_samples(self):
        """Test with insufficient samples."""
        n = 3
        X = np.random.randn(n, 5)
        y = np.random.randn(n)

        result = fit_multivariate_ols(y, X)

        assert np.all(np.isnan(result["beta"]))
        assert np.all(np.isnan(result["se"]))


class TestFitBetaMle:
    """Tests for fit_beta_mle function."""

    def test_uniform_pvalues(self):
        """Test with uniform p-values (should give alpha, beta ~ 1)."""
        np.random.seed(42)
        pvals = np.random.uniform(0, 1, 1000)

        alpha, beta_param = fit_beta_mle(pvals)

        # For uniform distribution, both should be close to 1
        assert 0.8 < alpha < 1.2
        assert 0.8 < beta_param < 1.2

    def test_small_pvalues(self):
        """Test with mostly small p-values."""
        np.random.seed(42)
        # Beta(0.5, 5) gives mostly small p-values
        pvals = np.random.beta(0.5, 5, 1000)

        alpha, beta_param = fit_beta_mle(pvals)

        # Alpha should be < 1, beta should be > 1
        assert alpha < 1.5
        assert beta_param > 1.0

    def test_empty_array(self):
        """Test with empty array."""
        pvals = np.array([])

        alpha, beta_param = fit_beta_mle(pvals)

        assert alpha == 1.0
        assert beta_param == 1.0

    def test_with_nans(self):
        """Test that NaNs are filtered out."""
        np.random.seed(42)
        pvals = np.random.uniform(0, 1, 100)
        pvals_with_nan = np.concatenate([pvals, [np.nan, np.nan, np.nan]])

        alpha1, beta1 = fit_beta_mle(pvals)
        alpha2, beta2 = fit_beta_mle(pvals_with_nan)

        np.testing.assert_allclose(alpha1, alpha2, rtol=0.01)
        np.testing.assert_allclose(beta1, beta2, rtol=0.01)

    def test_extreme_values(self):
        """Test with p-values at boundaries."""
        pvals = np.array([0.0, 0.0, 1.0, 1.0, 0.5])

        alpha, beta_param = fit_beta_mle(pvals)

        # Should not crash and return valid values
        assert np.isfinite(alpha)
        assert np.isfinite(beta_param)
        assert alpha > 0
        assert beta_param > 0


class TestVariantStandardization:
    """Tests for shared variant standardization helpers."""

    def test_standardize_variants_applies_coverage_filter(self):
        """Variants with insufficient observed samples should be dropped."""
        d_raw = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, np.nan],
                [5.0, np.nan, np.nan, np.nan, np.nan],
                [2.0, 2.0, 2.0, 2.0, 2.0],
            ]
        )

        d_std, obs_masks, keep_idx, sd_vec = standardize_variants(
            d_raw,
            coverage_tau=0.6,
            n_total=5,
            min_obs=3,
        )

        assert keep_idx.tolist() == [0]
        assert len(obs_masks) == 1
        assert obs_masks[0].tolist() == [0, 1, 2, 3]
        assert d_std.shape == (1, 5)
        assert np.isnan(d_std[0, 4])
        np.testing.assert_allclose(np.nanmean(d_std[0]), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.nanstd(d_std[0]), 1.0, atol=1e-10)
        np.testing.assert_allclose(sd_vec, np.array([np.std([1.0, 2.0, 3.0, 4.0])]))

    def test_standardize_variants_bootstrap_keeps_subsample_eligible_rows(self):
        """Bootstrap standardization should only enforce subsample eligibility."""
        d_raw = np.array(
            [
                [1.0, 3.0, 5.0, np.nan],
                [4.0, 4.0, 4.0, 4.0],
                [7.0, np.nan, np.nan, np.nan],
            ]
        )

        d_std, obs_masks, keep_idx = standardize_variants_bootstrap(
            d_raw,
            min_obs_boot=2,
        )

        assert keep_idx.tolist() == [0]
        assert len(obs_masks) == 1
        assert obs_masks[0].tolist() == [0, 1, 2]
        assert d_std.shape == (1, 4)
        assert np.isnan(d_std[0, 3])
        np.testing.assert_allclose(np.nanmean(d_std[0]), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.nanstd(d_std[0]), 1.0, atol=1e-10)


class TestStatisticalConsistency:
    """Tests for statistical consistency across functions."""

    def test_f_test_p_value_consistency(self):
        """Test that F-test p-value matches scipy calculation."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 * x + np.random.randn(n)

        X_null = np.ones((n, 1))
        X_alt = np.column_stack([np.ones(n), x])

        result = fit_ols_and_test(y, X_null, X_alt)

        # Manually compute p-value from F-stat
        df1 = 1
        df2 = n - 2
        p_manual = float(f_dist.sf(result["f_stat"], df1, df2))

        np.testing.assert_allclose(result["p_value"], p_manual, rtol=1e-10)

    def test_residuals_orthogonal_to_predictors(self):
        """Test that residuals are orthogonal to predictors."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)

        y_hat, residuals = fit_ols_null(y, X)

        # X^T @ residuals should be zero
        np.testing.assert_allclose(X.T @ residuals, np.zeros(2), atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
