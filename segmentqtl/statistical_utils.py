#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.optimize import minimize, newton
from scipy.special import loggamma
from scipy.stats import beta, f
from sklearn.linear_model import LinearRegression


def residualize(regression_data: pd.DataFrame):
    """
    Residualize the GEX and cur_genotypes columns by removing the variance explained by covariates.

    Parameters:
    - regression_data: The input dataframe with GEX, cur_genotypes, and covariates.

    Returns:
    - Residualized GEX and genotypes.
    """
    gex = regression_data["GEX"].to_numpy().reshape(-1, 1)
    cur_genotypes = regression_data["cur_genotypes"].to_numpy().reshape(-1, 1)
    covariates = regression_data.drop(columns=["GEX", "cur_genotypes"]).to_numpy()

    model = LinearRegression()

    model.fit(covariates, gex)
    gex_residuals = gex - model.predict(covariates)

    model.fit(covariates, cur_genotypes)
    cur_genotypes_residuals = cur_genotypes - model.predict(covariates)

    return gex_residuals.flatten(), cur_genotypes_residuals.flatten()


def get_tstat2(corr: float, df: int):
    """Calculate t-statistic squared from correlation and degrees of freedom.

    Parameters:
    - corr: Pearson correlation
    - df: Degrees of freedom

    Returns:
    - t-statistic squared
    """
    return df * corr**2 / (1 - corr**2)


def get_pvalue_from_tstat2(tstat2: float, df: int):
    """Calculate the p-value from the t-statistic and degrees of freedom.

    Parameters:
    - tstat2: t-statistic squared
    - df: Degrees of freedom

    Returns:
    - p-value
    """
    # Use the F-distribution survival function (sf) for the upper tail probability
    return f.sf(tstat2, 1, df)


def get_pvalue_from_corr(r2: float, df: int):
    """Calculate p-value from correlation r2 and degrees of freedom."""
    tstat2 = get_tstat2(np.sqrt(r2), df)
    return f.sf(tstat2, 1, df)


def beta_shape1_from_dof(r2_values, dof):
    """Estimate Beta shape1 parameter from moment matching."""
    pvals = np.array([get_pvalue_from_corr(r2, dof) for r2 in r2_values])
    mean_p = np.mean(pvals)
    var_p = np.var(pvals)
    return mean_p * (mean_p * (1 - mean_p) / var_p - 1.0)


def beta_log_likelihood(pvals, shape1, shape2):
    """Negative log-likelihood for the Beta distribution."""
    log_beta = loggamma(shape1) + loggamma(shape2) - loggamma(shape1 + shape2)
    return (
        (1.0 - shape1) * np.sum(np.log(pvals))
        + (1.0 - shape2) * np.sum(np.log(1.0 - pvals))
        + len(pvals) * log_beta
    )


def optimize_dof(r2_perm, dof_init, tol=1e-4):
    """
    Optimize degrees of freedom such that Beta shape1 ≈ 1.
    """

    def target(log_dof):
        return np.log(beta_shape1_from_dof(r2_perm, np.exp(log_dof)))

    log_dof_init = np.log(dof_init)
    try:
        log_true_dof = newton(target, log_dof_init, tol=tol, maxiter=50)
        return np.exp(log_true_dof)
    except RuntimeError:
        print("Warning: Newton's method failed, using fallback minimization.")
        res = minimize(
            lambda x: np.abs(beta_shape1_from_dof(r2_perm, x) - 1.0),
            dof_init,
            method="Nelder-Mead",
            tol=tol,
        )
        return res.x[0]


def fit_beta_parameters(r2_perm, dof):
    """
    Fit Beta distribution parameters to permutation p-values.
    """
    pvals = np.array([get_pvalue_from_corr(r2, dof) for r2 in r2_perm])
    mean_p, var_p = np.mean(pvals), np.var(pvals)

    # Initial Beta parameter estimates
    beta_shape1 = mean_p * (mean_p * (1 - mean_p) / var_p - 1)
    beta_shape2 = beta_shape1 * (1 / mean_p - 1)

    # Refine using log-likelihood minimization
    res = minimize(
        lambda s: beta_log_likelihood(pvals, s[0], s[1]),
        [beta_shape1, beta_shape2],
        method="Nelder-Mead",
    )
    beta_shape1, beta_shape2 = res.x
    return beta_shape1, beta_shape2, pvals


def adjust_p_values(r2_perm, r2_nominal, dof_init=10, tol=1e-4):
    """
    Calculate Beta-approximated p-values from permutation results.
    """
    # Optimize DOF
    optimized_dof = optimize_dof(r2_perm, dof_init, tol)

    # Fit Beta parameters
    beta_shape1, beta_shape2, pvals_perm = fit_beta_parameters(r2_perm, optimized_dof)

    # Calculate p-value for nominal r2
    pval_nominal = get_pvalue_from_corr(r2_nominal, optimized_dof)
    adjusted_pval = beta.cdf(pval_nominal, beta_shape1, beta_shape2)

    return adjusted_pval


def get_slope(corr: float, phenotype_sd: np.ndarray, genotype_sd: np.ndarray):
    """Calculate the slope.

    Parameters:
    - corr: Pearson correlation
    - phenotype_sd: Standard deviation of phenotypes
    - genotype_sd: Standard deviation of genotypes

    Returns:
    - slope
    """
    if genotype_sd < 1e-16 or phenotype_sd < 1e-16:
        return 0
    else:
        return corr * phenotype_sd / genotype_sd


def calculate_slope_and_se(regression_data: pd.DataFrame, corr: float):
    """
    Calculate the slope and its standard error.

    Parameters:
    regression_data: A dataframe with residualized "GEX" and "cur_genotypes" columns.
    corr: The correlation between residualized "GEX" and "cur_genotypes".

    Returns:
    slope: The slope of the linear relationship.
    slope_se: The standard error of the slope.
    """
    sample_count = len(regression_data)
    covariate_count = regression_data.shape[1] - 2

    df = sample_count - 2 - covariate_count

    tstat2 = get_tstat2(corr, df)

    gex_residuals = regression_data["GEX"].to_numpy()
    cur_genotypes_residuals = regression_data["cur_genotypes"].to_numpy()

    # Calculate standard deviations of phenotype (GEX) and genotype (cur_genotypes)
    # using Bessel's correction (ddof=1)
    phenotype_sd = np.std(gex_residuals, ddof=1)
    genotype_sd = np.std(cur_genotypes_residuals, ddof=1)

    slope = get_slope(corr, phenotype_sd, genotype_sd)

    slope_se = abs(slope) / np.sqrt(tstat2) if tstat2 > 0 else np.inf

    return slope, slope_se


def calculate_pvalue(df: pd.DataFrame, corr: float):
    """
    Calculate the p-value using the residualized data and correlation.

    Parameters:
    df: A dataframe with residualized "GEX" and "cur_genotypes" columns.
    corr: The correlation between residualized "GEX" and "cur_genotypes".

    Returns:
    pval: The p-value for testing whether the slope is different from 0.
    """
    sample_count = len(df)
    covariate_count = df.shape[1] - 2

    df = sample_count - 2 - covariate_count

    tstat2 = get_tstat2(corr, df)

    pval = get_pvalue_from_tstat2(tstat2, df)

    return pval
