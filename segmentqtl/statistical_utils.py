#!/usr/bin/env python

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import f
from sklearn.linear_model import LinearRegression
from tensorflow_probability import distributions


def residualize(regression_data: pd.DataFrame):
    """
    Residualize the GEX and cur_genotypes columns by removing the variance explained by covariates.

    Parameters:
    - regression_data: The input dataframe with GEX, cur_genotypes, and covariates.

    Returns:
    - residualized_df: A dataframe with residualized GEX and cur_genotypes.
    """
    gex = regression_data["GEX"].to_numpy().reshape(-1, 1)
    cur_genotypes = regression_data["cur_genotypes"].to_numpy().reshape(-1, 1)
    covariates = regression_data.drop(columns=["GEX", "cur_genotypes"]).to_numpy()

    model = LinearRegression()

    model.fit(covariates, gex)
    gex_residuals = gex - model.predict(covariates)

    model.fit(covariates, cur_genotypes)
    cur_genotypes_residuals = cur_genotypes - model.predict(covariates)

    residualized_df = pd.DataFrame(
        {
            "GEX": gex_residuals.flatten(),
            "cur_genotypes": cur_genotypes_residuals.flatten(),
        }
    )

    return residualized_df


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


def calculate_beta_parameters(perm_p_values: np.ndarray):
    """
    Calculate beta parameters for the array of p-value obtained from permutations.

    Parameters:
    - perm_p_values: Array of permutation p-values

    Returns:
    Tuple of
    - Beta parameter 1
    - Beta parameter 2
    """
    if perm_p_values is None or len(perm_p_values) == 0:
        return np.nan, np.nan

    perm_p_values_tensor = tf.constant(perm_p_values, dtype=tf.float64)

    mean_p_value = tf.reduce_mean(perm_p_values_tensor)
    var_p_value = tf.math.reduce_variance(perm_p_values_tensor)

    beta_shape1 = mean_p_value * (mean_p_value * (1 - mean_p_value) / var_p_value - 1)
    beta_shape2 = beta_shape1 * (1 / mean_p_value - 1)

    return beta_shape1, beta_shape2


def adjust_p_values(nominal_p_value: float, beta_shape1: float, beta_shape2: float):
    """
    Adjust p-values for multiple comparisons.

    Parameters:
    - nominal_p_value: The p-value from nominal pass
    - beta_shape1: Beta parameter 1
    - beta_shape2: Beta parameter 2

    Returns:
    - Adjusted p-value
    """
    if np.isnan(nominal_p_value) or np.isnan(beta_shape1) or np.isnan(beta_shape2):
        return np.nan

    nom_pval_tensor = tf.constant(nominal_p_value, dtype=tf.float64)

    beta_dist = distributions.Beta(beta_shape1, beta_shape2)
    adjusted_p_value = beta_dist.cdf(nom_pval_tensor)

    return adjusted_p_value.numpy()
