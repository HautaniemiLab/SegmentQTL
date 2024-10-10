#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions


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


def ols_reg_loglike(X: np.ndarray, Y: np.ndarray, R2_value: bool = False):
    """
    Ordinary least square regression and log-likelihood calculation. Optionally calculate the R2-value

    Parameters:
    - X: The dependent variables with the intercept term
    - Y: The independent variable
    - R2_value: Whether to calculate the R2-value, default false

    Returns:
    - log-likelihood of the model
    - (Optionally) R2-value
    """
    n = len(Y)
    X_tf = tf.constant(X, dtype=tf.float64)
    Y_tf = tf.constant(Y, dtype=tf.float64)

    coefficients = tf.linalg.lstsq(X_tf, Y_tf)
    Y_pred = tf.matmul(X_tf, coefficients)
    residuals = Y_tf - Y_pred
    SSR = tf.reduce_sum(tf.square(residuals))
    sigma2 = (1 / n) * SSR

    loglike_res = -(n / 2) * tf.math.log(sigma2) - (1 / (2 * sigma2)) * SSR

    if R2_value:
        Y_mean = tf.reduce_mean(Y_tf)
        SS = tf.reduce_sum(tf.square(Y_tf - Y_mean))
        R2 = 1 - SSR / SS
        return loglike_res, R2

    return loglike_res
