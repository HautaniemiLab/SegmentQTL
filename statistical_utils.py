#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def calculate_beta_parameters(perm_p_values):
    if perm_p_values is None or len(perm_p_values) == 0:
        return np.nan, np.nan

    perm_p_values_tensor = tf.constant(perm_p_values, dtype=tf.float64)

    # Calculate mean and variance of permutation p-values
    mean_p_value = tf.reduce_mean(perm_p_values_tensor)
    var_p_value = tf.math.reduce_variance(perm_p_values_tensor)

    # Calculate beta distribution parameters
    beta_shape1 = mean_p_value * (mean_p_value * (1 - mean_p_value) / var_p_value - 1)
    beta_shape2 = beta_shape1 * (1 / mean_p_value - 1)

    return beta_shape1, beta_shape2


def adjust_p_values(nominal_p_value, beta_shape1, beta_shape2):
    if np.isnan(nominal_p_value) or np.isnan(beta_shape1) or np.isnan(beta_shape2):
        return np.nan

    nom_pval_tensor = tf.constant(nominal_p_value, dtype=tf.float64)

    beta_dist = tfp.distributions.Beta(beta_shape1, beta_shape2)

    # Calculate the adjusted p-values using the beta distribution
    adjusted_p_value = beta_dist.cdf(nom_pval_tensor)

    return adjusted_p_value.numpy()


def ols_reg_loglike(X, Y, R2_value=False):
    n = len(Y)

    # Convert to TensorFlow tensors
    X_tf = tf.constant(X, dtype=tf.float64)
    Y_tf = tf.constant(Y, dtype=tf.float64)

    # Compute coefficients using TensorFlow
    coefficients = tf.linalg.lstsq(X_tf, Y_tf)

    # Compute predictions
    Y_pred = tf.matmul(X_tf, coefficients)

    # Compute residuals
    residuals = Y_tf - Y_pred

    # Compute sum of squares of residuals
    SSR = tf.reduce_sum(tf.square(residuals))

    sigma2 = (1 / n) * SSR

    loglike_res = -(n / 2) * tf.math.log(sigma2) - (1 / (2 * sigma2)) * SSR

    if R2_value:
        Y_mean = tf.reduce_mean(Y_tf)

        # Sum of squares
        SS = tf.reduce_sum(tf.square(Y_tf - Y_mean))

        R2 = 1 - SSR / SS
        return loglike_res, R2

    return loglike_res
