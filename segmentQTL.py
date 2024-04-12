#!/usr/bin/env python

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import warnings
import tensorflow as tf
import tensorflow_probability as tfp
import time

warnings.simplefilter(action='ignore', category=FutureWarning)

class SegmentQTL:
  def __init__(self, chromosome, copynumber, quantifications, covariates, ascat, genotype, out_dir = "./", num_cores=1):
    self.chromosome = chromosome   # Needs to have 'chr' prefix

    self.copy_number_df = pd.read_csv(copynumber, index_col=0)

    self.quan = pd.read_csv(quantifications, index_col=3)
    self.quan = self.quan[self.quan.chr == self.chromosome]

    self.samples = self.quan.columns.to_numpy()[3:]

    self.cov = pd.read_csv(covariates)
    self.cov = self.cov[self.cov.index == "tissue"]  # This is just for my case, generally use all covs in file

    self.ascat = pd.read_csv(ascat, index_col=0)
    self.ascat = self.ascat[self.ascat.chr == self.chromosome]
    self.ascat = self.ascat[self.ascat.index.isin(self.samples)]

    self.genotype = pd.read_csv(genotype, index_col=0)
    self.genotype = self.genotype.loc[:, self.genotype.columns.isin(self.samples)]
    self.genotype = self.genotype[self.samples]
    #self.genotype = self.genotype.loc[:, self.genotype.columns.isin(self.samples.tolist())]

    self.out_dir = out_dir

    self.num_cores = num_cores

    #self.genotype.to_csv('Python_genotype.csv')


  def start_end_gene_segment(self, gene_index):
    seg_start = self.quan['start'].iloc[gene_index] - 500000
    seg_end = self.quan['end'].iloc[gene_index] + 500000
    return [seg_start, seg_end]
  
  def get_variants_for_gene_segment(self, current_start, current_end):
    positions = self.genotype.index.str.extract(r'chr(?:[1-9]|1[0-9]|2[0-2]|X):(\d+):', expand=False).astype(int)
    subset_condition = (positions > current_start) & (positions < current_end)
    variants = self.genotype.loc[subset_condition]
    return variants
  
  def gene_variants_common_segment(self, start, end, variants):
    start += 500000
    end -= 500000
    
    for cur_sample in self.samples:
        cur_seg = self.ascat.loc[(self.ascat.index == cur_sample) & 
                             (self.ascat['startpos'] <= start) & 
                             (self.ascat['endpos'] >= start)]
        
        for variant_id in variants.index:
            variant_pos = int(variant_id.split(":")[1])
            
            # First check if gene is located on one segment
            if len(cur_seg) != 1:
                #variants[cur_sample] = None
                variants.loc[:, cur_sample] = None
                break
            # Then check if the variant is on the same segment
            elif cur_seg['startpos'].values[0] > variant_pos or cur_seg['endpos'].values[0] < variant_pos:
                variants.loc[variants.index == variant_id, cur_sample] = None
    
    return variants

  
  def ols_reg_loglike(self, X, Y, R2_value=False):

    # Add bias term to input features
    # Experiment with adding intercept before calling this function
    #X_with_bias = np.column_stack((np.ones(len(X)), X))
    n = len(Y)

    # Convert to TensorFlow tensors
    X_tf = tf.constant(X, dtype=tf.float32)   # Was X_with_bias
    Y_tf = tf.constant(Y, dtype=tf.float32)

    # Compute coefficients using TensorFlow
    coefficients = tf.linalg.lstsq(X_tf, Y_tf)

    # Compute predictions
    Y_pred = tf.matmul(X_tf, coefficients)

    # Compute residuals
    residuals = Y_tf - Y_pred

    # Compute sum of squares of residuals
    SSR = tf.reduce_sum(tf.square(residuals))

    sigma2 = (1/n) * SSR

    loglike_res = -(n/2) * tf.math.log(sigma2) - (1/(2*sigma2)) * SSR

    if R2_value:
      Y_mean = tf.reduce_mean(Y_tf)

      # Sum of squares
      SS = tf.reduce_sum(tf.square(Y_tf - Y_mean))

      R2 = 1 - SSR / SS

      return loglike_res, R2
       
       ###########################################

        #df1 = X.shape[1] - 1  # Number of predictors (excluding intercept)
        #df2 = n - X.shape[1] + 1  # Total number of observations minus number of predictors, plus 1 for intercept

        # Calculate F-statistic
        #F_statistic = (SSR / df1) / ((1 / n) * SSR)

        # Approximate F-distribution using SigmoidBeta distribution
        #d1 = df1
        #d2 = df2
        #f_dist = tfp.bijectors.Chain([tfp.bijectors.Scale(d2 / d1), tfp.bijectors.Exp()])(tfp.distributions.SigmoidBeta(d1 / 2, d2 / 2))

        # Calculate p-value using the approximate F-distribution
        #p_value = 1 - f_dist.cdf(F_statistic)
        #return loglike_res, p_value

    return loglike_res
  
  
  #def calc_pvalue(self, sigma2, df1, df2, SSR):
  #  """
  #  Calculate the p-value using the F-distribution.
  #  Args:
  #  - sigma2: Mean squared error.
  #  - df1: Degrees of freedom for the numerator.
  #  - df2: Degrees of freedom for the denominator.
  #  - SSR: Sum of squares of residuals.
  #  Returns:
  #  - p-value.
  #  """
  #  F_statistic = (SSR / df1) / (sigma2 / df2)
  #  p_value = 1 - dist.FisherSnedecor(df1, df2).cdf(F_statistic)
  #  return p_value.item()  # Convert from tensor to Python float
  
  
  #def calc_pvalue(self, SSR, X_tf, n):
  #  df_residuals = tf.cast(n - tf.shape(X_tf)[1], dtype=tf.float32)
  #  F_statistic = (SSR / df_residuals) / ((1 / tf.cast(n, dtype=tf.float32)) * tf.cast(SSR, dtype=tf.float32))

    # Compute p-value using betainc function
  #  df1 = tf.cast(tf.shape(X_tf)[1], dtype=tf.float32)
  #  df2 = tf.cast(n - tf.shape(X_tf)[1], dtype=tf.float32)
  #  p_value = 1 - tf.math.betainc(0.5 * df1, 0.5 * df2, df1 * F_statistic / (df1 * F_statistic + df2))

  #  return p_value


  def gene_variant_regressions(self, gene_index, current_gene, transf_variants):
    associations = []
    GEX = self.quan.iloc[gene_index, 3:].values.astype(float)
    CN = self.copy_number_df.loc[current_gene].values.flatten()

    cov_values = [self.cov.loc[covariate].values.flatten().astype(float) for covariate in self.cov.index]

    for variant_index, variant_values in zip(transf_variants.index, transf_variants.values):
      cur_genotypes = variant_values

      data_dict = {'GEX': GEX, 'CN': CN, 'cur_genotypes': cur_genotypes}
      data_dict.update({covariate: cov_value for covariate, cov_value in zip(self.cov.index, cov_values)})

      current_data = pd.DataFrame(data_dict)
      current_data['GEX'] = pd.to_numeric(current_data['GEX'], errors='coerce')
      current_data = current_data.dropna()

      # Make sure that all variables have more than one unique value
      if any(current_data[col].nunique() < 2 for col in current_data.columns):
        continue

        #################################################################
        # Here the data is ready (current_data) and regression can begin
        #################################################################

      #if (current_gene == 'ENSG00000177663' and variant_index == 'chr22:16585144:G:A'):
      #  print(current_data)
      #  current_data.to_csv('Python_current_data.csv')
      #  print(transf_variants)
      #  transf_variants.to_csv('Python_transf.csv')

      Y = current_data['GEX'].values.reshape(-1, 1)
      X = np.column_stack((np.ones(len(Y)), current_data.drop(columns=['GEX'])))
      #X = current_data.drop(columns=['GEX'])

      #X_nested = current_data.drop(columns=['GEX', 'cur_genotypes'])
      X_nested = np.column_stack((np.ones(len(Y)), current_data.drop(columns=['GEX', 'cur_genotypes'])))
      #X_intercept = np.ones((len(Y), 1))  # Intercept-only model/null model

      loglike_res, R2_value = self.ols_reg_loglike(X, Y, R2_value=True)
      loglike_nested = self.ols_reg_loglike(X_nested, Y)
      #loglike_intercept = self.ols_reg_loglike(X_intercept, Y)

      likelihood_ratio_stat = -2 * (loglike_nested - loglike_res)
      #likelihood_ratio_stat_intercept = -2 * (loglike_intercept - loglike_res)

      df = 1 # There should be 1 difference in degrees of freedoms as genotypes are dropped from nested model

      # Compute p-value using chi-squared distribution
      pr_over_chi_squared = 1 - tf.math.exp(tf.math.log(tfp.distributions.Chi2(df).cdf(likelihood_ratio_stat)))

      associations.append({
            'gene': current_gene,
            'variant': variant_index,
            'R2_value': R2_value.numpy(),
            'stats': likelihood_ratio_stat.numpy(),
            'log_likelihood_full': loglike_res.numpy(),
            'log_likelihood_nested': loglike_nested.numpy(),
            'pr_over_chi_squared': pr_over_chi_squared.numpy()
      })
  
    return pd.DataFrame(associations)

  
  def calculate_associations(self):
    start = time.time()

    limit = self.quan.shape[0]  # For testing, use small number, eg. 3
    # Use a list comprehension to generate a list of delayed function calls
    # Each call returns a DataFrame directly

    #for i in range(limit):
    #  self.calculate_associations_helper(i)

    full_associations = Parallel(n_jobs=self.num_cores)(
        delayed(lambda gene_index: self.calculate_associations_helper(gene_index))(index) for index in range(limit)
    )

    end = time.time()
    print("The time of execution: ",(end-start)/60, " min")

    # Concatenate the list of DataFrames into one DataFrame
    return pd.concat(full_associations)

  def calculate_associations_helper(self, gene_index):
    print(gene_index, "/", self.quan.shape[0]-1)
    current_gene = self.quan.index[gene_index]
    current_start, current_end = self.start_end_gene_segment(gene_index)
    current_variants = self.get_variants_for_gene_segment(current_start, current_end)
    transf_variants = self.gene_variants_common_segment(current_start, current_end,current_variants)
    cur_associations = self.gene_variant_regressions(gene_index, current_gene, transf_variants)
    return cur_associations


testing = SegmentQTL("chr22", 
                     "segmentQTL_inputs/copynumber.csv", 
                     "segmentQTL_inputs/quantifications.csv", 
                     "segmentQTL_inputs/covariates.csv", 
                     "segmentQTL_inputs/ascat.csv",
                     "segmentQTL_inputs/genotypes/chr22.csv").calculate_associations().to_csv('chr22.csv')

