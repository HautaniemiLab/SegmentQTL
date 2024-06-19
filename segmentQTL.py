#!/usr/bin/env python

# from joblib import Parallel, delayed
# import cProfile
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import torch

# from profilehooks import profile
from torch.distributions import Chi2


class SegmentQTL:
    def __init__(
        self,
        chromosome,
        copynumber,
        quantifications,
        covariates,
        ascat,
        genotype,
        out_dir="./",
        num_cores=5,
    ):
        self.chromosome = chromosome  # Needs to have 'chr' prefix

        self.copy_number_df = pd.read_csv(copynumber, index_col=0)

        self.quan = pd.read_csv(quantifications, index_col=3)
        self.quan = self.quan[self.quan.chr == self.chromosome]

        self.samples = self.quan.columns.to_numpy()[3:]

        self.cov = pd.read_csv(covariates)
        self.cov = self.cov[
            self.cov.index == "tissue"
        ]  # This is just for my case, generally use all covs in file

        self.ascat = pd.read_csv(ascat, index_col=0)
        self.ascat = self.ascat[self.ascat.chr == self.chromosome]
        self.ascat = self.ascat[self.ascat.index.isin(self.samples)]

        self.genotype = pd.read_csv(genotype, index_col=0)
        self.genotype = self.genotype.loc[:, self.genotype.columns.isin(self.samples)]
        self.genotype = self.genotype[self.samples]

        self.out_dir = out_dir

        self.num_cores = num_cores

    def start_end_gene_segment(self, gene_index):
        seg_start = self.quan["start"].iloc[gene_index] - 500000
        seg_end = self.quan["end"].iloc[gene_index] + 500000
        return [seg_start, seg_end]

    def get_variants_for_gene_segment(self, current_start, current_end):
        positions = self.genotype.index.str.extract(
            r"chr(?:[1-9]|1[0-9]|2[0-2]|X):(\d+):", expand=False
        ).astype(int)
        subset_condition = (positions > current_start) & (positions < current_end)
        variants = self.genotype.loc[subset_condition]
        return variants

    def gene_variants_common_segment(self, start, end, variants):
        start += 500000
        end -= 500000

        index_array = variants.index.astype(str).values
        variant_pos = [int(index.split(":")[1]) for index in index_array]

        for cur_sample in variants.columns:
            cur_seg = self.ascat.loc[
                (self.ascat.index == cur_sample)
                & (self.ascat["startpos"] <= start)
                & (self.ascat["endpos"] >= start)
            ]

            # If the gene falls onto multiple segments, assign whole col to nan
            if len(cur_seg) != 1:
                variants = variants.assign(cur_sample=np.nan)
                continue

            # Find the lower and upper bounds for the current position
            lower_bound = cur_seg["startpos"].values[0]
            upper_bound = cur_seg["endpos"].values[0]

            # Create mask for positions that are outside of the bounds to set them to nan
            within_bounds = (variant_pos >= lower_bound) & (variant_pos <= upper_bound)
            mask_cur_col = ~within_bounds

            variants = variants.copy()
            variants.loc[mask_cur_col, cur_sample] = np.nan

        return variants

    def calculate_beta_parameters(self, perm_p_values):
        if perm_p_values is None or len(perm_p_values) == 0:
            return np.nan, np.nan

        perm_p_values_tensor = tf.constant(perm_p_values, dtype=tf.float64)

        # Calculate mean and variance of permutation p-values
        mean_p_value = tf.reduce_mean(perm_p_values_tensor)
        var_p_value = tf.math.reduce_variance(perm_p_values_tensor)

        # Calculate beta distribution parameters
        beta_shape1 = mean_p_value * (
            mean_p_value * (1 - mean_p_value) / var_p_value - 1
        )
        beta_shape2 = beta_shape1 * (1 / mean_p_value - 1)

        return beta_shape1, beta_shape2

    def adjust_p_values(self, nominal_p_value, beta_shape1, beta_shape2):
        if np.isnan(nominal_p_value) or np.isnan(beta_shape1) or np.isnan(beta_shape2):
            return np.nan

        nom_pval_tensor = tf.constant(nominal_p_value, dtype=tf.float64)

        beta_dist = tfp.distributions.Beta(beta_shape1, beta_shape2)

        # Calculate the adjusted p-values using the beta distribution
        adjusted_p_value = beta_dist.cdf(nom_pval_tensor)

        return adjusted_p_value.numpy()

    # @profile(filename="gene_variant_regressions_permutations.prof")
    def gene_variant_regressions_permutations(
        self, gene_index, transf_variants, permutations
    ):
        permutations_list = []

        perm_indices = np.random.choice(
            range(self.quan.shape[0]), permutations, replace=False
        )

        for index in perm_indices:
            # Perform association testing with the permuted gene index
            associations = self.gene_variant_regressions(index, transf_variants)

            # Store the results of the permutation
            permutations_list.append(associations)

        permutations_results = pd.concat(permutations_list)

        # perm_p_values = permutations_results.loc[:, "pr_over_chi_squared"].values
        perm_p_values = permutations_results["pr_over_chi_squared"].dropna().values

        # Calculate beta parameters
        beta_shape1, beta_shape2 = self.calculate_beta_parameters(perm_p_values)

        # Perform nominal association testing for the actual data
        actual_associations = self.gene_variant_regressions(gene_index, transf_variants)
        actual_p_value = actual_associations.loc[:, "pr_over_chi_squared"].values[0]

        # Adjust p-values using beta approximation
        adjusted_p_values = self.adjust_p_values(
            actual_p_value, beta_shape1, beta_shape2
        )

        # Add adjusted p-values to actual associations
        actual_associations["p_adj"] = adjusted_p_values

        return actual_associations

    # @profile(filename="loglike_new.prof")
    def ols_reg_loglike(self, X, Y, R2_value=False):
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

    # @profile(filename="program.prof")
    def best_variant_data(self, gene_index, transf_variants):
        current_gene = self.quan.index[gene_index]
        GEX = pd.to_numeric(self.quan.iloc[gene_index, 3:], errors="coerce").values
        CN = self.copy_number_df.loc[current_gene].values.flatten()

        cov_values = [
            pd.to_numeric(self.cov.loc[covariate], errors="coerce").values.flatten()
            for covariate in self.cov.index
        ]

        best_corr = 0
        data_best_corr = pd.DataFrame()
        best_variant = ""

        for variant_index, cur_genotypes in zip(
            transf_variants.index, transf_variants.values
        ):
            # Check for shape mismatch
            # TODO: Why there exists mismatches?
            lengths = [len(GEX), len(CN), len(cur_genotypes)] + [
                len(cov_value) for cov_value in cov_values
            ]
            if len(set(lengths)) != 1:
                continue  # Skip this variant if lengths do not match

            # Filter out rows with NaNs in any of the required columns
            mask = ~np.isnan(GEX) & ~np.isnan(CN) & ~np.isnan(cur_genotypes)
            for cov_value in cov_values:
                mask &= ~np.isnan(cov_value)

            # Apply mask to all columns
            if np.sum(mask) < 2:  # If less than 2 valid rows, skip this variant
                continue

            GEX_filtered = GEX[mask]
            CN_filtered = CN[mask]
            cur_genotypes_filtered = cur_genotypes[mask]
            cov_values_filtered = [cov_value[mask] for cov_value in cov_values]

            # Ensure each column has more than one unique value
            if (
                len(np.unique(GEX_filtered)) < 2
                or len(np.unique(cur_genotypes_filtered)) < 2
            ):
                continue

            # Calculate Pearson correlation
            corr = np.corrcoef(GEX_filtered, cur_genotypes_filtered)[0, 1]

            if np.abs(corr) > np.abs(best_corr):
                data_dict = {
                    "GEX": GEX_filtered,
                    "CN": CN_filtered,
                    "cur_genotypes": cur_genotypes_filtered,
                }

                for covariate, cov_value_filtered in zip(
                    self.cov.index, cov_values_filtered
                ):
                    data_dict[covariate] = cov_value_filtered

                best_corr = corr
                data_best_corr = pd.DataFrame(data_dict)
                best_variant = variant_index

        return best_variant, data_best_corr

    def gene_variant_regressions(self, gene_index, transf_variants):
        associations = []
        current_gene = self.quan.index[gene_index]
        best_variant, data_best_corr = self.best_variant_data(
            gene_index, transf_variants
        )

        def create_association(
            gene,
            variant,
            R2_value,
            likelihood_ratio_stat,
            loglike_res,
            loglike_nested,
            pr_over_chi_squared,
        ):
            return {
                "gene": gene,
                "variant": variant,
                "R2_value": R2_value,
                "likelihood_ratio_stat": likelihood_ratio_stat,
                "log_likelihood_full": loglike_res,
                "log_likelihood_nested": loglike_nested,
                "pr_over_chi_squared": pr_over_chi_squared,
            }

        if data_best_corr.empty:
            associations.append(
                create_association(
                    current_gene, best_variant, np.nan, np.nan, np.nan, np.nan, np.nan
                )
            )
            return pd.DataFrame(associations)

        Y = data_best_corr["GEX"].values.reshape(-1, 1)

        X = np.column_stack((np.ones(len(Y)), data_best_corr.iloc[:, 1:]))
        X_nested = np.column_stack(
            (np.ones(len(Y)), data_best_corr.drop(columns=["GEX", "cur_genotypes"]))
        )

        loglike_res, R2_value = self.ols_reg_loglike(X, Y, R2_value=True)
        loglike_nested = self.ols_reg_loglike(X_nested, Y)

        likelihood_ratio_stat = -2 * (loglike_nested - loglike_res)

        if np.isnan(likelihood_ratio_stat):
            print("Likelihood ratio nan:", current_gene)
            associations.append(
                create_association(
                    current_gene, best_variant, np.nan, np.nan, np.nan, np.nan, np.nan
                )
            )
        else:
            # TODO: Double-check the appropriate degrees of freedom
            # I used 1, because it is the difference between full and nested models after genotypes are dropped
            chi2_dist = Chi2(1)
            pr_over_chi_squared = 1 - torch.exp(
                torch.log(
                    chi2_dist.cdf(
                        torch.tensor(likelihood_ratio_stat.numpy(), dtype=torch.float64)
                    )
                )
            )

            associations.append(
                create_association(
                    current_gene,
                    best_variant,
                    R2_value.numpy(),
                    likelihood_ratio_stat.numpy(),
                    loglike_res.numpy(),
                    loglike_nested.numpy(),
                    pr_over_chi_squared.item(),
                )
            )

        return pd.DataFrame(associations)

    def calculate_associations(self):
        start = time.time()

        limit = self.quan.shape[0]  # For testing, use small number, eg. 3

        # Set the start method to 'spawn' for multiprocessing.Pool
        mp.set_start_method("spawn")

        # Create a multiprocessing Pool
        pool = mp.Pool(processes=self.num_cores)

        # Map the gene indices to the helper function using the Pool
        full_associations = pool.map(self.calculate_associations_helper, range(limit))

        # Close the Pool
        pool.close()
        pool.join()

        end = time.time()
        print("The time of execution: ", (end - start) / 60, " min")

        # Concatenate the list of DataFrames into one DataFrame
        return pd.concat(full_associations), (end - start) / 60

    def calculate_associations_helper(self, gene_index):
        print(gene_index + 1, "/", self.quan.shape[0])
        current_start, current_end = self.start_end_gene_segment(gene_index)
        current_variants = self.get_variants_for_gene_segment(
            current_start, current_end
        )

        transf_variants = self.gene_variants_common_segment(
            current_start, current_end, current_variants
        )

        cur_associations = self.gene_variant_regressions_permutations(
            gene_index, transf_variants, 100
        )

        return cur_associations


def main():
    with open("elapsed_times.txt", "a") as f:
        # Loop over chromosomes
        for chr in [
            22
        ]:  # range(1, 23):  # range(1, 23):  # range(1, 22) will loop over 1 to 21
            # Format file paths
            copynumber_file = "segmentQTL_inputs/copynumber.csv"
            quantifications_file = "segmentQTL_inputs/quantifications.csv"
            covariates_file = "segmentQTL_inputs/covariates.csv"
            ascat_file = "segmentQTL_inputs/ascat.csv"
            genotypes_file = f"segmentQTL_inputs/genotypes/chr{chr}_adjusted.csv"

            # Call SegmentQTL and measure elapsed time
            mapping, elapsed_time = SegmentQTL(
                f"chr{chr}",
                copynumber_file,
                quantifications_file,
                covariates_file,
                ascat_file,
                genotypes_file,
            ).calculate_associations()

            mapping["chr"] = chr

            # Save elapsed time to text file
            f.write(f"Elapsed time for chr{chr}: {elapsed_time} minutes\n")

            # Save mapping DataFrame to CSV
            mapping.to_csv(f"test_perm_chr{chr}.csv")


if __name__ == "__main__":
    main()
    # cProfile.run("main()")
