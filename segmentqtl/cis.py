#!/usr/bin/env python

import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import torch
from statistical_utils import (
    adjust_p_values,
    calculate_beta_parameters,
    ols_reg_loglike,
)
from torch.distributions import Chi2


class Cis:
    def __init__(
        self,
        chromosome,
        mode,
        copynumber,
        quantifications,
        covariates,
        segmentation,
        genotype,
        num_permutations,
        num_cores,
    ):
        self.chromosome = chromosome

        self.copy_number_df = pd.read_csv(copynumber, index_col=0)

        self.full_quan = pd.read_csv(quantifications, index_col=3)
        self.quan = self.full_quan[self.full_quan.chr == self.chromosome]

        self.samples = self.quan.columns.to_numpy()[3:]

        self.cov = pd.read_csv(covariates)

        self.segmentation = pd.read_csv(segmentation, index_col=0)
        self.segmentation = self.segmentation[self.segmentation.chr == self.chromosome]
        self.segmentation = self.segmentation[
            self.segmentation.index.isin(self.samples)
        ]

        self.genotype = pd.read_csv(genotype, index_col=0)
        self.genotype = self.genotype.loc[:, self.genotype.columns.isin(self.samples)]
        self.genotype = self.genotype[self.samples]

        self.num_cores = num_cores

        if mode == "nominal":
            self.num_permutations = 0
        else:
            self.num_permutations = num_permutations

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
            cur_seg = self.segmentation.loc[
                (self.segmentation.index == cur_sample)
                & (self.segmentation["startpos"] <= start)
                & (self.segmentation["endpos"] >= start)
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

    def gene_variant_regressions_permutations(self, gene_index, transf_variants):
        # Perform nominal association testing for the actual data
        actual_associations = self.gene_variant_regressions(
            gene_index, transf_variants, self.quan
        )

        if self.num_permutations == 0:
            return actual_associations

        permutations_list = []

        perm_indices = np.random.choice(
            range(self.full_quan.shape[0]), self.num_permutations, replace=False
        )

        for index in perm_indices:
            # Perform association testing with the permuted gene index
            associations = self.gene_variant_regressions(
                index, transf_variants, self.full_quan
            )

            # Store the results of the permutation
            permutations_list.append(associations)

        permutations_results = pd.concat(permutations_list)

        perm_p_values = permutations_results["pr_over_chi_squared"].dropna().values
        beta_shape1, beta_shape2 = calculate_beta_parameters(perm_p_values)
        actual_p_value = actual_associations.loc[:, "pr_over_chi_squared"].values[0]

        # Adjust p-values using beta approximation
        adjusted_p_values = adjust_p_values(actual_p_value, beta_shape1, beta_shape2)

        # Add adjusted p-values to actual associations
        actual_associations["p_adj"] = adjusted_p_values

        return actual_associations

    def best_variant_data(self, gene_index, transf_variants, quantifications):
        current_gene = quantifications.index[gene_index]
        GEX = pd.to_numeric(
            quantifications.iloc[gene_index, 3:], errors="coerce"
        ).values
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

    def gene_variant_regressions(self, gene_index, transf_variants, quantifications):
        associations = []
        current_gene = quantifications.index[gene_index]
        best_variant, data_best_corr = self.best_variant_data(
            gene_index, transf_variants, quantifications
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

        loglike_res, R2_value = ols_reg_loglike(X, Y, R2_value=True)
        loglike_nested = ols_reg_loglike(X_nested, Y)

        likelihood_ratio_stat = -2 * (loglike_nested - loglike_res)

        if np.isnan(likelihood_ratio_stat) or likelihood_ratio_stat.numpy() < 0:
            print("Likelihood ratio nan:", current_gene)
            associations.append(
                create_association(
                    current_gene, best_variant, np.nan, np.nan, np.nan, np.nan, np.nan
                )
            )
        else:
            # Degrees of freedom = 1, because it is the difference between full and nested models after genotypes are dropped
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

        limit = 5  # self.quan.shape[0]  # For testing, use small number, eg. 3

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
        return pd.concat(full_associations)

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
            gene_index, transf_variants
        )

        return cur_associations
