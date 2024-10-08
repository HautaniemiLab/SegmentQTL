#!/usr/bin/env python

import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import torch
from torch.distributions import Chi2

from statistical_utils import (
    adjust_p_values,
    calculate_beta_parameters,
    ols_reg_loglike,
)


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
        all_variants_mode,
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

        self.all_variants_mode = all_variants_mode

        self.num_cores = num_cores

        if mode == "nominal":
            self.num_permutations = 0
        else:
            self.num_permutations = num_permutations

    def start_end_gene_window(self, gene_index: int):
        """
        Find position of the window of a given gene.

        Parameters:
        - gene_index: Index of the desired gene on the quantification file

        Returns:
        - Tuple of window_start and window_end, which define the start and end positions of the window
        """
        window_start = self.quan["start"].iloc[gene_index] - 500000
        window_end = self.quan["end"].iloc[gene_index] + 500000
        return [window_start, window_end]

    def get_variants_for_gene_window(self, current_start: int, current_end: int):
        """
        Find all the variants inside a window of a gene.

        Parameters:
        - current_start: Start position of a window
        - current_end: End position of a window

        Returns:
        - variants: Subset of genotype dataframe that contains only those variants that are inside
            the given window
        """
        positions = self.genotype.index.str.extract(
            r"chr(?:[1-9]|1[0-9]|2[0-2]|X):(\d+):", expand=False
        ).astype(int)
        subset_condition = (positions > current_start) & (positions < current_end)
        variants = self.genotype.loc[subset_condition]
        return variants

    def gene_variants_common_segment(
        self, start: int, end: int, variants: pd.DataFrame
    ):
        """
        Filter variants to ensure that the gene and variants that are in the same
        window are also on a same segment

        Parameters:
        - start: Start position of a window
        - end: End position of a window
        - variants: Subset of genotype file. Only variants that are in the same window as
            the gene of interest

        Returns:
        - variants: Subset of genotype dataframe that is filtered and masked by segmentation and window.
        """
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

    def gene_variant_regressions_permutations(
        self,
        gene_index: int,
        transf_variants: pd.DataFrame,
        variant: str,
        regression_data: pd.DataFrame,
    ):
        """
        Perform permutations to obtain adjusted p-values. In case of 0 permutations,
        do only nominal pass.

        Parameters:
        - gene_index: Index of a gene of interest on the quantification file.
        - transf_variants: Dataframe of transformed variants that are processed for
            window and segmentation.

        Returns:
        - actual_associations: Dataframe of association testing results for a gene.
            When > 0 permutations are used, also adjusted p-values are provided.
        """

        # best_variant, data_best_corr = self.best_variant_data(
        #    gene_index, transf_variants, self.quan
        # )

        # Perform nominal association testing for the actual data
        actual_associations = self.gene_variant_regressions(
            gene_index, self.quan, variant, regression_data
        )

        # TODO: Possibly add plotting here?

        if self.num_permutations == 0:
            return actual_associations

        permutations_list = []

        perm_indices = np.random.choice(
            range(self.full_quan.shape[0]), self.num_permutations, replace=False
        )

        for index in perm_indices:
            # Perform association testing with the permuted gene index
            perm_data = self.permutation_data(
                gene_index, index, transf_variants, variant
            )
            associations = self.gene_variant_regressions(
                index, self.full_quan, "", perm_data
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

    def permutation_data(
        self,
        gene_index: int,
        perm_index: int,
        transf_variants: pd.DataFrame,
        best_variant: str,
    ):
        """
        Find data for association testing for permutations. In this case all
        dependent variable values are fixed, only the phenotype levels are
        permuted.

        Parameters:
        - gene_index: Index of the actual gene on the quantification file.
        - perm_index: Index of a gene on the quantification file that is used for permutation.
        - transf_variants: Dataframe of transformed variants that are processed for
            window and segmentation.
        - best_variant: ID of the variant that has strongest correlation to the
            actual gene (not permuted)

        Returns:
        - perm_data: Dataframe of data linked with the fixed variant and permuted gene
        """
        if not best_variant:
            return pd.DataFrame()

        # Gene expression levels from the perm index
        GEX = pd.to_numeric(self.full_quan.iloc[perm_index, 3:], errors="coerce").values

        current_gene = self.quan.index[gene_index]
        CN = self.copy_number_df.loc[current_gene].values.flatten()
        cov_values = [
            pd.to_numeric(self.cov.loc[covariate], errors="coerce").values.flatten()
            for covariate in self.cov.index
        ]

        cur_genotypes = transf_variants.loc[best_variant]

        # Check for shape mismatch
        lengths = [len(GEX), len(CN), len(cur_genotypes)] + [
            len(cov_value) for cov_value in cov_values
        ]
        if len(set(lengths)) != 1:
            return pd.DataFrame()

        # Filter out rows with NaNs in any of the required columns
        mask = ~np.isnan(GEX) & ~np.isnan(CN) & ~np.isnan(cur_genotypes)
        for cov_value in cov_values:
            mask &= ~np.isnan(cov_value)

        GEX_filtered = GEX[mask]
        CN_filtered = CN[mask]
        cur_genotypes_filtered = cur_genotypes[mask]
        cov_values_filtered = [cov_value[mask] for cov_value in cov_values]

        # Ensure each column has more than one unique value
        if (
            len(np.unique(GEX_filtered)) < 2
            or len(np.unique(cur_genotypes_filtered)) < 2
        ):
            return pd.DataFrame()

        data_dict = {
            "GEX": GEX_filtered,
            "CN": CN_filtered,
            "cur_genotypes": cur_genotypes_filtered,
        }

        for covariate, cov_value_filtered in zip(self.cov.index, cov_values_filtered):
            data_dict[covariate] = cov_value_filtered

        perm_data = pd.DataFrame(data_dict)

        return perm_data

    def check_data(self, GEX_filtered, cur_genotypes_filtered):
        # Ensure each column has more than one unique value
        if (
            len(np.unique(GEX_filtered)) < 2
            or len(np.unique(cur_genotypes_filtered)) < 2
        ):
            return False

        bins = [0, 0.34, 0.67, 1]
        genotype_groups = pd.cut(cur_genotypes_filtered, bins=bins, include_lowest=True)
        group_counts = genotype_groups.value_counts()

        # TODO: Which threshold?
        threshold = 10  # Minimum number of group members

        # Check that there is enough variation in genotypes to examine the
        # differences imposed by different genotypes on phenotype levels
        # if all(group_counts < threshold):
        #    return False

        # Ensure that the groups are sorted by bin intervals
        # TODO: Check that this is safe to omit
        # sorted_group_counts = group_counts.sort_index()

        # Check if the first and third groups are > threshold and second group > threshold
        if (
            group_counts.iloc[0] < threshold
            or group_counts.iloc[2] < threshold
            or group_counts.iloc[1] < threshold
        ):
            return False

        return True

    def best_variant_data(
        self,
        gene_index: int,
        transf_variants: pd.DataFrame,
        quantifications: pd.DataFrame,
    ):
        """
        Find variant and linked data for a gene that has strongest Pearson
        correlation with the independent variable.

        Parameters:
        - gene_index: Index of a gene of interest on the quantification file.
        - transf_variants: Dataframe of transformed variants that are processed for
            window and segmentation.
        - quantifications: Dataframe of quantifications.

        Returns:
        - best_variant: Id of the variant with strongest correlation
        - data_best_corr: Dataframe of data linked with the chosen variant
        """
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

            # TODO: Test which threshold to use
            if np.sum(mask) < 30:  # If less than 30 valid rows, skip this variant
                continue

            # Apply mask to all columns
            GEX_filtered = GEX[mask]
            CN_filtered = CN[mask]
            cur_genotypes_filtered = cur_genotypes[mask]
            cov_values_filtered = [cov_value[mask] for cov_value in cov_values]

            # TODO: verify that this works as intended
            if not self.check_data(GEX_filtered, cur_genotypes_filtered):
                return data_best_corr

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

    def data_all_variants(self, GEX, CN, cov_values, cur_genotypes):
        data_all_variants = pd.DataFrame()

        # Check for shape mismatch
        lengths = [len(GEX), len(CN), len(cur_genotypes)] + [
            len(cov_value) for cov_value in cov_values
        ]
        if len(set(lengths)) != 1:
            return data_all_variants  # Skip this variant if lengths do not match

        # Filter out rows with NaNs in any of the required columns
        mask = ~np.isnan(GEX) & ~np.isnan(CN) & ~np.isnan(cur_genotypes)
        for cov_value in cov_values:
            mask &= ~np.isnan(cov_value)

        # TODO: Test which threshold to use
        if np.sum(mask) < 30:  # If less than 30 valid rows, skip this variant
            return data_all_variants

        # Apply mask to all columns
        GEX_filtered = GEX[mask]
        CN_filtered = CN[mask]
        cur_genotypes_filtered = cur_genotypes[mask]
        cov_values_filtered = [cov_value[mask] for cov_value in cov_values]

        if not self.check_data(GEX_filtered, cur_genotypes_filtered):
            return data_all_variants

        data_dict = {
            "GEX": GEX_filtered,
            "CN": CN_filtered,
            "cur_genotypes": cur_genotypes_filtered,
        }

        for covariate, cov_value_filtered in zip(self.cov.index, cov_values_filtered):
            data_dict[covariate] = cov_value_filtered

        data_all_variants = pd.DataFrame(data_dict)

        return data_all_variants

    def process_all_variants(self, gene_index, transf_variants):
        current_gene = self.quan.index[gene_index]
        GEX = pd.to_numeric(self.quan.iloc[gene_index, 3:], errors="coerce").values
        CN = self.copy_number_df.loc[current_gene].values.flatten()

        cov_values = [
            pd.to_numeric(self.cov.loc[covariate], errors="coerce").values.flatten()
            for covariate in self.cov.index
        ]

        df_res_list = []

        for variant_index, cur_genotypes in zip(
            transf_variants.index, transf_variants.values
        ):
            regression_data = self.data_all_variants(GEX, CN, cov_values, cur_genotypes)
            perm_res = self.gene_variant_regressions_permutations(
                gene_index, transf_variants, variant_index, regression_data
            )
            df_res_list.append(perm_res)

        return pd.concat(df_res_list, ignore_index=True)

    def gene_variant_regressions(
        self,
        gene_index: int,
        quantifications: pd.DataFrame,
        best_variant: str,
        data_best_corr: pd.DataFrame,
    ):
        """
        Find associations between the gene expression values of a gene and variants
        by performing regressions. Using ordinary least square regression,
        log-likelihood calculations, and likelihood ratio test to pinpoint the effect of genotypes.

        Parameters:
        - gene_index: Index of a gene of interest on the quantification file.
        - quantifications: Dataframe of quantifications.
        - best_variant: The ID of the variant whose genotypes have the strongest correlation to phenotype levels
        - data_best_corr: The data associated with the best_variant

        Returns:
        - associations dataframe with statistics of the strenghts of associations
        """
        associations = []
        current_gene = quantifications.index[gene_index]

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

        if len(data_best_corr) == 0:
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
        """
        Calculate associations for gene indices using multiprocessing.

        Steps:
        1. Initializes the multiprocessing pool with the specified number of cores.
        2. Maps gene indices to the helper function using the pool.
        3. Closes the pool and waits for the processes to complete.
        4. Concatenates the resulting DataFrames from each process into one DataFrame.

        Returns:
        - full_associations: A concatenated dataframe containing the association results
            for all gene indices.
        """
        start = time.time()

        limit = 3  # self.quan.shape[0]  # For testing, use small number, eg. 3

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

    def calculate_associations_helper(self, gene_index: int):
        """
        Helper function to calculate associations for a single gene index.

        This function performs several steps to calculate the associations for a
        specific gene index:
        1. Prints the current progress of the calculation.
        2. Determines the start and end positions for the gene window.
        3. Retrieves the variants within the gene window.
        4. Transforms the variants based on a common segment.
        5. Performs regressions to calculate associations.

        Parameters:
        - gene_index (int): The index of the gene for which associations are being calculated.

        Returns:
        - A dataframe containing the association results for the specified gene index.
        """
        print(gene_index + 1, "/", self.quan.shape[0])
        current_start, current_end = self.start_end_gene_window(gene_index)
        current_variants = self.get_variants_for_gene_window(current_start, current_end)

        transf_variants = self.gene_variants_common_segment(
            current_start, current_end, current_variants
        )

        if self.all_variants_mode:
            return self.process_all_variants(gene_index, transf_variants)
        else:
            best_variant, data_best_corr = self.best_variant_data(
                gene_index, transf_variants, self.quan
            )

            return self.gene_variant_regressions_permutations(
                gene_index, transf_variants, best_variant, data_best_corr
            )
