#!/usr/bin/env python

from multiprocessing import Pool, set_start_method
from os import path
from time import time

import numpy as np
import pandas as pd

from plotting_utils import box_and_whisker
from statistical_utils import (
    adjust_p_values,
    calculate_pvalue,
    calculate_slope_and_se,
    residualize,
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
        perm_method,
        num_permutations,
        window,
        num_cores,
        plot_threshold,
        plot_dir,
    ):
        self.chromosome = chromosome

        self.copy_number_df = self.load_and_validate_file(copynumber, index_col=0)

        self.full_quan = self.load_and_validate_file(quantifications, index_col=3)
        self.quan = self.full_quan[self.full_quan["chr"] == self.chromosome]

        self.samples = self.quan.columns.to_numpy()[3:]

        self.cov = self.load_and_validate_file(covariates, index_col=None)

        self.segmentation = self.load_and_validate_file(segmentation, index_col=0)
        self.segmentation = self.segmentation[self.segmentation.chr == self.chromosome]
        self.segmentation = self.segmentation[
            self.segmentation.index.isin(self.samples)
        ]

        self.genotype = self.load_and_validate_file(genotype, index_col=0)
        self.genotype = self.genotype.loc[:, self.genotype.columns.isin(self.samples)]
        self.genotype = self.genotype[self.samples]

        if isinstance(all_variants_mode, str):
            # Check if the gene ID given with --all_variants exists in quantification df
            if all_variants_mode in self.quan.index:
                self.quan = self.quan[self.quan.index == all_variants_mode]
                self.all_variants_mode = True
            else:
                raise ValueError(
                    f"Gene ID '{all_variants_mode}' not found in the quantification file under the specified chromosome."
                )
        else:
            self.all_variants_mode = all_variants_mode

        self.window = window

        self.num_cores = num_cores

        self.perm_method = perm_method
        if not (perm_method == "beta" or perm_method == "direct"):
            raise ValueError(
                f"Invalid perm_method selected: '{perm_method}'. Please select beta or direct."
            )

        self.plot_threshold = plot_threshold
        self.plot_dir = plot_dir

        if mode == "nominal":
            self.num_permutations = 0
        else:
            self.num_permutations = num_permutations

    def load_and_validate_file(self, file_path: str, index_col: int):
        """
        Load a CSV file and validate its existence and content.

        Parameters:
        - file_path: Path to file

        Returns:
        - Dataframe from contents of the CSV file

        Raises:
        - FileNotFoundError: If the file does not exist at the given path.
        - ValueError: If the CSV file is empty (i.e., has no rows).
        """
        if not path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")

        df = pd.read_csv(file_path, index_col=index_col)

        if df.shape[0] == 0:
            raise ValueError(f"File '{file_path}' has no rows.")

        return df

    def start_end_gene_window(self, gene_index: int):
        """
        Find position of the window of a given gene.

        Parameters:
        - gene_index: Index of the desired gene on the quantification file

        Returns:
        - Tuple of window_start and window_end, which define the start and end positions of the window
        """
        window_start = self.quan["start"].iloc[gene_index] - self.window
        window_end = self.quan["end"].iloc[gene_index] + self.window
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
        window are also on a same segment.

        Parameters:
        - start: Start position of a window
        - end: End position of a window
        - variants: Subset of genotype file. Only variants that are in the same window as
            the gene of interest

        Returns:
        - variants: Subset of genotype dataframe that is filtered and masked by segmentation and window.
        """
        start += self.window
        end -= self.window

        index_array = variants.index.astype(str).to_numpy()
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
            lower_bound = cur_seg["startpos"].to_numpy()[0]
            upper_bound = cur_seg["endpos"].to_numpy()[0]

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
        - variant: Variant id
        - regression_data: Dataframe with current gene expression levels, genotypes,
            and covariates

        Returns:
        - actual_associations: Dataframe of association testing results for a gene.
            When > 0 permutations are used, also adjusted p-values are provided.
        """
        # Perform nominal association testing for the actual data
        actual_associations = self.gene_variant_regressions(
            gene_index, self.quan, variant, regression_data
        )

        if self.num_permutations == 0:
            return actual_associations

        if self.perm_method == "beta":
            perm_indices = np.random.choice(
                range(self.full_quan.shape[0]), self.num_permutations, replace=False
            )

            r2_perm = np.zeros(self.num_permutations)

            for index in range(self.num_permutations):
                # Perform association testing with the permuted gene index
                perm_data = self.permutation_data(
                    gene_index, perm_indices[index], transf_variants, variant
                )

                perm_gex, perm_genotypes = residualize(perm_data)

                r_perm = np.corrcoef(perm_gex, perm_genotypes)[0, 1]

                r2_perm[index] = r_perm**2

            nom_gex, nom_genotypes = residualize(regression_data)

            nominal_r = np.corrcoef(nom_gex, nom_genotypes)[0, 1]
            nominal_r2 = np.power(nominal_r, 2)

            # Adjust p-values using beta approximation
            adjusted_p_value = adjust_p_values(r2_perm, nominal_r2)

        else:
            permuted_correlations = []

            # In direct scheme, permute whole dataset
            for index in range(self.full_quan.shape[0]):
                perm_data = self.permutation_data(
                    gene_index, index, transf_variants, variant
                )

                residualized_data = residualize(perm_data)

                perm_corr = np.corrcoef(
                    residualized_data["cur_genotypes"], residualized_data["GEX"]
                )[0, 1]
                permuted_correlations.append(perm_corr)

            actual_corr = np.corrcoef(
                regression_data["cur_genotypes"], regression_data["GEX"]
            )[0, 1]

            permuted_correlations = np.array(permuted_correlations)
            adjusted_p_value = (
                np.sum(np.abs(permuted_correlations) > np.abs(actual_corr))
                / self.full_quan.shape[0]
            )

        # Add adjusted p-values to actual associations
        actual_associations["p_adj"] = adjusted_p_value

        return actual_associations

    def permutation_data(
        self,
        gene_index: int,
        perm_index: int,
        transf_variants: pd.DataFrame,
        variant: str,
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
        - variant: Variant ID

        Returns:
        - perm_data: Dataframe of data linked with the fixed variant and permuted gene
        """
        if not variant:
            return pd.DataFrame()

        # Gene expression levels from the perm index
        GEX = pd.to_numeric(
            self.full_quan.iloc[perm_index, 3:], errors="coerce"
        ).to_numpy()

        current_gene = self.quan.index[gene_index]
        CN = self.copy_number_df.loc[current_gene].to_numpy().flatten()
        cov_values = [
            pd.to_numeric(self.cov.loc[covariate], errors="coerce").to_numpy().flatten()
            for covariate in self.cov.index
        ]

        cur_genotypes = transf_variants.loc[variant]

        GEX_filtered, CN_filtered, cur_genotypes_filtered, cov_values_filtered = (
            self.filter_arrays(GEX, CN, cur_genotypes, cov_values)
        )

        if not any(GEX_filtered):
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

    def check_grouping(self, cur_genotypes_filtered: np.ndarray):
        """
        Find if the genotype dosages have adequate variation in the data.

        Parameters:
        - cur_genotypes_filtered: Array of genotype dosages

        Returns:
        - Boolean value showing if there are enough instances in the different genotype groups.
        """
        bins = [0, 0.34, 0.67, 1]
        genotype_groups = pd.cut(cur_genotypes_filtered, bins=bins, include_lowest=True)
        group_counts = genotype_groups.value_counts()

        threshold = 10  # Minimum number of group members

        # Check if at least two groups exceed the threshold
        groups_exceeding_threshold = (group_counts > threshold).sum()

        if groups_exceeding_threshold < 2:
            return False

        return True

    def filter_arrays(
        self,
        GEX: np.ndarray,
        CN: np.ndarray,
        cur_genotypes: np.ndarray,
        cov_values: np.ndarray,
    ):
        """
        Filter data arrays and do validity checks.

        Parameters:
        - GEX: Gene expression levels
        - CN: Gene copy numbers
        - cur_genotypes: Genotype dosages
        - cov_values: All other covariate values
        - group_check: Whether to check representation of values in the middle
            and in the tails in genotypes

        Returns:
        Tuple of:
        - GEX_filtered: Filtered gene expression values
        - CN_filtered: Filtered copy numbers
        - cur_genotypes_filtered: Filtered genotypes dosages
        - cov_values_filtered: Filtered covariate values
        """
        # Check for shape mismatch
        lengths = [len(GEX), len(CN), len(cur_genotypes)] + [
            len(cov_value) for cov_value in cov_values
        ]
        if len(set(lengths)) != 1:
            return [], [], [], []

        # Filter out rows with NaNs in any of the required columns
        mask = ~np.isnan(GEX) & ~np.isnan(CN) & ~np.isnan(cur_genotypes)
        for cov_value in cov_values:
            mask &= ~np.isnan(cov_value)

        if np.sum(mask) < 30:  # If less than 30 valid rows, skip this variant
            return [], [], [], []

        GEX_filtered = GEX[mask]
        CN_filtered = CN[mask]
        cur_genotypes_filtered = cur_genotypes[mask]
        cov_values_filtered = [cov_value[mask] for cov_value in cov_values]

        # Ensure each column has more than one unique value
        if (
            len(np.unique(GEX_filtered)) < 2
            or len(np.unique(CN_filtered)) < 2
            or len(np.unique(cur_genotypes_filtered)) < 2
            or len(np.unique(cov_values_filtered)) < 2
        ):
            return [], [], [], []

        if not self.check_grouping(cur_genotypes_filtered):
            return [], [], [], []

        return GEX_filtered, CN_filtered, cur_genotypes_filtered, cov_values_filtered

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
        ).to_numpy()
        CN = self.copy_number_df.loc[current_gene].to_numpy().flatten()

        cov_values = [
            pd.to_numeric(self.cov.loc[covariate], errors="coerce").to_numpy().flatten()
            for covariate in self.cov.index
        ]

        best_corr = 0
        data_best_corr = pd.DataFrame()
        best_variant = ""

        for variant_index, cur_genotypes in zip(
            transf_variants.index, transf_variants.to_numpy()
        ):
            GEX_filtered, CN_filtered, cur_genotypes_filtered, cov_values_filtered = (
                self.filter_arrays(GEX, CN, cur_genotypes, cov_values)
            )

            if not any(GEX_filtered):
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

    def data_all_variants(
        self,
        GEX: np.ndarray,
        CN: np.ndarray,
        cov_values: np.ndarray,
        cur_genotypes: np.ndarray,
    ):
        """
        Process data for association testing when in all variants mode.

        Parameters:
        - GEX: Gene expression levels.
        - CN: Gene copy numbers
        - cov_values: All other covariate values
        - cur_genotypes: Genotype dosages

        Returns:
        - Dataframe of filtered regression data.
        """
        GEX_filtered, CN_filtered, cur_genotypes_filtered, cov_values_filtered = (
            self.filter_arrays(GEX, CN, cur_genotypes, cov_values)
        )

        if not any(GEX_filtered):
            return pd.DataFrame()

        data_dict = {
            "GEX": GEX_filtered,
            "CN": CN_filtered,
            "cur_genotypes": cur_genotypes_filtered,
        }

        for covariate, cov_value_filtered in zip(self.cov.index, cov_values_filtered):
            data_dict[covariate] = cov_value_filtered

        return pd.DataFrame(data_dict)

    def process_all_variants(self, gene_index: int, transf_variants: pd.DataFrame):
        """
        Conduct association testing for all variants in a window instead of selecting
        only best correlated variant. Construct regression data and then run the
        regressions.

        Parameters:
        - gene_index: Index of a gene of interest on the quantification file.
        - transf_variants: Dataframe of transformed variants that are processed for
            window and segmentation.

        Returns:
        - Dataframe with all association testing results for a gene.
        """
        current_gene = self.quan.index[gene_index]
        GEX = pd.to_numeric(self.quan.iloc[gene_index, 3:], errors="coerce").to_numpy()
        CN = self.copy_number_df.loc[current_gene].to_numpy().flatten()

        cov_values = [
            pd.to_numeric(self.cov.loc[covariate], errors="coerce").to_numpy().flatten()
            for covariate in self.cov.index
        ]

        df_res_list = []

        for variant_index, cur_genotypes in zip(
            transf_variants.index, transf_variants.to_numpy()
        ):
            regression_data = self.data_all_variants(GEX, CN, cov_values, cur_genotypes)
            perm_res = self.gene_variant_regressions_permutations(
                gene_index, transf_variants, variant_index, regression_data
            )
            df_res_list.append(perm_res)

            if self.plot_threshold != -1:
                p_value = -1
                if (self.num_permutations) > 0:
                    p_value = perm_res["p_adj"][0]
                else:
                    p_value = perm_res["nominal_p"][0]

                if not np.isnan(p_value) and p_value < self.plot_threshold:
                    gene_name = self.quan.index[gene_index]
                    box_and_whisker(
                        regression_data, gene_name, variant_index, self.plot_dir
                    )

        return pd.concat(df_res_list, ignore_index=True)

    def gene_variant_regressions(
        self,
        gene_index: int,
        quantifications: pd.DataFrame,
        variant: str,
        regression_data: pd.DataFrame,
    ):
        """
        Find associations between the gene expression values of a gene and variants
        by performing regressions. Using ordinary least square regression,
        log-likelihood calculations, and likelihood ratio test to pinpoint the effect of genotypes.

        Parameters:
        - gene_index: Index of a gene of interest on the quantification file.
        - quantifications: Dataframe of quantifications.
        - variant: Variant ID
        - regression_data: Regression data for current gene variant pair including covariates

        Returns:
        - associations dataframe with statistics of the strenghts of associations
        """
        associations = []
        current_gene = quantifications.index[gene_index]

        def create_association(gene, variant, slope, slope_se, p_value):
            return {
                "gene": gene,
                "variant": variant,
                "number_of_samples": regression_data.shape[0],
                "slope": slope,
                "slope_se": slope_se,
                "nominal_p": p_value,
            }

        if len(regression_data) == 0:
            associations.append(
                create_association(current_gene, variant, np.nan, np.nan, np.nan)
            )
            return pd.DataFrame(associations)

        gex, genotypes = residualize(regression_data)

        corr = np.corrcoef(gex, genotypes)[0, 1]

        residualized_df = pd.DataFrame(
            {
                "GEX": gex,
                "cur_genotypes": genotypes,
            }
        )

        slope, slope_se = calculate_slope_and_se(residualized_df, corr)

        pval = calculate_pvalue(residualized_df, corr)

        associations.append(
            create_association(current_gene, variant, slope, slope_se, pval)
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
        start = time()

        limit = self.quan.shape[0]  # For testing, use small number, eg. 3

        set_start_method("spawn")
        pool = Pool(processes=self.num_cores)

        # Map the gene indices to the helper function using the Pool
        full_associations = pool.map(self.calculate_associations_helper, range(limit))

        pool.close()
        pool.join()

        end = time()
        print("The time of execution: ", (end - start) / 60, " min")

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

            association_res = self.gene_variant_regressions_permutations(
                gene_index, transf_variants, best_variant, data_best_corr
            )

            if self.plot_threshold != -1:
                p_value = -1
                if (self.num_permutations) > 0:
                    p_value = association_res["p_adj"][0]
                else:
                    p_value = association_res["nominal_p"][0]

                if not np.isnan(p_value) and p_value < self.plot_threshold:
                    gene_name = self.quan.index[gene_index]
                    box_and_whisker(
                        data_best_corr, gene_name, best_variant, self.plot_dir
                    )

            return association_res
