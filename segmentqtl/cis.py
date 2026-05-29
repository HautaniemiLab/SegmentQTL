from multiprocessing import Pool
from os import path
from time import time
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .segment_utils import (
    filter_variants_to_common_segment,
    phenotype_window_bounds,
    variants_in_window,
)
from .statistical_utils import (
    check_d_variance,
    fit_ols_and_test,
)
from .statistical_utils import (
    gene_variant_regressions as run_gene_variant_regressions,
)
from .statistical_utils import (
    gene_variant_regressions_permutations as run_gene_variant_regressions_permutations,
)


class Cis:
    # Chromosome ordering for trans negative control (chr+1, wrapping)
    _CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

    def __init__(
        self,
        chromosome,
        mode,
        phenotype_covariate,
        perm_covariate,
        quantifications,
        covariates,
        segmentation,
        genotype_alt,
        genotype_ref,
        all_variants_mode,
        perm_method,
        num_permutations,
        window,
        num_cores,
        record_aic,
        neg_control=False,
        neg_control_max_variants=2000,
        genotypes_dir=None,
    ):
        self.chromosome = chromosome

        # Load phenotype-level covariate (optional)
        if phenotype_covariate is not None:
            self.phenotype_covariate_df = self.load_and_validate_file(
                phenotype_covariate, index_col=0
            )
        else:
            self.phenotype_covariate_df = None

        # Load permutation-only covariate (optional, for FL residualization)
        if perm_covariate is not None:
            self.perm_covariate_df = self.load_and_validate_file(
                perm_covariate, index_col=0
            )
        else:
            self.perm_covariate_df = None

        self.full_quan = self.load_and_validate_file(quantifications, index_col=3)
        self.quan = self.full_quan[self.full_quan["chr"] == self.chromosome]

        self.samples = self.quan.columns.to_numpy()[3:]

        # Load sample-level covariates (optional)
        if covariates is not None:
            self.cov = self.load_and_validate_file(covariates, index_col=None)
        else:
            self.cov = None

        self.segmentation = self.load_and_validate_file(segmentation, index_col=0)
        self.segmentation = self.segmentation[self.segmentation.chr == self.chromosome]
        self.segmentation = self.segmentation[
            self.segmentation.index.isin(self.samples)
        ]

        # Load both genotype matrices
        self.geno_alt = self.load_and_validate_file(genotype_alt, index_col=0)
        self.geno_alt = self.geno_alt.loc[:, self.geno_alt.columns.isin(self.samples)]
        self.geno_alt = self.geno_alt[self.samples]

        self.geno_ref = self.load_and_validate_file(genotype_ref, index_col=0)
        self.geno_ref = self.geno_ref.loc[:, self.geno_ref.columns.isin(self.samples)]
        self.geno_ref = self.geno_ref[self.samples]

        # Ensure both genotype matrices have same variants and samples (aligned)
        common_variants = self.geno_alt.index.intersection(self.geno_ref.index)
        self.geno_alt = self.geno_alt.loc[common_variants]
        self.geno_ref = self.geno_ref.loc[common_variants]

        # Reindex phenotype-level covariates to self.samples for alignment
        if self.phenotype_covariate_df is not None:
            self.phenotype_covariate_df = self.phenotype_covariate_df.loc[
                :, self.phenotype_covariate_df.columns.isin(self.samples)
            ][self.samples]
        if self.perm_covariate_df is not None:
            self.perm_covariate_df = self.perm_covariate_df.loc[
                :, self.perm_covariate_df.columns.isin(self.samples)
            ][self.samples]

        # Precompute variant positions once as NumPy array
        variant_index_array = self.geno_alt.index.astype(str).to_numpy()
        self.variant_positions = np.fromiter(
            (int(s.split(":")[1]) for s in variant_index_array), dtype=np.int64
        )

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

        self.record_aic = record_aic

        if mode == "nominal":
            self.num_permutations = 0
        else:
            self.num_permutations = num_permutations

        # ── Negative control mode ──
        self.neg_control = neg_control
        self.neg_control_max_variants = neg_control_max_variants

        if neg_control:
            if genotypes_dir is None:
                raise ValueError(
                    "--genotypes directory path is required for trans negative control mode."
                )
            trans_chr = self._get_trans_chromosome(chromosome)
            print(
                f"[neg_control] Gene chromosome: {chromosome} → variant chromosome: {trans_chr}"
            )
            print(
                "[neg_control] Segment consistency filtering is not applied "
                "(not meaningful across chromosomes)."
            )
            trans_alt_file = f"{genotypes_dir}/{trans_chr}_ALTlr.csv"
            trans_ref_file = f"{genotypes_dir}/{trans_chr}_REFlr.csv"

            self.neg_geno_alt = self.load_and_validate_file(trans_alt_file, index_col=0)
            self.neg_geno_alt = self.neg_geno_alt.loc[
                :, self.neg_geno_alt.columns.isin(self.samples)
            ]
            self.neg_geno_alt = self.neg_geno_alt[self.samples]

            self.neg_geno_ref = self.load_and_validate_file(trans_ref_file, index_col=0)
            self.neg_geno_ref = self.neg_geno_ref.loc[
                :, self.neg_geno_ref.columns.isin(self.samples)
            ]
            self.neg_geno_ref = self.neg_geno_ref[self.samples]

            # Align to common variant set
            common_neg = self.neg_geno_alt.index.intersection(self.neg_geno_ref.index)
            self.neg_geno_alt = self.neg_geno_alt.loc[common_neg]
            self.neg_geno_ref = self.neg_geno_ref.loc[common_neg]

            print(f"[neg_control] Loaded {len(common_neg)} variants from {trans_chr}")

    def load_and_validate_file(self, file_path: str, index_col: Optional[int]):
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
        window_start, window_end = phenotype_window_bounds(
            self.quan,
            gene_index,
            self.window,
        )
        return [window_start, window_end]

    def get_variants_for_gene_window(self, current_start: int, current_end: int):
        """
        Find all the variants inside a window of a gene.

        Parameters:
        - current_start: Start position of a window
        - current_end: End position of a window

        Returns:
        - variants_alt: Subset of ALT genotype dataframe that contains only those variants
            that are inside the given window
        - variants_ref: Subset of REF genotype dataframe that contains only those variants
            that are inside the given window
        """
        return variants_in_window(
            self.geno_alt,
            self.geno_ref,
            self.variant_positions,
            current_start,
            current_end,
        )

    # ── Negative control helpers ──────────────────────────────────────────

    @classmethod
    def _get_trans_chromosome(cls, chromosome: str) -> str:
        """Return the next chromosome in the ordering (wraps around)."""
        if chromosome in cls._CHROMOSOMES:
            idx = cls._CHROMOSOMES.index(chromosome)
            return cls._CHROMOSOMES[(idx + 1) % len(cls._CHROMOSOMES)]
        raise ValueError(
            f"Unknown chromosome '{chromosome}' for trans negative control. "
            f"Expected one of {cls._CHROMOSOMES}."
        )

    def _subsample_variants(
        self,
        variants_alt: pd.DataFrame,
        variants_ref: pd.DataFrame,
        gene_index: int,
    ):
        """Randomly subsample to at most neg_control_max_variants (reproducible per gene)."""
        n_total = len(variants_alt)
        if n_total <= self.neg_control_max_variants:
            return variants_alt, variants_ref
        rng = np.random.default_rng(seed=42 + gene_index)
        idx = np.sort(rng.choice(n_total, self.neg_control_max_variants, replace=False))
        return variants_alt.iloc[idx], variants_ref.iloc[idx]

    def get_neg_control_variants(self, gene_index: int):
        """
        Fetch trans negative-control variants for a gene.

        Returns subsampled variants from the next chromosome. These have no
        plausible cis-regulatory relationship to the gene, making them a pure
        null for checking pipeline calibration.

        Returns:
        - variants_alt, variants_ref: DataFrames of (subsampled) trans genotypes
        """
        if len(self.neg_geno_alt) == 0:
            return pd.DataFrame(), pd.DataFrame()
        return self._subsample_variants(
            self.neg_geno_alt, self.neg_geno_ref, gene_index
        )

    def gene_variants_common_segment(
        self,
        start: int,
        end: int,
        variants_alt: pd.DataFrame,
        variants_ref: pd.DataFrame,
        variant_pos: np.ndarray,
    ):
        """
        Filter variants to ensure that the gene and variants that are in the same
        window are also on a same segment.

        Parameters:
        - start: Start position of a window
        - end: End position of a window
        - variants_alt: Subset of ALT genotype file. Only variants that are in the same window
            as the gene of interest
        - variants_ref: Subset of REF genotype file. Only variants that are in the same window
            as the gene of interest
        - variant_pos: Precomputed variant positions (sliced from self.variant_positions)

        Returns:
        - variants_alt, variants_ref: Filtered and masked subsets of genotype dataframes.
        """
        return filter_variants_to_common_segment(
            self.segmentation,
            self.window,
            start,
            end,
            variants_alt,
            variants_ref,
            variant_pos,
        )

    def gene_variant_regressions_permutations(
        self,
        gene_index: int,
        transf_variants_alt: pd.DataFrame,
        transf_variants_ref: pd.DataFrame,
        variant: str,
        regression_data: pd.DataFrame,
    ):
        """
        Perform association testing for the provided gene-variant regression_data and,
        if enabled (mode=perm), compute a scan-level permutation-adjusted p-value.
        """
        return run_gene_variant_regressions_permutations(
            gene_index=gene_index,
            quantifications=self.quan,
            variant=variant,
            regression_data=regression_data,
            transf_variants_alt=transf_variants_alt,
            transf_variants_ref=transf_variants_ref,
            phenotype_covariate_df=self.phenotype_covariate_df,
            perm_covariate_df=self.perm_covariate_df,
            cov=self.cov,
            num_permutations=self.num_permutations,
            perm_method=self.perm_method,
            record_aic=self.record_aic,
        )

    def filter_arrays(
        self,
        GEX: np.ndarray,
        altlr: np.ndarray,
        reflr: np.ndarray,
        phenotype_cov: Optional[np.ndarray],
        cov_values: list,
    ):
        """
        Filter data arrays and do validity checks.

        For s/d parameterization, we check that d = REFlr - ALTlr has variance
        (this is the allelic difference we're testing).

        Parameters:
        - GEX: Phenotype levels
        - altlr: ALTlr genotype values
        - reflr: REFlr genotype values
        - phenotype_cov: Phenotype-level covariate (None if not provided)
        - cov_values: Sample-level covariate values

        Returns:
        Tuple of:
        - GEX_filtered: Filtered phenotype values
        - altlr_filtered: Filtered ALTlr values
        - reflr_filtered: Filtered REFlr values
        - phenotype_cov_filtered: Filtered phenotype-level covariate (or None)
        - cov_values_filtered: Filtered sample-level covariate values
        """
        # Check for shape mismatch
        lengths = [len(GEX), len(altlr), len(reflr)] + [
            len(cov_value) for cov_value in cov_values
        ]
        if phenotype_cov is not None:
            lengths.append(len(phenotype_cov))

        if len(set(lengths)) != 1:
            _empty = np.array([])
            return _empty, _empty, _empty, None, []

        # Filter out rows with NaNs in any of the required columns
        mask = ~np.isnan(GEX) & ~np.isnan(altlr) & ~np.isnan(reflr)

        if phenotype_cov is not None:
            mask &= ~np.isnan(phenotype_cov)

        for cov_value in cov_values:
            mask &= ~np.isnan(cov_value)

        if np.sum(mask) < 30:  # If less than 30 valid rows, skip this variant
            _empty = np.array([])
            return _empty, _empty, _empty, None, []

        GEX_filtered = GEX[mask]
        altlr_filtered = altlr[mask]
        reflr_filtered = reflr[mask]
        phenotype_cov_filtered = (
            phenotype_cov[mask] if phenotype_cov is not None else None
        )
        cov_values_filtered = [cov_value[mask] for cov_value in cov_values]

        # Ensure GEX has variance
        if len(np.unique(GEX_filtered)) < 2:
            _empty = np.array([])
            return _empty, _empty, _empty, None, []

        # Check that d = REFlr - ALTlr has variance (this is what we're testing)
        d_filtered = reflr_filtered - altlr_filtered
        if not check_d_variance(d_filtered):
            _empty = np.array([])
            return _empty, _empty, _empty, None, []

        return (
            GEX_filtered,
            altlr_filtered,
            reflr_filtered,
            phenotype_cov_filtered,
            cov_values_filtered,
        )

    def best_variant_data(
        self,
        gene_index: int,
        transf_variants_alt: pd.DataFrame,
        transf_variants_ref: pd.DataFrame,
        quantifications: pd.DataFrame,
    ):
        """
        Find variant and linked data for a gene that has strongest test statistic.

        Uses s/d parameterization:
        - s = REFlr + ALTlr (total dosage, included in null model)
        - d = REFlr - ALTlr (allelic difference, the test predictor)

        Selects variant with largest |t_d| (1-df t-test on d coefficient).

        Parameters:
        - gene_index: Index of a gene of interest on the quantification file.
        - transf_variants_alt: Dataframe of transformed ALTlr variants
        - transf_variants_ref: Dataframe of transformed REFlr variants
        - quantifications: Dataframe of quantifications.

        Returns:
        - best_variant: Id of the variant with strongest test statistic
        - data_best: Dataframe of data linked with the chosen variant
        """
        current_gene = quantifications.index[gene_index]
        GEX = pd.to_numeric(
            quantifications.iloc[gene_index, 3:], errors="coerce"
        ).to_numpy()

        # In perm mode, exclude samples with missing perm_cov so the nominal
        # best-variant scan uses the same sample set as the permutation null.
        # perm_cov is NOT added as a predictor — only used to define the mask.
        if self.num_permutations > 0 and self.perm_covariate_df is not None:
            perm_cov = (
                self.perm_covariate_df.loc[current_gene]
                .to_numpy()
                .flatten()
                .astype(float)
            )
            GEX[np.isnan(perm_cov)] = np.nan

        # Get phenotype-level covariate if available
        phenotype_cov = None
        if self.phenotype_covariate_df is not None:
            phenotype_cov = (
                self.phenotype_covariate_df.loc[current_gene].to_numpy().flatten()
            )

        cov_values = []
        if self.cov is not None:
            cov_values = [
                pd.to_numeric(self.cov.loc[covariate], errors="coerce")
                .to_numpy()
                .flatten()
                for covariate in self.cov.index
            ]

        best_abs_t = -np.inf
        data_best = pd.DataFrame()
        best_variant = ""

        for variant_index in transf_variants_alt.index:
            altlr = transf_variants_alt.loc[variant_index].to_numpy()
            reflr = transf_variants_ref.loc[variant_index].to_numpy()

            (
                GEX_filtered,
                altlr_filtered,
                reflr_filtered,
                phenotype_cov_filtered,
                cov_values_filtered,
            ) = self.filter_arrays(GEX, altlr, reflr, phenotype_cov, cov_values)

            if len(GEX_filtered) == 0:
                continue

            # Compute s and d
            s = reflr_filtered + altlr_filtered
            d = reflr_filtered - altlr_filtered
            n = len(GEX_filtered)

            # Build design matrices for 1-df t-test on d
            y = GEX_filtered.astype(float)

            # Null model: Phenotype ~ s + phenotype_cov + sample_covariates
            cov_blocks = [s]
            if phenotype_cov_filtered is not None:
                cov_blocks.append(phenotype_cov_filtered)
            cov_blocks.extend(cov_values_filtered)

            X_null = np.column_stack(
                [np.ones(n)] + [np.asarray(c, dtype=float) for c in cov_blocks]
            )

            # Alt model: adds d
            X_alt = np.column_stack(
                [np.ones(n), s, d]
                + (
                    [phenotype_cov_filtered]
                    if phenotype_cov_filtered is not None
                    else []
                )
                + [np.asarray(c, dtype=float) for c in cov_values_filtered]
            )

            result = fit_ols_and_test(y, X_null, X_alt)

            # Extract t-statistic for d (index 2 in X_alt: intercept=0, s=1, d=2)
            beta_d = result["beta_alt"][2]
            se_d = result["se_alt"][2]

            if se_d > 0 and np.isfinite(se_d):
                t_stat = beta_d / se_d
            else:
                t_stat = np.nan

            abs_t = np.abs(t_stat) if np.isfinite(t_stat) else -np.inf

            if abs_t > best_abs_t:
                best_abs_t = abs_t

                # Build data dict for this variant (keeping ALTlr/REFlr for compatibility)
                data_dict = {
                    "GEX": GEX_filtered,
                    "ALTlr": altlr_filtered,
                    "REFlr": reflr_filtered,
                }

                if phenotype_cov_filtered is not None:
                    data_dict["phenotype_cov"] = phenotype_cov_filtered

                if self.cov is not None:
                    for covariate, cov_value_filtered in zip(
                        self.cov.index, cov_values_filtered
                    ):
                        data_dict[covariate] = cov_value_filtered

                data_best = pd.DataFrame(data_dict)
                best_variant = variant_index

        return best_variant, data_best

    def data_all_variants(
        self,
        GEX: np.ndarray,
        altlr: np.ndarray,
        reflr: np.ndarray,
        phenotype_cov: Optional[np.ndarray],
        cov_values: list,
    ):
        """
        Process data for association testing when in all variants mode.

        Parameters:
        - GEX: Phenotype levels.
        - altlr: ALTlr genotype values
        - reflr: REFlr genotype values
        - phenotype_cov: Phenotype-level covariate (None if not provided)
        - cov_values: Sample-level covariate values

        Returns:
        - Dataframe of filtered regression data.
        """
        (
            GEX_filtered,
            altlr_filtered,
            reflr_filtered,
            phenotype_cov_filtered,
            cov_values_filtered,
        ) = self.filter_arrays(GEX, altlr, reflr, phenotype_cov, cov_values)

        if len(GEX_filtered) == 0:
            return pd.DataFrame()

        data_dict = {
            "GEX": GEX_filtered,
            "ALTlr": altlr_filtered,
            "REFlr": reflr_filtered,
        }

        if phenotype_cov_filtered is not None:
            data_dict["phenotype_cov"] = phenotype_cov_filtered

        if self.cov is not None:
            for covariate, cov_value_filtered in zip(
                self.cov.index, cov_values_filtered
            ):
                data_dict[covariate] = cov_value_filtered

        df_data = pd.DataFrame(data_dict)

        return df_data

    def process_all_variants(
        self,
        gene_index: int,
        transf_variants_alt: pd.DataFrame,
        transf_variants_ref: pd.DataFrame,
    ):
        """
        Conduct association testing for all variants in a window instead of selecting
        only best correlated variant. Construct regression data and then run the
        regressions.

        Note: Permutation testing is skipped in all-variants mode.

        Parameters:
        - gene_index: Index of a gene of interest on the quantification file.
        - transf_variants_alt: Dataframe of transformed ALTlr variants
        - transf_variants_ref: Dataframe of transformed REFlr variants

        Returns:
        - Dataframe with association testing results for all variants (nominal only, no p_adj).
        """
        current_gene = self.quan.index[gene_index]
        GEX = pd.to_numeric(self.quan.iloc[gene_index, 3:], errors="coerce").to_numpy()

        phenotype_cov = None
        if self.phenotype_covariate_df is not None:
            phenotype_cov = (
                self.phenotype_covariate_df.loc[current_gene].to_numpy().flatten()
            )

        cov_values = []
        if self.cov is not None:
            cov_values = [
                pd.to_numeric(self.cov.loc[covariate], errors="coerce")
                .to_numpy()
                .flatten()
                for covariate in self.cov.index
            ]

        df_res_list = []

        for variant_index in transf_variants_alt.index:
            altlr = transf_variants_alt.loc[variant_index].to_numpy()
            reflr = transf_variants_ref.loc[variant_index].to_numpy()

            regression_data = self.data_all_variants(
                GEX, altlr, reflr, phenotype_cov, cov_values
            )

            nominal_res = self.gene_variant_regressions(
                gene_index,
                self.quan,
                variant_index,
                regression_data,
            )
            df_res_list.append(nominal_res)

        return pd.concat(df_res_list, ignore_index=True)

    def gene_variant_regressions(
        self,
        gene_index: int,
        quantifications: pd.DataFrame,
        variant: str,
        regression_data: pd.DataFrame,
    ):
        """
        Find associations between phenotype levels and variants using s/d parameterization.

        Model: Phenotype ~ s + d + covariates
        where s = REFlr + ALTlr (total dosage) and d = REFlr - ALTlr (allelic difference).

        Test: H0: β_d = 0 (1-df t-test), which tests if the two alleles have different
        effects on phenotype (i.e., molQTL signal).

        Parameters:
        - gene_index: Index of a gene of interest on the quantification file.
        - quantifications: Dataframe of quantifications.
        - variant: Variant ID
        - regression_data: Regression data for current gene-variant pair including ALTlr, REFlr,
            optional phenotype_cov, and sample-level covariates

        Returns:
        - associations: Dataframe with statistics (beta_s, se_s, beta_d, se_d, nominal_p)
        """
        return run_gene_variant_regressions(
            gene_index=gene_index,
            quantifications=quantifications,
            variant=variant,
            regression_data=regression_data,
            record_aic=self.record_aic,
        )

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

        with Pool(processes=self.num_cores) as pool:
            # Map the gene indices to the helper function using the Pool
            # and print the progress
            full_associations = list(
                tqdm(
                    pool.imap(self.calculate_associations_helper, range(limit)),
                    total=limit,
                )
            )

        end = time()
        print("The time of execution: ", (end - start) / 60, " min")
        print("")

        return pd.concat(full_associations)

    def calculate_associations_helper(self, gene_index: int):
        """
        Helper function to calculate associations for a single gene index.

        In trans negative-control mode (--neg_control), variants are drawn from a
        different chromosome (chr+1, wrapping) with no plausible cis-regulatory link
        to the gene. Segment consistency filtering is not applied because it is not
        meaningful across chromosomes.

        Parameters:
        - gene_index (int): The index of the gene for which associations are being calculated.

        Returns:
        - A dataframe containing the association results for the specified gene index.
        """
        if self.neg_control:
            # ── Trans negative control: skip segmentation filtering ──
            # Segment consistency filtering is not meaningful across chromosomes.
            transf_variants_alt, transf_variants_ref = self.get_neg_control_variants(
                gene_index
            )
            if transf_variants_alt.empty:
                return pd.DataFrame()
        else:
            # ── Normal cis mode ──
            current_start, current_end = self.start_end_gene_window(gene_index)
            current_variants_alt, current_variants_ref, variant_pos = (
                self.get_variants_for_gene_window(current_start, current_end)
            )

            transf_variants_alt, transf_variants_ref = (
                self.gene_variants_common_segment(
                    current_start,
                    current_end,
                    current_variants_alt,
                    current_variants_ref,
                    variant_pos,
                )
            )

        if self.all_variants_mode:
            result = self.process_all_variants(
                gene_index, transf_variants_alt, transf_variants_ref
            )
        else:
            best_variant, data_best = self.best_variant_data(
                gene_index, transf_variants_alt, transf_variants_ref, self.quan
            )

            # Guard: if no variant survived filtering, return NA result
            if best_variant == "" or data_best.empty:
                current_gene = self.quan.index[gene_index]
                row = {
                    "phenotype": current_gene,
                    "variant": np.nan,
                    "number_of_samples": 0,
                    "beta_s": np.nan,
                    "se_s": np.nan,
                    "beta_d": np.nan,
                    "se_d": np.nan,
                    "t_stat_d": np.nan,
                    "nominal_p": np.nan,
                    "r2_alt": np.nan,
                    "p_adj": np.nan,
                }
                if self.record_aic:
                    row["aic_null"] = np.nan
                    row["aic_alt"] = np.nan
                    row["delta_aic_alt_minus_null"] = np.nan
                return pd.DataFrame([row])

            association_res = self.gene_variant_regressions_permutations(
                gene_index,
                transf_variants_alt,
                transf_variants_ref,
                best_variant,
                data_best,
            )

            result = association_res

        return result
