from dataclasses import dataclass
from multiprocessing import Pool
from os import path
from time import time

import numpy as np
import pandas as pd
from scipy.stats import beta, f
from tqdm import tqdm

from statistical_utils import (
    calculate_aic_full_ols,
    fit_beta_mle,
    fit_ols_and_test,
    fit_ols_null,
)


@dataclass
class VariantFWLCache:
    """Cache for fast permutation testing using Frisch-Waugh-Lovell residualization.

    Optimized to:
    - Store idx_masky (indices into mask_y axis) instead of full sample indices
    - Store QcT (transposed) to avoid transpose per permutation
    - Store QgT instead of G_tilde to avoid matrix-vector multiply with G_tilde
    - Use projection-based RSS computation: rss1 = ||y_tilde - Qg @ (QgT @ y_tilde)||^2
    """

    idx_masky: (
        np.ndarray
    )  # indices into mask_y axis (not full sample axis); shape (n_i,)
    QcT: np.ndarray  # transposed reduced Q for covariates; shape (rank_c, n_i)
    Qg: np.ndarray  # reduced Q for residualized predictors; shape (n_i, p)
    QgT: np.ndarray  # transposed Qg; shape (p, n_i)
    df1: int  # numerator df (number of predictors)
    df2: int  # denominator df


class Cis:
    def __init__(
        self,
        chromosome,
        mode,
        phenotype_covariate,
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
    ):
        self.chromosome = chromosome

        # Load phenotype-level covariate (optional)
        if phenotype_covariate is not None:
            self.phenotype_covariate_df = self.load_and_validate_file(
                phenotype_covariate, index_col=0
            )
        else:
            self.phenotype_covariate_df = None

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
        - variants_alt: Subset of ALT genotype dataframe that contains only those variants
            that are inside the given window
        - variants_ref: Subset of REF genotype dataframe that contains only those variants
            that are inside the given window
        """
        subset_condition = (self.variant_positions > current_start) & (
            self.variant_positions < current_end
        )
        variants_alt = self.geno_alt.loc[subset_condition]
        variants_ref = self.geno_ref.loc[subset_condition]
        return variants_alt, variants_ref

    def gene_variants_common_segment(
        self,
        start: int,
        end: int,
        variants_alt: pd.DataFrame,
        variants_ref: pd.DataFrame,
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

        Returns:
        - variants_alt, variants_ref: Filtered and masked subsets of genotype dataframes.
        """
        gene_start = start + self.window
        gene_end = end - self.window

        index_array = variants_alt.index.astype(str).to_numpy()
        variant_pos = np.fromiter(
            (int(s.split(":")[1]) for s in index_array), dtype=np.int64
        )

        alt_arr = variants_alt.to_numpy(dtype=float, copy=True)
        ref_arr = variants_ref.to_numpy(dtype=float, copy=True)
        sample_cols = variants_alt.columns.to_numpy()
        n_variants, n_samples = alt_arr.shape

        seg_index = self.segmentation.index.to_numpy()
        seg_startpos = self.segmentation["startpos"].to_numpy()
        seg_endpos = self.segmentation["endpos"].to_numpy()

        for col_idx, cur_sample in enumerate(sample_cols):
            # Find segment containing gene_start for this sample
            seg_mask = (
                (seg_index == cur_sample)
                & (seg_startpos <= gene_start)
                & (seg_endpos >= gene_start)
            )
            seg_indices = np.flatnonzero(seg_mask)

            # With valid segmentation, gene_start should fall on exactly one segment
            if len(seg_indices) != 1:
                alt_arr[:, col_idx] = np.nan
                ref_arr[:, col_idx] = np.nan
                continue

            seg_idx = seg_indices[0]
            lower_bound = seg_startpos[seg_idx]
            upper_bound = seg_endpos[seg_idx]

            # Check if gene_end also falls within the same segment
            if not (lower_bound <= gene_end <= upper_bound):
                alt_arr[:, col_idx] = np.nan
                ref_arr[:, col_idx] = np.nan
                continue

            # Mask variants outside segment bounds
            outside_bounds = (variant_pos < lower_bound) | (variant_pos > upper_bound)
            alt_arr[outside_bounds, col_idx] = np.nan
            ref_arr[outside_bounds, col_idx] = np.nan

        # Reconstruct DataFrames from masked NumPy arrays
        variants_alt = pd.DataFrame(
            alt_arr, index=variants_alt.index, columns=sample_cols
        )
        variants_ref = pd.DataFrame(
            ref_arr, index=variants_ref.index, columns=sample_cols
        )

        return variants_alt, variants_ref

    def build_variant_fwl_caches(
        self,
        transf_variants_alt: pd.DataFrame,
        transf_variants_ref: pd.DataFrame,
        mask_y: np.ndarray,
        masky_pos: np.ndarray,
        phenotype_cov_full: np.ndarray,
        cov_values_full: list,
    ):
        """
        Pre-compute Frisch-Waugh-Lovell (FWL) caches for all variants in the window.

        Optimized to:
        - Build masks directly without calling filter_arrays() (avoids duplication)
        - Store idx_masky (indices into mask_y axis) instead of full sample indices
        - Store QcT (transposed) to avoid transpose per permutation
        - Store QgT instead of G_tilde for projection-based RSS computation
        - Convert DataFrames to NumPy once and iterate by row index (not label)
        - Use matrix_rank for df2 to handle rank-deficient covariates

        Parameters:
        - transf_variants_alt: DataFrame of transformed ALTlr variants
        - transf_variants_ref: DataFrame of transformed REFlr variants
        - mask_y: Boolean mask for samples passing phenotype+covariate filters
        - masky_pos: Mapping from full sample index -> mask_y position (-1 if not in mask_y)
        - phenotype_cov_full: Optional phenotype-level covariate
        - cov_values_full: List of sample-level covariates

        Returns:
        - Dictionary mapping variant_id -> VariantFWLCache
        """
        caches = {}

        # Pre-slice covariates to mask_y for efficiency
        phenotype_cov_masky = (
            phenotype_cov_full[mask_y] if phenotype_cov_full is not None else None
        )
        cov_values_masky = [cov_val[mask_y] for cov_val in cov_values_full]

        alt_mat = transf_variants_alt.to_numpy(dtype=float)
        ref_mat = transf_variants_ref.to_numpy(dtype=float)
        variant_ids = transf_variants_alt.index.to_numpy()

        for row_idx, variant_index in enumerate(variant_ids):
            altlr_full = alt_mat[row_idx]
            reflr_full = ref_mat[row_idx]

            # Build mask_full directly: start from mask_y and AND genotype masks
            mask_full = mask_y & ~np.isnan(altlr_full) & ~np.isnan(reflr_full)

            # Get indices in full sample axis
            idx_full = np.flatnonzero(mask_full)
            n_i = len(idx_full)

            # Check minimum sample size
            if n_i < 30:
                continue

            # Slice genotypes to filtered set
            altlr_f = altlr_full[idx_full]
            reflr_f = reflr_full[idx_full]

            # Check genotype variance
            if not self.check_grouping(altlr_f, reflr_f):
                continue

            # Convert idx_full to idx_masky (indices into mask_y axis)
            idx_masky = masky_pos[idx_full]
            # Sanity check: all indices should be valid (no -1 values)
            # since mask_full is a subset of mask_y
            assert np.all(idx_masky >= 0), "idx_masky contains invalid indices"

            # Build covariate matrix C (on filtered set, sliced from mask_y arrays)
            cov_blocks = []
            if phenotype_cov_masky is not None:
                cov_blocks.append(
                    np.asarray(phenotype_cov_masky[idx_masky], dtype=float)
                )
            for cov_val_masky in cov_values_masky:
                cov_blocks.append(np.asarray(cov_val_masky[idx_masky], dtype=float))

            C = (
                np.column_stack([np.ones(n_i)] + cov_blocks)
                if len(cov_blocks) > 0
                else np.ones((n_i, 1))
            )

            # Compute QR of C for FWL
            try:
                Qc, Rc = np.linalg.qr(C, mode="reduced")
            except np.linalg.LinAlgError:
                continue

            # Store QcT (transposed) to avoid transpose per permutation
            QcT = Qc.T

            # Stack predictors: ALTlr and REFlr on filtered set
            G = np.column_stack(
                [
                    np.asarray(altlr_f, dtype=float),
                    np.asarray(reflr_f, dtype=float),
                ]
            )

            # Residualize G w.r.t. C: G_tilde = (I - Qc Qc^T) G
            G_tilde = G - Qc @ (QcT @ G)

            # Compute QR of G_tilde
            try:
                Qg, Rg = np.linalg.qr(G_tilde, mode="reduced")
            except np.linalg.LinAlgError:
                continue

            # Store QgT (transposed)
            # The projection-based RSS computation uses: proj = Qg @ (QgT @ y_tilde)
            QgT = Qg.T

            # Degrees of freedom
            rank_c = np.linalg.matrix_rank(C)
            p = G.shape[1]  # 2 (ALTlr, REFlr)
            df1 = p
            df2 = n_i - rank_c - p

            if df2 <= 0:
                continue

            caches[variant_index] = VariantFWLCache(
                idx_masky=idx_masky,
                QcT=QcT,
                Qg=Qg,
                QgT=QgT,
                df1=df1,
                df2=df2,
            )

        return caches

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

        Permutation logic:
        - Keep genotypes fixed.
        - Permute this gene's phenotype across samples.
        - For each permutation, re-run the cis-window scan (all variants in the window),
          recording the best F-statistic and its degrees of freedom.
        - Compute one p_best per permutation from best F and df2.
        - "direct": empirical adjusted p-value from best-permutation p-values.
        - "beta": beta approximation fitted to best-permutation p-values (one p per permutation,
          derived from best F and its df2).
        """
        # Nominal association
        actual_associations = self.gene_variant_regressions(
            gene_index, self.quan, variant, regression_data
        )

        # No permutations requested, return nominal results
        if self.num_permutations == 0:
            return actual_associations

        # If no usable data for the nominal pair, cannot permute
        if regression_data.shape[0] == 0:
            actual_associations["p_adj"] = np.nan
            return actual_associations

        # Prepare fixed (unpermuted) inputs for the scan
        current_gene = self.quan.index[gene_index]

        # Full phenotype across samples (same ordering as genotype columns / self.samples)
        GEX_full = pd.to_numeric(
            self.quan.iloc[gene_index, 3:], errors="coerce"
        ).to_numpy(dtype=float)

        # Optional phenotype-level covariate (NOT permuted)
        phenotype_cov_full = None
        if self.phenotype_covariate_df is not None:
            phenotype_cov_full = (
                self.phenotype_covariate_df.loc[current_gene].to_numpy().flatten()
            ).astype(float)

        # Optional sample-level covariates (NOT permuted)
        cov_values_full = []
        cov_names = []
        if self.cov is not None:
            cov_names = list(self.cov.index)
            cov_values_full = [
                pd.to_numeric(self.cov.loc[covariate], errors="coerce")
                .to_numpy()
                .flatten()
                .astype(float)
                for covariate in cov_names
            ]

        # Keep the nominal BEST p-value for permutation adjustment
        nominal_best_p = float(actual_associations["nominal_p"].iloc[0])

        # Freedman-Lane (residual-based) permutation to preserve covariate structure
        # Build phenotype+covariate mask
        mask_y = ~np.isnan(GEX_full)
        if phenotype_cov_full is not None:
            mask_y &= ~np.isnan(phenotype_cov_full)
        for cov_val in cov_values_full:
            mask_y &= ~np.isnan(cov_val)

        # Create mapping from full sample index -> mask_y position
        # masky_pos[i] = position in mask_y axis, or -1 if not in mask_y
        n_samples = len(GEX_full)
        masky_pos = np.full(n_samples, -1, dtype=np.intp)
        masky_pos[mask_y] = np.arange(np.sum(mask_y))

        # Build FWL caches for fast permutation scanning
        variant_caches = self.build_variant_fwl_caches(
            transf_variants_alt,
            transf_variants_ref,
            mask_y,
            masky_pos,
            phenotype_cov_full,
            cov_values_full,
        )

        if not variant_caches:
            # No variants passed filtering; cannot permute
            actual_associations["p_adj"] = np.nan
            return actual_associations

        # Filter arrays by phenotype mask only (work in mask_y axis)
        y_masky = GEX_full[mask_y].astype(float)
        phenotype_cov_masky = (
            phenotype_cov_full[mask_y] if phenotype_cov_full is not None else None
        )
        cov_values_masky = [cov_val[mask_y] for cov_val in cov_values_full]

        # Build null design matrix (on mask_y filtered set)
        cov_blocks = []
        if phenotype_cov_masky is not None:
            cov_blocks.append(np.asarray(phenotype_cov_masky, dtype=float))
        for cov_val in cov_values_masky:
            cov_blocks.append(np.asarray(cov_val, dtype=float))

        X_null = (
            np.column_stack([np.ones(len(y_masky))] + cov_blocks)
            if len(cov_blocks) > 0
            else np.ones((len(y_masky), 1))
        )

        # Fit null model and compute residuals
        # Store yhat_masky and resid_masky for Freedman-Lane permutation
        yhat_masky, resid_masky = fit_ols_null(y_masky, X_null)

        # Permutation scan with cached decompositions
        best_p_perms = []
        n_masky = len(resid_masky)

        # Use per-gene RNG to avoid identical permutation sequences across
        # multiprocessing workers
        rng = np.random.default_rng(seed=12345 + gene_index)

        for _ in range(self.num_permutations):
            # Freedman-Lane: permute residuals in mask_y axis
            perm = rng.permutation(n_masky)
            y_perm_masky = yhat_masky + resid_masky[perm]

            best_F = -np.inf
            best_df1 = None
            best_df2 = None

            # Re-scan all variants using cached FWL decompositions
            for variant_index, cache in variant_caches.items():
                # Slice permuted phenotype using idx_masky (indices into mask_y axis)
                y_perm = y_perm_masky[cache.idx_masky]

                # FWL residualization w.r.t. covariates: y_tilde = (I - Qc Qc^T) y_perm
                # Using pre-transposed QcT to avoid transpose per iteration
                y_tilde = y_perm - cache.QcT.T @ (cache.QcT @ y_perm)

                # Projection-based RSS computation (no G_tilde needed):
                # proj = Qg @ (QgT @ y_tilde) gives the projection onto G_tilde column space
                # res = y_tilde - proj
                # rss1 = res @ res
                proj = cache.Qg @ (cache.QgT @ y_tilde)
                res = y_tilde - proj
                rss1 = float(np.dot(res, res))

                # Null RSS in residual space: y_tilde @ y_tilde
                rss0 = float(np.dot(y_tilde, y_tilde))

                if rss1 > 0 and cache.df2 > 0:
                    f_stat = ((rss0 - rss1) / cache.df1) / (rss1 / cache.df2)
                    f_stat = max(0.0, f_stat)  # Clamp negative F to 0
                else:
                    f_stat = np.nan

                if not np.isnan(f_stat) and f_stat > best_F:
                    best_F = f_stat
                    best_df1 = cache.df1
                    best_df2 = cache.df2

            # Compute p-value from best F and its df
            if best_df1 is not None and best_df2 is not None and best_F > -np.inf:
                p_best = float(f.sf(best_F, best_df1, best_df2))
            else:
                p_best = np.nan
            best_p_perms.append(p_best)

        adjusted_p_value = np.nan

        if self.perm_method == "direct":
            # Empirical adjusted p-value based on best-permutation p-values
            if not np.isnan(nominal_best_p) and len(best_p_perms) > 0:
                best_p_arr = np.asarray(best_p_perms, dtype=float)
                best_p_arr = best_p_arr[~np.isnan(best_p_arr)]
                if len(best_p_arr) > 0:
                    adjusted_p_value = float(
                        (1.0 + np.sum(best_p_arr <= nominal_best_p))
                        / (1.0 + len(best_p_arr))
                    )
        else:  # beta
            pbest = np.asarray(best_p_perms, dtype=float)
            pbest = pbest[~np.isnan(pbest)]
            if len(pbest) > 0 and not np.isnan(nominal_best_p):
                a, b = fit_beta_mle(pbest)
                # Clip nominal_best_p away from 0 and 1 for CDF stability
                nominal_best_p_clipped = np.clip(nominal_best_p, 1e-15, 1 - 1e-15)
                adjusted_p_value = float(beta.cdf(nominal_best_p_clipped, a, b))

        actual_associations["p_adj"] = adjusted_p_value
        return actual_associations

    def check_grouping(self, altlr_filtered: np.ndarray, reflr_filtered: np.ndarray):
        """
        Find if the genotype predictors have adequate variation in the data.

        Parameters:
        - altlr_filtered: Array of ALTlr values
        - reflr_filtered: Array of REFlr values

        Returns:
        - Boolean value showing if both predictors have sufficient variance
        """
        # For continuous predictors, check that standard deviation is non-trivial
        std_altlr = np.std(altlr_filtered)
        std_reflr = np.std(reflr_filtered)

        eps = 1e-10
        if std_altlr < eps or std_reflr < eps:
            return False

        return True

    def residualize_vector(self, y: np.ndarray, X: np.ndarray):
        """
        Residualize a vector with respect to a design matrix.

        Parameters:
        - y: Vector to residualize (n,)
        - X: Design matrix (n, p), typically includes intercept and covariates

        Returns:
        - Residuals: y - X @ beta (fitted)
        """
        if X.shape[0] != len(y):
            return np.array([])

        try:
            beta, *_ = np.linalg.lstsq(X, y.reshape(-1, 1), rcond=None)
            fitted = X @ beta
            return (y.reshape(-1, 1) - fitted).flatten()
        except np.linalg.LinAlgError:
            return np.array([])

    def filter_arrays(
        self,
        GEX: np.ndarray,
        altlr: np.ndarray,
        reflr: np.ndarray,
        phenotype_cov: np.ndarray,
        cov_values: list,
    ):
        """
        Filter data arrays and do validity checks.

        Parameters:
        - GEX: Gene expression levels
        - altlr: ALTlr genotype values
        - reflr: REFlr genotype values
        - phenotype_cov: Phenotype-level covariate (None if not provided)
        - cov_values: Sample-level covariate values

        Returns:
        Tuple of:
        - GEX_filtered: Filtered gene expression values
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
            return [], [], [], None, []

        # Filter out rows with NaNs in any of the required columns
        mask = ~np.isnan(GEX) & ~np.isnan(altlr) & ~np.isnan(reflr)

        if phenotype_cov is not None:
            mask &= ~np.isnan(phenotype_cov)

        for cov_value in cov_values:
            mask &= ~np.isnan(cov_value)

        if np.sum(mask) < 30:  # If less than 30 valid rows, skip this variant
            return [], [], [], None, []

        GEX_filtered = GEX[mask]
        altlr_filtered = altlr[mask]
        reflr_filtered = reflr[mask]
        phenotype_cov_filtered = (
            phenotype_cov[mask] if phenotype_cov is not None else None
        )
        cov_values_filtered = [cov_value[mask] for cov_value in cov_values]

        # Ensure each required column has more than one unique value
        if (
            len(np.unique(GEX_filtered)) < 2
            or len(np.unique(altlr_filtered)) < 2
            or len(np.unique(reflr_filtered)) < 2
        ):
            return [], [], [], None, []

        if not self.check_grouping(altlr_filtered, reflr_filtered):
            return [], [], [], None, []

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
        Find variant and linked data for a gene that has strongest test statistic
        (using 2-df F-test for ALTlr + REFlr).

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

        best_f_stat = -np.inf
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

            # Build data dict for this variant
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

            data_df = pd.DataFrame(data_dict)

            # Build design matrices for F-test
            y = data_df["GEX"].to_numpy(dtype=float)

            # Null model: GEX ~ phenotype_cov + sample_covariates (if any)
            null_cols = [
                col for col in data_df.columns if col not in ["GEX", "ALTlr", "REFlr"]
            ]
            X_null = np.column_stack(
                [np.ones(len(y))]
                + [data_df[col].to_numpy(dtype=float) for col in null_cols]
            )

            # Alt model: GEX ~ ALTlr + REFlr + phenotype_cov + sample_covariates
            X_alt = np.column_stack(
                [
                    np.ones(len(y)),
                    data_df["ALTlr"].to_numpy(dtype=float),
                    data_df["REFlr"].to_numpy(dtype=float),
                ]
                + [data_df[col].to_numpy(dtype=float) for col in null_cols]
            )

            result = fit_ols_and_test(y, X_null, X_alt)
            f_stat = result["f_stat"]

            if f_stat > best_f_stat and not np.isnan(f_stat):
                best_f_stat = f_stat
                data_best = data_df
                best_variant = variant_index

        return best_variant, data_best

    def data_all_variants(
        self,
        GEX: np.ndarray,
        altlr: np.ndarray,
        reflr: np.ndarray,
        phenotype_cov: np.ndarray,
        cov_values: list,
    ):
        """
        Process data for association testing when in all variants mode.

        Parameters:
        - GEX: Gene expression levels.
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
        Find associations between the gene expression values of a gene and variants
        by performing OLS regressions with F-test for ALTlr + REFlr jointly.

        Parameters:
        - gene_index: Index of a gene of interest on the quantification file.
        - quantifications: Dataframe of quantifications.
        - variant: Variant ID
        - regression_data: Regression data for current gene-variant pair including ALTlr, REFlr,
            optional phenotype_cov, and sample-level covariates

        Returns:
        - associations: Dataframe with statistics of the strengths of associations
        """
        associations = []
        current_gene = quantifications.index[gene_index]

        def create_association(
            gene,
            variant,
            beta_altlr,
            se_altlr,
            beta_reflr,
            se_reflr,
            p_value,
            r2_alt=None,
            aic_null=None,
            aic_alt=None,
            delta_aic=None,
        ):
            association = {
                "phenotype": gene,
                "variant": variant,
                "number_of_samples": regression_data.shape[0],
                "beta_altlr": beta_altlr,
                "se_altlr": se_altlr,
                "beta_reflr": beta_reflr,
                "se_reflr": se_reflr,
                "nominal_p": p_value,
                "r2_alt": r2_alt,
            }
            if self.record_aic:
                association["aic_null"] = aic_null
                association["aic_alt"] = aic_alt
                association["delta_aic_alt_minus_null"] = delta_aic
            return association

        if len(regression_data) == 0:
            associations.append(
                create_association(
                    current_gene,
                    variant,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )
            )
            return pd.DataFrame(associations)

        y = regression_data["GEX"].to_numpy(dtype=float)

        # Build design matrices
        # Null model: GEX ~ (phenotype_cov if present) + sample_covariates
        null_cols = [
            col
            for col in regression_data.columns
            if col not in ["GEX", "ALTlr", "REFlr"]
        ]
        X_null = np.column_stack(
            [np.ones(len(y))]
            + [regression_data[col].to_numpy(dtype=float) for col in null_cols]
        )

        # Alt model: GEX ~ ALTlr + REFlr + (phenotype_cov if present) + sample_covariates
        X_alt = np.column_stack(
            [
                np.ones(len(y)),
                regression_data["ALTlr"].to_numpy(dtype=float),
                regression_data["REFlr"].to_numpy(dtype=float),
            ]
            + [regression_data[col].to_numpy(dtype=float) for col in null_cols]
        )

        # Perform F-test
        result = fit_ols_and_test(y, X_null, X_alt)

        beta_altlr = result["beta_alt"][
            1
        ]  # Index 0 is intercept, 1 is ALTlr, 2 is REFlr
        se_altlr = result["se_alt"][1]
        beta_reflr = result["beta_alt"][2]
        se_reflr = result["se_alt"][2]
        pval = result["p_value"]
        r2_alt = result["r2_alt"]

        # AIC recording
        aic_null = aic_alt = delta_aic = None
        if self.record_aic:
            aic_null = calculate_aic_full_ols(
                y, X_null[:, 1:]
            )  # Exclude intercept (it's added by the function)
            aic_alt = calculate_aic_full_ols(y, X_alt[:, 1:])  # Exclude intercept
            delta_aic = aic_alt - aic_null

        associations.append(
            create_association(
                current_gene,
                variant,
                beta_altlr,
                se_altlr,
                beta_reflr,
                se_reflr,
                pval,
                r2_alt,
                aic_null,
                aic_alt,
                delta_aic,
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
        start = time()

        limit = self.quan.shape[0]  # For testing, use small number, eg. 3

        pool = Pool(processes=self.num_cores)

        # Map the gene indices to the helper function using the Pool
        # and print the progress
        full_associations = list(
            tqdm(
                pool.imap(self.calculate_associations_helper, range(limit)), total=limit
            )
        )

        pool.close()
        pool.join()

        end = time()
        print("The time of execution: ", (end - start) / 60, " min")
        print("")

        return pd.concat(full_associations)

    def calculate_associations_helper(self, gene_index: int):
        """
        Helper function to calculate associations for a single gene index.

        This function performs several steps to calculate the associations for a
        specific gene index:
        1. Determines the start and end positions for the gene window.
        2. Retrieves the variants within the gene window.
        3. Transforms the variants based on a common segment.
        4. Performs regressions to calculate associations.

        Parameters:
        - gene_index (int): The index of the gene for which associations are being calculated.

        Returns:
        - A dataframe containing the association results for the specified gene index.
        """
        current_start, current_end = self.start_end_gene_window(gene_index)
        current_variants_alt, current_variants_ref = self.get_variants_for_gene_window(
            current_start, current_end
        )

        transf_variants_alt, transf_variants_ref = self.gene_variants_common_segment(
            current_start, current_end, current_variants_alt, current_variants_ref
        )

        if self.all_variants_mode:
            result = self.process_all_variants(
                gene_index, transf_variants_alt, transf_variants_ref
            )
        else:
            best_variant, data_best = self.best_variant_data(
                gene_index, transf_variants_alt, transf_variants_ref, self.quan
            )

            association_res = self.gene_variant_regressions_permutations(
                gene_index,
                transf_variants_alt,
                transf_variants_ref,
                best_variant,
                data_best,
            )

            result = association_res

        return result
