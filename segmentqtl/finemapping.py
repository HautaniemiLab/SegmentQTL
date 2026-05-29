from multiprocessing import Pool
from os import path
from time import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

from segment_utils import (
    filter_variants_to_common_segment,
    phenotype_window_bounds,
    variants_in_window,
)
from statistical_utils import (
    adjusted_r_squared,
    ols_fit,
    r_squared,
    standardize_variants,
    standardize_variants_bootstrap,
)

# ======================================================================
# Numba-accelerated kernels for coordinate descent
# ======================================================================


def _obs_masks_to_csr(obs_masks: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert list-of-arrays obs_masks to CSR-like (indices, offsets)."""
    lengths = [len(m) for m in obs_masks]
    total = sum(lengths)
    offsets = np.empty(len(obs_masks) + 1, dtype=np.intp)
    offsets[0] = 0
    for i, ln in enumerate(lengths):
        offsets[i + 1] = offsets[i] + ln
    indices = np.empty(total, dtype=np.intp)
    for i, m in enumerate(obs_masks):
        indices[offsets[i] : offsets[i + 1]] = m
    return indices, offsets


@njit(cache=True)
def _compute_G(d_std, csr_idx, csr_off, p):
    """Precompute G_v = sum d_tilde_v^2 for each variant (observed entries)."""
    G = np.zeros(p)
    for v in range(p):
        s = csr_off[v]
        e = csr_off[v + 1]
        acc = 0.0
        for k in range(s, e):
            val = d_std[v, csr_idx[k]]
            acc += val * val
        G[v] = acc
    return G


@njit(cache=True)
def _init_residuals(r, beta, d_std, csr_idx, csr_off, p):
    """Subtract beta_v * d_std[v, obs] from residuals for non-zero betas."""
    for v in range(p):
        if beta[v] != 0.0:
            s = csr_off[v]
            e = csr_off[v + 1]
            bv = beta[v]
            for k in range(s, e):
                r[csr_idx[k]] -= bv * d_std[v, csr_idx[k]]


@njit(cache=True)
def _cd_sweep(d_std, r, beta, G, csr_idx, csr_off, p, lam1, lam2):
    """One full coordinate-descent pass. Returns max |change|."""
    max_change = 0.0
    for v in range(p):
        s = csr_off[v]
        e = csr_off[v + 1]
        if s == e or G[v] == 0.0:
            continue

        old = beta[v]

        # z_v = <d_v, r[obs]> + old * G_v
        z_v = 0.0
        for k in range(s, e):
            z_v += d_std[v, csr_idx[k]] * r[csr_idx[k]]
        z_v += old * G[v]

        # Soft threshold
        if z_v > lam1:
            new = (z_v - lam1) / (G[v] + lam2)
        elif z_v < -lam1:
            new = (z_v + lam1) / (G[v] + lam2)
        else:
            new = 0.0

        if new != old:
            diff = old - new
            for k in range(s, e):
                r[csr_idx[k]] += d_std[v, csr_idx[k]] * diff
            beta[v] = new
            change = abs(new - old)
            if change > max_change:
                max_change = change

    return max_change


@njit(cache=True)
def _subtract_beta_contributions(y_adj, beta, d_std, csr_idx, csr_off, p):
    """Subtract beta_v * d_std[v, obs_v] from y_adj in-place."""
    for v in range(p):
        if beta[v] != 0.0:
            s = csr_off[v]
            e = csr_off[v + 1]
            bv = beta[v]
            for k in range(s, e):
                y_adj[csr_idx[k]] -= bv * d_std[v, csr_idx[k]]


@njit(cache=True)
def _compute_lambda_max_grad(d_std, r0, csr_idx, csr_off, p):
    """Max |gradient| across variants at beta=0."""
    max_abs_grad = 0.0
    for v in range(p):
        s = csr_off[v]
        e = csr_off[v + 1]
        if s == e:
            continue
        g = 0.0
        for k in range(s, e):
            g += d_std[v, csr_idx[k]] * r0[csr_idx[k]]
        g = abs(g)
        if g > max_abs_grad:
            max_abs_grad = g
    return max_abs_grad


@njit(cache=True)
def _add_beta_contributions(pred, beta, d_std, csr_idx, csr_off, p):
    """Add beta_v * d_std[v, obs_v] to pred in-place (for test-fold prediction)."""
    for v in range(p):
        if beta[v] != 0.0:
            s = csr_off[v]
            e = csr_off[v + 1]
            bv = beta[v]
            for k in range(s, e):
                pred[csr_idx[k]] += bv * d_std[v, csr_idx[k]]


class MissingAwareElasticNet:
    """
    Coordinate-descent Elastic Net that handles per-entry missingness.

    The penalised objective (only variant coefficients beta_v are penalised):

        (1/2) sum_i (y_i - X_unpen_i @ theta
                      - sum_{v in O_i} beta_v * d_tilde_{v,i})^2
        + lam * alpha_en * sum_v |beta_v|
        + lam * (1 - alpha_en) / 2 * sum_v beta_v^2

    where O_i is the set of variants observed for sample i.

    Parameters:
    - alpha_en: Mixing parameter in [0, 1]. 1 = Lasso, 0 = Ridge.
    - max_iter: Maximum full coordinate-descent passes.
    - tol: Convergence tolerance (max |delta beta|).
    """

    def __init__(self, alpha_en: float = 0.5, max_iter: int = 1000, tol: float = 1e-6):
        self.alpha_en = alpha_en
        self.max_iter = max_iter
        self.tol = tol

    def _ols(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Ordinary least-squares: y = X theta.  Returns theta."""
        try:
            theta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            theta = np.zeros(X.shape[1])
        return theta

    def compute_lambda_max(
        self,
        y: np.ndarray,
        d_std: np.ndarray,
        obs_masks: List[np.ndarray],
        X_unpen: np.ndarray,
        csr_idx: Optional[np.ndarray] = None,
        csr_off: Optional[np.ndarray] = None,
    ) -> float:
        """
        Smallest lambda for which all beta_v = 0.

        At beta = 0 the sub-gradient condition requires
        |sum_{i in O_v} d_tilde_{v,i} r_i^{(0)}| <= lam * alpha_en
        for every v.
        """
        theta0 = self._ols(y, X_unpen)
        r0 = y - X_unpen @ theta0

        if csr_idx is None or csr_off is None:
            csr_idx, csr_off = _obs_masks_to_csr(obs_masks)

        max_abs_grad = _compute_lambda_max_grad(
            d_std, r0, csr_idx, csr_off, d_std.shape[0]
        )

        if self.alpha_en > 0:
            return max_abs_grad / self.alpha_en
        return max_abs_grad * 1e4  # Ridge: no finite lambda_max

    def fit(
        self,
        y: np.ndarray,
        d_std: np.ndarray,
        obs_masks: List[np.ndarray],
        X_unpen: np.ndarray,
        lam: float,
        beta_init: Optional[np.ndarray] = None,
        theta_init: Optional[np.ndarray] = None,
        csr_idx: Optional[np.ndarray] = None,
        csr_off: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Fit the model for a single lambda value.

        Parameters:
        - y: (n,) phenotype (centred recommended).
        - d_std: (p, n) standardised variant predictors, NaN where missing.
        - obs_masks: list of index arrays; obs_masks[v] = observed sample indices.
        - X_unpen: (n, q) unpenalised design matrix (intercept + covariates).
        - lam: penalty magnitude.
        - beta_init: optional warm-start for beta.
        - theta_init: optional warm-start for theta.
        - csr_idx: optional precomputed CSR indices from _obs_masks_to_csr.
        - csr_off: optional precomputed CSR offsets from _obs_masks_to_csr.

        Returns:
        - beta: (p,) penalised coefficients.
        - theta: (q,) unpenalised coefficients.
        - r: (n,) final residuals.
        - n_iter: number of iterations used.
        """
        p, n = d_std.shape
        lam1 = lam * self.alpha_en
        lam2 = lam * (1.0 - self.alpha_en)

        if csr_idx is None or csr_off is None:
            csr_idx, csr_off = _obs_masks_to_csr(obs_masks)

        # Initialise
        theta = theta_init.copy() if theta_init is not None else self._ols(y, X_unpen)
        beta = beta_init.copy() if beta_init is not None else np.zeros(p)

        # Initial residuals
        r = y - X_unpen @ theta
        _init_residuals(r, beta, d_std, csr_idx, csr_off, p)

        # Precompute G_v = sum d_tilde^2 per variant
        G = _compute_G(d_std, csr_idx, csr_off, p)

        # Coordinate descent
        n_iter = 0
        for it in range(self.max_iter):
            max_change = _cd_sweep(d_std, r, beta, G, csr_idx, csr_off, p, lam1, lam2)

            n_iter = it + 1

            # Refit unpenalised block every 5 passes
            if n_iter % 5 == 0:
                y_adj = y.copy()
                _subtract_beta_contributions(y_adj, beta, d_std, csr_idx, csr_off, p)
                theta = self._ols(y_adj, X_unpen)
                r = y_adj - X_unpen @ theta

            if max_change < self.tol:
                break

        # Final unpenalised refit
        y_adj = y.copy()
        _subtract_beta_contributions(y_adj, beta, d_std, csr_idx, csr_off, p)
        theta = self._ols(y_adj, X_unpen)
        r = y_adj - X_unpen @ theta

        return beta, theta, r, n_iter

    def fit_path_cv(
        self,
        y: np.ndarray,
        d_std: np.ndarray,
        obs_masks: List[np.ndarray],
        X_unpen: np.ndarray,
        n_lambda: int = 30,
        lambda_ratio: float = 0.01,
        n_folds: int = 5,
        cv_tau: float = 0.8,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Select lambda via K-fold CV with range-based tolerance.

        A log-spaced grid of lambdas is evaluated by K-fold CV. The selected
        lambda is the largest (most regularised) whose mean CV error is within
        cv_tau of the improvement range over the null model:
        threshold = CV_min + cv_tau * (CV_null - CV_min).
        If no lambda improves over the null, the null model is returned
        (zero hits).

        Parameters:
        - n_lambda: number of lambda grid points (log-spaced).
        - lambda_ratio: lam_min / lam_max.
        - n_folds: number of CV folds (default 5).
        - cv_tau: fraction of improvement over null that may be sacrificed
          for sparsity (default 0.8).
        - seed: random seed for fold assignment.

        Returns:
        - best_beta, best_theta, best_lambda, mean_cv_errors, lambdas,
          best_beta_refit
        """
        lam_max = self.compute_lambda_max(y, d_std, obs_masks, X_unpen)
        if lam_max <= 0:
            lam_max = 1.0
        lam_min = lam_max * lambda_ratio

        n = len(y)
        p = d_std.shape[0]

        # Precompute CSR for full obs_masks (used for final refit)
        csr_idx_full, csr_off_full = _obs_masks_to_csr(obs_masks)

        # Log-spaced grid: large (sparse) to small (dense)
        lambdas = np.geomspace(lam_max, lam_min, max(n_lambda, 2))

        # Assign folds
        rng = np.random.default_rng(seed)
        fold_ids = np.empty(n, dtype=int)
        perm = rng.permutation(n)
        for i, idx in enumerate(perm):
            fold_ids[idx] = i % n_folds

        cv_errors = np.zeros((len(lambdas), n_folds))

        for fold in range(n_folds):
            train_idx = np.where(fold_ids != fold)[0]
            test_idx = np.where(fold_ids == fold)[0]
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue

            y_train = y[train_idx]
            X_train = X_unpen[train_idx]
            d_train = d_std[:, train_idx]

            # Remap obs_masks to training-local indices
            train_bool = np.zeros(n, dtype=bool)
            train_bool[train_idx] = True
            old_to_train = np.full(n, -1, dtype=np.intp)
            old_to_train[train_idx] = np.arange(len(train_idx))

            obs_masks_train: List[np.ndarray] = []
            for v_mask in obs_masks:
                in_train = train_bool[v_mask]
                obs_masks_train.append(old_to_train[v_mask[in_train]])

            # Precompute CSR for training fold
            csr_idx_train, csr_off_train = _obs_masks_to_csr(obs_masks_train)

            # Build test-fold prediction obs_masks (remapped to test-local indices)
            old_to_test = np.full(n, -1, dtype=np.intp)
            old_to_test[test_idx] = np.arange(len(test_idx))
            test_bool = np.zeros(n, dtype=bool)
            test_bool[test_idx] = True

            obs_masks_test: List[np.ndarray] = []
            for v_mask in obs_masks:
                in_test = test_bool[v_mask]
                obs_masks_test.append(old_to_test[v_mask[in_test]])

            csr_idx_test, csr_off_test = _obs_masks_to_csr(obs_masks_test)

            # d_std remapped to test-local columns
            d_test = d_std[:, test_idx]

            # Warm-start along the lambda path
            beta_ws: Optional[np.ndarray] = None
            theta_ws: Optional[np.ndarray] = None

            for li, lam in enumerate(lambdas):
                beta, theta, _, _ = self.fit(
                    y_train,
                    d_train,
                    obs_masks_train,
                    X_train,
                    lam,
                    beta_init=beta_ws,
                    theta_init=theta_ws,
                    csr_idx=csr_idx_train,
                    csr_off=csr_off_train,
                )
                beta_ws = beta
                theta_ws = theta

                # Predict on test fold
                pred = X_unpen[test_idx] @ theta
                _add_beta_contributions(
                    pred, beta, d_test, csr_idx_test, csr_off_test, p
                )

                cv_errors[li, fold] = float(np.mean((y[test_idx] - pred) ** 2))

        # Range-based tolerance + null-model fallback
        mean_cv = cv_errors.mean(axis=1)

        # Null-model MSE = first grid point (lam_max, all betas zero)
        cv_null = mean_cv[0]
        idx_min = int(np.argmin(mean_cv))
        cv_min = mean_cv[idx_min]

        # If best CV is not better than null, return null (zero hits)
        if cv_min >= cv_null:
            best_lam = float(lambdas[0])
        else:
            threshold = cv_min + cv_tau * (cv_null - cv_min)
            # Largest lambda (= smallest index) with CV error ≤ threshold
            idx_sel = idx_min
            for i in range(len(lambdas)):
                if mean_cv[i] <= threshold:
                    idx_sel = i
                    break
            best_lam = float(lambdas[idx_sel])

        # Final refit on all data
        best_beta, best_theta, _, _ = self.fit(
            y,
            d_std,
            obs_masks,
            X_unpen,
            best_lam,
            csr_idx=csr_idx_full,
            csr_off=csr_off_full,
        )

        return (
            best_beta,
            best_theta,
            best_lam,
            mean_cv,
            lambdas,
            best_beta.copy(),
        )


class Finemapping:
    """
    SegmentQTL finemapping: segment-aware missing Elastic Net + stability
    selection.

    Uses the same data-loading and segment-filtering logic as Cis mode,
    then fits a joint missing-aware Elastic Net model across all variants in
    each phenotype window, with cross-validated lambda selection and stability
    selection for robust identification of causal variants.

    Parameters:
    - chromosome: Chromosome identifier (e.g. "chr1").
    - quantifications: Path to quantifications CSV.
    - covariates: Path to sample-level covariates CSV (optional).
    - segmentation: Path to segmentation CSV.
    - genotype_alt: Path to ALTlr genotype CSV for this chromosome.
    - genotype_ref: Path to REFlr genotype CSV for this chromosome.
    - copynumber: Path to phenotype-level copy-number covariate CSV
      (optional). This is CNlr -- included as an unpenalised predictor
      in the Elastic Net.
    - phenotype_covariate: Path to additional phenotype-level covariate CSV
      (optional; also unpenalised).
    - window: Cis-window size in bp.
    - num_cores: Number of parallel workers.
    - alpha_en: Elastic Net mixing (1 = Lasso, 0 = Ridge).
    - coverage_tau: Minimum fraction of samples observed for a variant
      to be retained.
    - n_bootstrap: Number of stability-selection resamples.
    - subsample_frac: Fraction of samples per resample.
    - n_lambda: Number of lambda grid points for CV-based selection.
    - lambda_ratio: lam_min / lam_max ratio for the grid.
    - cv_tau: Range-based tolerance for lambda selection
      (default 0.8 = sacrifice up to 80% of improvement over null for
      sparsity).
    - min_obs_boot: Hard minimum observed entries per variant inside
      each bootstrap subsample. Variants with fewer observations are
      skipped for that resample.
    - compute_r2: If True, compute R² for baseline (covariates only)
      and full (covariates + selected variants) models for each
      phenotype and include the values in the output.
    - r2_stability_threshold: Minimum stability score for variant
      selection in R² computation.
    """

    def __init__(
        self,
        chromosome: str,
        quantifications: str,
        covariates: Optional[str],
        segmentation: str,
        genotype_alt: str,
        genotype_ref: str,
        copynumber: Optional[str],
        phenotype_covariate: Optional[str],
        window: int,
        num_cores: int,
        alpha_en: float = 0.5,
        coverage_tau: float = 0.6,
        n_bootstrap: int = 200,
        subsample_frac: float = 0.8,
        n_lambda: int = 30,
        lambda_ratio: float = 0.01,
        cv_tau: float = 0.8,
        min_obs_boot: int = 20,
        compute_r2: bool = False,
        r2_stability_threshold: float = 0.6,
    ):
        self.chromosome = chromosome

        # Load phenotype data
        self.full_quan = self._load(quantifications, index_col=3)
        self.quan = self.full_quan[self.full_quan["chr"] == self.chromosome]
        quan_samples = self.quan.columns.to_numpy()[3:]

        # Load sample-level covariates (optional)
        self.cov = self._load(covariates, index_col=None) if covariates else None
        if self.cov is not None:
            # Keep only columns that are actual sample ids from quantifications.
            # This drops metadata columns such as a leading "sample" column.
            cov_sample_cols = [s for s in quan_samples if s in self.cov.columns]
            if len(cov_sample_cols) == 0:
                raise ValueError(
                    "[finemapping] covariates file has no sample columns matching "
                    f"quantifications for {self.chromosome}."
                )
            self.cov = self.cov.loc[:, cov_sample_cols]

        # Load genotypes
        self.geno_alt = self._load(genotype_alt, index_col=0)
        self.geno_ref = self._load(genotype_ref, index_col=0)

        # Pre-load optional sample-aligned tables so we can include them in
        # the canonical-sample intersection.
        cn_df: Optional[pd.DataFrame] = None
        if copynumber:
            cn_df = self._load(copynumber, index_col=0)
        pc_df: Optional[pd.DataFrame] = None
        if phenotype_covariate:
            pc_df = self._load(phenotype_covariate, index_col=0)

        # Canonical sample set = intersection of every column-aligned table.
        # Preserves quantifications order so y_full positional slicing stays
        # consistent. This is required when (val) genotypes / covariates are
        # a strict subset of the quantifications samples.
        sample_set = set(self.geno_alt.columns).intersection(self.geno_ref.columns)
        if self.cov is not None:
            sample_set &= set(self.cov.columns)
        if cn_df is not None:
            sample_set &= set(cn_df.columns)
        if pc_df is not None:
            sample_set &= set(pc_df.columns)
        self.samples = np.array(
            [s for s in quan_samples if s in sample_set], dtype=quan_samples.dtype
        )
        if len(self.samples) == 0:
            raise ValueError(
                f"[finemapping] No samples are common to quantifications, "
                f"genotypes, and the optional copynumber/phenotype_covariate "
                f"tables for {self.chromosome}. Check input cohort consistency."
            )
        if len(self.samples) < len(quan_samples):
            dropped = len(quan_samples) - len(self.samples)
            print(
                f"[finemapping] {self.chromosome}: dropped {dropped}/"
                f"{len(quan_samples)} quantification samples not present in "
                f"all input tables; using {len(self.samples)} samples."
            )

        # Re-slice quantifications data columns to the canonical sample set
        # so that the positional `iloc[..., 3:]` access yields y aligned with
        # self.samples.
        meta_cols = list(self.quan.columns[:3])
        self.quan = self.quan[meta_cols + list(self.samples)]

        # Load segmentation
        self.segmentation = self._load(segmentation, index_col=0)
        self.segmentation = self.segmentation[self.segmentation.chr == self.chromosome]
        self.segmentation = self.segmentation[
            self.segmentation.index.isin(self.samples)
        ]

        # Align genotypes to the canonical sample order
        self.geno_alt = self.geno_alt[self.samples]
        self.geno_ref = self.geno_ref[self.samples]

        common = self.geno_alt.index.intersection(self.geno_ref.index)
        self.geno_alt = self.geno_alt.loc[common]
        self.geno_ref = self.geno_ref.loc[common]

        idx_arr = self.geno_alt.index.astype(str).to_numpy()
        self.variant_positions = np.fromiter(
            (int(s.split(":")[1]) for s in idx_arr), dtype=np.int64
        )

        # Align optional sample tables to the canonical order
        self.cov = self.cov[self.samples] if self.cov is not None else None
        self.copynumber_df = cn_df[self.samples] if cn_df is not None else None
        self.phenotype_covariate_df = pc_df[self.samples] if pc_df is not None else None

        # Window & parallelism
        self.window = window
        self.num_cores = num_cores

        # EN hyper-parameters
        self.alpha_en = alpha_en
        self.coverage_tau = coverage_tau
        self.n_bootstrap = n_bootstrap
        self.subsample_frac = subsample_frac
        self.n_lambda = n_lambda
        self.lambda_ratio = lambda_ratio
        self.cv_tau = cv_tau
        self.min_obs_boot = min_obs_boot
        self.compute_r2_flag = compute_r2
        self.r2_stability_threshold = r2_stability_threshold

        self.bootstrap_nonzero_diagnostics = pd.DataFrame(
            columns=["phenotype", "variant", "bootstrap_iteration", "beta_full"]
        )

    def _load(self, fp: str, index_col) -> pd.DataFrame:
        if not path.exists(fp):
            raise FileNotFoundError(f"File '{fp}' not found.")
        df = pd.read_csv(fp, index_col=index_col)
        if df.shape[0] == 0:
            raise ValueError(f"File '{fp}' has no rows.")
        return df

    def _phenotype_window(self, pheno_index: int) -> Tuple[int, int]:
        """Cis-window boundaries for phenotype at pheno_index."""
        return phenotype_window_bounds(self.quan, pheno_index, self.window)

    def _variants_in_window(
        self, start: int, end: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        return variants_in_window(
            self.geno_alt,
            self.geno_ref,
            self.variant_positions,
            start,
            end,
        )

    def _segment_filter(
        self,
        start: int,
        end: int,
        variants_alt: pd.DataFrame,
        variants_ref: pd.DataFrame,
        variant_pos: np.ndarray,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Segment-consistency filter identical to Cis.gene_variants_common_segment.

        Masks genotype entries to NaN when the phenotype and variant do not
        sit on the same segment for a given sample.
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

    def _prepare_phenotype(self, pheno_index: int) -> Optional[dict]:
        """
        Build the d-matrix, phenotype vector, and unpenalised design for
        one phenotype.

        Returns None when the phenotype cannot be finemapped (too few
        data). Otherwise returns a dict with keys:
        - variant_ids: (p_window,) variant identifiers
        - d_raw: (p_window, n_good) raw d values (NaN = missing)
        - y: (n_good,) centred phenotype
        - X_unpen: (n_good, q) unpenalised design (intercept + covs)
        - n_samples: int
        - cnlr_col: column index of CNlr in X_unpen, or None
        - sample_ids: (n_good,) sample identifiers retained after masking
        - unpen_blocks_raw: list of (n_good,) raw, post-mask vectors for every
          non-intercept column of X_unpen, in column order. Useful for
          re-standardising the validation cohort with main-cohort
          parameters in frozen-mode validation.
        - unpen_transforms: list of (mu, sd, frozen_constant) tuples per
          non-intercept block. ``frozen_constant`` is None when the block
          was actually standardised on the main cohort (the validation
          frozen-mode rule is then ``(raw - mu) / sd``). For degenerate
          blocks (sd <= 1e-10) ``frozen_constant`` is the constant value
          the block carried during training; the validation frozen-mode
          rule is to replace the entire column by that constant so theta
          operates on the exact training feature.
        - unpen_names: human-readable label per non-intercept block, in
          the same order as ``unpen_transforms`` and the non-intercept
          columns of ``X_unpen`` (e.g. ``["CN", "phenotype_cov", "sex",
          "age"]``). The intercept column is unnamed (always present).
        """
        current_pheno = self.quan.index[pheno_index]

        # 1) Variants in window -> segment filter
        ws, we = self._phenotype_window(pheno_index)
        va, vr, vpos = self._variants_in_window(ws, we)
        if va.empty:
            return None
        va, vr = self._segment_filter(ws, we, va, vr, vpos)

        # 2) d = REFlr - ALTlr  (NaN propagation)
        alt_arr = va.to_numpy(dtype=float)
        ref_arr = vr.to_numpy(dtype=float)
        d_all = ref_arr - alt_arr  # (p_window, n_all_samples)
        variant_ids_all = va.index.to_numpy()

        # 3) Phenotype y
        y_full = pd.to_numeric(
            self.quan.iloc[pheno_index, 3:], errors="coerce"
        ).to_numpy(dtype=float)

        # 4) Phenotype-level CN (copynumber = CNlr)
        cnlr_full = None
        if self.copynumber_df is not None:
            if current_pheno in self.copynumber_df.index:
                cnlr_full = (
                    self.copynumber_df.loc[current_pheno]
                    .to_numpy()
                    .flatten()
                    .astype(float)
                )

        # 5) Phenotype-level covariate (optional additional unpenalised)
        phenotype_cov_full = None
        if self.phenotype_covariate_df is not None:
            if current_pheno in self.phenotype_covariate_df.index:
                phenotype_cov_full = (
                    self.phenotype_covariate_df.loc[current_pheno]
                    .to_numpy()
                    .flatten()
                    .astype(float)
                )

        # 6) Sample-level covariates
        cov_full: List[np.ndarray] = []
        cov_names: List[str] = []
        if self.cov is not None:
            for covariate in self.cov.index:
                cov_full.append(
                    pd.to_numeric(self.cov.loc[covariate], errors="coerce")
                    .to_numpy()
                    .flatten()
                    .astype(float)
                )
                cov_names.append(str(covariate))

        # Defensive check: all sample-aligned vectors must have the same length.
        expected_n = len(y_full)
        if cnlr_full is not None and len(cnlr_full) != expected_n:
            raise ValueError(
                f"[finemapping] CN covariate length mismatch for {current_pheno!r}: "
                f"expected {expected_n}, got {len(cnlr_full)}. "
                "Check sample columns across quantifications/covariates/genotypes."
            )
        if phenotype_cov_full is not None and len(phenotype_cov_full) != expected_n:
            raise ValueError(
                f"[finemapping] phenotype_covariate length mismatch for {current_pheno!r}: "
                f"expected {expected_n}, got {len(phenotype_cov_full)}. "
                "Check sample columns across quantifications/covariates/genotypes."
            )
        for cov_name, cv in zip(cov_names, cov_full):
            if len(cv) != expected_n:
                raise ValueError(
                    f"[finemapping] sample covariate {cov_name!r} length mismatch for "
                    f"{current_pheno!r}: expected {expected_n}, got {len(cv)}. "
                    "Check sample columns across quantifications/covariates/genotypes."
                )

        # 7) Global sample mask: non-NaN in y, CNlr, phenotype_cov, covariates
        #    (d is allowed to be NaN -- handled by EN as missing)
        mask = ~np.isnan(y_full)
        if cnlr_full is not None:
            mask &= ~np.isnan(cnlr_full)
        if phenotype_cov_full is not None:
            mask &= ~np.isnan(phenotype_cov_full)
        for cv in cov_full:
            mask &= ~np.isnan(cv)

        n_good = int(np.sum(mask))
        if n_good < 30:
            return None

        # Reduce to good samples
        y = y_full[mask]
        d_raw = d_all[:, mask]

        # Centre y
        y = y - np.mean(y)

        # 8) Build X_unpen = [intercept, CNlr_std, phenotype_cov_std, cov_std...]
        blocks: List[np.ndarray] = [np.ones(n_good)]
        unpen_blocks_raw: List[np.ndarray] = []
        unpen_transforms: List[Tuple[float, float, Optional[float]]] = []
        unpen_names: List[str] = []
        cnlr_col: Optional[int] = None

        def _record_block(raw_vec: np.ndarray) -> np.ndarray:
            unpen_blocks_raw.append(raw_vec.copy())
            mu = float(np.mean(raw_vec))
            sd = float(np.std(raw_vec))
            if sd > 1e-10:
                unpen_transforms.append((mu, sd, None))
                return (raw_vec - mu) / sd
            # Degenerate: training column was the constant ``mu``. Record
            # ``mu`` as the frozen constant so validation frozen-mode
            # substitutes the same constant column rather than passing
            # validation's own (possibly different) constant through.
            unpen_transforms.append((mu, 1.0, mu))
            return raw_vec

        if cnlr_full is not None:
            cnlr = _record_block(cnlr_full[mask])
            cnlr_col = len(blocks)
            unpen_names.append("CN")
            blocks.append(cnlr)

        if phenotype_cov_full is not None:
            pcov = _record_block(phenotype_cov_full[mask])
            unpen_names.append("phenotype_cov")
            blocks.append(pcov)

        for cv, cv_name in zip(cov_full, cov_names):
            blocks.append(_record_block(cv[mask]))
            unpen_names.append(cv_name)

        X_unpen = np.column_stack(blocks)
        sample_ids = self.samples[mask]

        return {
            "variant_ids": variant_ids_all,
            "d_raw": d_raw,
            "y": y,
            "X_unpen": X_unpen,
            "n_samples": n_good,
            "cnlr_col": cnlr_col,
            "sample_ids": sample_ids,
            "unpen_blocks_raw": unpen_blocks_raw,
            "unpen_transforms": unpen_transforms,
            "unpen_names": unpen_names,
        }

    def _stability_selection(
        self,
        phenotype: str,
        kept_ids: np.ndarray,
        y: np.ndarray,
        d_raw_kept: np.ndarray,
        X_unpen: np.ndarray,
        lam_selected: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]]]:
        """
        Bootstrap stability selection at a fixed lambda.

        Variant predictors are re-standardised inside each subsample (observed
        entries only) but the variant set is frozen from the full-data
        coverage filter. Bootstrap subsamples only skip a variant when it has
        too few observed entries (< min_obs_boot) or zero variance, so that
        pi_v reflects "selected when fit is attempted", not "selected AND
        passed a second coverage filter".

        Parameters:
        - y: (n,) centred phenotype.
        - d_raw_kept: (p_kept, n) raw d values for kept variants, NaN = missing.
        - X_unpen: (n, q) unpenalised design.
        - lam_selected: penalty selected on full data.
        - rng: random generator (seeded per phenotype).

        Returns:
        - pi_v: (p_kept,) selection probability (fraction of bootstraps
          where beta_v != 0).
        - mean_beta: (p_kept,) mean beta among selections (standardised units).
        - sign_consist: (p_kept,) fraction of selections with beta > 0.
        - bootstrap_nonzero_rows: list of diagnostic records for each
          bootstrap/variant pair where beta_full != 0.
        """
        n = len(y)
        p_kept = d_raw_kept.shape[0]
        n_sub = max(30, int(n * self.subsample_frac))

        sel_count = np.zeros(p_kept)
        beta_sum = np.zeros(p_kept)
        pos_count = np.zeros(p_kept)
        n_effective_boot = 0
        bootstrap_nonzero_rows: List[Dict[str, object]] = []

        en = MissingAwareElasticNet(alpha_en=self.alpha_en, max_iter=500, tol=1e-5)

        for boot_idx in range(self.n_bootstrap):
            sub = np.sort(rng.choice(n, n_sub, replace=False))
            y_sub = y[sub]
            X_sub = X_unpen[sub]
            d_raw_sub = d_raw_kept[:, sub]

            # Re-standardise on subsample (no coverage filter applied)
            d_sub, obs_sub, keep_inner = standardize_variants_bootstrap(
                d_raw_sub, min_obs_boot=self.min_obs_boot
            )
            if len(keep_inner) == 0:
                continue

            n_effective_boot += 1
            beta, _, _, _ = en.fit(y_sub, d_sub, obs_sub, X_sub, lam_selected)

            for j, orig_v in enumerate(keep_inner):
                if abs(beta[j]) > 1e-10:
                    sel_count[orig_v] += 1
                    beta_sum[orig_v] += beta[j]
                    if beta[j] > 0:
                        pos_count[orig_v] += 1
                    bootstrap_nonzero_rows.append(
                        {
                            "phenotype": phenotype,
                            "variant": kept_ids[orig_v],
                            "bootstrap_iteration": boot_idx + 1,
                            "beta_full": float(beta[j]),
                        }
                    )

        pi_v = sel_count / max(n_effective_boot, 1)
        with np.errstate(invalid="ignore"):
            mean_beta = np.where(sel_count > 0, beta_sum / sel_count, 0.0)
            sign_consist = np.where(sel_count > 0, pos_count / sel_count, np.nan)

        return pi_v, mean_beta, sign_consist, bootstrap_nonzero_rows

    def _empty_result(self, phenotype: str, n_samples: int = 0) -> pd.DataFrame:
        row = {
            "phenotype": phenotype,
            "variant": np.nan,
            "n_samples": n_samples,
            "n_variants": 0,
            "n_obs": 0,
            "median_n_obs": np.nan,
            "mean_d": np.nan,
            "sd_d": np.nan,
            "frac_alt_gain": np.nan,
            "frac_ref_gain": np.nan,
            "frac_balanced": np.nan,
            "stability_score": np.nan,
            "mean_beta": np.nan,
            "sign_consistency": np.nan,
            "lambda_selected": np.nan,
            "lambda_bic": np.nan,
            "beta_full": np.nan,
            "beta_full_raw": np.nan,
            "beta_cnlr": np.nan,
            "effect_interpretation": np.nan,
        }
        return pd.DataFrame([row])

    def _empty_bootstrap_diagnostics(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["phenotype", "variant", "bootstrap_iteration", "beta_full"]
        )

    def _process_phenotype(
        self, pheno_index: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[dict]]:
        """Full finemapping pipeline for one phenotype."""
        current_pheno = self.quan.index[pheno_index]
        data = self._prepare_phenotype(pheno_index)
        if data is None:
            return (
                self._empty_result(current_pheno),
                self._empty_bootstrap_diagnostics(),
                None,
            )

        variant_ids = data["variant_ids"]
        d_raw = data["d_raw"]
        y = data["y"]
        X_unpen = data["X_unpen"]
        n_samples = data["n_samples"]
        cnlr_col = data["cnlr_col"]

        # Standardise & coverage filter (full data)
        d_std, obs_masks, keep_idx, sd_vec, _mu_vec = standardize_variants(
            d_raw, self.coverage_tau, n_samples
        )
        if len(keep_idx) == 0:
            return (
                self._empty_result(current_pheno, n_samples),
                self._empty_bootstrap_diagnostics(),
                None,
            )

        kept_ids = variant_ids[keep_idx]
        p_kept = len(kept_ids)

        # Compute median n_obs across kept variants for diagnostics
        median_n_obs = float(np.median([len(m) for m in obs_masks]))

        # CV-based lambda selection on full data
        en = MissingAwareElasticNet(alpha_en=self.alpha_en)
        best_beta, best_theta, lam_selected, _, _, best_beta_refit = en.fit_path_cv(
            y,
            d_std,
            obs_masks,
            X_unpen,
            n_lambda=self.n_lambda,
            lambda_ratio=self.lambda_ratio,
            cv_tau=self.cv_tau,
            seed=42 + pheno_index,
        )

        # Extract CNlr coefficient from unpenalised block
        beta_cnlr = float(best_theta[cnlr_col]) if cnlr_col is not None else np.nan

        # Stability selection
        rng = np.random.default_rng(seed=42 + pheno_index)
        d_raw_kept = d_raw[keep_idx]

        pi_v, mean_beta, sign_consist, bootstrap_nonzero_rows = (
            self._stability_selection(
                current_pheno,
                kept_ids,
                y,
                d_raw_kept,
                X_unpen,
                lam_selected,
                rng,
            )
        )

        # ── R² computation (optional) ──
        r2_info: Optional[dict] = None
        if self.compute_r2_flag:
            r2_info = self._compute_r2_from_fit(
                current_pheno,
                y,
                X_unpen,
                d_std,
                obs_masks,
                pi_v,
                best_beta_refit,
                kept_ids,
                n_samples,
            )

        # Assemble per-variant results
        d_raw_kept = d_raw[keep_idx]
        d_threshold = 0.1  # ~7% allelic ratio shift
        rows = []
        for j in range(p_kept):
            obs_d = d_raw_kept[j, obs_masks[j]]
            n_obs_j = len(obs_d)
            m_d = float(np.mean(obs_d))
            s_d = float(np.std(obs_d))
            b = best_beta_refit[j]

            # Allelic imbalance fractions
            frac_alt_gain = float(np.sum(obs_d < -d_threshold)) / n_obs_j
            frac_ref_gain = float(np.sum(obs_d > d_threshold)) / n_obs_j
            frac_balanced = float(np.sum(np.abs(obs_d) <= d_threshold)) / n_obs_j

            # Causal interpretation from beta sign alone
            if abs(b) < 1e-10:
                interp = "no effect"
            elif b < 0:
                interp = "ALT gain increases phenotype; REF gain decreases phenotype"
            else:
                interp = "REF gain increases phenotype; ALT gain decreases phenotype"

            rows.append(
                {
                    "phenotype": current_pheno,
                    "variant": kept_ids[j],
                    "n_samples": n_samples,
                    "n_variants": p_kept,
                    "n_obs": n_obs_j,
                    "median_n_obs": median_n_obs,
                    "mean_d": m_d,
                    "sd_d": s_d,
                    "frac_alt_gain": frac_alt_gain,
                    "frac_ref_gain": frac_ref_gain,
                    "frac_balanced": frac_balanced,
                    "stability_score": float(pi_v[j]),
                    "mean_beta": float(mean_beta[j]),
                    "sign_consistency": (
                        float(sign_consist[j])
                        if np.isfinite(sign_consist[j])
                        else np.nan
                    ),
                    "lambda_selected": float(lam_selected),
                    "lambda_bic": float(lam_selected),
                    "beta_full": float(b),
                    "beta_full_raw": float(b / sd_vec[j]),
                    "beta_cnlr": beta_cnlr,
                    "effect_interpretation": interp,
                }
            )

        bootstrap_diag_df = pd.DataFrame(
            bootstrap_nonzero_rows,
            columns=["phenotype", "variant", "bootstrap_iteration", "beta_full"],
        )

        return pd.DataFrame(rows), bootstrap_diag_df, r2_info

    def calculate_finemapping(self, phenotype_id: Optional[str] = None) -> pd.DataFrame:
        """
        Run finemapping for phenotypes on this chromosome.

        Parameters:
        - phenotype_id: optional phenotype identifier. If provided, only
          that phenotype is finemapped. Otherwise all phenotypes on the
          chromosome are processed.
        """
        start = time()

        if phenotype_id is not None:
            if phenotype_id not in self.quan.index:
                raise ValueError(
                    f"Phenotype '{phenotype_id}' not found on {self.chromosome}. "
                    f"Available: {self.quan.index.tolist()[:5]}{'...' if len(self.quan) > 5 else ''}"
                )
            loc = self.quan.index.get_loc(phenotype_id)
            assert isinstance(loc, int), f"Duplicate phenotype ID '{phenotype_id}'"
            pheno_indices: list[int] = [loc]
            desc = f"Finemapping {phenotype_id}"
        else:
            pheno_indices = list(range(self.quan.shape[0]))
            desc = "Finemapping"

        n_phenos = len(pheno_indices)

        if self.num_cores == 1 or n_phenos == 1:
            # Sequential: no pickling overhead
            results = [
                self._process_phenotype_safe(i) for i in tqdm(pheno_indices, desc=desc)
            ]
        else:
            with Pool(processes=self.num_cores) as pool:
                results = list(
                    tqdm(
                        pool.imap(self._process_phenotype_safe, pheno_indices),
                        total=n_phenos,
                        desc=desc,
                    )
                )

        elapsed = (time() - start) / 60
        print(f"Finemapping completed in {elapsed:.1f} min")

        main_results = [res[0] for res in results]
        diag_results = [res[1] for res in results]
        r2_rows = [res[2] for res in results if res[2] is not None]

        self.bootstrap_nonzero_diagnostics = pd.concat(diag_results, ignore_index=True)
        self.r2_results = pd.DataFrame(r2_rows) if r2_rows else pd.DataFrame()
        return pd.concat(main_results, ignore_index=True)

    def _process_phenotype_safe(
        self, pheno_index: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[dict]]:
        """Exception-safe wrapper for multiprocessing workers."""
        try:
            return self._process_phenotype(pheno_index)
        except Exception as e:
            current_pheno = self.quan.index[pheno_index]
            print(f"[finemapping] Error for phenotype {current_pheno}: {e}")
            return (
                self._empty_result(current_pheno),
                self._empty_bootstrap_diagnostics(),
                None,
            )

    # ==================================================================
    # R² computation (internal, called from _process_phenotype)
    # ==================================================================

    def _compute_r2_from_fit(
        self,
        phenotype: str,
        y: np.ndarray,
        X_unpen: np.ndarray,
        d_std: np.ndarray,
        obs_masks: List[np.ndarray],
        pi_v: np.ndarray,
        best_beta_refit: np.ndarray,
        kept_ids: np.ndarray,
        n_samples: int,
    ) -> Optional[dict]:
        """
        Compute baseline vs full-model R² using already-fitted results.

        Selects all stable variants (π ≥ r2_stability_threshold with
        non-zero beta), then fits a missing-aware unpenalised model
        (MissingAwareElasticNet with λ=0) for the full model and standard
        OLS for the baseline (covariates only).

        Predictions are computed with missing-aware logic: for each sample,
        only observed variant entries contribute to ŷ. R² is computed over
        all n samples — no imputation is performed.

        Returns a dict with one row of R² results for this phenotype.
        If no variants qualify for the full model, baseline R² is still
        returned and full-model/variant-related fields are left empty.
        """
        n = len(y)

        # Baseline model: always reported when R² computation is enabled.
        theta_base = ols_fit(y, X_unpen)
        yhat_base = X_unpen @ theta_base
        r2_base = r_squared(y, yhat_base)
        r2_base_adj = adjusted_r_squared(r2_base, n, X_unpen.shape[1] - 1)

        # Identify stable variants with non-zero effect.
        candidates = []
        for j in range(len(kept_ids)):
            if (
                pi_v[j] >= self.r2_stability_threshold
                and abs(best_beta_refit[j]) > 1e-10
            ):
                candidates.append(j)

        if len(candidates) == 0:
            return {
                "phenotype": phenotype,
                "n_samples": n_samples,
                "r2_baseline": r2_base,
                "r2_baseline_adj": r2_base_adj,
                "r2_full": np.nan,
                "r2_full_adj": np.nan,
                "delta_r2": np.nan,
                "delta_r2_adj": np.nan,
                "r2_n_variants": np.nan,
                "r2_variants": np.nan,
            }

        # Keep all stability-supported variants (no peak-based down-selection).
        sel_indices = np.array(candidates, dtype=int)

        # Variant IDs and obs masks for the selected subset
        selected_variant_ids = [kept_ids[j] for j in sel_indices]
        d_sel = d_std[sel_indices, :]  # (p_sel, n)
        sel_obs_masks = [obs_masks[j] for j in sel_indices]
        p_sel = len(sel_indices)

        # ── Full model: missing-aware unpenalised fit (EN with λ=0) ──
        en = MissingAwareElasticNet(alpha_en=0.5, max_iter=2000, tol=1e-8)
        csr_idx_sel, csr_off_sel = _obs_masks_to_csr(sel_obs_masks)
        beta_full, theta_full, _, _ = en.fit(
            y,
            d_sel,
            sel_obs_masks,
            X_unpen,
            lam=0.0,
            csr_idx=csr_idx_sel,
            csr_off=csr_off_sel,
        )

        # Missing-aware predictions: y_hat_i = X_unpen_i @ theta + sum_{j observed} d_j,i * beta_j
        yhat_full = X_unpen @ theta_full
        _add_beta_contributions(
            yhat_full, beta_full, d_sel, csr_idx_sel, csr_off_sel, p_sel
        )

        r2_full_val = r_squared(y, yhat_full)
        r2_full_adj = adjusted_r_squared(r2_full_val, n, X_unpen.shape[1] - 1 + p_sel)

        return {
            "phenotype": phenotype,
            "n_samples": n_samples,
            "r2_baseline": r2_base,
            "r2_baseline_adj": r2_base_adj,
            "r2_full": r2_full_val,
            "r2_full_adj": r2_full_adj,
            "delta_r2": r2_full_val - r2_base,
            "delta_r2_adj": r2_full_adj - r2_base_adj,
            "r2_n_variants": len(sel_indices),
            "r2_variants": ";".join(selected_variant_ids),
        }
