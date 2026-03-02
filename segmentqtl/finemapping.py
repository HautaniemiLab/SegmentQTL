"""
SegmentQTL finemapping mode: missing-aware Elastic Net with stability selection.

Performs statistical finemapping for molecular QTLs using:
1. Segment-aware filtering (phenotype & variant on same segment, identical to
   cis mode)
2. Missing-aware Elastic Net via coordinate descent (NaN entries from segment
   filtering are handled natively -- never imputed or set to zero)
3. BIC-based regularisation-path selection
4. Stability selection for robust variant identification
5. LD-aware clustering for credible-set-like output

References
----------
Meinshausen & Bühlmann (2010).  Stability selection.  JRSS-B.
Zou & Hastie (2005).  Regularization and variable selection via the Elastic Net.
"""

import warnings
from multiprocessing import Pool
from os import path
from time import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from tqdm import tqdm

# ---------------------------------------------------------------------------
#  Missing-aware Elastic Net coordinate-descent solver
# ---------------------------------------------------------------------------


class MissingAwareElasticNet:
    """Coordinate-descent Elastic Net that handles per-entry missingness.

    The penalised objective (only variant coefficients beta_v are penalised):

        (1/2) sum_i (y_i - X_unpen_i @ theta
                      - sum_{v in O_i} beta_v * d_tilde_{v,i})^2
        + lam * alpha_en * sum_v |beta_v|
        + lam * (1 - alpha_en) / 2 * sum_v beta_v^2

    where O_i is the set of variants observed for sample i.

    Parameters
    ----------
    alpha_en : float
        Mixing parameter in [0, 1].  1 = Lasso, 0 = Ridge.
    max_iter : int
        Maximum full coordinate-descent passes.
    tol : float
        Convergence tolerance (max |delta beta|).
    """

    def __init__(self, alpha_en: float = 0.5, max_iter: int = 1000, tol: float = 1e-6):
        self.alpha_en = alpha_en
        self.max_iter = max_iter
        self.tol = tol

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _soft_threshold(z: float, lam: float) -> float:
        """Soft-thresholding operator S(z, lam) = sign(z) max(|z| - lam, 0)."""
        return float(np.sign(z)) * max(abs(z) - lam, 0.0)

    @staticmethod
    def _ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Ordinary least-squares: y = X theta.  Returns theta."""
        try:
            theta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            theta = np.zeros(X.shape[1])
        return theta

    # ---- core methods ----------------------------------------------------

    def compute_lambda_max(
        self,
        y: np.ndarray,
        d_std: np.ndarray,
        obs_masks: List[np.ndarray],
        X_unpen: np.ndarray,
    ) -> float:
        """Smallest lambda for which all beta_v = 0.

        At beta = 0 the sub-gradient condition requires
        |sum_{i in O_v} d_tilde_{v,i} r_i^{(0)}| <= lam * alpha_en
        for every v.
        """
        theta0 = self._ols(y, X_unpen)
        r0 = y - X_unpen @ theta0

        max_abs_grad = 0.0
        for v, idx in enumerate(obs_masks):
            if len(idx) == 0:
                continue
            g = abs(np.dot(d_std[v, idx], r0[idx]))
            if g > max_abs_grad:
                max_abs_grad = g

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Fit the model for a single lambda value.

        Parameters
        ----------
        y         : (n,)  phenotype (centred recommended).
        d_std     : (p, n) standardised variant predictors, NaN where missing.
        obs_masks : list of index arrays; obs_masks[v] = observed sample indices.
        X_unpen   : (n, q) unpenalised design matrix (intercept + covariates).
        lam       : penalty magnitude.
        beta_init : optional warm-start for beta.
        theta_init: optional warm-start for theta.

        Returns
        -------
        beta   : (p,) penalised coefficients.
        theta  : (q,) unpenalised coefficients.
        r      : (n,) final residuals.
        n_iter : number of iterations used.
        """
        p, n = d_std.shape
        lam1 = lam * self.alpha_en
        lam2 = lam * (1.0 - self.alpha_en)

        # --- initialise ---
        theta = theta_init.copy() if theta_init is not None else self._ols(y, X_unpen)
        beta = beta_init.copy() if beta_init is not None else np.zeros(p)

        # Initial residuals
        r = y - X_unpen @ theta
        for v in range(p):
            if beta[v] != 0.0:
                idx = obs_masks[v]
                if len(idx) > 0:
                    r[idx] -= beta[v] * d_std[v, idx]

        # Precompute G_v = sum d_tilde^2 per variant
        G = np.zeros(p)
        for v in range(p):
            idx = obs_masks[v]
            if len(idx) > 0:
                G[v] = np.sum(d_std[v, idx] ** 2)

        # --- coordinate descent ---
        n_iter = 0
        for it in range(self.max_iter):
            max_change = 0.0
            for v in range(p):
                idx = obs_masks[v]
                if len(idx) == 0 or G[v] == 0.0:
                    continue

                d_v = d_std[v, idx]
                old = beta[v]

                # z_v = <d_tilde_v, partial_resid_v>
                #     = <d_tilde_v, r> + old * G_v
                z_v = np.dot(d_v, r[idx]) + old * G[v]

                new = self._soft_threshold(z_v, lam1) / (G[v] + lam2)

                if new != old:
                    r[idx] += d_v * (old - new)
                    beta[v] = new
                    change = abs(new - old)
                    if change > max_change:
                        max_change = change

            n_iter = it + 1

            # Refit unpenalised block every 5 passes
            if n_iter % 5 == 0:
                y_adj = y.copy()
                for v in range(p):
                    if beta[v] != 0.0:
                        idx = obs_masks[v]
                        if len(idx) > 0:
                            y_adj[idx] -= beta[v] * d_std[v, idx]
                theta = self._ols(y_adj, X_unpen)
                r = y_adj - X_unpen @ theta

            if max_change < self.tol:
                break

        # Final unpenalised refit
        y_adj = y.copy()
        for v in range(p):
            if beta[v] != 0.0:
                idx = obs_masks[v]
                if len(idx) > 0:
                    y_adj[idx] -= beta[v] * d_std[v, idx]
        theta = self._ols(y_adj, X_unpen)
        r = y_adj - X_unpen @ theta

        return beta, theta, r, n_iter

    @staticmethod
    def compute_bic(n: int, rss: float, n_nonzero: int, n_unpen: int) -> float:
        """BIC = n log(RSS / n) + k log(n),  k = n_nonzero + n_unpen.

        Note: n is the *global* sample count (all samples with non-missing
        phenotype and covariates), even though individual variants may have
        fewer observed entries due to segment-filtering missingness.  This is
        defensible because one residual is computed per sample regardless, but
        it means BIC may slightly favour phenotype windows with higher average
        variant coverage.
        """
        if rss <= 0.0 or n <= 0:
            return np.inf
        k = n_nonzero + n_unpen
        return n * np.log(rss / n) + k * np.log(n)

    def fit_path_bic(
        self,
        y: np.ndarray,
        d_std: np.ndarray,
        obs_masks: List[np.ndarray],
        X_unpen: np.ndarray,
        n_lambda: int = 100,
        lambda_ratio: float = 0.01,
        refit_gamma: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        """Fit Elastic Net along a warm-started lambda path; select lambda by
        relaxed-refit BIC.

        At each lambda the active set (non-zero EN coefficients) is
        identified, those variables are refitted with a *relaxed* pure-Ridge
        penalty ``gamma * lambda`` (alpha_en=0) to obtain a nearly-unbiased
        RSS while keeping the system well-conditioned.  BIC is computed on
        the relaxed-refit model.  This corrects the downward bias of
        penalised-path BIC caused by coefficient shrinkage (cf. Meinshausen
        2007, relaxed Lasso) without requiring a fully unpenalised OLS that
        can be singular when the active set is large relative to n.

        The active set is capped at ``n // 3`` variants (top-K by |beta|)
        to prevent near-saturated refits.

        Parameters
        ----------
        n_lambda     : number of lambda grid points.
        lambda_ratio : lam_min / lam_max.
        refit_gamma  : relaxation factor for refit penalty.  The active-set
                       refit uses ``gamma * lambda`` with alpha_en=0 (pure
                       Ridge).  Default 0.1 removes ~90 % of shrinkage
                       while providing robust stabilisation.

        Returns
        -------
        best_beta, best_theta, best_lambda, bic_path, lambda_path, best_beta_refit
        """
        lam_max = self.compute_lambda_max(y, d_std, obs_masks, X_unpen)
        if lam_max <= 0:
            lam_max = 1.0

        lam_min = lam_max * lambda_ratio
        lambda_path = np.exp(np.linspace(np.log(lam_max), np.log(lam_min), n_lambda))

        n = len(y)
        q = X_unpen.shape[1]
        p = d_std.shape[0]

        # Maximum active-set size for a reliable refit (avoid near-saturated systems).
        # Subtract q (unpenalised covariates) so the refit system
        # (q + top_k parameters) does not approach saturation.
        max_active_refit = max(1, (n - q) // 3)

        best_bic = np.inf
        best_beta = np.zeros(p)
        best_theta = self._ols(y, X_unpen)
        best_lambda = lambda_path[0]
        best_beta_refit = np.zeros(p)
        bic_path = np.full(n_lambda, np.inf)

        # Warm-start containers
        beta_ws: Optional[np.ndarray] = None
        theta_ws: Optional[np.ndarray] = None

        for i, lam in enumerate(lambda_path):
            beta, theta, r, _ = self.fit(
                y,
                d_std,
                obs_masks,
                X_unpen,
                lam,
                beta_init=beta_ws,
                theta_init=theta_ws,
            )

            active = np.flatnonzero(np.abs(beta) > 1e-10)
            n_nz = len(active)

            # Relaxed-refit BIC: refit active set with gamma * lam
            lam_refit = refit_gamma * lam
            refit_failed = False  # track convergence for BIC fallback

            if 0 < n_nz <= max_active_refit:
                d_active = d_std[active]
                obs_active = [obs_masks[v] for v in active]
                # Pure Ridge (alpha_en=0) for the refit: all of lam_refit
                # goes to the L2 term, maximising stabilisation.
                # Tolerance 1e-5 is sufficient for RSS/BIC; tighter values
                # stall due to the alternating theta-refit perturbation.
                refit_en = MissingAwareElasticNet(alpha_en=0.0, max_iter=5000, tol=1e-5)
                beta_rf, theta_rf, r_rf, n_iter_rf = refit_en.fit(
                    y,
                    d_active,
                    obs_active,
                    X_unpen,
                    lam=lam_refit,
                    beta_init=beta[active],
                    theta_init=theta,
                )
                if n_iter_rf == 5000:
                    warnings.warn(
                        f"Refit at lambda={lam:.4f} (active={n_nz}) "
                        f"did not converge in {n_iter_rf} iterations.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    refit_failed = True
                rss = float(np.dot(r_rf, r_rf))
                n_bic_params = n_nz  # all active vars were refitted
            elif n_nz > max_active_refit:
                # Too many active variables for full refit.
                # Refit only the top-K variants by |beta|.
                top_k = max_active_refit
                top_idx = active[np.argsort(-np.abs(beta[active]))[:top_k]]
                d_top = d_std[top_idx]
                obs_top = [obs_masks[v] for v in top_idx]
                refit_en = MissingAwareElasticNet(alpha_en=0.0, max_iter=5000, tol=1e-5)
                beta_rf_top, _, r_rf, n_iter_rf = refit_en.fit(
                    y,
                    d_top,
                    obs_top,
                    X_unpen,
                    lam=lam_refit,
                    beta_init=beta[top_idx],
                    theta_init=theta,
                )
                if n_iter_rf == 5000:
                    warnings.warn(
                        f"Refit at lambda={lam:.4f} (top-{top_k} of {n_nz}) "
                        f"did not converge in {n_iter_rf} iterations.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    refit_failed = True
                rss = float(np.dot(r_rf, r_rf))
                # BIC penalty matches the *refit* model size, not the
                # full EN active size, since RSS comes from the top-K refit.
                n_bic_params = top_k
                # Build full-p refitted beta using global indices (safe
                # against active-local / global index mix-ups).
                beta_rf_full = np.zeros(p)
                beta_rf_full[top_idx] = beta_rf_top
                beta_rf = beta_rf_full[active]
            else:
                rss = float(np.dot(r, r))
                beta_rf = None
                n_bic_params = 0

            # If refit did not converge, do not trust its RSS for BIC
            if refit_failed:
                bic = np.inf
            else:
                bic = self.compute_bic(n, rss, n_bic_params, q)
            bic_path[i] = bic

            if bic < best_bic:
                best_bic = bic
                best_beta = beta.copy()
                best_theta = theta.copy()
                best_lambda = lam
                # Build full-length refitted beta vector
                if beta_rf is not None:
                    best_beta_refit = np.zeros(p)
                    best_beta_refit[active] = beta_rf
                else:
                    best_beta_refit = beta.copy()

            # Warm start for next lambda
            beta_ws = beta
            theta_ws = theta

        return (
            best_beta,
            best_theta,
            best_lambda,
            bic_path,
            lambda_path,
            best_beta_refit,
        )


# ---------------------------------------------------------------------------
#  Variant standardisation with per-variant observed-only statistics
# ---------------------------------------------------------------------------


def standardize_variants(
    d_raw: np.ndarray,
    coverage_tau: float,
    n_total: int,
    min_obs: int = 30,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """Standardise each variant predictor d_v on its observed entries.

    Applies a coverage filter: only variants with at least
    ``max(min_obs, coverage_tau * n_total)`` observed samples are kept.

    Parameters
    ----------
    d_raw        : (p, n) allelic difference d = REFlr - ALTlr, NaN where missing.
    coverage_tau : minimum fraction of n_total required for a variant to be kept.
    n_total      : total number of samples (for coverage filter).
    min_obs      : hard minimum observation count.

    Returns
    -------
    d_std     : (p_kept, n) standardised, NaN where missing.
    obs_masks : list of ndarray -- observed sample indices per kept variant.
    keep_idx  : (p_kept,) original row indices that were retained.
    sd_vec    : (p_kept,) per-variant SD used for standardisation (for
                back-transforming coefficients to raw units).
    """
    p, n = d_raw.shape
    min_required = max(min_obs, int(coverage_tau * n_total))

    d_std_rows: List[np.ndarray] = []
    obs_masks: List[np.ndarray] = []
    keep: List[int] = []
    sd_list: List[float] = []

    for v in range(p):
        idx = np.flatnonzero(~np.isnan(d_raw[v]))
        if len(idx) < min_required:
            continue

        vals = d_raw[v, idx]
        mu = np.mean(vals)
        sd = np.std(vals)

        if sd < 1e-10:
            continue

        row = np.full(n, np.nan)
        row[idx] = (vals - mu) / sd

        d_std_rows.append(row)
        obs_masks.append(idx)
        keep.append(v)
        sd_list.append(float(sd))

    keep_idx = np.array(keep, dtype=int)
    sd_vec = np.array(sd_list, dtype=float)
    if len(keep) == 0:
        d_std = np.empty((0, n))
    else:
        d_std = np.vstack(d_std_rows)

    return d_std, obs_masks, keep_idx, sd_vec


def standardize_variants_bootstrap(
    d_raw: np.ndarray,
    min_obs_boot: int = 20,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Standardise variants for a bootstrap subsample *without* coverage filtering.

    Unlike :func:`standardize_variants`, this function does **not** apply the
    ``coverage_tau`` threshold.  It only drops a variant if the subsample has
    fewer than ``min_obs_boot`` observed entries or zero variance.  This ensures
    that the stability-selection frequency pi_v reflects "selected when fit is
    attempted", not "selected AND passed a second coverage filter".

    Parameters
    ----------
    d_raw         : (p, n_sub) raw d values for the subsample, NaN = missing.
    min_obs_boot  : hard minimum of observed entries per variant.

    Returns
    -------
    d_std     : (p_kept, n_sub) standardised, NaN where missing.
    obs_masks : list of ndarray -- observed indices per kept variant.
    keep_idx  : (p_kept,) original row indices that were retained.
    """
    p, n = d_raw.shape

    d_std_rows: List[np.ndarray] = []
    obs_masks: List[np.ndarray] = []
    keep: List[int] = []

    for v in range(p):
        idx = np.flatnonzero(~np.isnan(d_raw[v]))
        if len(idx) < min_obs_boot:
            continue

        vals = d_raw[v, idx]
        mu = np.mean(vals)
        sd = np.std(vals)

        if sd < 1e-10:
            continue

        row = np.full(n, np.nan)
        row[idx] = (vals - mu) / sd

        d_std_rows.append(row)
        obs_masks.append(idx)
        keep.append(v)

    keep_idx = np.array(keep, dtype=int)
    if len(keep) == 0:
        d_std = np.empty((0, n))
    else:
        d_std = np.vstack(d_std_rows)

    return d_std, obs_masks, keep_idx


# ---------------------------------------------------------------------------
#  LD computation & clustering
# ---------------------------------------------------------------------------


def compute_ld_matrix(d_std: np.ndarray, obs_masks: List[np.ndarray]) -> np.ndarray:
    """Pairwise r-squared matrix using overlap samples per pair.

    Uses boolean masks and vectorised overlap computation to avoid
    expensive Python-level set operations per pair.

    Parameters
    ----------
    d_std     : (p, n) standardised predictors (NaN where missing).
    obs_masks : list of observed-sample index arrays (sorted, unique).

    Returns
    -------
    r2 : (p, p) symmetric matrix.
    """
    p, n = d_std.shape
    r2 = np.eye(p)

    if p <= 1:
        return r2

    # Pre-build boolean observation masks (p x n) for fast overlap
    obs_bool = np.zeros((p, n), dtype=bool)
    for v in range(p):
        obs_bool[v, obs_masks[v]] = True

    for v in range(p):
        for w in range(v + 1, p):
            overlap_mask = obs_bool[v] & obs_bool[w]
            n_overlap = int(overlap_mask.sum())
            if n_overlap < 5:
                r2[v, w] = r2[w, v] = 0.0
                continue
            dv = d_std[v, overlap_mask]
            dw = d_std[w, overlap_mask]
            # Fast correlation via dot products
            dv_c = dv - dv.mean()
            dw_c = dw - dw.mean()
            ss_vw = np.dot(dv_c, dw_c)
            ss_vv = np.dot(dv_c, dv_c)
            ss_ww = np.dot(dw_c, dw_c)
            denom = ss_vv * ss_ww
            if denom > 0:
                c = ss_vw / np.sqrt(denom)
                r2[v, w] = r2[w, v] = c * c
            else:
                r2[v, w] = r2[w, v] = 0.0

    return r2


def ld_cluster_variants(
    r2: np.ndarray,
    stability_scores: np.ndarray,
    ld_threshold: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, float]]:
    """Hierarchical clustering on 1 - r-squared distance.

    Returns
    -------
    cluster_ids       : (p,) integer cluster labels (1-based).
    lead_mask         : (p,) boolean -- True for lead variant per cluster.
    cluster_stability : dict cluster_id -> max stability score in cluster.
    """
    p = r2.shape[0]
    if p == 0:
        return np.array([], dtype=int), np.array([], dtype=bool), {}

    if p == 1:
        return (
            np.ones(1, dtype=int),
            np.ones(1, dtype=bool),
            {1: float(stability_scores[0])},
        )

    dist = np.clip(1.0 - r2, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=1.0 - ld_threshold, criterion="distance")

    lead_mask = np.zeros(p, dtype=bool)
    cluster_stability: Dict[int, float] = {}

    for cid in np.unique(cluster_ids):
        members = np.flatnonzero(cluster_ids == cid)
        best = members[np.argmax(stability_scores[members])]
        lead_mask[best] = True
        cluster_stability[int(cid)] = float(np.max(stability_scores[members]))

    return cluster_ids, lead_mask, cluster_stability


# ---------------------------------------------------------------------------
#  Finemapping orchestrator
# ---------------------------------------------------------------------------


class Finemapping:
    """SegmentQTL finemapping: segment-aware missing Elastic Net + stability
    selection.

    Uses the same data-loading and segment-filtering logic as ``Cis`` mode,
    then fits a joint missing-aware Elastic Net model across all variants in
    each phenotype window, with BIC for lambda selection and stability
    selection for robust identification of causal variants.

    Parameters
    ----------
    chromosome          : Chromosome identifier (e.g. ``"chr1"``).
    quantifications     : Path to quantifications CSV.
    covariates          : Path to sample-level covariates CSV (optional).
    segmentation        : Path to segmentation CSV.
    genotype_alt        : Path to ALTlr genotype CSV for this chromosome.
    genotype_ref        : Path to REFlr genotype CSV for this chromosome.
    copynumber          : Path to phenotype-level copy-number covariate CSV
                          (optional).  This is CNlr -- included as an
                          *unpenalised* predictor in the Elastic Net.
    phenotype_covariate : Path to additional phenotype-level covariate CSV
                          (optional; also unpenalised).
    window              : Cis-window size in bp.
    num_cores           : Number of parallel workers.
    alpha_en            : Elastic Net mixing (1 = Lasso, 0 = Ridge).
    coverage_tau        : Minimum fraction of samples observed for a variant
                          to be retained.
    n_bootstrap         : Number of stability-selection resamples.
    subsample_frac      : Fraction of samples per resample.
    stability_threshold : Threshold on selection probability pi_v for the
                          ``is_stable`` flag.
    n_lambda            : Lambda-grid size for BIC path.
    lambda_ratio        : lam_min / lam_max ratio for the grid.
    ld_threshold        : r-squared threshold for LD clustering.
    min_obs_boot        : Hard minimum observed entries per variant inside
                          each bootstrap subsample.  Variants with fewer
                          observations are skipped for that resample.
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
        stability_threshold: float = 0.6,
        n_lambda: int = 100,
        lambda_ratio: float = 0.01,
        ld_threshold: float = 0.8,
        min_obs_boot: int = 20,
    ):
        self.chromosome = chromosome

        # ── phenotype data ──
        self.full_quan = self._load(quantifications, index_col=3)
        self.quan = self.full_quan[self.full_quan["chr"] == self.chromosome]
        self.samples = self.quan.columns.to_numpy()[3:]

        # ── sample-level covariates ──
        self.cov = self._load(covariates, index_col=None) if covariates else None

        # ── segmentation ──
        self.segmentation = self._load(segmentation, index_col=0)
        self.segmentation = self.segmentation[self.segmentation.chr == self.chromosome]
        self.segmentation = self.segmentation[
            self.segmentation.index.isin(self.samples)
        ]

        # ── genotypes ──
        self.geno_alt = self._load(genotype_alt, index_col=0)
        self.geno_alt = self.geno_alt.loc[:, self.geno_alt.columns.isin(self.samples)][
            self.samples
        ]

        self.geno_ref = self._load(genotype_ref, index_col=0)
        self.geno_ref = self.geno_ref.loc[:, self.geno_ref.columns.isin(self.samples)][
            self.samples
        ]

        common = self.geno_alt.index.intersection(self.geno_ref.index)
        self.geno_alt = self.geno_alt.loc[common]
        self.geno_ref = self.geno_ref.loc[common]

        idx_arr = self.geno_alt.index.astype(str).to_numpy()
        self.variant_positions = np.fromiter(
            (int(s.split(":")[1]) for s in idx_arr), dtype=np.int64
        )

        # ── phenotype-level copy-number covariate (CNlr) ──
        self.copynumber_df = self._load(copynumber, index_col=0) if copynumber else None

        # ── phenotype-level covariate (optional, additional unpenalised) ──
        self.phenotype_covariate_df = (
            self._load(phenotype_covariate, index_col=0)
            if phenotype_covariate
            else None
        )

        # ── window & parallelism ──
        self.window = window
        self.num_cores = num_cores

        # ── EN hyper-parameters ──
        self.alpha_en = alpha_en
        self.coverage_tau = coverage_tau
        self.n_bootstrap = n_bootstrap
        self.subsample_frac = subsample_frac
        self.stability_threshold = stability_threshold
        self.n_lambda = n_lambda
        self.lambda_ratio = lambda_ratio
        self.ld_threshold = ld_threshold
        self.min_obs_boot = min_obs_boot

    # ---- I/O (mirrors Cis) -----------------------------------------------

    @staticmethod
    def _load(fp: str, index_col) -> pd.DataFrame:
        if not path.exists(fp):
            raise FileNotFoundError(f"File '{fp}' not found.")
        df = pd.read_csv(fp, index_col=index_col)
        if df.shape[0] == 0:
            raise ValueError(f"File '{fp}' has no rows.")
        return df

    # ---- window / segment helpers (identical to Cis) ----------------------

    def _phenotype_window(self, pheno_index: int) -> Tuple[int, int]:
        """Cis-window boundaries for phenotype at *pheno_index*."""
        start = self.quan["start"].iloc[pheno_index] - self.window
        end = self.quan["end"].iloc[pheno_index] + self.window
        return start, end

    def _variants_in_window(
        self, start: int, end: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        mask = (self.variant_positions >= start) & (self.variant_positions <= end)
        return (
            self.geno_alt.loc[mask],
            self.geno_ref.loc[mask],
            self.variant_positions[mask],
        )

    def _segment_filter(
        self,
        start: int,
        end: int,
        variants_alt: pd.DataFrame,
        variants_ref: pd.DataFrame,
        variant_pos: np.ndarray,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Segment-consistency filter identical to ``Cis.gene_variants_common_segment``.

        Masks genotype entries to NaN when the phenotype and variant do not sit
        on the same segment for a given sample.
        """
        pheno_start = start + self.window
        pheno_end = end - self.window

        alt_arr = variants_alt.to_numpy(dtype=float, copy=True)
        ref_arr = variants_ref.to_numpy(dtype=float, copy=True)
        sample_cols = variants_alt.columns.to_numpy()

        seg_index = self.segmentation.index.to_numpy()
        seg_startpos = self.segmentation["startpos"].to_numpy()
        seg_endpos = self.segmentation["endpos"].to_numpy()

        for col_idx, cur_sample in enumerate(sample_cols):
            seg_mask = (
                (seg_index == cur_sample)
                & (seg_startpos <= pheno_start)
                & (seg_endpos >= pheno_start)
            )
            seg_indices = np.flatnonzero(seg_mask)

            if len(seg_indices) != 1:
                alt_arr[:, col_idx] = np.nan
                ref_arr[:, col_idx] = np.nan
                continue

            seg_idx = seg_indices[0]
            lb = seg_startpos[seg_idx]
            ub = seg_endpos[seg_idx]

            if not (lb <= pheno_end <= ub):
                alt_arr[:, col_idx] = np.nan
                ref_arr[:, col_idx] = np.nan
                continue

            outside = (variant_pos < lb) | (variant_pos > ub)
            alt_arr[outside, col_idx] = np.nan
            ref_arr[outside, col_idx] = np.nan

        variants_alt = pd.DataFrame(
            alt_arr, index=variants_alt.index, columns=sample_cols
        )
        variants_ref = pd.DataFrame(
            ref_arr, index=variants_ref.index, columns=sample_cols
        )
        return variants_alt, variants_ref

    # ---- data preparation for one phenotype ------------------------------

    def _prepare_phenotype(self, pheno_index: int) -> Optional[dict]:
        """Build the d-matrix, phenotype vector, and unpenalised design for
        one phenotype.

        Returns ``None`` when the phenotype cannot be finemapped (too few
        data).  Otherwise returns a dict with keys:

        * ``variant_ids`` -- (p_window,) variant identifiers
        * ``d_raw``       -- (p_window, n_good) raw d values (NaN = missing)
        * ``y``           -- (n_good,) centred phenotype
        * ``X_unpen``     -- (n_good, q) unpenalised design (intercept + covs)
        * ``n_samples``   -- int
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
        if self.cov is not None:
            cov_full = [
                pd.to_numeric(self.cov.loc[c], errors="coerce")
                .to_numpy()
                .flatten()
                .astype(float)
                for c in self.cov.index
            ]

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

        if cnlr_full is not None:
            cnlr = cnlr_full[mask]
            mu_cn, sd_cn = float(np.mean(cnlr)), float(np.std(cnlr))
            if sd_cn > 1e-10:
                cnlr = (cnlr - mu_cn) / sd_cn
            blocks.append(cnlr)

        if phenotype_cov_full is not None:
            pcov = phenotype_cov_full[mask]
            mu_pc, sd_pc = float(np.mean(pcov)), float(np.std(pcov))
            if sd_pc > 1e-10:
                pcov = (pcov - mu_pc) / sd_pc
            blocks.append(pcov)

        for cv in cov_full:
            c = cv[mask]
            mu_c, sd_c = float(np.mean(c)), float(np.std(c))
            if sd_c > 1e-10:
                c = (c - mu_c) / sd_c
            blocks.append(c)

        X_unpen = np.column_stack(blocks)

        return {
            "variant_ids": variant_ids_all,
            "d_raw": d_raw,
            "y": y,
            "X_unpen": X_unpen,
            "n_samples": n_good,
        }

    # ---- stability selection ---------------------------------------------

    def _stability_selection(
        self,
        y: np.ndarray,
        d_raw_kept: np.ndarray,
        X_unpen: np.ndarray,
        lam_bic: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bootstrap stability selection at a fixed lambda.

        Variant predictors are re-standardised inside each subsample (observed
        entries only) but the variant set is **frozen** from the full-data
        coverage filter.  Bootstrap subsamples only skip a variant when it has
        too few observed entries (< ``min_obs_boot``) or zero variance, so that
        pi_v reflects "selected when fit is attempted", not "selected AND
        passed a second coverage filter".

        Parameters
        ----------
        y           : (n,) centred phenotype.
        d_raw_kept  : (p_kept, n) raw d values for kept variants, NaN = missing.
        X_unpen     : (n, q) unpenalised design.
        lam_bic     : penalty selected on full data.
        rng         : random generator (seeded per phenotype).

        Returns
        -------
        pi_v          : (p_kept,) selection probability (fraction of bootstraps
                        where beta_v != 0).
        mean_beta     : (p_kept,) mean beta among selections (standardised units).
        sign_consist  : (p_kept,) fraction of selections with beta > 0.
        """
        n = len(y)
        p_kept = d_raw_kept.shape[0]
        n_sub = max(30, int(n * self.subsample_frac))

        sel_count = np.zeros(p_kept)
        beta_sum = np.zeros(p_kept)
        pos_count = np.zeros(p_kept)
        n_effective_boot = 0

        en = MissingAwareElasticNet(alpha_en=self.alpha_en, max_iter=500, tol=1e-5)

        for _ in range(self.n_bootstrap):
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
            beta, _, _, _ = en.fit(y_sub, d_sub, obs_sub, X_sub, lam_bic)

            for j, orig_v in enumerate(keep_inner):
                if abs(beta[j]) > 1e-10:
                    sel_count[orig_v] += 1
                    beta_sum[orig_v] += beta[j]
                    if beta[j] > 0:
                        pos_count[orig_v] += 1

        pi_v = sel_count / max(n_effective_boot, 1)
        with np.errstate(invalid="ignore"):
            mean_beta = np.where(sel_count > 0, beta_sum / sel_count, 0.0)
            sign_consist = np.where(sel_count > 0, pos_count / sel_count, np.nan)

        return pi_v, mean_beta, sign_consist

    # ---- per-phenotype processing ----------------------------------------

    def _empty_result(self, phenotype: str, n_samples: int = 0) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "phenotype": phenotype,
                    "variant": np.nan,
                    "n_samples": n_samples,
                    "n_variants": 0,
                    "n_obs": 0,
                    "median_n_obs": np.nan,
                    "stability_score": np.nan,
                    "is_stable": False,
                    "mean_beta": np.nan,
                    "sign_consistency": np.nan,
                    "ld_cluster": np.nan,
                    "is_lead": False,
                    "cluster_stability": np.nan,
                    "lambda_bic": np.nan,
                    "beta_full": np.nan,
                    "beta_full_raw": np.nan,
                }
            ]
        )

    def _process_phenotype(self, pheno_index: int) -> pd.DataFrame:
        """Full finemapping pipeline for one phenotype."""
        current_pheno = self.quan.index[pheno_index]
        data = self._prepare_phenotype(pheno_index)
        if data is None:
            return self._empty_result(current_pheno)

        variant_ids = data["variant_ids"]
        d_raw = data["d_raw"]
        y = data["y"]
        X_unpen = data["X_unpen"]
        n_samples = data["n_samples"]

        # --- standardise & coverage filter (full data) ---
        d_std, obs_masks, keep_idx, sd_vec = standardize_variants(
            d_raw, self.coverage_tau, n_samples
        )
        if len(keep_idx) == 0:
            return self._empty_result(current_pheno, n_samples)

        kept_ids = variant_ids[keep_idx]
        p_kept = len(kept_ids)

        # Compute median n_obs across kept variants for diagnostics
        median_n_obs = float(np.median([len(m) for m in obs_masks]))

        # --- BIC lambda-path on full data ---
        en = MissingAwareElasticNet(alpha_en=self.alpha_en)
        best_beta, best_theta, lam_bic, _, _, best_beta_refit = en.fit_path_bic(
            y,
            d_std,
            obs_masks,
            X_unpen,
            n_lambda=self.n_lambda,
            lambda_ratio=self.lambda_ratio,
        )

        # --- stability selection ---
        rng = np.random.default_rng(seed=42 + pheno_index)
        d_raw_kept = d_raw[keep_idx]

        pi_v, mean_beta, sign_consist = self._stability_selection(
            y,
            d_raw_kept,
            X_unpen,
            lam_bic,
            rng,
        )

        # --- LD clustering ---
        r2 = compute_ld_matrix(d_std, obs_masks)
        cluster_ids, lead_mask, cluster_stab = ld_cluster_variants(
            r2,
            pi_v,
            self.ld_threshold,
        )

        # --- assemble per-variant results ---
        rows = []
        for j in range(p_kept):
            cid = int(cluster_ids[j])
            rows.append(
                {
                    "phenotype": current_pheno,
                    "variant": kept_ids[j],
                    "n_samples": n_samples,
                    "n_variants": p_kept,
                    "n_obs": len(obs_masks[j]),
                    "median_n_obs": median_n_obs,
                    "stability_score": float(pi_v[j]),
                    "is_stable": bool(pi_v[j] >= self.stability_threshold),
                    "mean_beta": float(mean_beta[j]),
                    "sign_consistency": (
                        float(sign_consist[j])
                        if np.isfinite(sign_consist[j])
                        else np.nan
                    ),
                    "ld_cluster": cid,
                    "is_lead": bool(lead_mask[j]),
                    "cluster_stability": cluster_stab.get(cid, 0.0),
                    "lambda_bic": float(lam_bic),
                    "beta_full": float(best_beta_refit[j]),
                    "beta_full_raw": float(best_beta_refit[j] / sd_vec[j]),
                }
            )

        return pd.DataFrame(rows)

    # ---- public entry point ----------------------------------------------

    def calculate_finemapping(self, phenotype_id: Optional[str] = None) -> pd.DataFrame:
        """Run finemapping for phenotypes on this chromosome.

        Parameters
        ----------
        phenotype_id : optional phenotype identifier.  If provided, only
                       that phenotype is finemapped.  Otherwise all
                       phenotypes on the chromosome are processed.
        """
        start = time()

        if phenotype_id is not None:
            if phenotype_id not in self.quan.index:
                raise ValueError(
                    f"Phenotype '{phenotype_id}' not found on {self.chromosome}. "
                    f"Available: {self.quan.index.tolist()[:5]}{'...' if len(self.quan) > 5 else ''}"
                )
            pheno_indices = [self.quan.index.get_loc(phenotype_id)]
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
        return pd.concat(results, ignore_index=True)

    def _process_phenotype_safe(self, pheno_index: int) -> pd.DataFrame:
        """Exception-safe wrapper for multiprocessing workers."""
        try:
            return self._process_phenotype(pheno_index)
        except Exception as e:
            current_pheno = self.quan.index[pheno_index]
            print(f"[finemapping] Error for phenotype {current_pheno}: {e}")
            return self._empty_result(current_pheno)
