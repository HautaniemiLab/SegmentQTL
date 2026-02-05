from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta, f
from scipy.stats import t as t_dist


@dataclass
class VariantFWLCache:
    """Cache for fast permutation testing using Frisch-Waugh-Lovell residualization.

    For s/d parameterization:
    - s = REFlr + ALTlr (sum, captures total allelic dosage)
    - d = REFlr - ALTlr (difference, captures allelic imbalance = molQTL signal)

    The null model includes s (along with covariates), and we test whether d adds
    explanatory power: H0: β_d = 0.

    Optimized to:
    - Store idx_masky (indices into mask_y axis) instead of full sample indices
    - Store QcT (transposed) to avoid transpose per permutation
    - Store q_d (normalized residualized d vector) for single-predictor t-test
    - Use projection-based RSS computation for 1-df test
    """

    idx_masky: (
        np.ndarray
    )  # indices into mask_y axis (not full sample axis); shape (n_i,)
    QcT: np.ndarray  # transposed reduced Q for covariates+s; shape (rank_c, n_i)
    q_d: np.ndarray  # normalized residualized d vector; shape (n_i,)
    df2: int  # denominator df (n - p_null - 1)


def check_d_variance(d_filtered: np.ndarray, eps: float = 1e-10) -> bool:
    """
    Check if the allelic difference (d = REFlr - ALTlr) has adequate variation.

    Parameters:
    - d_filtered: Array of d values (REFlr - ALTlr)
    - eps: Minimum standard deviation threshold

    Returns:
    - Boolean value showing if d has sufficient variance for testing
    """
    std_d = np.std(d_filtered)
    return std_d >= eps


def check_grouping(
    altlr_filtered: np.ndarray, reflr_filtered: np.ndarray, eps: float = 1e-10
) -> bool:
    """
    Find if the genotype predictors have adequate variation in the data.
    For s/d parameterization, we check that d = REFlr - ALTlr has variance.

    Parameters:
    - altlr_filtered: Array of ALTlr values
    - reflr_filtered: Array of REFlr values
    - eps: Minimum standard deviation threshold

    Returns:
    - Boolean value showing if d (allelic difference) has sufficient variance
    """
    d = reflr_filtered - altlr_filtered
    return check_d_variance(d, eps)


def build_variant_fwl_caches(
    transf_variants_alt: pd.DataFrame,
    transf_variants_ref: pd.DataFrame,
    mask_y: np.ndarray,
    masky_pos: np.ndarray,
    phenotype_cov_full: np.ndarray | None,
    cov_values_full: list,
    min_samples: int = 30,
):
    """
    Pre-compute Frisch-Waugh-Lovell (FWL) caches for all variants in the window.

    Uses s/d parameterization:
    - s = REFlr + ALTlr (included in null model with covariates)
    - d = REFlr - ALTlr (the test predictor)

    The cache stores the residualized d vector (after projecting out covariates + s)
    for fast 1-df t-test computation during permutation.

    Parameters:
    - transf_variants_alt: DataFrame of transformed ALTlr variants
    - transf_variants_ref: DataFrame of transformed REFlr variants
    - mask_y: Boolean mask for samples passing phenotype+covariate filters
    - masky_pos: Mapping from full sample index -> mask_y position (-1 if not in mask_y)
    - phenotype_cov_full: Optional phenotype-level covariate
    - cov_values_full: List of sample-level covariates
    - min_samples: Minimum number of samples required for a variant to be cached

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
        if n_i < min_samples:
            continue

        # Slice genotypes to filtered set
        altlr_f = altlr_full[idx_full]
        reflr_f = reflr_full[idx_full]

        # Compute s and d
        s_f = reflr_f + altlr_f
        d_f = reflr_f - altlr_f

        # Check d variance (this is what we're testing)
        if not check_d_variance(d_f):
            continue

        # Convert idx_full to idx_masky (indices into mask_y axis)
        idx_masky = masky_pos[idx_full]
        # Sanity check: all indices should be valid (no -1 values)
        # since mask_full is a subset of mask_y
        assert np.all(idx_masky >= 0), "idx_masky contains invalid indices"

        # Build null model covariate matrix C (includes intercept, s, phenotype_cov, sample covariates)
        # s is part of the null model because we want to test d controlling for total dosage
        cov_blocks = [np.asarray(s_f, dtype=float)]  # s is always included
        if phenotype_cov_masky is not None:
            cov_blocks.append(np.asarray(phenotype_cov_masky[idx_masky], dtype=float))
        for cov_val_masky in cov_values_masky:
            cov_blocks.append(np.asarray(cov_val_masky[idx_masky], dtype=float))

        C = np.column_stack([np.ones(n_i)] + cov_blocks)

        # Compute QR of C for FWL
        try:
            Qc, Rc = np.linalg.qr(C, mode="reduced")
        except np.linalg.LinAlgError:
            continue

        # Store QcT (transposed) to avoid transpose per permutation
        QcT = Qc.T

        # Residualize d w.r.t. C: d_tilde = (I - Qc Qc^T) d
        d_vec = np.asarray(d_f, dtype=float)
        d_tilde = d_vec - Qc @ (QcT @ d_vec)

        # Normalize d_tilde for fast projection
        d_tilde_norm = np.linalg.norm(d_tilde)
        if d_tilde_norm < 1e-10:
            # d is collinear with covariates+s, skip
            continue
        q_d = d_tilde / d_tilde_norm

        # Degrees of freedom
        rank_c = np.linalg.matrix_rank(C)
        df2 = n_i - rank_c - 1  # -1 for the d predictor

        if df2 <= 0:
            continue

        caches[variant_index] = VariantFWLCache(
            idx_masky=idx_masky,
            QcT=QcT,
            q_d=q_d,
            df2=df2,
        )

    return caches


def fit_ols_null(y: np.ndarray, X: np.ndarray) -> tuple:
    """
    Fit OLS null model and return fitted values and residuals.
    Uses fast normal equations solver (matching fit_ols_and_test() path).

    Parameters:
    - y: Outcome vector (n,)
    - X: Design matrix (n, p), typically includes intercept and covariates

    Returns:
    - (y_hat, residuals): Tuple of fitted values and residuals, both 1D arrays
    """
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    X = np.asarray(X, dtype=float)

    n = y.shape[0]
    p = X.shape[1]

    if n <= p:
        # Not enough samples; return NaNs
        return np.full(n, np.nan), np.full(n, np.nan)

    # Fast OLS via normal equations
    XtX = X.T @ X
    Xty = X.T @ y

    try:
        # Try Cholesky for speed
        L = np.linalg.cholesky(XtX)
        z = np.linalg.solve(L, Xty)
        beta = np.linalg.solve(L.T, z)
    except np.linalg.LinAlgError:
        try:
            # Fallback to solve
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # Last resort: lstsq
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    y_hat = (X @ beta).flatten()
    residuals = (y - X @ beta).flatten()

    return y_hat, residuals


def residualize_vector(y: np.ndarray, X: np.ndarray) -> np.ndarray:
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


def calculate_aic_full_ols(y: np.ndarray, X: np.ndarray) -> float:
    """
    AIC for Gaussian OLS with MLE sigma^2 = RSS/n.

    Matches statsmodels OLS aic (up to floating rounding):
      AIC = n*(log(2*pi) + 1 + log(RSS/n)) + 2*k
    where k is number of regression parameters (including intercept).
    """
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    X = np.asarray(X, dtype=float)

    n = y.shape[0]
    if n == 0:
        return np.nan

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Add intercept
    Xd = np.column_stack([np.ones((n, 1)), X])
    k = Xd.shape[1]  # includes intercept

    if n <= k:
        return np.nan

    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    rss = float(np.sum(resid**2))

    if rss <= 0:
        return np.nan

    return float(n * (np.log(2.0 * np.pi) + 1.0 + np.log(rss / n)) + 2.0 * k)


def fit_ols_and_test(y: np.ndarray, X_null: np.ndarray, X_alt: np.ndarray):
    """
    Fit OLS models and perform partial F-test.

    NOTE: This function is called in the inner loop of cis-scans. Using
    np.linalg.lstsq (SVD) per variant is expensive, so we solve the normal
    equations (X'X) beta = X'y with np.linalg.solve / Cholesky where possible,
    and fall back to lstsq only if the Gram matrix is singular/ill-conditioned.

    Parameters:
    - y: Outcome vector (n,)
    - X_null: Null model design matrix (n, p_null) - includes intercept
    - X_alt: Alt model design matrix (n, p_alt) - includes intercept and extra predictors

    Returns:
    - Dictionary with:
        - beta_alt: Full set of OLS coefficients for alt model
        - se_alt: Standard errors for alt model coefficients
        - rss_null: Residual sum of squares for null model
        - rss_alt: Residual sum of squares for alt model
        - f_stat: F-statistic for partial F-test
        - p_value: p-value from partial F-test
        - r2_alt: R² from alt model
    """
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    X_null = np.asarray(X_null, dtype=float)
    X_alt = np.asarray(X_alt, dtype=float)

    n = y.shape[0]
    p_null = X_null.shape[1]
    p_alt = X_alt.shape[1]

    if n <= p_alt:
        return {
            "beta_alt": np.full(p_alt, np.nan),
            "se_alt": np.full(p_alt, np.nan),
            "rss_null": np.nan,
            "rss_alt": np.nan,
            "f_stat": np.nan,
            "p_value": np.nan,
            "r2_alt": np.nan,
        }

    def _ols_fast(X: np.ndarray, y_: np.ndarray):
        """Return (beta, rss, inv_xtx) with fast path via normal equations."""
        # Try normal equations (small p) first: solve (X'X)beta = X'y
        XtX = X.T @ X
        Xty = X.T @ y_
        try:
            # Prefer Cholesky for speed/stability if positive definite
            L = np.linalg.cholesky(XtX)
            # Solve L * z = X'y, then L.T * beta = z
            z = np.linalg.solve(L, Xty)
            beta = np.linalg.solve(L.T, z)
            # Inverse via Cholesky factors: inv(XtX) = inv(L.T) @ inv(L)
            Linv = np.linalg.solve(L, np.eye(L.shape[0]))
            inv_xtx = Linv.T @ Linv
        except np.linalg.LinAlgError:
            try:
                beta = np.linalg.solve(XtX, Xty)
                inv_xtx = np.linalg.inv(XtX)
            except np.linalg.LinAlgError:
                beta, *_ = np.linalg.lstsq(X, y_, rcond=None)
                inv_xtx = None

        resid = y_ - X @ beta
        rss = float(np.sum(resid**2))
        return beta, rss, inv_xtx

    # Fit null and alt
    beta_null, rss_null, _ = _ols_fast(X_null, y)
    beta_alt, rss_alt, inv_xtx_alt = _ols_fast(X_alt, y)

    # Standard errors for alt model
    sigma2_hat = rss_alt / (n - p_alt)
    if inv_xtx_alt is None:
        # fallback: try direct inverse; if still fails, return NaNs
        try:
            inv_xtx_alt = np.linalg.inv(X_alt.T @ X_alt)
        except np.linalg.LinAlgError:
            inv_xtx_alt = None

    if inv_xtx_alt is None:
        se_alt = np.full(p_alt, np.nan)
    else:
        se_alt = np.sqrt(
            np.clip(np.diag(inv_xtx_alt) * sigma2_hat, a_min=0.0, a_max=None)
        )

    # Partial F-test
    df1 = p_alt - p_null
    df2 = n - p_alt

    # Guard for numerical issues (can happen when rss_alt ~ 0)
    if (
        df1 <= 0
        or df2 <= 0
        or not np.isfinite(rss_null)
        or not np.isfinite(rss_alt)
        or rss_alt <= 0
    ):
        f_stat = np.nan
        p_value = np.nan
    else:
        f_stat = ((rss_null - rss_alt) / df1) / (rss_alt / df2)
        p_value = float(f.sf(f_stat, df1, df2)) if np.isfinite(f_stat) else np.nan

    # R² for alt model
    y_mean = float(np.mean(y))
    tss = float(np.sum((y - y_mean) ** 2))
    r2_alt = 1.0 - (rss_alt / tss) if tss > 0 else 0.0

    return {
        "beta_alt": beta_alt.flatten(),
        "se_alt": se_alt,
        "rss_null": rss_null,
        "rss_alt": rss_alt,
        "f_stat": f_stat,
        "p_value": p_value,
        "r2_alt": r2_alt,
    }


def fit_multivariate_ols(y: np.ndarray, X: np.ndarray):
    """
    Fit multivariate OLS model.

    Parameters:
    - y: Outcome vector (n,)
    - X: Design matrix (n, p) - should include intercept

    Returns:
    - Dictionary with:
        - beta: OLS coefficients
        - se: Standard errors
        - r2: R² value
        - rss: Residual sum of squares
    """
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    X = np.asarray(X, dtype=float)

    n = y.shape[0]
    p = X.shape[1]

    if n <= p:
        return {
            "beta": np.full(p, np.nan),
            "se": np.full(p, np.nan),
            "r2": np.nan,
            "rss": np.nan,
        }

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(np.sum(resid**2))

    sigma2_hat = rss / (n - p)

    try:
        var_covar_matrix = np.linalg.inv(X.T @ X) * sigma2_hat
        se = np.sqrt(np.diag(var_covar_matrix))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)

    y_mean = np.mean(y)
    tss = np.sum((y - y_mean) ** 2)
    r2 = 1.0 - (rss / tss) if tss > 0 else 0.0

    return {
        "beta": beta.flatten(),
        "se": se,
        "r2": r2,
        "rss": rss,
    }


def fit_beta_mle(pvals: np.ndarray) -> tuple:
    """
    Fit Beta distribution parameters to observed p-values using MLE.

    Parameters:
    - pvals: Array of p-values from permutation null distribution

    Returns:
    - (alpha, beta): Beta distribution shape parameters
    """
    pvals = np.asarray(pvals, dtype=float)
    pvals = pvals[~np.isnan(pvals)]

    if len(pvals) == 0:
        return 1.0, 1.0

    # Clip p-values away from 0 and 1 to avoid log(0) in Beta MLE
    pvals = np.clip(pvals, 1e-15, 1 - 1e-15)

    try:
        # Use scipy.stats.beta.fit for fast and stable MLE
        # floc=0, fscale=1 fixes the support to [0, 1]
        alpha, beta_shape, _, _ = beta.fit(pvals, floc=0, fscale=1)
        return alpha, beta_shape
    except Exception:
        # Fallback if fit fails (e.g., all values identical after clipping)
        return 1.0, 1.0


def gene_variant_regressions(
    gene_index: int,
    quantifications: pd.DataFrame,
    variant: str,
    regression_data: pd.DataFrame,
    record_aic: bool = False,
):
    """
    Find associations between phenotype levels and variants using s/d parameterization.

    Model: Phenotype ~ s + d + covariates
    where s = REFlr + ALTlr (total dosage) and d = REFlr - ALTlr (allelic difference).

    The test is whether β_d ≠ 0 (1-df t-test), which tests if the two alleles
    have different effects on phenotype (i.e., molQTL signal).

    Parameters:
    - gene_index: Index of a gene of interest on the quantification file.
    - quantifications: Dataframe of quantifications.
    - variant: Variant ID
    - regression_data: Regression data for current gene-variant pair including ALTlr, REFlr,
        optional phenotype_cov, and sample-level covariates
    - record_aic: Whether to compute AIC statistics

    Returns:
    - associations: Dataframe with statistics including beta_d, se_d, nominal_p (for H0: β_d = 0)
    """
    associations = []
    current_gene = quantifications.index[gene_index]

    def create_association(
        gene,
        variant_id,
        n_samples,
        beta_s,
        se_s,
        beta_d,
        se_d,
        t_stat_d,
        p_value,
        r2_alt=None,
        aic_null=None,
        aic_alt=None,
        delta_aic=None,
    ):
        association = {
            "phenotype": gene,
            "variant": variant_id,
            "number_of_samples": n_samples,
            "beta_s": beta_s,
            "se_s": se_s,
            "beta_d": beta_d,
            "se_d": se_d,
            "t_stat_d": t_stat_d,
            "nominal_p": p_value,
            "r2_alt": r2_alt,
        }
        if record_aic:
            association["aic_null"] = aic_null
            association["aic_alt"] = aic_alt
            association["delta_aic_alt_minus_null"] = delta_aic
        return association

    if len(regression_data) == 0:
        associations.append(
            create_association(
                current_gene,
                variant,
                0,
                np.nan,
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
    n = len(y)

    # Compute s and d from ALTlr and REFlr
    altlr = regression_data["ALTlr"].to_numpy(dtype=float)
    reflr = regression_data["REFlr"].to_numpy(dtype=float)
    s = reflr + altlr
    d = reflr - altlr

    # Build design matrices
    # Null model: Phenotype ~ s + (phenotype_cov if present) + sample_covariates
    cov_cols = [
        col for col in regression_data.columns if col not in ["GEX", "ALTlr", "REFlr"]
    ]
    X_null = np.column_stack(
        [np.ones(n), s]
        + [regression_data[col].to_numpy(dtype=float) for col in cov_cols]
    )

    # Alt model: Phenotype ~ s + d + (phenotype_cov if present) + sample_covariates
    X_alt = np.column_stack(
        [np.ones(n), s, d]
        + [regression_data[col].to_numpy(dtype=float) for col in cov_cols]
    )

    # Perform F-test (which is equivalent to t-test squared for 1-df)
    result = fit_ols_and_test(y, X_null, X_alt)

    # Extract coefficients: index 0=intercept, 1=s, 2=d, then covariates
    beta_s = result["beta_alt"][1]
    se_s = result["se_alt"][1]
    beta_d = result["beta_alt"][2]
    se_d = result["se_alt"][2]

    # Compute t-statistic for d
    if se_d > 0 and np.isfinite(se_d):
        t_stat_d = beta_d / se_d
    else:
        t_stat_d = np.nan

    # p-value for t-test on d (2-sided)
    df_resid = n - X_alt.shape[1]
    if df_resid > 0 and np.isfinite(t_stat_d):
        pval = float(2 * t_dist.sf(np.abs(t_stat_d), df_resid))
    else:
        pval = result["p_value"]  # Fallback to F-test p-value

    r2_alt = result["r2_alt"]

    # AIC recording
    aic_null = aic_alt = delta_aic = None
    if record_aic:
        aic_null = calculate_aic_full_ols(
            y, X_null[:, 1:]
        )  # Exclude intercept (it's added by the function)
        aic_alt = calculate_aic_full_ols(y, X_alt[:, 1:])  # Exclude intercept
        delta_aic = aic_alt - aic_null

    associations.append(
        create_association(
            current_gene,
            variant,
            n,
            beta_s,
            se_s,
            beta_d,
            se_d,
            t_stat_d,
            pval,
            r2_alt,
            aic_null,
            aic_alt,
            delta_aic,
        )
    )

    return pd.DataFrame(associations)


def gene_variant_regressions_permutations(
    gene_index: int,
    quantifications: pd.DataFrame,
    variant: str,
    regression_data: pd.DataFrame,
    transf_variants_alt: pd.DataFrame,
    transf_variants_ref: pd.DataFrame,
    phenotype_covariate_df: pd.DataFrame | None,
    perm_covariate_df: pd.DataFrame | None,
    cov: pd.DataFrame | None,
    num_permutations: int,
    perm_method: str,
    record_aic: bool = False,
):
    """
    Perform association testing using s/d parameterization with permutation adjustment.

    Uses Freedman-Lane permutation to test H0: β_d = 0 (allelic difference has no effect).

    Permutation logic:
    - Global null model for residualization: Phenotype ~ perm_covariate + sample_covariates
      (perm_covariate is typically gene-level CN to remove CN-driven structure)
    - Fit null model, compute residuals
    - Permute residuals, create pseudo-phenotype
    - For each permutation, scan all variants testing d while controlling for s
    - Compute adjusted p-value from permutation null distribution

    Note: perm_covariate_df is used ONLY for FL residualization, not in nominal model.
    """
    # Nominal association
    actual_associations = gene_variant_regressions(
        gene_index,
        quantifications,
        variant,
        regression_data,
        record_aic=record_aic,
    )

    # No permutations requested, return nominal results
    if num_permutations == 0:
        return actual_associations

    # If no usable data for the nominal pair, cannot permute
    if regression_data.shape[0] == 0:
        actual_associations["p_adj"] = np.nan
        return actual_associations

    # Prepare fixed (unpermuted) inputs for the scan
    current_gene = quantifications.index[gene_index]

    # Full phenotype across samples (same ordering as genotype columns / quantifications)
    GEX_full = pd.to_numeric(
        quantifications.iloc[gene_index, 3:], errors="coerce"
    ).to_numpy(dtype=float)

    # Optional phenotype-level covariate (NOT permuted)
    phenotype_cov_full = None
    if phenotype_covariate_df is not None:
        phenotype_cov_full = (
            phenotype_covariate_df.loc[current_gene].to_numpy().flatten()
        ).astype(float)

    # Optional sample-level covariates (NOT permuted)
    cov_values_full = []
    if cov is not None:
        cov_values_full = [
            pd.to_numeric(cov.loc[covariate], errors="coerce")
            .to_numpy()
            .flatten()
            .astype(float)
            for covariate in cov.index
        ]

    # Optional permutation-only covariate (for FL residualization, e.g., gene-level CN)
    perm_cov_full = None
    if perm_covariate_df is not None:
        perm_cov_full = (
            perm_covariate_df.loc[current_gene].to_numpy().flatten()
        ).astype(float)

    # Keep the nominal BEST p-value for permutation adjustment
    nominal_best_p = float(actual_associations["nominal_p"].iloc[0])

    # Freedman-Lane (residual-based) permutation to preserve covariate structure
    # Build phenotype+covariate mask
    mask_y = ~np.isnan(GEX_full)
    if phenotype_cov_full is not None:
        mask_y &= ~np.isnan(phenotype_cov_full)
    if perm_cov_full is not None:
        mask_y &= ~np.isnan(perm_cov_full)
    for cov_val in cov_values_full:
        mask_y &= ~np.isnan(cov_val)

    # Create mapping from full sample index -> mask_y position
    # masky_pos[i] = position in mask_y axis, or -1 if not in mask_y
    n_samples = len(GEX_full)
    masky_pos = np.full(n_samples, -1, dtype=np.intp)
    masky_pos[mask_y] = np.arange(np.sum(mask_y))

    # Build FWL caches for fast permutation scanning
    variant_caches = build_variant_fwl_caches(
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
    perm_cov_masky = perm_cov_full[mask_y] if perm_cov_full is not None else None
    cov_values_masky = [cov_val[mask_y] for cov_val in cov_values_full]

    # Build null design matrix for FL residualization (on mask_y filtered set)
    # This includes perm_covariate (gene-level CN) to remove CN-driven structure
    # before permuting, making residuals approximately exchangeable under H0: β_d = 0
    cov_blocks = []
    if perm_cov_masky is not None:
        cov_blocks.append(np.asarray(perm_cov_masky, dtype=float))
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

    for _ in range(num_permutations):
        # Freedman-Lane: permute residuals in mask_y axis
        perm = rng.permutation(n_masky)
        y_perm_masky = yhat_masky + resid_masky[perm]

        best_abs_t = -np.inf
        best_df2 = None

        # Re-scan all variants using cached FWL decompositions
        for cache in variant_caches.values():
            # Slice permuted phenotype using idx_masky (indices into mask_y axis)
            y_perm = y_perm_masky[cache.idx_masky]

            # FWL residualization w.r.t. covariates+s: y_tilde = (I - Qc Qc^T) y_perm
            # Using pre-transposed QcT to avoid transpose per iteration
            y_tilde = y_perm - cache.QcT.T @ (cache.QcT @ y_perm)

            # For 1-df t-test on d:
            # t_d = (q_d @ y_tilde) / sigma_hat
            # where sigma_hat = sqrt(RSS / df2) and RSS = ||y_tilde - (q_d @ y_tilde) * q_d||^2
            coef_d = float(np.dot(cache.q_d, y_tilde))
            res = y_tilde - coef_d * cache.q_d
            rss = float(np.dot(res, res))

            if rss > 0 and cache.df2 > 0:
                sigma_hat = np.sqrt(rss / cache.df2)
                if sigma_hat > 1e-15:
                    t_stat = coef_d / sigma_hat
                else:
                    t_stat = np.nan
            else:
                t_stat = np.nan

            abs_t = np.abs(t_stat) if np.isfinite(t_stat) else -np.inf
            if abs_t > best_abs_t:
                best_abs_t = abs_t
                best_df2 = cache.df2

        # Compute p-value from best |t| and its df
        if best_df2 is not None and best_abs_t > -np.inf:
            p_best = float(2 * t_dist.sf(best_abs_t, best_df2))
        else:
            p_best = np.nan
        best_p_perms.append(p_best)

    adjusted_p_value = np.nan

    if perm_method == "direct":
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
