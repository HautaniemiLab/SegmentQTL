import numpy as np
from scipy.stats import beta, f


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
