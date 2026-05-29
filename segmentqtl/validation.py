"""
Validation mode: assess generalization of finemapping models from a main cohort
to an independent validation cohort.

Two scoring modes are supported:

- ``recalibrated`` (default): freeze the genetic component (selected variants,
  betas, and the main-cohort centering/scaling applied to validation d) and
  refit the unpenalised block (intercept, CN, covariate effects) on the
  validation cohort. Answers: "Does the genetic component transfer once
  cohort-level baseline / covariate drift is allowed to adjust?"

- ``frozen``: apply the discovery model exactly as learned. Genetic betas,
  preprocessing, and main-cohort theta are all reused; nothing is refit on
  validation. Answers: "Does the discovery model transfer as built?"

Both scoring modes apply the **main-cohort mean and SD** to standardise
validation d (REFlr - ALTlr), so that the frozen betas operate on the same
feature definition they were trained on. Predictions use the same
missing-aware rule as `Finemapping`: only observed variant entries
contribute for each sample.

In addition to RMSE/MAE/R²/calibration, this module reports:

- Calibration joint Wald test (HC3-robust) for H0: a=0, b=1.
- Genetic transfer slope rho from y ~ X_unpen + rho * g_frozen.
- Burden-stratified R² (terciles of `n_obs_used`).
- Optional paired bootstrap CIs for R² and calibration slope.
"""

import os
from multiprocessing import Pool
from time import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, pearsonr, spearmanr
from scipy.stats import t as t_dist
from tqdm import tqdm

from finemapping import (
    Finemapping,
    MissingAwareElasticNet,
    _add_beta_contributions,
    _obs_masks_to_csr,
    _subtract_beta_contributions,
)
from statistical_utils import ols_fit, r_squared, standardize_variants

# ----------------------------------------------------------------------
# Stat helpers
# ----------------------------------------------------------------------


_FINEMAP_REQUIRED_COLS = (
    "phenotype",
    "variant",
    "mean_d",
    "sd_d",
    "stability_score",
    "lambda_selected",
    "beta_full",
)


def _load_finemap_results(finemap_results_dir: str, chromosome: str) -> pd.DataFrame:
    """
    Load a previously written ``finemap_<chr>.csv`` so that validation can
    reuse main-cohort betas/mu/sd/lambda/stability scores instead of
    re-fitting the Elastic Net.

    Hard-errors if the file is missing or required columns are absent.
    """
    fname = os.path.join(finemap_results_dir, f"finemap_{chromosome}.csv")
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"[validation] --finemap_results_dir does not contain "
            f"finemap_{chromosome}.csv (looked at {fname})."
        )
    df = pd.read_csv(fname)
    missing = [c for c in _FINEMAP_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[validation] {fname} is missing required columns: {missing}. "
            f"Re-run finemap with the current SegmentQTL version to regenerate it."
        )
    return df


def _calibration_hc3(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    """
    Fit y = a + b * yhat, return (a, b) and HC3-robust joint Wald test for
    H0: a=0, b=1.
    """
    n = len(y)
    if n < 3 or np.std(yhat) < 1e-12:
        return {
            "calibration_intercept": float("nan"),
            "calibration_slope": float("nan"),
            "calibration_joint_wald": float("nan"),
            "calibration_joint_pval": float("nan"),
        }
    X = np.column_stack([np.ones(n), yhat])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y
    e = y - X @ beta_hat

    # Hat-matrix diagonal h_ii = X_i (X'X)^-1 X_i'
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    h = np.clip(h, 0.0, 1.0 - 1e-8)
    # HC3 weights e_i^2 / (1 - h_ii)^2
    omega = (e**2) / (1.0 - h) ** 2
    meat = (X.T * omega) @ X
    V = XtX_inv @ meat @ XtX_inv

    diff = beta_hat - np.array([0.0, 1.0])
    try:
        wald = float(diff @ np.linalg.solve(V, diff))
    except np.linalg.LinAlgError:
        wald = float("nan")
    pval = float("nan") if not np.isfinite(wald) else float(1.0 - chi2.cdf(wald, df=2))
    return {
        "calibration_intercept": float(beta_hat[0]),
        "calibration_slope": float(beta_hat[1]),
        "calibration_joint_wald": wald,
        "calibration_joint_pval": pval,
    }


def _genetic_transfer_slope(
    y: np.ndarray, X_unpen: np.ndarray, g_frozen: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Fit y ~ X_unpen + rho * g_frozen and return
    ``(rho_hat, p_rho, rho_adj_hat, p_rho_adj)`` where:

    - ``rho_hat`` is the genetic transfer slope. Should be ~1 if the
      frozen genetic component transfers cleanly after local adjustment
      of the unpenalised block. The p-value tests H0: rho = 0
      (i.e. "no genetic signal transfers").
    - ``rho_adj_hat = rho_hat - 1`` is the **adjustment** to the frozen
      unit slope. Should be ~0 if the frozen component transfers
      cleanly. ``p_rho_adj`` tests H0: rho = 1 (equivalently
      rho_adj = 0), i.e. "no scale adjustment of the genetic component
      is needed". This is the natural transferability null for a
      fully-frozen model, where the genetic-component slope is fixed
      at 1 by construction.

    The two parametrisations share the same SE; only the test statistic
    recenters.
    """
    n = len(y)
    nan_tuple = (float("nan"), float("nan"), float("nan"), float("nan"))
    if n < X_unpen.shape[1] + 2 or np.std(g_frozen) < 1e-12:
        return nan_tuple
    X = np.column_stack([X_unpen, g_frozen])
    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
    except np.linalg.LinAlgError:
        return nan_tuple
    beta = XtX_inv @ X.T @ y
    rho = float(beta[-1])
    rho_adj = rho - 1.0
    e = y - X @ beta
    df = n - X.shape[1]
    if df <= 0:
        return rho, float("nan"), rho_adj, float("nan")
    sigma2 = float(e @ e) / df
    se_rho = float(np.sqrt(sigma2 * XtX_inv[-1, -1]))
    if se_rho <= 0:
        return rho, float("nan"), rho_adj, float("nan")
    t = rho / se_rho
    t_adj = rho_adj / se_rho
    pval = float(2.0 * (1.0 - t_dist.cdf(abs(t), df=df)))
    pval_adj = float(2.0 * (1.0 - t_dist.cdf(abs(t_adj), df=df)))
    return rho, pval, rho_adj, pval_adj


def _phenotype_passes_support_filter(
    pheno_rows: pd.DataFrame,
    definition: str,
    min_stability: float,
) -> bool:
    """Support filter applied to a phenotype's rows from finemap_<chr>.csv.

    Two definitions are supported:

    - ``stability`` (recommended): pass if at least one variant has
      ``stability_score >= min_stability``. Aligns with the validation
      bootstrap mask threshold; selects phenotypes where the discovery
      EN solution is bootstrap-supported.
    - ``selected``: pass if at least one variant has ``beta_full != 0``,
      i.e. the EN at lambda-min selected at least one variant. Looser;
      retains many lambda-min-selected variants that did not survive
      stability bootstrapping.

    The filter is applied at the phenotype level (any variant suffices)
    so that downstream scoring still uses the full kept variant set.
    """
    if pheno_rows.empty:
        return False
    if definition == "stability":
        s = pheno_rows["stability_score"].to_numpy(dtype=float)
        return bool(np.any(np.isfinite(s) & (s >= min_stability)))
    if definition == "selected":
        b = pheno_rows["beta_full"].to_numpy(dtype=float)
        return bool(np.any(np.isfinite(b) & (np.abs(b) > 0.0)))
    raise ValueError(
        f"Unknown support_definition {definition!r}; expected 'stability' or 'selected'."
    )


def _permutation_pvals(
    y: np.ndarray,
    yhat_obs: np.ndarray,
    X_unpen: np.ndarray,
    g_frozen: np.ndarray,
    validation_mode: str,
    n_perm: int,
    seed: int,
    obs_r2: float,
    obs_rho: float,
) -> Dict[str, float]:
    """Permutation p-values for r2_descriptive and rho.

    Shuffles validation phenotype labels across samples ``n_perm`` times,
    keeping the genetic component ``g_frozen`` and unpenalised design
    ``X_unpen`` fixed. For each permuted ``y_p``:

    * Compute ``sse_p`` against either the frozen prediction (frozen mode,
      ``yhat_obs`` invariant to permutation) or the recalibrated
      prediction obtained by re-fitting ``y_p - g_frozen ~ X_unpen``
      (recalibrated mode);
    * Compute ``rho_p`` as the last coefficient of the augmented
      regression ``y_p ~ [X_unpen, g_frozen]``.

    The reported p-value for each statistic is

        p = (1 + #{null >= observed}) / (1 + n_perm)

    one-sided in the natural direction (larger r2/rho =>
    stronger transport). NaN observed statistic ⇒ NaN p-value.

    Parameters
    ----------
    y : observed validation phenotype.
    yhat_obs : observed prediction (used in frozen mode to seed sse_p).
    X_unpen : unpenalised block on validation samples.
    g_frozen : frozen genetic component on validation samples.
    validation_mode : 'recalibrated' or 'frozen'.
    n_perm : number of permutations (caller skips this if 0).
    seed : RNG seed (deterministic per phenotype).
    obs_* : observed values of the two statistics, computed once on
        the unshuffled data; reused as the comparison threshold.

    Returns
    -------
    dict with ``r2_perm_pval`` and ``rho_perm_pval``.
    """
    out = {
        "r2_perm_pval": float("nan"),
        "rho_perm_pval": float("nan"),
    }
    if n_perm <= 0:
        return out
    n = len(y)
    if n < X_unpen.shape[1] + 2:
        return out

    rng = np.random.default_rng(seed)

    # Pre-factor designs so each permutation costs only matvecs.
    XtX_inv = np.linalg.pinv(X_unpen.T @ X_unpen)
    X_aug = np.column_stack([X_unpen, g_frozen])
    if np.std(g_frozen) > 1e-12:
        XtX_aug_inv = np.linalg.pinv(X_aug.T @ X_aug)
        aug_solver = XtX_aug_inv @ X_aug.T  # last row picks rho
    else:
        aug_solver = None  # rho not estimable

    sst = float(np.sum((y - y.mean()) ** 2))  # invariant to permutation

    # Counters of null exceedances (>= observed). One-sided 'null at least
    # as extreme as observed' with extremeness defined as 'larger' since
    # both statistics are signed-positive under transport.
    cnt_r2 = 0
    cnt_rho = 0
    valid_r2 = 0
    valid_rho = 0

    for _ in range(n_perm):
        idx = rng.permutation(n)
        y_p = y[idx]

        # Full-model prediction depends on mode.
        if validation_mode == "frozen":
            # yhat does not depend on y; reuse observed prediction.
            yhat_p = yhat_obs
        else:
            # Recalibrated: theta_recal_p = XtX_inv @ X.T @ (y_p - g_frozen).
            theta_p = XtX_inv @ (X_unpen.T @ (y_p - g_frozen))
            yhat_p = X_unpen @ theta_p + g_frozen
        resid_p = y_p - yhat_p
        sse_p = float(resid_p @ resid_p)

        if sst > 0:
            r2_p = 1.0 - sse_p / sst
            valid_r2 += 1
            if np.isfinite(obs_r2) and r2_p >= obs_r2:
                cnt_r2 += 1
        if aug_solver is not None:
            beta_aug = aug_solver @ y_p
            rho_p = float(beta_aug[-1])
            valid_rho += 1
            if np.isfinite(obs_rho) and rho_p >= obs_rho:
                cnt_rho += 1

    if np.isfinite(obs_r2) and valid_r2 > 0:
        out["r2_perm_pval"] = (1 + cnt_r2) / (1 + valid_r2)
    if np.isfinite(obs_rho) and valid_rho > 0:
        out["rho_perm_pval"] = (1 + cnt_rho) / (1 + valid_rho)
    return out


def _burden_tercile_r2(
    y: np.ndarray, yhat: np.ndarray, n_obs_used: np.ndarray
) -> Tuple[float, float, float]:
    """Return R² within low/mid/high terciles of `n_obs_used`."""
    n = len(y)
    if n < 9:
        return float("nan"), float("nan"), float("nan")
    q1, q2 = np.quantile(n_obs_used, [1 / 3, 2 / 3])
    masks = [n_obs_used <= q1, (n_obs_used > q1) & (n_obs_used <= q2), n_obs_used > q2]
    out = []
    for m in masks:
        if int(m.sum()) < 3 or np.var(y[m]) <= 0:
            out.append(float("nan"))
        else:
            out.append(float(r_squared(y[m], yhat[m])))
    return out[0], out[1], out[2]


def _paired_bootstrap_ci(
    y: np.ndarray,
    yhat: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    """Paired bootstrap 95% CIs for out-of-sample R² and calibration slope."""
    n = len(y)
    if n < 5 or n_boot <= 0:
        return {
            "r2_ci_lo": float("nan"),
            "r2_ci_hi": float("nan"),
            "cal_slope_ci_lo": float("nan"),
            "cal_slope_ci_hi": float("nan"),
        }
    rng = np.random.default_rng(seed)
    r2s = np.empty(n_boot)
    slopes = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, yhb = y[idx], yhat[idx]
        if np.var(yb) <= 0 or np.std(yhb) < 1e-12:
            r2s[b] = np.nan
            slopes[b] = np.nan
            continue
        r2s[b] = r_squared(yb, yhb)
        # OLS slope
        xc = yhb - yhb.mean()
        yc = yb - yb.mean()
        denom = float(xc @ xc)
        slopes[b] = float(xc @ yc) / denom if denom > 0 else np.nan
    return {
        "r2_ci_lo": float(np.nanquantile(r2s, 0.025)),
        "r2_ci_hi": float(np.nanquantile(r2s, 0.975)),
        "cal_slope_ci_lo": float(np.nanquantile(slopes, 0.025)),
        "cal_slope_ci_hi": float(np.nanquantile(slopes, 0.975)),
    }


def _build_audit_rows(
    phenotype: str,
    kept_ids_main: np.ndarray,
    shared_main_pos: List[int],
    beta_refit_main: np.ndarray,
    mu_vec_main: np.ndarray,
    sd_vec_main: np.ndarray,
    theta_main: np.ndarray,
    theta_val: Optional[np.ndarray],
    theta_base: Optional[np.ndarray],
    unpen_names: List[str],
    main_unpen_transforms: List[Tuple[float, float, Optional[float]]],
    val_unpen_transforms: List[Tuple[float, float, Optional[float]]],
) -> pd.DataFrame:
    """
    Assemble a long-format audit table for a single phenotype.

    Schema: phenotype, block, name, value_main, value_val, shared_with_val.

    Blocks emitted:
    - ``beta``: refit genetic coefficient in standardised d-units (one row
      per main-kept variant; post-stability-mask if --validate_with_bootstrap).
    - ``variant_mu``, ``variant_sd``: main-cohort centring/scaling on d
      (always identical across cohorts -- value_val is NaN).
    - ``theta``: full-model unpenalised-block coefficient. value_val is the
      validation refit when validation_mode == "recalibrated", else NaN.
    - ``theta_baseline``: baseline-model (covariates-only) unpenalised-block
      coefficient, fit on validation data. Always present when baseline is computed.
    - ``unpen_mu``, ``unpen_sd``: per-cohort centring/scaling for
      non-intercept unpenalised columns. Both columns populated to expose
      cohort drift even when frozen-mode prediction uses only the main
      values.

    The intercept theta/theta_baseline rows use name == "intercept"; non-intercept rows
    use the labels from ``unpen_names`` (e.g. CN, phenotype_cov, sex).
    """
    shared_set = set(int(j) for j in shared_main_pos)
    rows: List[Dict[str, object]] = []

    for j, vid in enumerate(kept_ids_main):
        shared = j in shared_set
        rows.append(
            {
                "phenotype": phenotype,
                "block": "beta",
                "name": str(vid),
                "value_main": float(beta_refit_main[j]),
                "value_val": np.nan,
                "shared_with_val": shared,
            }
        )
        rows.append(
            {
                "phenotype": phenotype,
                "block": "variant_mu",
                "name": str(vid),
                "value_main": float(mu_vec_main[j]),
                "value_val": np.nan,
                "shared_with_val": shared,
            }
        )
        rows.append(
            {
                "phenotype": phenotype,
                "block": "variant_sd",
                "name": str(vid),
                "value_main": float(sd_vec_main[j]),
                "value_val": np.nan,
                "shared_with_val": shared,
            }
        )

    # Intercept (always position 0 of theta) -- full model
    rows.append(
        {
            "phenotype": phenotype,
            "block": "theta",
            "name": "intercept",
            "value_main": float(theta_main[0]),
            "value_val": (float(theta_val[0]) if theta_val is not None else np.nan),
            "shared_with_val": True,
        }
    )

    # Intercept (always position 0 of theta) -- baseline model
    if theta_base is not None:
        rows.append(
            {
                "phenotype": phenotype,
                "block": "theta_baseline",
                "name": "intercept",
                "value_main": np.nan,
                "value_val": float(theta_base[0]),
                "shared_with_val": True,
            }
        )

    # Non-intercept unpenalised columns. unpen_names is keyed to the main
    # cohort; align validation transforms by index up to the available
    # length, padding with NaN if validation has fewer blocks (mismatched
    # configurations are allowed in recalibrated mode).
    n_unpen = len(unpen_names)
    for k in range(n_unpen):
        name = unpen_names[k]
        v_main_th = float(theta_main[k + 1])
        v_val_th = float(theta_val[k + 1]) if theta_val is not None else np.nan
        v_base_th = float(theta_base[k + 1]) if theta_base is not None else np.nan
        rows.append(
            {
                "phenotype": phenotype,
                "block": "theta",
                "name": name,
                "value_main": v_main_th,
                "value_val": v_val_th,
                "shared_with_val": True,
            }
        )

        if theta_base is not None:
            rows.append(
                {
                    "phenotype": phenotype,
                    "block": "theta_baseline",
                    "name": name,
                    "value_main": np.nan,
                    "value_val": v_base_th,
                    "shared_with_val": True,
                }
            )
        mu_m, sd_m, _ = main_unpen_transforms[k]
        if k < len(val_unpen_transforms):
            mu_v, sd_v, _ = val_unpen_transforms[k]
            mu_v = float(mu_v)
            sd_v = float(sd_v)
        else:
            mu_v = np.nan
            sd_v = np.nan
        rows.append(
            {
                "phenotype": phenotype,
                "block": "unpen_mu",
                "name": name,
                "value_main": float(mu_m),
                "value_val": mu_v,
                "shared_with_val": True,
            }
        )
        rows.append(
            {
                "phenotype": phenotype,
                "block": "unpen_sd",
                "name": name,
                "value_main": float(sd_m),
                "value_val": sd_v,
                "shared_with_val": True,
            }
        )

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Validation class
# ----------------------------------------------------------------------


class Validation:
    """
    Validate a finemapping model fit on a main cohort against an independent
    validation cohort.

    The class composes two ``Finemapping`` instances (one per cohort) and
    reuses their data-loading, segment-filtering, and per-phenotype
    preparation logic. For each phenotype:

    1. Fit the missing-aware Elastic Net on the main cohort (CV-selected
       lambda) and obtain refit betas.
    2. Apply the main-cohort mean / SD to standardise validation d for the
       same variant set.
    3. Score the validation cohort under the chosen ``validation_mode``
       (``recalibrated`` or ``frozen``) and compute predictive metrics,
       calibration, transport tests, and (optional) bootstrap CIs.

    Bootstrap stability selection is skipped by default for speed; the
    refit betas from the CV solution are used directly.
    """

    def __init__(
        self,
        chromosome: str,
        # Main cohort paths
        quantifications_main: str,
        covariates_main: Optional[str],
        segmentation_main: str,
        genotype_alt_main: str,
        genotype_ref_main: str,
        copynumber_main: Optional[str],
        phenotype_covariate_main: Optional[str],
        # Validation cohort paths
        quantifications_val: str,
        covariates_val: Optional[str],
        segmentation_val: str,
        genotype_alt_val: str,
        genotype_ref_val: str,
        copynumber_val: Optional[str],
        phenotype_covariate_val: Optional[str],
        # Hyperparameters
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
        # Validation-specific flags
        validation_mode: str = "recalibrated",
        validate_with_bootstrap: bool = False,
        validation_stability_threshold: float = 0.6,
        bootstrap_ci: bool = False,
        n_boot_ci: int = 1000,
        save_model_audit: bool = False,
        finemap_results_dir: Optional[str] = None,
        restrict_to_supported_phenotypes: bool = False,
        support_definition: str = "stability",
        support_min_stability: float = 0.6,
        n_permutations: int = 0,
    ):
        if validation_mode not in ("recalibrated", "frozen"):
            raise ValueError(
                f"validation_mode must be 'recalibrated' or 'frozen', got {validation_mode!r}"
            )
        if support_definition not in ("stability", "selected"):
            raise ValueError(
                f"support_definition must be 'stability' or 'selected', got {support_definition!r}"
            )
        if restrict_to_supported_phenotypes and finemap_results_dir is None:
            raise ValueError(
                "--restrict_to_supported_phenotypes requires --finemap_results_dir to be set."
            )
        self.chromosome = chromosome
        self.validation_mode = validation_mode
        self.validate_with_bootstrap = validate_with_bootstrap
        self.validation_stability_threshold = validation_stability_threshold
        self.bootstrap_ci = bootstrap_ci
        self.n_boot_ci = n_boot_ci
        self.save_model_audit = save_model_audit
        self.finemap_results_dir = finemap_results_dir
        self.restrict_to_supported_phenotypes = restrict_to_supported_phenotypes
        self.support_definition = support_definition
        self.support_min_stability = float(support_min_stability)
        self.n_permutations = int(n_permutations)
        if finemap_results_dir is not None:
            self._finemap_main_df = _load_finemap_results(
                finemap_results_dir, chromosome
            )
        else:
            self._finemap_main_df = None

        common_kwargs = dict(
            window=window,
            num_cores=1,  # finemapping is driven phenotype-by-phenotype here
            alpha_en=alpha_en,
            coverage_tau=coverage_tau,
            n_bootstrap=n_bootstrap,
            subsample_frac=subsample_frac,
            n_lambda=n_lambda,
            lambda_ratio=lambda_ratio,
            cv_tau=cv_tau,
            min_obs_boot=min_obs_boot,
        )

        self.fm_main = Finemapping(
            chromosome,
            quantifications_main,
            covariates_main,
            segmentation_main,
            genotype_alt_main,
            genotype_ref_main,
            copynumber_main,
            phenotype_covariate_main,
            **common_kwargs,
        )
        self.fm_val = Finemapping(
            chromosome,
            quantifications_val,
            covariates_val,
            segmentation_val,
            genotype_alt_val,
            genotype_ref_val,
            copynumber_val,
            phenotype_covariate_val,
            **common_kwargs,
        )

        # Outer-level parallelism (over phenotypes)
        self.num_cores = num_cores

        # Hyperparameters needed by per-phenotype work
        self.alpha_en = alpha_en
        self.coverage_tau = coverage_tau
        self.n_lambda = n_lambda
        self.lambda_ratio = lambda_ratio
        self.cv_tau = cv_tau

    # ------------------------------------------------------------------
    # Per-phenotype pipeline
    # ------------------------------------------------------------------

    def _empty_metrics(self, phenotype: str) -> Dict[str, object]:
        return {
            "phenotype": phenotype,
            "n_samples_main": 0,
            "n_samples_val": 0,
            "n_variants_main_kept": 0,
            "n_variants_shared": 0,
            "n_variants_stable": np.nan,
            "median_n_obs_val": np.nan,
            "rmse": np.nan,
            "mse": np.nan,
            "mae": np.nan,
            "r2_descriptive": np.nan,
            "r2_baseline_val": np.nan,
            "r2_full_val": np.nan,
            "pearson_r": np.nan,
            "pearson_pval": np.nan,
            "spearman_rho": np.nan,
            "spearman_pval": np.nan,
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
            "calibration_joint_wald": np.nan,
            "calibration_joint_pval": np.nan,
            "rho_genetic_transfer": np.nan,
            "rho_genetic_transfer_pval": np.nan,
            "rho_genetic_adjustment": np.nan,
            "rho_genetic_adjustment_pval": np.nan,
            "r2_burden_low": np.nan,
            "r2_burden_mid": np.nan,
            "r2_burden_high": np.nan,
            "r2_ci_lo": np.nan,
            "r2_ci_hi": np.nan,
            "cal_slope_ci_lo": np.nan,
            "cal_slope_ci_hi": np.nan,
            "mean_bias": np.nan,
            "sd_residual": np.nan,
            "lambda_selected": np.nan,
            "validation_mode": self.validation_mode,
            "r2_perm_pval": np.nan,
            "rho_perm_pval": np.nan,
        }

    def _process_phenotype(
        self, pheno_index_main: int
    ) -> Tuple[Optional[Dict[str, object]], pd.DataFrame, pd.DataFrame]:
        current_pheno = self.fm_main.quan.index[pheno_index_main]

        # Map phenotype to validation cohort
        if current_pheno not in self.fm_val.quan.index:
            return (
                self._empty_metrics(current_pheno),
                pd.DataFrame(),
                pd.DataFrame(),
            )
        pheno_index_val = int(self.fm_val.quan.index.get_loc(current_pheno))

        data_main = self.fm_main._prepare_phenotype(pheno_index_main)
        if data_main is None:
            return (
                self._empty_metrics(current_pheno),
                pd.DataFrame(),
                pd.DataFrame(),
            )
        data_val = self.fm_val._prepare_phenotype(pheno_index_val)
        if data_val is None:
            return (
                self._empty_metrics(current_pheno),
                pd.DataFrame(),
                pd.DataFrame(),
            )

        # ── Fit on main ──
        d_std_main, obs_masks_main, keep_idx_main, sd_vec_main, mu_vec_main = (
            standardize_variants(
                data_main["d_raw"], self.coverage_tau, data_main["n_samples"]
            )
        )
        if len(keep_idx_main) == 0:
            return (
                self._empty_metrics(current_pheno),
                pd.DataFrame(),
                pd.DataFrame(),
            )
        kept_ids_main = data_main["variant_ids"][keep_idx_main]

        # ── Source main-cohort betas / lambda / stability ──
        # Either by re-fitting the EN here or by loading a pre-computed
        # finemap_<chr>.csv. The latter is the recommended workflow when
        # validation is run after a finemap pass.
        #
        # theta_main is NOT taken from either source. It is always
        # recomputed below by missing-aware OLS at the final (possibly
        # stability-masked) beta. Without masking this is mathematically
        # identical to the EN unpenalised joint refit; with masking it
        # correctly recalibrates the unpenalised block to the masked
        # genetic component (otherwise frozen-mode predictions are
        # biased).
        pi_v_from_finemap: Optional[np.ndarray] = None
        if self._finemap_main_df is not None:
            pheno_rows = self._finemap_main_df[
                self._finemap_main_df["phenotype"] == current_pheno
            ]
            if len(pheno_rows) == 0:
                return (
                    self._empty_metrics(current_pheno),
                    pd.DataFrame(),
                    pd.DataFrame(),
                )
            if pheno_rows["variant"].isna().all():
                return (
                    self._empty_metrics(current_pheno),
                    pd.DataFrame(),
                    pd.DataFrame(),
                )

            # Support filter: discovery-side restriction to phenotypes whose
            # finemap solution carries at least one bootstrap-supported
            # (or, optionally, lambda-min-selected) variant. Skipped when
            # --restrict_to_supported_phenotypes is off. Filter is applied
            # here so the
            # phenotype's full variant set is still used downstream.
            if (
                self.restrict_to_supported_phenotypes
                and not _phenotype_passes_support_filter(
                    pheno_rows,
                    self.support_definition,
                    self.support_min_stability,
                )
            ):
                # Skip unsupported phenotypes entirely (no metrics row).
                return None, pd.DataFrame(), pd.DataFrame()

            csv_variants = pheno_rows["variant"].to_numpy()
            if len(csv_variants) != len(kept_ids_main) or not np.array_equal(
                csv_variants.astype(kept_ids_main.dtype, copy=False), kept_ids_main
            ):
                raise RuntimeError(
                    f"[validation] finemap result variants for {current_pheno!r} "
                    f"do not match main-cohort kept variants. CSV has "
                    f"{len(csv_variants)} variants, main cohort kept "
                    f"{len(kept_ids_main)}. Inputs (quantifications, genotypes, "
                    f"coverage_tau) must match the finemap run that produced "
                    f"--finemap_results_dir."
                )
            csv_mu = pheno_rows["mean_d"].to_numpy(dtype=float)
            csv_sd = pheno_rows["sd_d"].to_numpy(dtype=float)
            if not np.allclose(
                csv_mu, mu_vec_main, rtol=1e-6, atol=1e-8
            ) or not np.allclose(csv_sd, sd_vec_main, rtol=1e-6, atol=1e-8):
                raise RuntimeError(
                    f"[validation] finemap result mu/sd for {current_pheno!r} "
                    f"do not match recomputed values. Inputs may have changed "
                    f"since finemap was run."
                )
            beta_refit_main = pheno_rows["beta_full"].to_numpy(dtype=float)
            lam_selected = float(pheno_rows["lambda_selected"].iloc[0])
            pi_v_from_finemap = pheno_rows["stability_score"].to_numpy(dtype=float)
        else:
            en = MissingAwareElasticNet(alpha_en=self.alpha_en)
            _, _, lam_selected, _, _, beta_refit_main = en.fit_path_cv(
                data_main["y"],
                d_std_main,
                obs_masks_main,
                data_main["X_unpen"],
                n_lambda=self.n_lambda,
                lambda_ratio=self.lambda_ratio,
                cv_tau=self.cv_tau,
                seed=42 + pheno_index_main,
            )

        # Optional stability-selection mask on main betas
        unstable_main: Optional[np.ndarray] = None
        if self.validate_with_bootstrap:
            if pi_v_from_finemap is not None:
                pi_v_main = pi_v_from_finemap
            else:
                rng = np.random.default_rng(seed=42 + pheno_index_main)
                d_raw_main_kept = data_main["d_raw"][keep_idx_main]
                pi_v_main, _, _, _ = self.fm_main._stability_selection(
                    current_pheno,
                    kept_ids_main,
                    data_main["y"],
                    d_raw_main_kept,
                    data_main["X_unpen"],
                    lam_selected,
                    rng,
                )
            unstable_main = pi_v_main < self.validation_stability_threshold
            beta_refit_main = beta_refit_main.copy()
            beta_refit_main[unstable_main] = 0.0

        # Recompute theta_main from missing-aware OLS at the FINAL beta
        # (after any stability masking). This restores joint consistency
        # between theta and the genetic component used in frozen mode.
        n_main_kept = len(kept_ids_main)
        csr_idx_main, csr_off_main = _obs_masks_to_csr(obs_masks_main)
        y_adj_main = data_main["y"].copy()
        _subtract_beta_contributions(
            y_adj_main,
            beta_refit_main,
            d_std_main,
            csr_idx_main,
            csr_off_main,
            n_main_kept,
        )
        theta_main = ols_fit(y_adj_main, data_main["X_unpen"])

        # ── Apply main preprocessing to validation ──
        # Map kept_ids_main into val variant index
        val_ids = data_val["variant_ids"]
        val_id_to_idx = {vid: i for i, vid in enumerate(val_ids)}
        shared_main_pos: List[int] = []
        shared_val_pos: List[int] = []
        for j, vid in enumerate(kept_ids_main):
            i_val = val_id_to_idx.get(vid)
            if i_val is not None:
                shared_main_pos.append(j)
                shared_val_pos.append(i_val)
        if len(shared_main_pos) == 0:
            return (
                self._empty_metrics(current_pheno),
                pd.DataFrame(),
                pd.DataFrame(),
            )

        shared_main_pos_arr = np.array(shared_main_pos, dtype=int)
        shared_val_pos_arr = np.array(shared_val_pos, dtype=int)
        n_shared = len(shared_main_pos_arr)

        d_raw_val_shared = data_val["d_raw"][shared_val_pos_arr]  # (n_shared, n_val)
        mu_main_shared = mu_vec_main[shared_main_pos_arr]
        sd_main_shared = sd_vec_main[shared_main_pos_arr]
        beta_shared = beta_refit_main[shared_main_pos_arr]

        # Standardise val d using MAIN cohort mu/sd (NaN preserved)
        d_val_rescaled = (d_raw_val_shared - mu_main_shared[:, None]) / sd_main_shared[
            :, None
        ]

        # Build obs_masks (per-variant index lists into val sample axis)
        obs_masks_val_shared = [
            np.flatnonzero(~np.isnan(d_val_rescaled[k])) for k in range(n_shared)
        ]
        # NaN -> 0 so that masked sums are correct (mask controls inclusion)
        d_val_rescaled = np.where(np.isnan(d_val_rescaled), 0.0, d_val_rescaled)

        median_n_obs_val = float(
            np.median([len(m) for m in obs_masks_val_shared]) if n_shared else np.nan
        )

        # Per-sample observed-variant burden (over the shared, kept set)
        n_val = data_val["n_samples"]
        n_obs_used = np.zeros(n_val, dtype=int)
        for m in obs_masks_val_shared:
            n_obs_used[m] += 1

        # CSR for fast missing-aware ops
        csr_idx, csr_off = _obs_masks_to_csr(obs_masks_val_shared)

        # ── Genetic component (frozen) ──
        # g_i = sum_{v observed for i} beta_v * d_tilde_{v,i}
        g_frozen = np.zeros(n_val)
        _add_beta_contributions(
            g_frozen, beta_shared, d_val_rescaled, csr_idx, csr_off, n_shared
        )

        y_val = data_val["y"]
        X_unpen_val = data_val["X_unpen"]

        # ── Score under chosen mode ──
        if self.validation_mode == "frozen":
            # Reuse main theta verbatim. The unpenalised design must match
            # exactly between cohorts; otherwise frozen scoring is undefined.
            if X_unpen_val.shape[1] != len(theta_main):
                raise ValueError(
                    f"[validation] frozen mode requires identical covariate "
                    f"structure between cohorts. Phenotype {current_pheno!r}: "
                    f"main has {len(theta_main)} unpenalised columns, "
                    f"validation has {X_unpen_val.shape[1]}. "
                    f"Use --validation_mode recalibrated or align covariates."
                )
            unpen_main_transforms = data_main["unpen_transforms"]
            unpen_blocks_val_raw = data_val["unpen_blocks_raw"]
            if len(unpen_main_transforms) != len(unpen_blocks_val_raw):
                raise ValueError(
                    f"[validation] frozen mode unpenalised block count mismatch "
                    f"for phenotype {current_pheno!r}: main has "
                    f"{len(unpen_main_transforms)} blocks, validation has "
                    f"{len(unpen_blocks_val_raw)}."
                )
            # Re-standardise validation unpenalised columns with the exact
            # MAIN-cohort training rule so theta_main operates on the same
            # feature definition it was learned on. For degenerate main
            # blocks (constant column during training) the validation
            # column is replaced by that same constant rather than passing
            # validation's own (possibly different) constant through.
            blocks_frozen: List[np.ndarray] = [np.ones(n_val)]
            for raw_val_block, (mu_main, sd_main, frozen_const) in zip(
                unpen_blocks_val_raw, unpen_main_transforms
            ):
                if frozen_const is not None:
                    blocks_frozen.append(
                        np.full(n_val, float(frozen_const), dtype=float)
                    )
                    continue
                if sd_main <= 0:
                    raise ValueError(
                        f"[validation] frozen mode encountered non-positive "
                        f"main-cohort sd for an unpenalised block (phenotype "
                        f"{current_pheno!r}). Cannot apply main standardisation."
                    )
                blocks_frozen.append((raw_val_block - mu_main) / sd_main)
            X_unpen_val_frozen = np.column_stack(blocks_frozen)
            yhat = X_unpen_val_frozen @ theta_main + g_frozen
            theta_val: Optional[np.ndarray] = None
        else:
            # Recalibrated: refit unpenalised block on validation, betas frozen
            y_adj = y_val.copy()
            _subtract_beta_contributions(
                y_adj, beta_shared, d_val_rescaled, csr_idx, csr_off, n_shared
            )
            theta_val = ols_fit(y_adj, X_unpen_val)
            yhat = X_unpen_val @ theta_val + g_frozen

        # ── Metrics ──
        residuals = y_val - yhat
        sse = float(residuals @ residuals)
        sst = float(np.sum((y_val - y_val.mean()) ** 2))
        r2_desc = 1.0 - sse / sst if sst > 0 else np.nan

        # Baseline (covariates-only) refit on validation. Used for the
        # audit table's theta_baseline rows and for the per-sample
        # baseline residuals in the residuals CSV. Documents validation-
        # cohort baseline drift.
        theta_base = ols_fit(y_val, X_unpen_val)
        yhat_base = X_unpen_val @ theta_base
        sse_base = float(np.sum((y_val - yhat_base) ** 2))
        r2_baseline_val = 1.0 - sse_base / sst if sst > 0 else np.nan
        # Full-model R² on validation. Identical to r2_descriptive above;
        # exposed under an explicit name for side-by-side comparison with
        # the baseline (covariates-only) R².
        r2_full_val = r2_desc

        rmse = float(np.sqrt(sse / n_val)) if n_val else np.nan
        mae = float(np.mean(np.abs(residuals))) if n_val else np.nan
        mse = sse / n_val if n_val else np.nan

        try:
            pr, pp = pearsonr(y_val, yhat)
        except ValueError:
            pr, pp = np.nan, np.nan
        try:
            sr, sp = spearmanr(y_val, yhat)
        except ValueError:
            sr, sp = np.nan, np.nan

        cal = _calibration_hc3(y_val, yhat)
        # Note: rho_genetic_transfer regresses y on [X_unpen_val, g_frozen]
        # using validation's own X_unpen_val even in frozen mode. It is
        # therefore a "genetic transfer slope after local nuisance
        # adjustment" diagnostic, not a fully frozen statistic. Frozen
        # transferability of theta is assessed via R²/calibration on
        # ``yhat``; rho complements those by isolating the genetic signal.
        rho, rho_p, rho_adj, rho_adj_p = _genetic_transfer_slope(
            y_val, X_unpen_val, g_frozen
        )
        r2_lo, r2_mid, r2_hi = _burden_tercile_r2(y_val, yhat, n_obs_used)

        ci = (
            _paired_bootstrap_ci(
                y_val, yhat, n_boot=self.n_boot_ci, seed=42 + pheno_index_main
            )
            if self.bootstrap_ci
            else {
                "r2_ci_lo": np.nan,
                "r2_ci_hi": np.nan,
                "cal_slope_ci_lo": np.nan,
                "cal_slope_ci_hi": np.nan,
            }
        )

        # n_variants_stable: stable variants among the *shared* main-kept
        # variants (i.e. those that actually contribute to validation
        # prediction). NaN when --validate_with_bootstrap is off.
        if unstable_main is None:
            n_stable_shared: float = float("nan")
        else:
            shared_main_pos_arr = np.asarray(shared_main_pos, dtype=int)
            n_stable_shared = float(int(np.sum(~unstable_main[shared_main_pos_arr])))

        # Permutation null on validation phenotype labels. Genetic
        # component g_frozen and the unpenalised design X_unpen are held
        # fixed; only y is shuffled. Provides empirical p-values for
        # r2_descriptive and rho_genetic_transfer at the exact n_val
        # sample size, which the asymptotic Wald p-values cannot deliver
        # here. Uses a phenotype-deterministic seed so results are
        # reproducible and not synchronised across genes.
        perm_pvals = _permutation_pvals(
            y=y_val,
            yhat_obs=yhat,
            X_unpen=X_unpen_val,
            g_frozen=g_frozen,
            validation_mode=self.validation_mode,
            n_perm=self.n_permutations,
            seed=20240601 + int(pheno_index_main),
            obs_r2=float(r2_desc) if np.isfinite(r2_desc) else float("nan"),
            obs_rho=float(rho)
            if rho is not None and np.isfinite(rho)
            else float("nan"),
        )

        metrics = {
            "phenotype": current_pheno,
            "n_samples_main": int(data_main["n_samples"]),
            "n_samples_val": int(n_val),
            "n_variants_main_kept": int(len(kept_ids_main)),
            "n_variants_shared": int(n_shared),
            "n_variants_stable": (
                int(n_stable_shared) if not np.isnan(n_stable_shared) else np.nan
            ),
            "median_n_obs_val": median_n_obs_val,
            "rmse": rmse,
            "mse": float(mse),
            "mae": mae,
            "r2_descriptive": float(r2_desc) if np.isfinite(r2_desc) else np.nan,
            "r2_baseline_val": float(r2_baseline_val)
            if np.isfinite(r2_baseline_val)
            else np.nan,
            "r2_full_val": float(r2_full_val) if np.isfinite(r2_full_val) else np.nan,
            "pearson_r": float(pr) if pr is not None else np.nan,
            "pearson_pval": float(pp) if pp is not None else np.nan,
            "spearman_rho": float(sr) if sr is not None else np.nan,
            "spearman_pval": float(sp) if sp is not None else np.nan,
            "rho_genetic_transfer": rho,
            "rho_genetic_transfer_pval": rho_p,
            "rho_genetic_adjustment": rho_adj,
            "rho_genetic_adjustment_pval": rho_adj_p,
            "r2_burden_low": r2_lo,
            "r2_burden_mid": r2_mid,
            "r2_burden_high": r2_hi,
            "mean_bias": float(np.mean(residuals)),
            "sd_residual": float(np.std(residuals, ddof=1)) if n_val > 1 else np.nan,
            "lambda_selected": float(lam_selected),
            "validation_mode": self.validation_mode,
            **cal,
            **ci,
            **perm_pvals,
        }

        # Per-sample residual rows. Sample ids come from the prepared dict
        # so they reflect the same mask used to build y_val (non-NaN y, CN,
        # phenotype covariate, and sample covariates).
        sample_ids_val = data_val["sample_ids"]
        if len(sample_ids_val) != n_val:
            raise RuntimeError(
                f"[validation] sample id alignment failed for {current_pheno!r}: "
                f"got {len(sample_ids_val)} ids for {n_val} samples. "
                f"This indicates an upstream masking inconsistency."
            )

        residuals_df = pd.DataFrame(
            {
                "phenotype": current_pheno,
                "sample_id": sample_ids_val,
                "y": y_val,
                "y_hat": yhat,
                "y_hat_base": yhat_base,
                "residual": residuals,
                "residual_base": y_val - yhat_base,
                "n_obs_used": n_obs_used,
            }
        )

        # ── Optional model audit rows ──
        if self.save_model_audit:
            audit_df = _build_audit_rows(
                phenotype=current_pheno,
                kept_ids_main=kept_ids_main,
                shared_main_pos=shared_main_pos,
                beta_refit_main=beta_refit_main,
                mu_vec_main=mu_vec_main,
                sd_vec_main=sd_vec_main,
                theta_main=theta_main,
                theta_val=theta_val,
                theta_base=theta_base,
                unpen_names=data_main.get("unpen_names", []),
                main_unpen_transforms=data_main["unpen_transforms"],
                val_unpen_transforms=data_val["unpen_transforms"],
            )
        else:
            audit_df = pd.DataFrame()

        return metrics, residuals_df, audit_df

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def calculate_validation(
        self, phenotype_id: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run validation across phenotypes on this chromosome.

        Returns:
                - metrics_df: one row per scored phenotype (filtered phenotypes
                    may be omitted, e.g. when
                    ``restrict_to_supported_phenotypes`` is enabled).
        - residuals_df: one row per (phenotype, validation sample) with
          y (actual), y_hat (full model prediction), y_hat_base (baseline
          model prediction), residual and residual_base.
        - audit_df: one row per (phenotype, parameter); empty when
          ``save_model_audit`` is False. Includes full model theta and
          baseline model theta_baseline coefficients for comparison.
        """
        start = time()
        if phenotype_id is not None:
            if phenotype_id not in self.fm_main.quan.index:
                raise ValueError(
                    f"Phenotype '{phenotype_id}' not found on {self.chromosome} in main cohort."
                )
            pheno_indices = [int(self.fm_main.quan.index.get_loc(phenotype_id))]
            desc = f"Validating {phenotype_id}"
        else:
            pheno_indices = list(range(self.fm_main.quan.shape[0]))
            desc = "Validating"

        n_phenos = len(pheno_indices)

        if self.num_cores == 1 or n_phenos == 1:
            results = [
                self._process_phenotype(i) for i in tqdm(pheno_indices, desc=desc)
            ]
        else:
            with Pool(processes=self.num_cores) as pool:
                results = list(
                    tqdm(
                        pool.imap(self._process_phenotype, pheno_indices),
                        total=n_phenos,
                        desc=desc,
                    )
                )

        elapsed = (time() - start) / 60
        print(f"Validation completed in {elapsed:.1f} min")

        metrics_rows = [r[0] for r in results if r[0] is not None]
        metrics_df = (
            pd.DataFrame(metrics_rows)
            if metrics_rows
            else pd.DataFrame(columns=list(self._empty_metrics("").keys()))
        )
        non_empty = [r[1] for r in results if not r[1].empty]
        residuals_df = (
            pd.concat(non_empty, ignore_index=True)
            if non_empty
            else pd.DataFrame(
                columns=[
                    "phenotype",
                    "sample_id",
                    "y",
                    "y_hat",
                    "residual",
                    "n_obs_used",
                ]
            )
        )
        non_empty_audit = [r[2] for r in results if not r[2].empty]
        audit_df = (
            pd.concat(non_empty_audit, ignore_index=True)
            if non_empty_audit
            else pd.DataFrame(
                columns=[
                    "phenotype",
                    "block",
                    "name",
                    "value_main",
                    "value_val",
                    "shared_with_val",
                ]
            )
        )
        return metrics_df, residuals_df, audit_df
