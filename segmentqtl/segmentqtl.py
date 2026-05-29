from argparse import ArgumentParser
from os import makedirs, path

from cis import Cis
from finemapping import Finemapping
from validation import Validation


def main():
    parser = ArgumentParser(description="Perform QTL cis-mapping")
    parser.add_argument(
        "--mode",
        type=str,
        default="validate",
        help="Nominal (nominal), permutation (perm), finemapping (finemap), or validation (validate)",
    )
    parser.add_argument(
        "--chromosome",
        type=str,
        help="Chromosome number or X with or without chr prefix",
    )
    parser.add_argument(
        "--phenotype_covariate",
        type=str,
        default=None,
        help="Path to phenotype-level covariate CSV file. Optional.",
    )
    parser.add_argument(
        "--copynumber",
        type=str,
        default=None,
        help="Path to phenotype-level copy-number covariate CSV (e.g. CNlr). "
        "In perm mode: used for Freedman-Lane residualization (removes CN-driven "
        "structure before permuting for proper exchangeability). "
        "In finemap mode: included as an unpenalised predictor in the Elastic Net model.",
    )
    parser.add_argument(
        "--quantifications",
        type=str,
        default=None,
        help="Path to quantifications CSV file",
    )
    parser.add_argument(
        "--covariates",
        default=None,
        type=str,
        help="Path to covariates CSV file",
    )
    parser.add_argument(
        "--segmentation",
        default=None,
        type=str,
        help="Path to file with segmentation data",
    )
    parser.add_argument(
        "--genotypes",
        default=None,
        type=str,
        help="Path to genotypes directory",
    )
    parser.add_argument(
        "--all_variants",
        nargs="?",
        const=True,
        default=False,
        help="Test all applicable variants for a given gene. Provide a gene ID or use without a value to process all genes.",
    )
    parser.add_argument(
        "--perm_method",
        type=str,
        default="beta",
        help="Method used in permutation. Options: beta and direct.",
    )
    parser.add_argument(
        "--num_permutations",
        type=int,
        default=5000,
        help="Number of permutations to be run on each phenotype",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1000000,
        help="Window size",
    )
    parser.add_argument(
        "--num_cores",
        type=int,
        default=1,
        help="Number of cores to be used in the computation",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory where intermediate results are saved",
    )
    parser.add_argument(
        "--record_aic",
        action="store_true",
        help="Record AIC scores for associations.",
    )

    parser.add_argument(
        "--neg_control",
        action="store_true",
        help="Run trans negative control mode. For each gene on chromosome c, tests "
        "variants from chromosome c+1 (wrapping) instead of cis variants. "
        "Segment consistency filtering is not applied (not meaningful across chromosomes). "
        "Results should show uniform p-values if the pipeline is well-calibrated.",
    )
    parser.add_argument(
        "--neg_control_max_variants",
        type=int,
        default=2000,
        help="Maximum number of trans negative-control variants to test per gene (subsampled randomly). Default: 2000.",
    )

    # ── Finemapping-specific arguments ──
    parser.add_argument(
        "--alpha_en",
        type=float,
        default=0.5,
        help="Elastic Net mixing parameter (1=Lasso, 0=Ridge). Default: 0.5.",
    )
    parser.add_argument(
        "--coverage_tau",
        type=float,
        default=0.6,
        help="Minimum fraction of samples observed for a variant. Default: 0.6.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=200,
        help="Number of stability-selection bootstrap resamples. Default: 200.",
    )
    parser.add_argument(
        "--subsample_frac",
        type=float,
        default=0.8,
        help="Fraction of samples per bootstrap resample. Default: 0.8.",
    )
    parser.add_argument(
        "--n_lambda",
        type=int,
        default=30,
        help="Number of lambda grid points for CV-based selection. Default: 30.",
    )
    parser.add_argument(
        "--lambda_ratio",
        type=float,
        default=0.01,
        help="Lower-bound ratio lam_min/lam_max for lambda grid. Default: 0.01.",
    )
    parser.add_argument(
        "--cv_tau",
        type=float,
        default=0.8,
        help="Range-based CV tolerance: fraction of improvement over null sacrificed for sparsity (0.8 = keep lambdas within 80%% of range). Default: 0.8.",
    )
    parser.add_argument(
        "--min_obs_boot",
        type=int,
        default=20,
        help="Minimum observed entries per variant inside each bootstrap subsample. Default: 20.",
    )
    parser.add_argument(
        "--phenotype_id",
        type=str,
        # default="ENSG00000086232",
        help="Phenotype ID to process (finemap or validate mode). If not provided, all phenotypes on the chromosome are processed.",
    )
    parser.add_argument(
        "--compute_r2",
        action="store_true",
        help="(finemap mode) Compute R² for baseline vs full model and include in output.",
    )
    parser.add_argument(
        "--r2_stability_threshold",
        type=float,
        default=0.6,
        help="(finemap mode) Minimum stability score for variant selection in R² computation. Default: 0.6.",
    )

    # ── Validation-mode arguments ──
    parser.add_argument(
        "--val_quantifications",
        type=str,
        default=None,
        help="(validate mode) Path to validation cohort quantifications CSV.",
    )
    parser.add_argument(
        "--val_covariates",
        type=str,
        default=None,
        help="(validate mode) Path to validation cohort sample covariates CSV.",
    )
    parser.add_argument(
        "--val_segmentation",
        type=str,
        default=None,
        help="(validate mode) Path to validation cohort segmentation CSV.",
    )
    parser.add_argument(
        "--val_genotypes",
        type=str,
        default=None,
        help="(validate mode) Path to validation cohort genotypes directory.",
    )
    parser.add_argument(
        "--val_copynumber",
        type=str,
        default=None,
        help="(validate mode) Path to validation cohort phenotype-level CN covariate CSV.",
    )
    parser.add_argument(
        "--val_phenotype_covariate",
        type=str,
        default=None,
        help="(validate mode) Path to validation cohort additional phenotype-level covariate CSV.",
    )
    parser.add_argument(
        "--validation_mode",
        type=str,
        default="recalibrated",
        choices=["recalibrated", "frozen"],
        help="(validate mode) Scoring strategy. 'recalibrated' (default) freezes variant betas + main-cohort preprocessing and refits the unpenalised block on validation. 'frozen' applies the discovery model exactly as learned (also reuses main theta).",
    )
    parser.add_argument(
        "--validate_with_bootstrap",
        action="store_true",
        help="(validate mode) Run stability-selection bootstraps on the main cohort. Default off (uses CV-selected refit betas).",
    )
    parser.add_argument(
        "--validation_stability_threshold",
        type=float,
        default=0.6,
        help="(validate mode) When --validate_with_bootstrap is set, mask main-cohort betas to zero for variants with stability score below this threshold. Default: 0.6.",
    )
    parser.add_argument(
        "--save_model_audit",
        action="store_true",
        help="(validate mode) Write a long-format audit CSV of the final per-phenotype model (genetic betas, variant mu/sd, unpenalised theta and mu/sd) for both main and validation cohorts. Output: validate_model_<chr>.csv",
    )
    parser.add_argument(
        "--bootstrap_ci",
        action="store_true",
        help="(validate mode) Compute paired bootstrap 95%% CIs for R² and calibration slope.",
    )
    parser.add_argument(
        "--n_boot_ci",
        type=int,
        default=1000,
        help="(validate mode) Number of bootstrap resamples when --bootstrap_ci is set. Default: 1000.",
    )
    parser.add_argument(
        "--finemap_results_dir",
        type=str,
        default=None,
        help="(validate mode) Directory containing pre-computed finemap_<chr>.csv from a prior `--mode finemap` run. When provided, validation reuses the main-cohort betas, mu, sd, lambda, and stability scores from that file instead of refitting the Elastic Net. Hard-errors if variant order or mu/sd disagree with the current main-cohort inputs.",
    )
    parser.add_argument(
        "--restrict_to_supported_phenotypes",
        action="store_true",
        help="(validate mode) Restrict scoring to discovery-supported phenotypes (phenotypes with at least one variant passing --support_definition). Requires --finemap_results_dir. Unsupported phenotypes are skipped and do not appear in validation outputs.",
    )
    parser.add_argument(
        "--support_definition",
        type=str,
        choices=["stability", "selected"],
        default="stability",
        help="(validate mode) Discovery support definition. 'stability' (default): phenotype has any variant with stability_score >= --support_min_stability. 'selected': phenotype has any variant with non-zero beta at lambda_selected. Only used when --restrict_to_supported_phenotypes is set.",
    )
    parser.add_argument(
        "--support_min_stability",
        type=float,
        default=0.6,
        help="(validate mode) Stability-score threshold for support_definition='stability'. Default: 0.6.",
    )
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=0,
        help="(validate mode) Permutation-null replicates. When >0, validation phenotype labels are shuffled K times per gene (genetic component held fixed) and empirical p-values are reported in r2_perm_pval, rho_perm_pval. Use 1000 for FDR-quality nulls; 0 disables (default).",
    )

    args = parser.parse_args()

    out_dir = args.out_dir
    if not out_dir.endswith("/"):
        out_dir = out_dir + "/"

    mode = args.mode
    if mode in ("nominal", "perm", "finemap"):
        chromosome = args.chromosome
        if not chromosome.startswith("chr"):
            chromosome = "chr" + chromosome

        phenotype_covariate_file = args.phenotype_covariate
        copynumber_file = args.copynumber
        quantifications_file = args.quantifications
        covariates_file = args.covariates
        segmentation_file = args.segmentation
        genotype_alt_file = f"{args.genotypes}/{chromosome}_ALTlr.csv"
        genotype_ref_file = f"{args.genotypes}/{chromosome}_REFlr.csv"
        all_variants_mode = args.all_variants
        perm_method = args.perm_method
        num_permutations = args.num_permutations
        window = args.window
        num_cores = args.num_cores
        record_aic = args.record_aic
        neg_control = args.neg_control
        neg_control_max_variants = args.neg_control_max_variants

        # ── Finemapping mode ──
        if mode == "finemap":
            finemapper = Finemapping(
                chromosome,
                quantifications_file,
                covariates_file,
                segmentation_file,
                genotype_alt_file,
                genotype_ref_file,
                copynumber_file,
                phenotype_covariate_file,
                window,
                num_cores,
                alpha_en=args.alpha_en,
                coverage_tau=args.coverage_tau,
                n_bootstrap=args.n_bootstrap,
                subsample_frac=args.subsample_frac,
                n_lambda=args.n_lambda,
                lambda_ratio=args.lambda_ratio,
                cv_tau=args.cv_tau,
                min_obs_boot=args.min_obs_boot,
                compute_r2=args.compute_r2,
                r2_stability_threshold=args.r2_stability_threshold,
            )
            mapping = finemapper.calculate_finemapping(phenotype_id=args.phenotype_id)

            mapping["chr"] = chromosome

            if not path.exists(out_dir):
                makedirs(out_dir)

            if args.phenotype_id is not None:
                fname = f"{out_dir}finemap_{chromosome}_{args.phenotype_id}.csv"
                diag_fname = f"{out_dir}finemap_bootstrap_nonzero_{chromosome}_{args.phenotype_id}.csv"
                r2_fname = f"{out_dir}finemap_r2_{chromosome}_{args.phenotype_id}.csv"
            else:
                fname = f"{out_dir}finemap_{chromosome}.csv"
                diag_fname = f"{out_dir}finemap_bootstrap_nonzero_{chromosome}.csv"
                r2_fname = f"{out_dir}finemap_r2_{chromosome}.csv"

            mapping.to_csv(fname, index=False)
            finemapper.bootstrap_nonzero_diagnostics.to_csv(diag_fname, index=False)
            if not finemapper.r2_results.empty:
                finemapper.r2_results.to_csv(r2_fname, index=False)
            return

        # Validate: permutation mode with all_variants is not supported
        if mode == "perm" and all_variants_mode:
            raise ValueError(
                "--mode perm is not compatible with --all_variants. "
                "Permutation testing computes gene-level scan p-values (best variant in window). "
                "Use --mode nominal for per-variant association testing without permutation adjustment."
            )

        # Validate: permutation mode requires copynumber for proper FL residualization
        if mode == "perm" and copynumber_file is None:
            raise ValueError("--mode perm requires --copynumber to be specified.")

        # Validate: neg_control with all_variants is not supported in perm mode
        if neg_control and all_variants_mode and mode == "perm":
            raise ValueError(
                "--neg_control with --all_variants is only supported in nominal mode."
            )

        # Perform cis-mapping, nominal or with permutations
        mapping = Cis(
            chromosome,
            mode,
            phenotype_covariate_file,
            copynumber_file,
            quantifications_file,
            covariates_file,
            segmentation_file,
            genotype_alt_file,
            genotype_ref_file,
            all_variants_mode,
            perm_method,
            num_permutations,
            window,
            num_cores,
            record_aic,
            neg_control=neg_control,
            neg_control_max_variants=neg_control_max_variants,
            genotypes_dir=args.genotypes,
        ).calculate_associations()

        mapping["chr"] = chromosome

        if not path.exists(out_dir):
            makedirs(out_dir)

        # Build output filename
        if neg_control:
            prefix = "negctrl_trans"
        else:
            prefix = ""

        if isinstance(all_variants_mode, str):
            fname = f"{out_dir}{prefix + '_' if prefix else ''}{all_variants_mode}_{mode}_{chromosome}.csv"
        elif mode == "nominal":
            fname = f"{out_dir}{prefix + '_' if prefix else ''}{mode}_{chromosome}.csv"
        else:
            fname = f"{out_dir}{prefix + '_' if prefix else ''}{mode}_{chromosome}_{num_permutations}.csv"

        mapping.to_csv(fname, index=False)

    elif mode == "validate":
        chromosome = args.chromosome
        if not chromosome.startswith("chr"):
            chromosome = "chr" + chromosome

        # Validate required validation-cohort args
        required = {
            "--val_quantifications": args.val_quantifications,
            "--val_segmentation": args.val_segmentation,
            "--val_genotypes": args.val_genotypes,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(
                f"--mode validate requires {', '.join(missing)} to be specified."
            )

        genotype_alt_main = f"{args.genotypes}/{chromosome}_ALTlr.csv"
        genotype_ref_main = f"{args.genotypes}/{chromosome}_REFlr.csv"
        genotype_alt_val = f"{args.val_genotypes}/{chromosome}_ALTlr.csv"
        genotype_ref_val = f"{args.val_genotypes}/{chromosome}_REFlr.csv"

        validator = Validation(
            chromosome,
            args.quantifications,
            args.covariates,
            args.segmentation,
            genotype_alt_main,
            genotype_ref_main,
            args.copynumber,
            args.phenotype_covariate,
            args.val_quantifications,
            args.val_covariates,
            args.val_segmentation,
            genotype_alt_val,
            genotype_ref_val,
            args.val_copynumber,
            args.val_phenotype_covariate,
            args.window,
            args.num_cores,
            alpha_en=args.alpha_en,
            coverage_tau=args.coverage_tau,
            n_bootstrap=args.n_bootstrap,
            subsample_frac=args.subsample_frac,
            n_lambda=args.n_lambda,
            lambda_ratio=args.lambda_ratio,
            cv_tau=args.cv_tau,
            min_obs_boot=args.min_obs_boot,
            validation_mode=args.validation_mode,
            validate_with_bootstrap=args.validate_with_bootstrap,
            validation_stability_threshold=args.validation_stability_threshold,
            bootstrap_ci=args.bootstrap_ci,
            n_boot_ci=args.n_boot_ci,
            save_model_audit=args.save_model_audit,
            finemap_results_dir=args.finemap_results_dir,
            restrict_to_supported_phenotypes=args.restrict_to_supported_phenotypes,
            support_definition=args.support_definition,
            support_min_stability=args.support_min_stability,
            n_permutations=args.n_permutations,
        )
        metrics_df, residuals_df, audit_df = validator.calculate_validation(
            phenotype_id=args.phenotype_id
        )
        metrics_df["chr"] = chromosome
        residuals_df["chr"] = chromosome
        if not audit_df.empty:
            audit_df["chr"] = chromosome

        if not path.exists(out_dir):
            makedirs(out_dir)

        if args.phenotype_id is not None:
            fname = f"{out_dir}validate_{chromosome}_{args.phenotype_id}.csv"
            rname = f"{out_dir}validate_residuals_{chromosome}_{args.phenotype_id}.csv"
            mname = f"{out_dir}validate_model_{chromosome}_{args.phenotype_id}.csv"
        else:
            fname = f"{out_dir}validate_{chromosome}.csv"
            rname = f"{out_dir}validate_residuals_{chromosome}.csv"
            mname = f"{out_dir}validate_model_{chromosome}.csv"

        metrics_df.to_csv(fname, index=False)
        residuals_df.to_csv(rname, index=False)
        if args.save_model_audit:
            audit_df.to_csv(mname, index=False)
        return

    else:
        print(
            f"Invalid mode: {mode}, please select nominal, perm, finemap, or validate."
        )
