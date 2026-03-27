from argparse import ArgumentParser
from os import makedirs, path

from cis import Cis
from finemapping import Finemapping


def main():
    parser = ArgumentParser(description="Perform QTL cis-mapping")
    parser.add_argument(
        "--mode",
        type=str,
        default="perm",
        help="Nominal (nominal), permutation (perm), or finemapping (finemap)",
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
        help="Path to quantifications CSV file",
    )
    parser.add_argument(
        "--covariates",
        type=str,
        help="Path to covariates CSV file",
    )
    parser.add_argument(
        "--segmentation",
        type=str,
        help="Path to file with segmentation data",
    )
    parser.add_argument(
        "--genotypes",
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
        default=500000,
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
        default=False,
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
        "--stability_threshold",
        type=float,
        default=0.6,
        help="Selection probability threshold for stable variants. Default: 0.6.",
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
        default=None,
        help="Phenotype ID to finemap (finemap mode only). If not provided, all phenotypes on the chromosome are finemapped.",
    )

    args = parser.parse_args()

    out_dir = args.out_dir
    if not out_dir.endswith("/"):
        out_dir = out_dir + "/"

    mode = args.mode
    if mode == "nominal" or mode == "perm" or mode == "finemap":
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
                stability_threshold=args.stability_threshold,
                n_lambda=args.n_lambda,
                lambda_ratio=args.lambda_ratio,
                cv_tau=args.cv_tau,
                min_obs_boot=args.min_obs_boot,
            )
            mapping = finemapper.calculate_finemapping(phenotype_id=args.phenotype_id)

            mapping["chr"] = chromosome

            if not path.exists(out_dir):
                makedirs(out_dir)

            if args.phenotype_id is not None:
                fname = f"{out_dir}finemap_{chromosome}_{args.phenotype_id}.csv"
                diag_fname = f"{out_dir}finemap_bootstrap_nonzero_{chromosome}_{args.phenotype_id}.csv"
            else:
                fname = f"{out_dir}finemap_{chromosome}.csv"
                diag_fname = f"{out_dir}finemap_bootstrap_nonzero_{chromosome}.csv"

            mapping.to_csv(fname, index=False)
            finemapper.bootstrap_nonzero_diagnostics.to_csv(diag_fname, index=False)
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

    else:
        print(f"Invalid mode: {mode}, please select nominal, perm, or finemap.")
