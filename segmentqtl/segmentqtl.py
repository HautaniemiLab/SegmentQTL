from argparse import ArgumentParser
from os import makedirs, path

from cis import Cis
from fdr_correction import fdr


def main():
    parser = ArgumentParser(description="Perform QTL cis-mapping")
    parser.add_argument(
        "--mode",
        type=str,
        default="perm",
        help="Nominal (nominal) or permutation (perm) mapping or fdr correction (fdr)",
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
        "--perm_covariate",
        type=str,
        default=None,
        help="Path to phenotype-level covariate (e.g., gene-level CN) used ONLY for "
        "Freedman-Lane residualization in permutation mode. Not included in nominal model. "
        "If provided, removes CN-driven structure before permuting for proper exchangeability.",
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
        "--fdr_out",
        type=str,
        help="File path to which fdr corrected full results are saved to. Must be a csv file.",
    )

    parser.add_argument(
        "--record_aic",
        action="store_true",
        help="Record AIC scores for associations.",
    )

    args = parser.parse_args()

    out_dir = args.out_dir
    if not out_dir.endswith("/"):
        out_dir = out_dir + "/"

    mode = args.mode
    if mode == "nominal" or mode == "perm":
        chromosome = args.chromosome
        if not chromosome.startswith("chr"):
            chromosome = "chr" + chromosome

        phenotype_covariate_file = args.phenotype_covariate
        perm_covariate_file = args.perm_covariate
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

        # Validate: permutation mode with all_variants is not supported
        if mode == "perm" and all_variants_mode:
            raise ValueError(
                "--mode perm is not compatible with --all_variants. "
                "Permutation testing computes gene-level scan p-values (best variant in window). "
                "Use --mode nominal for per-variant association testing without permutation adjustment."
            )

        # Validate: permutation mode requires perm_covariate for proper FL residualization
        if mode == "perm" and perm_covariate_file is None:
            raise ValueError("--mode perm requires --perm_covariate to be specified.")

        # Perform cis-mapping, nominal or with permutations
        mapping = Cis(
            chromosome,
            mode,
            phenotype_covariate_file,
            perm_covariate_file,
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
        ).calculate_associations()

        mapping["chr"] = chromosome

        if not path.exists(out_dir):
            makedirs(out_dir)

        if isinstance(all_variants_mode, str):
            mapping.to_csv(
                f"{out_dir}{all_variants_mode}_{mode}_{chromosome}.csv", index=False
            )
        elif mode == "nominal":
            mapping.to_csv(f"{out_dir}{mode}_{chromosome}.csv", index=False)
        else:
            mapping.to_csv(
                f"{out_dir}{mode}_{chromosome}_{num_permutations}.csv", index=False
            )

    elif mode == "fdr":
        out_path = args.fdr_out
        fdr_corrected_res = fdr(out_dir)
        fdr_corrected_res.to_csv(out_path, index=False)
    else:
        print(f"Invalid mode: {mode}, please select nominal, perm, or fdr.")
