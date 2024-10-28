#!/usr/bin/env python

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
        default="10",
        help="Chromosome number or X with or without chr prefix",
    )
    parser.add_argument(
        "--copynumber",
        type=str,
        default="../segmentQTL_inputs/copynumber.csv",
        help="Path to copynumber CSV file",
    )
    parser.add_argument(
        "--quantifications",
        type=str,
        default="../segmentQTL_inputs/quantifications.csv",
        help="Path to quantifications CSV file",
    )
    parser.add_argument(
        "--covariates",
        type=str,
        default="../segmentQTL_inputs/covariates.csv",
        help="Path to covariates CSV file",
    )
    parser.add_argument(
        "--segmentation",
        type=str,
        default="../segmentQTL_inputs/purple.csv",
        help="Path to file with segmentation data",
    )
    parser.add_argument(
        "--genotypes",
        type=str,
        default="../segmentQTL_inputs/genotypes",
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
        "--num_permutations",
        type=int,
        default=8000,
        help="Number of permutations to be run on each phenotype",
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
        default="/results/",
        help="Directory where intermediate results are saved",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="../mod_fdr_corrected_res.csv",
        help="File path to which fdr corrected full results are saved to. Must be a csv file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Threshold value for fdr cutoff.",
    )
    parser.add_argument(
        "--plot_threshold",
        type=float,
        default=-1,
        help="Threshold p-value for creating a plot.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="plots/",
        help="Directory for plots.",
    )

    args = parser.parse_args()

    out_dir = args.out_dir
    if not out_dir.endswith("/"):
        out_dir = out_dir + "/"

    plot_dir = args.plot_dir
    if not plot_dir.endswith("/"):
        plot_dir = plot_dir + "/"

    mode = args.mode
    if mode == "nominal" or mode == "perm":
        chromosome = args.chromosome
        if not chromosome.startswith("chr"):
            chromosome = "chr" + chromosome

        copynumber_file = args.copynumber
        quantifications_file = args.quantifications
        covariates_file = args.covariates
        segmentation_file = args.segmentation
        genotypes_file = f"{args.genotypes}/{chromosome}.csv"
        all_variants_mode = args.all_variants
        num_permutations = args.num_permutations
        num_cores = args.num_cores
        plot_threshold = args.plot_threshold

        # Perform cis-mapping, nominal or with permutations
        mapping = Cis(
            chromosome,
            mode,
            copynumber_file,
            quantifications_file,
            covariates_file,
            segmentation_file,
            genotypes_file,
            all_variants_mode,
            num_permutations,
            num_cores,
            plot_threshold,
            plot_dir,
        ).calculate_associations()

        mapping["chr"] = chromosome

        if not path.exists(out_dir):
            makedirs(out_dir)

        if mode == "nominal":
            mapping.to_csv(f"{out_dir}{mode}_{chromosome}.csv", index=False)
        else:
            mapping.to_csv(
                f"{out_dir}{mode}_{chromosome}_{num_permutations}.csv", index=False
            )

    elif mode == "fdr":
        out_path = args.out
        threshold = args.threshold
        fdr_corrected_res = fdr(out_dir, threshold)
        fdr_corrected_res.to_csv(out_path, index=False)
    else:
        print(f"Invalid mode: {mode}, please select nominal, perm, or fdr.")


if __name__ == "__main__":
    main()
