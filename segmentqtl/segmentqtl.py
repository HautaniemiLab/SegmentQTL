#!/usr/bin/env python

import argparse
import os

from cis import Cis


def main():
    parser = argparse.ArgumentParser(description="Perform QTL cis-mapping")
    parser.add_argument(
        "--mode",
        type=str,
        default="perm",
        help="Nominal (nominal) or permutation (perm) mapping",
    )
    parser.add_argument(
        "--chromosome",
        type=str,
        default="21",
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
        action="store_true",
        help="Test all applicable variants per gene instead of only the best correlated one",
    )
    parser.add_argument(
        "--num_permutations",
        type=int,
        default=10,
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
        default="../results/",
        help="Directory where intermediate results are saved",
    )

    args = parser.parse_args()

    chromosome = args.chromosome
    if not chromosome.startswith("chr"):
        chromosome = "chr" + chromosome

    mode = args.mode
    copynumber_file = args.copynumber
    quantifications_file = args.quantifications
    covariates_file = args.covariates
    segmentation_file = args.segmentation
    genotypes_file = f"{args.genotypes}/{chromosome}_with_id.csv"
    all_variants_mode = args.all_variants
    num_permutations = args.num_permutations
    num_cores = args.num_cores
    out_dir = args.out_dir

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
    ).calculate_associations()

    mapping["chr"] = chromosome

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save mapping DataFrame to CSV
    mapping.to_csv(f"{out_dir}{mode}_{chromosome}_{num_permutations}.csv", index=False)


if __name__ == "__main__":
    main()
