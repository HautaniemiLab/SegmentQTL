from glob import glob
from os import path

import pandas as pd
from statsmodels.stats.multitest import multipletests


def combine_chromosome(outdir: str):
    """
    Combine all csv files from the given directory.

    Parameters:
    - outdir: Directory to which the mapping results have been saved.

    Returns:
    - combined_df: Dataframe with data from all csv files from the folder.
    """
    csv_files = glob(path.join(outdir, "*.csv"))

    dfs = []

    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.dropna()

    return combined_df


def fdr(outdir: str, threshold: float):
    """
    Perform Benjamini Hochberg false discovery rate correction to mapping results.

    Parameters:
    - outdir: Directory to which the mapping results have been saved.
    - threshold: Cutoff value for fdr correction.

    Returns:
    - full_res: Dataframe with all mapping results including a column for fdr corrected p-values.
    """
    full_res = combine_chromosome(outdir)
    perm_pvals = full_res["p_adj"]
    _, bh_p_values, _, _ = multipletests(perm_pvals, method="fdr_bh", alpha=threshold)
    full_res["fdr"] = bh_p_values
    return full_res
