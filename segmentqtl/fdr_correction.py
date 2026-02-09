from glob import glob
from os import path

import pandas as pd
from multipy.fdr import qvalue


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

    # Use p_adj (permutation-adjusted) if available, otherwise fall back to nominal_p
    if "p_adj" in combined_df.columns and combined_df["p_adj"].notna().any():
        combined_df["pval_for_fdr"] = combined_df["p_adj"]
    elif "nominal_p" in combined_df.columns:
        combined_df["pval_for_fdr"] = combined_df["nominal_p"]
    else:
        raise ValueError(
            "No suitable p-value column found. Expected 'p_adj' or 'nominal_p'."
        )

    combined_df = combined_df.dropna(subset=["pval_for_fdr"])

    return combined_df


def fdr(outdir: str):
    """
    Perform Storey-Tibshirani q-value false discovery rate correction to mapping results.

    Parameters:
    - outdir: Directory to which the mapping results have been saved.

    Returns:
    - full_res: Dataframe with all mapping results including a column for q-values.
    """
    full_res = combine_chromosome(outdir)
    perm_pvals = full_res["pval_for_fdr"].values
    _, qvals = qvalue(perm_pvals, verbose=False)
    full_res["fdr"] = qvals
    full_res = full_res.drop(columns=["pval_for_fdr"])
    return full_res
