import glob
import os

import pandas as pd
from statsmodels.stats.multitest import multipletests


def combine_chromosome(outdir):
    csv_files = glob.glob(os.path.join(outdir, "*.csv"))

    dfs = []

    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.dropna()

    combined_df.to_csv(f"{outdir}full_result.csv", index=False)

    return combined_df


def fdr(outdir, threshold):
    full_res = combine_chromosome(outdir)
    perm_pvals = full_res["p_adj"]
    _, bh_p_values, _, _ = multipletests(perm_pvals, method="fdr_bh", alpha=threshold)
    full_res["fdr"] = bh_p_values
    return full_res
