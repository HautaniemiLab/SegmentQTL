from os import makedirs, path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def box_and_whisker(df: pd.DataFrame, gene_name: str, variant: str, output_folder: str):
    """
    Create a box-and-whisker plot with significance bars and Kruskal-Wallis test for grouped data.

    Parameters:
    - df: Dataframe containing 'GEX' and 'cur_genotypes' columns.
    - gene_name: Name of the gene that is used for the plot title and file name.
    - output_folder: Path to the folder where the plot should be saved.
    """
    bins = [0, 0.34, 0.67, 1]
    df["Dosage"] = pd.cut(df["cur_genotypes"], bins=bins, include_lowest=True)

    grouped_data = [
        df.loc[df["Dosage"] == category, "GEX"]
        for category in df["Dosage"].cat.categories
    ]

    xticklabels = [str(category) for category in df["Dosage"].cat.categories]
    ax = plt.axes()
    bp = ax.boxplot(grouped_data, widths=0.6, patch_artist=False)

    ax.set_title(gene_name, fontsize=14)
    ax.set_ylabel("Expression (log(TMM + 1))")
    ax.set_xticklabels(xticklabels)
    ax.tick_params(axis="x", which="major", length=0)

    plt.setp(bp["medians"], color="k")

    _, kruskal_p = stats.kruskal(*grouped_data)
    ax.text(
        0,
        -0.1,
        f"Kruskal-Wallis p = {kruskal_p:.3g}",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        ha="left",
    )

    # Check pairwise significance
    significant_combinations = []
    ls = list(range(1, len(grouped_data) + 1))
    combinations = [
        (ls[x], ls[x + y]) for y in reversed(ls) for x in range(len(ls) - y)
    ]

    for c in combinations:
        data1 = grouped_data[c[0] - 1]
        data2 = grouped_data[c[1] - 1]
        _, p = stats.mannwhitneyu(data1, data2, alternative="two-sided")
        significant_combinations.append([c, p])

    bottom, top = ax.get_ylim()
    all_data = [item for sublist in grouped_data for item in sublist]
    max_data_point = max(all_data)
    min_data_point = min(all_data)
    ax.set_ylim(min_data_point - 0.1, max_data_point + 1)

    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]

    for i, significant_combination in enumerate(significant_combinations):
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        level = len(significant_combinations) - i
        spacing_factor = 0.05
        bar_height = (yrange * (0.08 + spacing_factor * level)) + top
        bar_tips = bar_height - (yrange * (0.0001 + spacing_factor * 0.5))

        plt.plot(
            [x1, x1, x2, x2], [bar_tips, bar_height, bar_height, bar_tips], lw=1, c="k"
        )
        p = significant_combination[1]
        plt.text(
            (x1 + x2) * 0.5,
            bar_height + (yrange * 0.01),
            f"p = {p:.3g}",
            ha="center",
            fontsize=7,
        )

    # Annotate sample size below each box
    for i, dataset in enumerate(grouped_data):
        sample_size = len(dataset)
        ax.text(i + 1, bottom, rf"n = {sample_size}", ha="center", size="x-small")

    if not path.exists(output_folder):
        makedirs(output_folder)

    plt.autoscale(axis="y")
    filename = path.join(output_folder, f"{gene_name}_{variant}.png")
    plt.savefig(filename, dpi=300)
    plt.clf()
