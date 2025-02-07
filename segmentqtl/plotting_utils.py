import os

import pandas as pd
import plotly.graph_objects as go
from scipy import stats


def box_and_whisker(df: pd.DataFrame, gene_name: str, variant: str, output_folder: str):
    """
    Create a box-and-whisker plot with significance bars and Kruskal-Wallis test for grouped data using plotly.

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

    # Create the box plot using plotly
    fig = go.Figure()

    for i, data in enumerate(grouped_data):
        fig.add_trace(
            go.Box(y=data, boxmean="sd", name=xticklabels[i], marker_color="lightblue")
        )

    fig.update_layout(
        title=gene_name,
        yaxis_title="Expression (log(TMM + 1))",
        xaxis_title="Dosage",
        showlegend=False,
    )

    # Perform Kruskal-Wallis test
    _, kruskal_p = stats.kruskal(*grouped_data)
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        text=f"Kruskal-Wallis p = {kruskal_p:.3g}",
        showarrow=False,
        font=dict(size=12),
        xref="paper",
        yref="paper",
    )

    # Check pairwise significance using Mann-Whitney U test
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

    # Annotate significance bars
    yrange = max([max(data) for data in grouped_data]) - min(
        [min(data) for data in grouped_data]
    )
    for i, significant_combination in enumerate(significant_combinations):
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        level = len(significant_combinations) - i
        bar_height = yrange * (0.08 + 0.05 * level)
        bar_tips = bar_height - (yrange * (0.0001 + 0.5))

        fig.add_trace(
            go.Scatter(
                x=[x1, x1, x2, x2],
                y=[bar_tips, bar_height, bar_height, bar_tips],
                mode="lines",
                line=dict(color="black"),
                showlegend=False,
            )
        )
        p = significant_combination[1]
        fig.add_annotation(
            x=(x1 + x2) * 0.5,
            y=bar_height + (yrange * 0.01),
            text=f"p = {p:.3g}",
            showarrow=False,
            font=dict(size=7),
        )

    # Annotate sample size below each box
    for i, dataset in enumerate(grouped_data):
        sample_size = len(dataset)
        fig.add_annotation(
            x=i,
            y=-0.1,
            text=f"n = {sample_size}",
            showarrow=False,
            font=dict(size=8),
            xref="x",
            yref="y",
        )

    # Save the plot to a file
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = os.path.join(output_folder, f"{gene_name}_{variant}.html")
    fig.write_html(filename)
