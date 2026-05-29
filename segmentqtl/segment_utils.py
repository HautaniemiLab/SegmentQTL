from typing import Tuple

import numpy as np
import pandas as pd


def phenotype_window_bounds(
    quan: pd.DataFrame,
    index: int,
    window: int,
) -> Tuple[int, int]:
    """Return cis-window start/end for a phenotype row index."""
    start = int(quan["start"].iloc[index] - window)
    end = int(quan["end"].iloc[index] + window)
    return start, end


def variants_in_window(
    geno_alt: pd.DataFrame,
    geno_ref: pd.DataFrame,
    variant_positions: np.ndarray,
    start: int,
    end: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Slice ALT/REF genotype matrices and positions to a genomic window."""
    mask = (variant_positions >= start) & (variant_positions <= end)
    return geno_alt.loc[mask], geno_ref.loc[mask], variant_positions[mask]


def filter_variants_to_common_segment(
    segmentation: pd.DataFrame,
    window: int,
    start: int,
    end: int,
    variants_alt: pd.DataFrame,
    variants_ref: pd.DataFrame,
    variant_pos: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mask genotype entries where phenotype and variants are not on same segment.

    For each sample, both phenotype boundaries (start+window and end-window)
    must lie inside one segment. Variants outside that segment are set to NaN.
    """
    pheno_start = start + window
    pheno_end = end - window

    alt_arr = variants_alt.to_numpy(dtype=float, copy=True)
    ref_arr = variants_ref.to_numpy(dtype=float, copy=True)
    sample_cols = variants_alt.columns.to_numpy()

    seg_index = segmentation.index.to_numpy()
    seg_startpos = segmentation["startpos"].to_numpy()
    seg_endpos = segmentation["endpos"].to_numpy()

    for col_idx, cur_sample in enumerate(sample_cols):
        seg_mask = (
            (seg_index == cur_sample)
            & (seg_startpos <= pheno_start)
            & (seg_endpos >= pheno_start)
        )
        seg_indices = np.flatnonzero(seg_mask)

        if len(seg_indices) != 1:
            alt_arr[:, col_idx] = np.nan
            ref_arr[:, col_idx] = np.nan
            continue

        seg_idx = seg_indices[0]
        lb = seg_startpos[seg_idx]
        ub = seg_endpos[seg_idx]

        if not (lb <= pheno_end <= ub):
            alt_arr[:, col_idx] = np.nan
            ref_arr[:, col_idx] = np.nan
            continue

        outside = (variant_pos < lb) | (variant_pos > ub)
        alt_arr[outside, col_idx] = np.nan
        ref_arr[outside, col_idx] = np.nan

    return (
        pd.DataFrame(alt_arr, index=variants_alt.index, columns=sample_cols),
        pd.DataFrame(ref_arr, index=variants_ref.index, columns=sample_cols),
    )
