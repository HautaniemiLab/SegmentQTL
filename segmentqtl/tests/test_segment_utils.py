"""
Tests for segment_utils.py

This module tests:
- phenotype_window_bounds: cis-window boundary calculation
- variants_in_window: genotype slicing to genomic window
- filter_variants_to_common_segment: segment-consistency masking
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from segment_utils import (
    filter_variants_to_common_segment,
    phenotype_window_bounds,
    variants_in_window,
)


class TestPhenotypeWindowBounds:
    """Tests for phenotype_window_bounds."""

    def test_basic_window(self):
        quan = pd.DataFrame(
            {
                "chr": ["chr1"],
                "start": [1000000],
                "end": [1001000],
                "gene_id": ["GENE001"],
                "S1": [5.0],
            }
        )
        start, end = phenotype_window_bounds(quan, 0, window=500000)
        assert start == 1000000 - 500000
        assert end == 1001000 + 500000

    def test_multiple_genes(self):
        quan = pd.DataFrame(
            {
                "chr": ["chr1", "chr1"],
                "start": [100, 200],
                "end": [150, 250],
                "gene_id": ["G1", "G2"],
                "S1": [1.0, 2.0],
            }
        )
        s0, e0 = phenotype_window_bounds(quan, 0, window=10)
        s1, e1 = phenotype_window_bounds(quan, 1, window=10)
        assert s0 == 90
        assert e0 == 160
        assert s1 == 190
        assert e1 == 260

    def test_zero_window(self):
        quan = pd.DataFrame(
            {
                "chr": ["chr1"],
                "start": [500],
                "end": [600],
                "gene_id": ["G"],
                "S1": [0.0],
            }
        )
        start, end = phenotype_window_bounds(quan, 0, window=0)
        assert start == 500
        assert end == 600


class TestVariantsInWindow:
    """Tests for variants_in_window."""

    def setup_method(self):
        self.positions = np.array([100, 200, 300, 400, 500])
        self.alt = pd.DataFrame(
            np.arange(15).reshape(5, 3).astype(float),
            index=[f"v{p}" for p in self.positions],
            columns=["S1", "S2", "S3"],
        )
        self.ref = pd.DataFrame(
            np.arange(15, 30).reshape(5, 3).astype(float),
            index=[f"v{p}" for p in self.positions],
            columns=["S1", "S2", "S3"],
        )

    def test_full_window(self):
        a, r, p = variants_in_window(self.alt, self.ref, self.positions, 100, 500)
        assert len(a) == 5
        assert len(r) == 5
        np.testing.assert_array_equal(p, self.positions)

    def test_partial_window(self):
        a, r, p = variants_in_window(self.alt, self.ref, self.positions, 200, 400)
        assert len(a) == 3
        np.testing.assert_array_equal(p, [200, 300, 400])

    def test_no_variants(self):
        a, r, p = variants_in_window(self.alt, self.ref, self.positions, 600, 700)
        assert len(a) == 0
        assert len(r) == 0
        assert len(p) == 0

    def test_single_variant(self):
        a, r, p = variants_in_window(self.alt, self.ref, self.positions, 300, 300)
        assert len(a) == 1
        np.testing.assert_array_equal(p, [300])


class TestFilterVariantsToCommonSegment:
    """Tests for filter_variants_to_common_segment."""

    def _make_seg(self, sample_ids, startpos_list, endpos_list, chromosome="chr1"):
        return pd.DataFrame(
            {
                "chr": [chromosome] * len(sample_ids),
                "startpos": startpos_list,
                "endpos": endpos_list,
            },
            index=sample_ids,
        )

    def test_all_on_same_segment(self):
        """When all samples share one big segment, no NaNs should appear."""
        samples = ["S1", "S2", "S3"]
        seg = self._make_seg(samples, [0, 0, 0], [10000, 10000, 10000])

        positions = np.array([100, 200, 300])
        alt = pd.DataFrame(np.ones((3, 3)), index=["v1", "v2", "v3"], columns=samples)
        ref = pd.DataFrame(
            np.ones((3, 3)) * 2, index=["v1", "v2", "v3"], columns=samples
        )

        # window=500, start=gene_start-500=0, end=gene_end+500=1000
        # pheno_start = start + window = 500, pheno_end = end - window = 500
        a, r = filter_variants_to_common_segment(seg, 500, 0, 1000, alt, ref, positions)

        assert not np.any(np.isnan(a.to_numpy()))
        assert not np.any(np.isnan(r.to_numpy()))

    def test_sample_off_segment(self):
        """A sample whose segment doesn't cover the phenotype should be all NaN."""
        samples = ["S1", "S2"]
        # S1 covers the phenotype, S2 does not
        seg = self._make_seg(samples, [0, 5000], [10000, 6000])

        positions = np.array([100])
        alt = pd.DataFrame([[1.0, 2.0]], index=["v1"], columns=samples)
        ref = pd.DataFrame([[3.0, 4.0]], index=["v1"], columns=samples)

        # window=500, start=0, end=1000 → pheno_start=500, pheno_end=500
        a, r = filter_variants_to_common_segment(seg, 500, 0, 1000, alt, ref, positions)

        # S1 should be preserved, S2 should be NaN
        assert not np.isnan(a.iloc[0, 0])  # S1
        assert np.isnan(a.iloc[0, 1])  # S2

    def test_variant_outside_segment(self):
        """Variants outside the sample's segment should be NaN for that sample."""
        samples = ["S1"]
        seg = self._make_seg(samples, [100], [300])

        positions = np.array([50, 200, 400])
        alt = pd.DataFrame(
            [[1.0], [2.0], [3.0]], index=["v1", "v2", "v3"], columns=samples
        )
        ref = pd.DataFrame(
            [[4.0], [5.0], [6.0]], index=["v1", "v2", "v3"], columns=samples
        )

        # window=100, start=100, end=300 → pheno_start=200, pheno_end=200
        a, r = filter_variants_to_common_segment(
            seg, 100, 100, 300, alt, ref, positions
        )

        assert np.isnan(a.iloc[0, 0])  # position 50 outside [100, 300]
        assert not np.isnan(a.iloc[1, 0])  # position 200 inside [100, 300]
        assert np.isnan(a.iloc[2, 0])  # position 400 outside [100, 300]

    def test_preserves_shape_and_labels(self):
        """Output should have the same shape and labels as input."""
        samples = ["S1", "S2"]
        seg = self._make_seg(samples, [0, 0], [10000, 10000])
        positions = np.array([100, 200])
        alt = pd.DataFrame(
            [[1, 2], [3, 4]], index=["v1", "v2"], columns=samples, dtype=float
        )
        ref = pd.DataFrame(
            [[5, 6], [7, 8]], index=["v1", "v2"], columns=samples, dtype=float
        )

        a, r = filter_variants_to_common_segment(seg, 500, 0, 1000, alt, ref, positions)

        assert a.shape == alt.shape
        assert list(a.columns) == list(alt.columns)
        assert list(a.index) == list(alt.index)
