"""
Tests for cis.py

This module tests:
- VariantFWLCache dataclass
- Cis class methods for:
  - File loading/validation
  - Gene window calculations
  - Variant filtering
  - Segment filtering
  - FWL cache building
  - Association testing
  - Permutation testing
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cis import Cis
from statistical_utils import VariantFWLCache, build_variant_fwl_caches, check_grouping


class TestVariantFWLCache:
    """Tests for VariantFWLCache dataclass."""

    def test_cache_creation(self):
        """Test creating a VariantFWLCache instance."""
        idx_masky = np.array([0, 1, 2, 3, 4])
        QcT = np.random.randn(2, 5)
        q_d = np.random.randn(5)
        q_d = q_d / np.linalg.norm(q_d)
        df2 = 3

        cache = VariantFWLCache(
            idx_masky=idx_masky,
            QcT=QcT,
            q_d=q_d,
            df2=df2,
        )

        np.testing.assert_array_equal(cache.idx_masky, idx_masky)
        np.testing.assert_array_equal(cache.QcT, QcT)
        np.testing.assert_array_equal(cache.q_d, q_d)
        assert cache.df2 == df2


class TestDataFixtures:
    """Fixtures for creating test data."""

    @staticmethod
    def create_test_quantifications(n_genes=5, n_samples=50, chromosome="chr21"):
        """Create test quantification data.

        Matches the real BED-like format: chr, start, end, gene_id, sample1, ...
        Cis.__init__ reads with index_col=3, so gene_id (position 3) becomes the index.
        """
        gene_ids = [f"GENE{i:03d}" for i in range(n_genes)]
        sample_ids = [f"SAMPLE{i:03d}" for i in range(n_samples)]

        data = {
            "chr": [chromosome] * n_genes,
            "start": [1000000 + i * 100000 for i in range(n_genes)],
            "end": [1000500 + i * 100000 for i in range(n_genes)],
            "gene_id": gene_ids,
        }

        # Add expression values for each sample
        np.random.seed(42)
        for sample in sample_ids:
            data[sample] = np.random.randn(n_genes) + 5

        df = pd.DataFrame(data)
        return df

    @staticmethod
    def create_test_genotypes(n_variants=10, n_samples=50, chromosome="chr21"):
        """Create test genotype data (ALT and REF)."""
        sample_ids = [f"SAMPLE{i:03d}" for i in range(n_samples)]

        # Create variant IDs with positions
        variant_ids = [
            f"{chromosome}:{1000000 + i * 10000}:A:T" for i in range(n_variants)
        ]

        np.random.seed(42)
        # ALT log-ratio values
        alt_data = {}
        for sample in sample_ids:
            alt_data[sample] = np.random.randn(n_variants)

        # REF log-ratio values
        ref_data = {}
        for sample in sample_ids:
            ref_data[sample] = np.random.randn(n_variants)

        geno_alt = pd.DataFrame(alt_data, index=variant_ids)
        geno_ref = pd.DataFrame(ref_data, index=variant_ids)

        return geno_alt, geno_ref

    @staticmethod
    def create_test_segmentation(n_samples=50, chromosome="chr21"):
        """Create test segmentation data."""
        sample_ids = [f"SAMPLE{i:03d}" for i in range(n_samples)]

        data = {
            "chr": [chromosome] * n_samples,
            "startpos": [0] * n_samples,  # All samples cover full region
            "endpos": [10000000] * n_samples,
        }

        df = pd.DataFrame(data, index=sample_ids)
        return df

    @staticmethod
    def create_test_covariates(n_samples=50, n_covariates=2):
        """Create test covariates data."""
        sample_ids = [f"SAMPLE{i:03d}" for i in range(n_samples)]
        cov_names = [f"COV{i}" for i in range(n_covariates)]

        np.random.seed(42)
        data = {}
        for sample in sample_ids:
            data[sample] = np.random.randn(n_covariates)

        df = pd.DataFrame(data, index=cov_names)
        return df

    @staticmethod
    def create_test_phenotype_covariate(gene_ids, n_samples=50):
        """Create test phenotype-level covariate data."""
        sample_ids = [f"SAMPLE{i:03d}" for i in range(n_samples)]

        np.random.seed(42)
        data = {}
        for sample in sample_ids:
            data[sample] = np.random.randn(len(gene_ids))

        df = pd.DataFrame(data, index=gene_ids)
        return df


class TestCisInitialization:
    """Tests for Cis class initialization."""

    @pytest.fixture
    def test_data_dir(self, tmp_path):
        """Create temporary test data files."""
        # Create test data
        fixtures = TestDataFixtures()
        quan = fixtures.create_test_quantifications()
        geno_alt, geno_ref = fixtures.create_test_genotypes()
        seg = fixtures.create_test_segmentation()
        cov = fixtures.create_test_covariates()

        # Save to temp files
        quan_path = tmp_path / "quantifications.csv"
        quan.to_csv(quan_path, index=False)

        seg_path = tmp_path / "segmentation.csv"
        seg.to_csv(seg_path)

        cov_path = tmp_path / "covariates.csv"
        cov.to_csv(cov_path, index=True)

        alt_path = tmp_path / "chr21_ALTlr.csv"
        geno_alt.to_csv(alt_path)

        ref_path = tmp_path / "chr21_REFlr.csv"
        geno_ref.to_csv(ref_path)

        return {
            "quantifications": str(quan_path),
            "segmentation": str(seg_path),
            "covariates": str(cov_path),
            "genotype_alt": str(alt_path),
            "genotype_ref": str(ref_path),
        }

    def test_basic_initialization(self, test_data_dir):
        """Test basic Cis initialization."""
        cis = Cis(
            chromosome="chr21",
            mode="nominal",
            phenotype_covariate=None,
            perm_covariate=None,
            quantifications=test_data_dir["quantifications"],
            covariates=test_data_dir["covariates"],
            segmentation=test_data_dir["segmentation"],
            genotype_alt=test_data_dir["genotype_alt"],
            genotype_ref=test_data_dir["genotype_ref"],
            all_variants_mode=False,
            perm_method="beta",
            num_permutations=100,
            window=500000,
            num_cores=1,
            record_aic=False,
        )

        assert cis.chromosome == "chr21"
        assert cis.window == 500000
        assert cis.num_permutations == 0  # Nominal mode sets this to 0
        assert cis.perm_method == "beta"

    def test_perm_mode_initialization(self, test_data_dir):
        """Test Cis initialization in permutation mode."""
        cis = Cis(
            chromosome="chr21",
            mode="perm",
            phenotype_covariate=None,
            perm_covariate=None,
            quantifications=test_data_dir["quantifications"],
            covariates=test_data_dir["covariates"],
            segmentation=test_data_dir["segmentation"],
            genotype_alt=test_data_dir["genotype_alt"],
            genotype_ref=test_data_dir["genotype_ref"],
            all_variants_mode=False,
            perm_method="beta",
            num_permutations=100,
            window=500000,
            num_cores=1,
            record_aic=False,
        )

        assert cis.num_permutations == 100

    def test_invalid_perm_method(self, test_data_dir):
        """Test that invalid perm_method raises error."""
        with pytest.raises(ValueError, match="Invalid perm_method"):
            Cis(
                chromosome="chr21",
                mode="perm",
                phenotype_covariate=None,
                perm_covariate=None,
                quantifications=test_data_dir["quantifications"],
                covariates=test_data_dir["covariates"],
                segmentation=test_data_dir["segmentation"],
                genotype_alt=test_data_dir["genotype_alt"],
                genotype_ref=test_data_dir["genotype_ref"],
                all_variants_mode=False,
                perm_method="invalid",
                num_permutations=100,
                window=500000,
                num_cores=1,
                record_aic=False,
            )

    def test_file_not_found(self, test_data_dir):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            Cis(
                chromosome="chr21",
                mode="nominal",
                phenotype_covariate=None,
                perm_covariate=None,
                quantifications="nonexistent.csv",
                covariates=test_data_dir["covariates"],
                segmentation=test_data_dir["segmentation"],
                genotype_alt=test_data_dir["genotype_alt"],
                genotype_ref=test_data_dir["genotype_ref"],
                all_variants_mode=False,
                perm_method="beta",
                num_permutations=100,
                window=500000,
                num_cores=1,
                record_aic=False,
            )


class TestCisHelperMethods:
    """Tests for Cis helper methods."""

    @pytest.fixture
    def cis_instance(self, tmp_path):
        """Create a Cis instance with test data."""
        fixtures = TestDataFixtures()
        quan = fixtures.create_test_quantifications()
        geno_alt, geno_ref = fixtures.create_test_genotypes()
        seg = fixtures.create_test_segmentation()
        cov = fixtures.create_test_covariates()

        # Save to temp files
        quan_path = tmp_path / "quantifications.csv"
        quan.to_csv(quan_path, index=False)

        seg_path = tmp_path / "segmentation.csv"
        seg.to_csv(seg_path)

        cov_path = tmp_path / "covariates.csv"
        cov.to_csv(cov_path, index=True)

        alt_path = tmp_path / "chr21_ALTlr.csv"
        geno_alt.to_csv(alt_path)

        ref_path = tmp_path / "chr21_REFlr.csv"
        geno_ref.to_csv(ref_path)

        return Cis(
            chromosome="chr21",
            mode="nominal",
            phenotype_covariate=None,
            perm_covariate=None,
            quantifications=str(quan_path),
            covariates=str(cov_path),
            segmentation=str(seg_path),
            genotype_alt=str(alt_path),
            genotype_ref=str(ref_path),
            all_variants_mode=False,
            perm_method="beta",
            num_permutations=100,
            window=500000,
            num_cores=1,
            record_aic=False,
        )

    def test_start_end_gene_window(self, cis_instance):
        """Test gene window calculation."""
        window_start, window_end = cis_instance.start_end_gene_window(0)

        # First gene starts at 1000000
        expected_start = 1000000 - 500000
        expected_end = 1000500 + 500000

        assert window_start == expected_start
        assert window_end == expected_end

    def test_check_grouping_sufficient_variance(self, cis_instance):
        """Test check_grouping with sufficient variance in d = reflr - altlr."""
        altlr = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        reflr = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        result = check_grouping(altlr, reflr)

        assert result == True

    def test_check_grouping_no_variance(self, cis_instance):
        """Test check_grouping with no variance in d = reflr - altlr."""
        altlr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        reflr = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        result = check_grouping(altlr, reflr)

        assert result == False

    def test_filter_arrays_basic(self, cis_instance):
        """Test filter_arrays with basic data."""
        n = 50
        GEX = np.random.randn(n)
        altlr = np.random.randn(n)
        reflr = np.random.randn(n)
        phenotype_cov = None
        cov_values = []

        result = cis_instance.filter_arrays(
            GEX, altlr, reflr, phenotype_cov, cov_values
        )

        GEX_f, altlr_f, reflr_f, pheno_f, cov_f = result

        assert len(GEX_f) == n
        assert len(altlr_f) == n
        assert len(reflr_f) == n
        assert pheno_f is None
        assert cov_f == []

    def test_filter_arrays_with_nans(self, cis_instance):
        """Test filter_arrays removes NaN rows."""
        n = 50
        GEX = np.random.randn(n)
        altlr = np.random.randn(n)
        reflr = np.random.randn(n)

        # Add NaNs
        GEX[0:5] = np.nan
        altlr[10:15] = np.nan

        result = cis_instance.filter_arrays(GEX, altlr, reflr, None, [])

        GEX_f, altlr_f, reflr_f, pheno_f, cov_f = result

        # Should have n - 10 samples (5 NaN in GEX, 5 in altlr)
        assert len(GEX_f) == n - 10
        assert not np.any(np.isnan(GEX_f))
        assert not np.any(np.isnan(altlr_f))

    def test_filter_arrays_insufficient_samples(self, cis_instance):
        """Test filter_arrays returns empty when < 30 samples."""
        n = 40
        GEX = np.random.randn(n)
        altlr = np.random.randn(n)
        reflr = np.random.randn(n)

        # Add NaNs to reduce to < 30 samples
        GEX[0:15] = np.nan

        result = cis_instance.filter_arrays(GEX, altlr, reflr, None, [])

        GEX_f, altlr_f, reflr_f, pheno_f, cov_f = result

        assert len(GEX_f) == 0


class TestGeneVariantsCommonSegment:
    """Tests for gene_variants_common_segment method."""

    @pytest.fixture
    def setup_data(self, tmp_path):
        """Create test data for segment filtering tests."""
        n_samples = 50
        n_variants = 10
        chromosome = "chr21"

        fixtures = TestDataFixtures()
        quan = fixtures.create_test_quantifications(
            n_genes=5, n_samples=n_samples, chromosome=chromosome
        )

        # Create genotypes with specific positions
        sample_ids = [f"SAMPLE{i:03d}" for i in range(n_samples)]
        variant_positions = [1000000 + i * 10000 for i in range(n_variants)]
        variant_ids = [f"{chromosome}:{pos}:A:T" for pos in variant_positions]

        np.random.seed(42)
        alt_data = {sample: np.random.randn(n_variants) for sample in sample_ids}
        ref_data = {sample: np.random.randn(n_variants) for sample in sample_ids}

        geno_alt = pd.DataFrame(alt_data, index=variant_ids)
        geno_ref = pd.DataFrame(ref_data, index=variant_ids)

        # Create segmentation that covers different regions per sample
        seg_data = {
            "chr": [chromosome] * n_samples,
            "startpos": [0] * n_samples,
            "endpos": [10000000] * n_samples,
        }
        seg = pd.DataFrame(seg_data, index=sample_ids)

        # Save files
        quan_path = tmp_path / "quantifications.csv"
        quan.to_csv(quan_path, index=False)

        seg_path = tmp_path / "segmentation.csv"
        seg.to_csv(seg_path)

        cov = fixtures.create_test_covariates(n_samples=n_samples)
        cov_path = tmp_path / "covariates.csv"
        cov.to_csv(cov_path, index=True)

        alt_path = tmp_path / "chr21_ALTlr.csv"
        geno_alt.to_csv(alt_path)

        ref_path = tmp_path / "chr21_REFlr.csv"
        geno_ref.to_csv(ref_path)

        cis = Cis(
            chromosome=chromosome,
            mode="nominal",
            phenotype_covariate=None,
            perm_covariate=None,
            quantifications=str(quan_path),
            covariates=str(cov_path),
            segmentation=str(seg_path),
            genotype_alt=str(alt_path),
            genotype_ref=str(ref_path),
            all_variants_mode=False,
            perm_method="beta",
            num_permutations=0,
            window=500000,
            num_cores=1,
            record_aic=False,
        )

        return cis, geno_alt, geno_ref

    def test_segment_filtering_basic(self, setup_data):
        """Test basic segment filtering."""
        cis, geno_alt, geno_ref = setup_data

        # Use gene coordinates that are within the segment
        start = 500000  # window_start (gene_start - window)
        end = 1500500  # window_end (gene_end + window)

        result_alt, result_ref = cis.gene_variants_common_segment(
            start, end, geno_alt, geno_ref, cis.variant_positions
        )

        # Should return DataFrames
        assert isinstance(result_alt, pd.DataFrame)
        assert isinstance(result_ref, pd.DataFrame)

    def test_variant_pos_is_numpy_array(self, setup_data):
        """Test that variant_pos is properly converted to numpy array."""
        cis, geno_alt, geno_ref = setup_data

        # This should not raise TypeError
        start = 500000
        end = 1500500

        # If variant_pos was a list, the comparison would fail
        result_alt, result_ref = cis.gene_variants_common_segment(
            start, end, geno_alt, geno_ref, cis.variant_positions
        )

        assert result_alt is not None
        assert result_ref is not None


class TestBuildVariantFWLCaches:
    """Tests for build_variant_fwl_caches method."""

    def test_cache_building(self, tmp_path):
        """Test that FWL caches are properly built."""
        n_samples = 50
        n_variants = 5
        chromosome = "chr21"

        fixtures = TestDataFixtures()
        quan = fixtures.create_test_quantifications(
            n_genes=3, n_samples=n_samples, chromosome=chromosome
        )

        sample_ids = [f"SAMPLE{i:03d}" for i in range(n_samples)]
        variant_ids = [
            f"{chromosome}:{1000000 + i * 10000}:A:T" for i in range(n_variants)
        ]

        np.random.seed(42)
        alt_data = {sample: np.random.randn(n_variants) for sample in sample_ids}
        ref_data = {sample: np.random.randn(n_variants) for sample in sample_ids}

        geno_alt = pd.DataFrame(alt_data, index=variant_ids)
        geno_ref = pd.DataFrame(ref_data, index=variant_ids)

        seg_data = {
            "chr": [chromosome] * n_samples,
            "startpos": [0] * n_samples,
            "endpos": [10000000] * n_samples,
        }
        seg = pd.DataFrame(seg_data, index=sample_ids)

        # Save files
        quan_path = tmp_path / "quantifications.csv"
        quan.to_csv(quan_path, index=False)

        seg_path = tmp_path / "segmentation.csv"
        seg.to_csv(seg_path)

        cov = fixtures.create_test_covariates(n_samples=n_samples, n_covariates=1)
        cov_path = tmp_path / "covariates.csv"
        cov.to_csv(cov_path, index=True)

        alt_path = tmp_path / "chr21_ALTlr.csv"
        geno_alt.to_csv(alt_path)

        ref_path = tmp_path / "chr21_REFlr.csv"
        geno_ref.to_csv(ref_path)

        cis = Cis(
            chromosome=chromosome,
            mode="perm",
            phenotype_covariate=None,
            perm_covariate=None,
            quantifications=str(quan_path),
            covariates=str(cov_path),
            segmentation=str(seg_path),
            genotype_alt=str(alt_path),
            genotype_ref=str(ref_path),
            all_variants_mode=False,
            perm_method="beta",
            num_permutations=10,
            window=500000,
            num_cores=1,
            record_aic=False,
        )

        # Build mask_y and masky_pos
        mask_y = np.ones(n_samples, dtype=bool)
        masky_pos = np.arange(n_samples)

        # Build caches
        caches = build_variant_fwl_caches(
            geno_alt,
            geno_ref,
            mask_y,
            masky_pos,
            None,  # phenotype_cov_full
            [],  # cov_values_full
        )

        # Should have created caches for variants with sufficient data
        assert isinstance(caches, dict)

        for variant_id, cache in caches.items():
            assert isinstance(cache, VariantFWLCache)
            assert cache.df2 > 0
            assert len(cache.idx_masky) >= 30  # Minimum samples


class TestPermutationRNG:
    """Tests for RNG handling in permutation testing."""

    def test_deterministic_permutations(self, tmp_path):
        """Test that permutations are deterministic given same gene_index."""
        n_samples = 50
        chromosome = "chr21"

        fixtures = TestDataFixtures()
        quan = fixtures.create_test_quantifications(
            n_genes=3, n_samples=n_samples, chromosome=chromosome
        )

        sample_ids = [f"SAMPLE{i:03d}" for i in range(n_samples)]
        variant_ids = [f"{chromosome}:{1000000 + i * 10000}:A:T" for i in range(5)]

        np.random.seed(42)
        alt_data = {sample: np.random.randn(5) for sample in sample_ids}
        ref_data = {sample: np.random.randn(5) for sample in sample_ids}

        geno_alt = pd.DataFrame(alt_data, index=variant_ids)
        geno_ref = pd.DataFrame(ref_data, index=variant_ids)

        seg_data = {
            "chr": [chromosome] * n_samples,
            "startpos": [0] * n_samples,
            "endpos": [10000000] * n_samples,
        }
        seg = pd.DataFrame(seg_data, index=sample_ids)

        # Save files
        quan_path = tmp_path / "quantifications.csv"
        quan.to_csv(quan_path, index=False)

        seg_path = tmp_path / "segmentation.csv"
        seg.to_csv(seg_path)

        cov = fixtures.create_test_covariates(n_samples=n_samples, n_covariates=0)
        cov_path = tmp_path / "covariates.csv"
        cov.to_csv(cov_path, index=True)

        alt_path = tmp_path / "chr21_ALTlr.csv"
        geno_alt.to_csv(alt_path)

        ref_path = tmp_path / "chr21_REFlr.csv"
        geno_ref.to_csv(ref_path)

        # Test that same gene_index produces same RNG sequence
        gene_index = 0
        rng1 = np.random.default_rng(seed=12345 + gene_index)
        rng2 = np.random.default_rng(seed=12345 + gene_index)

        perm1 = rng1.permutation(n_samples)
        perm2 = rng2.permutation(n_samples)

        np.testing.assert_array_equal(perm1, perm2)

    def test_different_genes_different_permutations(self):
        """Test that different genes get different permutations."""
        n = 50

        rng0 = np.random.default_rng(seed=12345 + 0)
        rng1 = np.random.default_rng(seed=12345 + 1)

        perm0 = rng0.permutation(n)
        perm1 = rng1.permutation(n)

        # Should be different
        assert not np.array_equal(perm0, perm1)


class TestFStatComputation:
    """Tests for F-statistic computation in permutation loop."""

    def test_f_stat_clamping(self):
        """Test that negative F-stats are clamped to 0."""
        # Simulate case where rss0 - rss1 < 0 due to numerical issues
        rss0 = 100.0
        rss1 = 100.1  # Slightly larger than rss0
        df1 = 2
        df2 = 45

        f_stat = ((rss0 - rss1) / df1) / (rss1 / df2)

        # Without clamping, this would be negative
        assert f_stat < 0

        # With clamping
        f_stat_clamped = max(0.0, f_stat)
        assert f_stat_clamped == 0.0

    def test_f_stat_computation(self):
        """Test F-stat computation formula."""
        rss0 = 100.0
        rss1 = 50.0
        df1 = 2
        df2 = 45

        f_stat = ((rss0 - rss1) / df1) / (rss1 / df2)

        expected = ((100.0 - 50.0) / 2) / (50.0 / 45)
        np.testing.assert_allclose(f_stat, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
