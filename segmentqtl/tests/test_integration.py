"""
Integration tests for SegmentQTL

This module tests end-to-end workflows:
- Full association testing pipeline
- Permutation testing with actual data

"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cis import Cis


class TestEndToEndWorkflow:
    """Integration tests for full analysis workflow."""

    @pytest.fixture
    def full_test_data(self, tmp_path):
        """Create comprehensive test dataset for integration testing."""
        np.random.seed(42)

        n_samples = 60
        n_genes = 5
        n_variants = 20
        chromosome = "chr21"

        sample_ids = [f"SAMPLE{i:03d}" for i in range(n_samples)]
        gene_ids = [f"GENE{i:03d}" for i in range(n_genes)]

        # Create gene positions
        gene_starts = [1000000 + i * 200000 for i in range(n_genes)]
        gene_ends = [start + 1000 for start in gene_starts]

        # Create quantifications with some genes having real effects
        quan_data = {
            "chr": [chromosome] * n_genes,
            "start": gene_starts,
            "end": gene_ends,
            "gene_id": gene_ids,
        }

        # Generate expression data
        base_expression = np.random.randn(n_genes, n_samples) * 2 + 10

        for i, sample in enumerate(sample_ids):
            quan_data[sample] = base_expression[:, i]

        quan = pd.DataFrame(quan_data)

        # Create variant positions within gene windows
        variant_positions = []
        for gene_start in gene_starts:
            # Add variants around each gene
            for offset in [-100000, -50000, 0, 50000]:
                variant_positions.append(gene_start + offset)

        variant_ids = [f"{chromosome}:{pos}:A:T" for pos in variant_positions]

        # Create genotype data with some true effects
        alt_data = {}
        ref_data = {}

        for i, sample in enumerate(sample_ids):
            # Generate genotypes
            alt_vals = np.random.randn(len(variant_ids))
            ref_vals = np.random.randn(len(variant_ids))

            alt_data[sample] = alt_vals
            ref_data[sample] = ref_vals

        geno_alt = pd.DataFrame(alt_data, index=variant_ids)
        geno_ref = pd.DataFrame(ref_data, index=variant_ids)

        # Create segmentation covering all regions
        seg_data = {
            "chr": [chromosome] * n_samples,
            "startpos": [0] * n_samples,
            "endpos": [10000000] * n_samples,
        }
        seg = pd.DataFrame(seg_data, index=sample_ids)

        # Create covariates
        cov_data = {}
        for sample in sample_ids:
            cov_data[sample] = np.random.randn(2)
        cov = pd.DataFrame(cov_data, index=["COV1", "COV2"])

        # Save all files
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
            "tmp_path": tmp_path,
            "n_genes": n_genes,
            "n_samples": n_samples,
        }

    def test_nominal_mode_workflow(self, full_test_data):
        """Test nominal association testing workflow."""
        cis = Cis(
            chromosome="chr21",
            mode="nominal",
            phenotype_covariate=None,
            perm_covariate=None,
            quantifications=full_test_data["quantifications"],
            covariates=full_test_data["covariates"],
            segmentation=full_test_data["segmentation"],
            genotype_alt=full_test_data["genotype_alt"],
            genotype_ref=full_test_data["genotype_ref"],
            all_variants_mode=False,
            perm_method="beta",
            num_permutations=0,
            window=500000,
            num_cores=1,
            record_aic=False,
        )

        # Run single gene analysis
        result = cis.calculate_associations_helper(0)

        assert result is not None
        assert "phenotype" in result.columns
        assert "variant" in result.columns
        assert "nominal_p" in result.columns
        assert "beta_s" in result.columns
        assert "beta_d" in result.columns

    def test_permutation_mode_workflow(self, full_test_data):
        """Test permutation-based association testing workflow."""
        cis = Cis(
            chromosome="chr21",
            mode="perm",
            phenotype_covariate=None,
            perm_covariate=None,
            quantifications=full_test_data["quantifications"],
            covariates=full_test_data["covariates"],
            segmentation=full_test_data["segmentation"],
            genotype_alt=full_test_data["genotype_alt"],
            genotype_ref=full_test_data["genotype_ref"],
            all_variants_mode=False,
            perm_method="beta",
            num_permutations=50,  # Small number for testing
            window=500000,
            num_cores=1,
            record_aic=False,
        )

        # Run single gene analysis
        result = cis.calculate_associations_helper(0)

        assert result is not None
        assert "p_adj" in result.columns

        # p_adj should be between 0 and 1
        p_adj = result["p_adj"].values[0]
        if not np.isnan(p_adj):
            assert 0 <= p_adj <= 1

    def test_direct_permutation_method(self, full_test_data):
        """Test direct permutation method."""
        cis = Cis(
            chromosome="chr21",
            mode="perm",
            phenotype_covariate=None,
            perm_covariate=None,
            quantifications=full_test_data["quantifications"],
            covariates=full_test_data["covariates"],
            segmentation=full_test_data["segmentation"],
            genotype_alt=full_test_data["genotype_alt"],
            genotype_ref=full_test_data["genotype_ref"],
            all_variants_mode=False,
            perm_method="direct",
            num_permutations=50,
            window=500000,
            num_cores=1,
            record_aic=False,
        )

        result = cis.calculate_associations_helper(0)

        assert result is not None
        assert "p_adj" in result.columns

    def test_aic_recording(self, full_test_data):
        """Test AIC recording option."""
        cis = Cis(
            chromosome="chr21",
            mode="nominal",
            phenotype_covariate=None,
            perm_covariate=None,
            quantifications=full_test_data["quantifications"],
            covariates=full_test_data["covariates"],
            segmentation=full_test_data["segmentation"],
            genotype_alt=full_test_data["genotype_alt"],
            genotype_ref=full_test_data["genotype_ref"],
            all_variants_mode=False,
            perm_method="beta",
            num_permutations=0,
            window=500000,
            num_cores=1,
            record_aic=True,
        )

        result = cis.calculate_associations_helper(0)

        assert result is not None
        # Should have AIC columns when record_aic=True
        if len(result) > 0:
            assert "aic_null" in result.columns or result.empty
            assert "aic_alt" in result.columns or result.empty


class TestDataQuality:
    """Tests for data quality and edge cases."""

    @pytest.fixture
    def minimal_test_data(self, tmp_path):
        """Create minimal test dataset."""
        np.random.seed(42)

        n_samples = 35  # Just above minimum
        chromosome = "chr21"

        sample_ids = [f"SAMPLE{i:03d}" for i in range(n_samples)]

        # Single gene
        quan_data = {
            "chr": [chromosome],
            "start": [1000000],
            "end": [1001000],
            "gene_id": ["GENE001"],
        }
        for sample in sample_ids:
            quan_data[sample] = [np.random.randn() + 10]

        quan = pd.DataFrame(quan_data)

        # Few variants
        variant_ids = [f"{chromosome}:{1000000 + i * 10000}:A:T" for i in range(3)]

        alt_data = {sample: np.random.randn(3) for sample in sample_ids}
        ref_data = {sample: np.random.randn(3) for sample in sample_ids}

        geno_alt = pd.DataFrame(alt_data, index=variant_ids)
        geno_ref = pd.DataFrame(ref_data, index=variant_ids)

        seg_data = {
            "chr": [chromosome] * n_samples,
            "startpos": [0] * n_samples,
            "endpos": [10000000] * n_samples,
        }
        seg = pd.DataFrame(seg_data, index=sample_ids)

        # No covariates
        cov = pd.DataFrame({sample: [0.0] for sample in sample_ids}, index=["dummy"])

        # Save files
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

    def test_minimal_data(self, minimal_test_data):
        """Test with minimal data."""
        cis = Cis(
            chromosome="chr21",
            mode="nominal",
            phenotype_covariate=None,
            perm_covariate=None,
            quantifications=minimal_test_data["quantifications"],
            covariates=minimal_test_data["covariates"],
            segmentation=minimal_test_data["segmentation"],
            genotype_alt=minimal_test_data["genotype_alt"],
            genotype_ref=minimal_test_data["genotype_ref"],
            all_variants_mode=False,
            perm_method="beta",
            num_permutations=0,
            window=500000,
            num_cores=1,
            record_aic=False,
        )

        result = cis.calculate_associations_helper(0)

        # Should complete without error
        assert result is not None


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_extreme_values(self, tmp_path):
        """Test with extreme expression values."""
        np.random.seed(42)

        n_samples = 50
        chromosome = "chr21"

        sample_ids = [f"SAMPLE{i:03d}" for i in range(n_samples)]

        # Create expression with extreme values
        quan_data = {
            "chr": [chromosome],
            "start": [1000000],
            "end": [1001000],
            "gene_id": ["GENE001"],
        }
        extreme_expr = np.random.randn(n_samples) * 1000 + 10000
        for i, sample in enumerate(sample_ids):
            quan_data[sample] = [extreme_expr[i]]

        quan = pd.DataFrame(quan_data)

        variant_ids = [f"{chromosome}:{1000000}:A:T"]

        alt_data = {sample: np.random.randn(1) for sample in sample_ids}
        ref_data = {sample: np.random.randn(1) for sample in sample_ids}

        geno_alt = pd.DataFrame(alt_data, index=variant_ids)
        geno_ref = pd.DataFrame(ref_data, index=variant_ids)

        seg_data = {
            "chr": [chromosome] * n_samples,
            "startpos": [0] * n_samples,
            "endpos": [10000000] * n_samples,
        }
        seg = pd.DataFrame(seg_data, index=sample_ids)

        cov = pd.DataFrame({sample: [0.0] for sample in sample_ids}, index=["dummy"])

        # Save files
        quan.to_csv(tmp_path / "quantifications.csv", index=False)
        seg.to_csv(tmp_path / "segmentation.csv")
        cov.to_csv(tmp_path / "covariates.csv", index=True)
        geno_alt.to_csv(tmp_path / "chr21_ALTlr.csv")
        geno_ref.to_csv(tmp_path / "chr21_REFlr.csv")

        cis = Cis(
            chromosome="chr21",
            mode="nominal",
            phenotype_covariate=None,
            perm_covariate=None,
            quantifications=str(tmp_path / "quantifications.csv"),
            covariates=str(tmp_path / "covariates.csv"),
            segmentation=str(tmp_path / "segmentation.csv"),
            genotype_alt=str(tmp_path / "chr21_ALTlr.csv"),
            genotype_ref=str(tmp_path / "chr21_REFlr.csv"),
            all_variants_mode=False,
            perm_method="beta",
            num_permutations=0,
            window=500000,
            num_cores=1,
            record_aic=False,
        )

        # Should not crash with extreme values
        result = cis.calculate_associations_helper(0)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
