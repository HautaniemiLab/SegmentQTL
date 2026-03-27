"""
Shared fixtures and configuration for SegmentQTL tests.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def random_seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def sample_ids():
    """Standard sample IDs for testing."""
    return [f"SAMPLE{i:03d}" for i in range(50)]


@pytest.fixture
def chromosome():
    """Standard chromosome for testing."""
    return "chr21"


@pytest.fixture
def standard_test_data(tmp_path, sample_ids, chromosome):
    """
    Create a standard test dataset that can be shared across tests.

    Returns dictionary with file paths and metadata.
    """
    np.random.seed(42)

    n_samples = len(sample_ids)
    n_genes = 3
    n_variants = 10

    # Gene metadata
    gene_ids = [f"GENE{i:03d}" for i in range(n_genes)]
    gene_starts = [1000000 + i * 100000 for i in range(n_genes)]
    gene_ends = [start + 1000 for start in gene_starts]

    # Quantification data
    quan_data = {
        "chr": [chromosome] * n_genes,
        "start": gene_starts,
        "end": gene_ends,
    }
    for sample in sample_ids:
        quan_data[sample] = np.random.randn(n_genes) + 10

    quan = pd.DataFrame(quan_data, index=gene_ids)

    # Variant positions
    variant_positions = []
    for gene_start in gene_starts:
        for offset in [-50000, 0, 50000]:
            pos = gene_start + offset
            if pos not in variant_positions:
                variant_positions.append(pos)
    variant_positions = sorted(variant_positions)[:n_variants]

    variant_ids = [f"{chromosome}:{pos}:A:T" for pos in variant_positions]

    # Genotype data
    geno_alt = pd.DataFrame(
        {sample: np.random.randn(n_variants) for sample in sample_ids},
        index=variant_ids,
    )
    geno_ref = pd.DataFrame(
        {sample: np.random.randn(n_variants) for sample in sample_ids},
        index=variant_ids,
    )

    # Segmentation - all samples in one segment
    seg = pd.DataFrame(
        {
            "chr": [chromosome] * n_samples,
            "startpos": [0] * n_samples,
            "endpos": [10000000] * n_samples,
        },
        index=sample_ids,
    )

    # Covariates
    cov = pd.DataFrame(
        {sample: np.random.randn(2) for sample in sample_ids}, index=["COV1", "COV2"]
    )

    # Save files
    quan_path = tmp_path / "quantifications.csv"
    quan.to_csv(quan_path)

    seg_path = tmp_path / "segmentation.csv"
    seg.to_csv(seg_path)

    cov_path = tmp_path / "covariates.csv"
    cov.to_csv(cov_path, index=True)

    alt_path = tmp_path / f"{chromosome}_ALTlr.csv"
    geno_alt.to_csv(alt_path)

    ref_path = tmp_path / f"{chromosome}_REFlr.csv"
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
        "n_variants": n_variants,
        "gene_ids": gene_ids,
        "variant_ids": variant_ids,
        "chromosome": chromosome,
    }


@pytest.fixture
def small_test_arrays():
    """Small arrays for unit testing statistical functions."""
    np.random.seed(42)
    n = 30

    X = np.column_stack(
        [
            np.ones(n),
            np.random.randn(n),
            np.random.randn(n),
        ]
    )

    y = X @ np.array([1.0, 0.5, -0.3]) + np.random.randn(n) * 0.5

    return {
        "X": X,
        "y": y,
        "n": n,
        "p": X.shape[1],
    }


# Helper functions for creating test data


def create_simple_quan(tmp_path, chromosome, n_samples, n_genes=1):
    """Create simple quantification file."""
    np.random.seed(42)
    sample_ids = [f"SAMPLE{i:03d}" for i in range(n_samples)]
    gene_ids = [f"GENE{i:03d}" for i in range(n_genes)]

    data = {
        "chr": [chromosome] * n_genes,
        "start": [1000000 + i * 100000 for i in range(n_genes)],
        "end": [1001000 + i * 100000 for i in range(n_genes)],
    }
    for sample in sample_ids:
        data[sample] = np.random.randn(n_genes) + 10

    df = pd.DataFrame(data, index=gene_ids)
    path = tmp_path / "quantifications.csv"
    df.to_csv(path)
    return str(path)


def create_simple_genotype(tmp_path, chromosome, n_samples, n_variants=5):
    """Create simple genotype files."""
    np.random.seed(42)
    sample_ids = [f"SAMPLE{i:03d}" for i in range(n_samples)]
    variant_ids = [f"{chromosome}:{1000000 + i * 10000}:A:T" for i in range(n_variants)]

    alt_data = {sample: np.random.randn(n_variants) for sample in sample_ids}
    ref_data = {sample: np.random.randn(n_variants) for sample in sample_ids}

    alt_df = pd.DataFrame(alt_data, index=variant_ids)
    ref_df = pd.DataFrame(ref_data, index=variant_ids)

    alt_path = tmp_path / f"{chromosome}_ALTlr.csv"
    ref_path = tmp_path / f"{chromosome}_REFlr.csv"

    alt_df.to_csv(alt_path)
    ref_df.to_csv(ref_path)

    return str(alt_path), str(ref_path)
