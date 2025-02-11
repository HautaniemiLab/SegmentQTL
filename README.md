# SegmentQTL

## Overview

**SegmentQTL** is a segmentation-aware molecular quantitative trait loci (molQTL) analysis tool designed for copy-number–driven cancers. It incorporates genomic segmentation data to improve QTL mapping accuracy by filtering out associations disrupted by structural variations. This approach prevents spurious signals caused by breakpoints, ensuring biologically meaningful genotype-phenotype associations.

SegmentQTL supports both **nominal** and **permutation-based** association testing, along with **false discovery rate (FDR) correction**. The tool efficiently processes large datasets, leveraging multi-core parallelization and supporting continuous genotype dosage data to enhance analysis precision.

## Features

- **Segmentation-aware QTL mapping**: Filters out associations where variants and genes are separated by breakpoints.
- **Multiple modes**: Nominal p-value testing, permutation-based testing, and FDR correction.
- **Parallelization support**: Users can specify the number of CPU cores to accelerate computations.
- **Permutation testing options**: Supports beta approximation and direct permutation methods.
- **Structured input format**: Accepts genotype, segmentation, covariate, copy number, and quantification data in CSV format.
- **Customizable window size**: Users can specify the genomic window for cis-mapping.
- **Plotting functionality**: Option to generate QTL plots for visualizing genotype-phenotype associations.

## Usage

SegmentQTL is executed via the command line with various options to control input data, analysis modes, and computational resources. The key arguments are:

### Required Arguments:
- `--mode`  
  - Specifies the analysis mode:  
    - `nominal`: Perform nominal association testing.  
    - `perm`: Perform permutation-based testing.  
    - `fdr`: Apply FDR correction to existing results.  
- `--chromosome`  
  - Chromosome number (e.g., `21` or `X`). Supports `chr` prefix (e.g., `chr21`).
- `--genotypes`  
  - Path to genotype data directory.
- `--quantifications`  
  - Path to CSV file containing phenotype quantifications (e.g., gene expression). Note: Provide file with quantification for whole genome. This is needed for reliable permutations even if SegmentQTL processes one chromosome at a time.
- `--covariates`  
  - Path to CSV file with sample level covariate data.
- `--copynumber`  
  - Path to CSV file with copy number data.
- `--segmentation`  
  - Path to segmentation file with breakpoint data.

### Optional Arguments:
- `--all_variants`  
  - Test all variants for a given phenotype. Provide a phenotype ID or use without a value to process all phenotypes.
- `--perm_method`  
  - Method used for permutation (`beta` or `direct`).
- `--num_permutations`  
  - Number of permutations per phenotype (default: `8000`).
- `--window`  
  - Window size in base pairs for cis-mapping (default: `1,000,000` bp).
- `--num_cores`  
  - Number of CPU cores to use for parallel processing (default: `1`).
- `--out_dir`  
  - Directory where results are saved.
- `--fdr_out`  
  - File path for saving FDR-corrected results. Must have .csv file extension.
- `--plot_threshold`  
  - P-value threshold for generating plots (`-1` disables plotting).
- `--plot_dir`  
  - Directory for saving generated plots.

---

## Examples

### 1. Nominal Mapping
Run a nominal association test for chromosome 7 using 4 CPU cores:

```bash
python segmentqtl.py --mode nominal --chromosome 7 --num_cores 4 \
    --genotypes path/to/genotypes --quantifications path/to/quantifications.csv \
    --covariates path/to/covariates.csv --copynumber path/to/copynumber.csv \
    --segmentation path/to/segmentation.csv --out_dir results/
```

### 2. Permutation-Based Mapping
Perform 5,000 permutations using the beta approximation method on chromosome X with 8 CPU cores:

```bash
python segmentqtl.py --mode perm --chromosome X --num_permutations 5000 \
    --perm_method beta --num_cores 8 \
    --genotypes path/to/genotypes --quantifications path/to/quantifications.csv \
    --covariates path/to/covariates.csv --copynumber path/to/copynumber.csv \
    --segmentation path/to/segmentation.csv --out_dir results/
```

### 3. FDR Correction

Apply false discovery rate (FDR) correction to previously computed results:

```bash
python segmentqtl.py --mode fdr --out_dir path/to/computedRes --fdr_out corrected_results.csv
```

### 4. Testing All Variants for a Specific Phenotype

Run SegmentQTL for all variants of a given phenotype id (e.g., gene TP53):

```bash
python segmentqtl.py --mode nominal --all_variants TP53 \
    --chromosome 17 --num_cores 4 \
    --genotypes path/to/genotypes --quantifications path/to/quantifications.csv \
    --covariates path/to/covariates.csv --copynumber path/to/copynumber.csv \
    --segmentation path/to/segmentation.csv --out_dir results/
```
This option work with nominal and permutation mode.

### 5. Generating Plots for Significant Associations

Generate plots for all associations with p-values below 0.05:

```bash
python segmentqtl.py --mode nominal --plot_threshold 0.05 --plot_dir plots/ \
    --chromosome 7 --num_cores 4 \
    --genotypes path/to/genotypes --quantifications path/to/quantifications.csv \
    --covariates path/to/covariates.csv --copynumber path/to/copynumber.csv \
    --segmentation path/to/segmentation.csv --out_dir results/
```



