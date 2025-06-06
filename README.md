# <img src="images/segmentQTLcircle.png" alt="segmentLogo" align="right" height="138" style="margin-left: 0.5em" /> SegmentQTL

**SegmentQTL** is a segmentation-aware molecular quantitative trait loci (molQTL) analysis tool designed for copy-number–driven cancers. It incorporates genomic segmentation data to improve QTL mapping accuracy by filtering out associations disrupted by structural variations. This approach prevents spurious signals caused by breakpoints, ensuring biologically meaningful genotype-phenotype associations.

SegmentQTL supports both **nominal** and **permutation-based** association testing, along with **false discovery rate (FDR) correction**. The tool efficiently processes large datasets, leveraging multi-core parallelization and supporting continuous genotype dosage data to enhance analysis precision.

<img src="images/bySegmentFiltering.png" alt="variantFiltering" width="500"/>

## Contents

- [Installation](#installation)
- [Usage](#usage)
- [Input File Formats](#input-file-formats)
  - [1. Genotype Files (Per-Chromosome CSVs)](#1-genotype-files-per-chromosome-csvs)
  - [2. Phenotype Quantifications (CSV)](#2-phenotype-quantifications-csv)
  - [3. Covariate File (CSV)](#3-covariate-file-csv)
  - [4. Copy Number File (CSV)](#4-copy-number-file-csv)
  - [5. Segmentation File (CSV)](#5-segmentation-file-csv)
- [Output Format](#output-format)
- [Examples](#examples)

## Installation

Requiring preinstalled Python and pip (Python package installer).

```bash
git clone https://github.com/HautaniemiLab/SegmentQTL.git
cd SegmentQTL

# (Optional, but recommended) Create a virtual environment
python -m venv <my-venv>
source <my-venv>/bin/activate

pip install -r requirements.txt
```

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
 
## Input File Formats

SegmentQTL requires five main input files: genotypes, quantifications, covariates, copy number data, and segmentation information. Below are the required formats and examples for each input.

---

#### 1. Genotype Files (Per-Chromosome CSVs)

The `--genotypes` argument should point to a directory containing per-chromosome genotype files, typically named chr1.csv, chr2.csv, ..., chr22.csv, chrX.csv

Each file corresponds to one chromosome and contains genotype dosages for multiple samples.

##### **Required Columns:**
- **`ID`**: Variant identifier in the format `chr:pos:ref:alt` (e.g., `chr8:123456:A:G`).
- **`<sample1>`**, **`<sample2>`**, ...: Sample-specific dosage values. Dosages are continuous values between `0` and `1`.

##### **Example File Format (chr8.csv):**

| ID                | sample1 | sample2 | sample3 |
|-------------------|---------|---------|---------|
| chr8:123456:A:G   | 0.32    | 0.45    | 0.10    |
| chr8:123789:T:C   | 0.76    | 0.88    | 0.34    |
| chr8:124000:G:T   | 0.00    | 0.05    | 0.50    |

---

#### 2. Phenotype Quantifications (CSV)

The `--quantifications` argument should point to a CSV file containing normalized phenotype levels (e.g., gene expression) for all samples across the genome.

##### **Required Columns:**
- **`chr`**: Chromosome where the phenotype is located (e.g., `chr1`, `chrX`).
- **`start`**: Start position of the phenotype.
- **`end`**: End position of the phenotype.
- **`gene_id`**: Unique identifier for the phenotype (e.g., Ensembl gene ID).

##### **Additional Columns:**
- **`<sample1>`**, **`<sample2>`**, ...: Normalized phenotype values per sample.

##### **Example File Format:**

| chr    | start   | end     | gene_id       | sample1 | sample2 | sample3 |
|--------|---------|---------|---------------|---------|---------|---------|
| chr8   | 123000  | 124000  | ENSG00000123  | 1.21    | 0.98    | 1.34    |
| chr8   | 130000  | 132000  | ENSG00000456  | 0.87    | 1.05    | 0.92    |

**Note**: Provide quantifications for the **entire genome**, even if only one chromosome is analyzed at a time. This ensures correct permutation testing and FDR correction.

---

#### 3. Covariate File (CSV)

The `--covariates` argument should point to a CSV file containing covariate values for each sample. First row has `n` entries (samples); subsequent rows have `n + 1` entries (covariate name + values).

##### **Structure:**
- **Row 1**: Sample IDs only (e.g., `sample1,sample2,sample3`)
- **Row 2+**: First cell is the covariate name, followed by values for each sample.

---

#### 4. Copy Number File (CSV)

The `--copynumber` argument should point to a CSV file containing phenotype-level copy number values for each sample.

##### **Required Columns:**
- **`gene_id`**: Ensembl gene ID or equivalent identifier.

##### **Additional Columns:**
- **`<sample1>`**, **`<sample2>`**, ...: Copy number values per sample.

##### **Example File Format:**

| gene_id       | sample1 | sample2 | sample3 |
|---------------|---------|---------|---------|
| ENSG00000123  | 2.10    | 1.85    | 1.92    |
| ENSG00000456  | 1.75    | 2.30    | 2.00    |

---

#### 5. Segmentation File (CSV)

The `--segmentation` argument should point to a CSV file with structural segmentation data for each sample. This is used to determine if a variant and gene are on the same intact genomic segment.

##### **Required Columns:**
- **`sample`**: Sample ID.
- **`chr`**: Chromosome identifier.
- **`startpos`**: Start coordinate of the segment.
- **`endpos`**: End coordinate of the segment.

##### **Example File Format:**

| sample   | chr   | startpos | endpos  |
|----------|-------|----------|---------|
| sample1  | chr8  | 100000   | 200000  |
| sample1  | chr8  | 200001   | 300000  |
| sample2  | chr8  | 120000   | 250000  |

 
## Output Format

The primary output file of SegmentQTL is a CSV containing gene-variant associations.

### Output Columns

| Column Name          | Description                                                                            |
|----------------------|----------------------------------------------------------------------------------------|
| `phenotype`          | Phenotype identifier.                                                                  |
| `variant`            | Variant identifier.                                                                    |
| `number_of_samples`  | Effective number of samples used in the association test after the segment filtering.  |
| `slope`              | Estimated regression coefficient (effect size) for the genotype–phenotype association. |
| `slope_se`           | Standard error of the slope estimate.                                                  |
| `nominal_p`          | P-value from the nominal association test.                                             |
| `p_adj`              | Permutation adjusted p-value.                                                          |
| `chr`                | Chromosome where the gene and variant are located.                                     |
| `fdr`                | FDR corrected p-value.                                                                 |


---

## Examples

These examples assume you're in the root of the `SegmentQTL` folder.

First, unzip the provided mock dataset:

```bash
unzip mock.zip
```

### 1. Nominal Mapping
Run a nominal association test for chromosome 8 using 4 CPU cores:

```bash
python -m segmentqtl --mode nominal --chromosome 8 --num_cores 4 \
    --genotypes mock/genotypes --quantifications mock/quantifications.csv \
    --covariates mock/covariates.csv --copynumber mock/copynumbers.csv \
    --segmentation mock/segments.csv --out_dir results/
```

### 2. Permutation-Based Mapping
Perform 25 permutations using the beta approximation method:

```bash
python -m segmentqtl --mode perm --chromosome 8 --num_permutations 25 \
    --perm_method beta --num_cores 4 \
    --genotypes mock/genotypes --quantifications mock/quantifications.csv \
    --covariates mock/covariates.csv --copynumber mock/copynumbers.csv \
    --segmentation mock/segments.csv --out_dir results/
```
Note that number of permutations should not exceed the number of phenotypes in the full dataset.

### 3. FDR Correction

Apply false discovery rate (FDR) correction to previously computed results:

```bash
python -m segmentqtl --mode fdr --out_dir results/ --fdr_out corrected_results.csv
```

### 4. Testing All Variants for a Specific Phenotype

Run SegmentQTL for all variants of a given phenotype id:

```bash
python -m segmentqtl --mode nominal --all_variants ENSG00000003987 \
    --chromosome 8 --num_cores 1 \
    --genotypes mock/genotypes --quantifications mock/quantifications.csv \
    --covariates mock/covariates.csv --copynumber mock/copynumbers.csv \
    --segmentation mock/segments.csv --out_dir results/
```

### 5. Generating QTL Plots

Generate QTL plots for all tested phenotypes:

```bash
python -m segmentqtl --mode perm --plot_threshold 1 --plot_dir plots/ \
    --chromosome 8 --num_cores 4 --num_permutations 25 \
    --genotypes mock/genotypes --quantifications mock/quantifications.csv \
    --covariates mock/covariates.csv --copynumber mock/copynumbers.csv \
    --segmentation mock/segments.csv --out_dir results/
```



