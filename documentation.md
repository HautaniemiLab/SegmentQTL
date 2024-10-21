# SegmentQTL Documentation
&nbsp;

<a id="cis"></a>

# cis

<a id="cis.Cis"></a>

## Cis Objects

```python
class Cis()
```

<a id="cis.Cis.load_and_validate_file"></a>

#### load\_and\_validate\_file

```python
def load_and_validate_file(file_path: str, index_col: int)
```

Load a CSV file and validate its existence and content.

**Arguments**:

  - file_path: Path to file
  

**Returns**:

  - Dataframe from contents of the CSV file
  

**Raises**:

  - FileNotFoundError: If the file does not exist at the given path.
  - ValueError: If the CSV file is empty (i.e., has no rows).

<a id="cis.Cis.start_end_gene_window"></a>

#### start\_end\_gene\_window

```python
def start_end_gene_window(gene_index: int)
```

Find position of the window of a given gene.

**Arguments**:

  - gene_index: Index of the desired gene on the quantification file
  

**Returns**:

  - Tuple of window_start and window_end, which define the start and end positions of the window

<a id="cis.Cis.get_variants_for_gene_window"></a>

#### get\_variants\_for\_gene\_window

```python
def get_variants_for_gene_window(current_start: int, current_end: int)
```

Find all the variants inside a window of a gene.

**Arguments**:

  - current_start: Start position of a window
  - current_end: End position of a window
  

**Returns**:

  - variants: Subset of genotype dataframe that contains only those variants that are inside
  the given window

<a id="cis.Cis.gene_variants_common_segment"></a>

#### gene\_variants\_common\_segment

```python
def gene_variants_common_segment(start: int, end: int, variants: pd.DataFrame)
```

Filter variants to ensure that the gene and variants that are in the same
window are also on a same segment.

**Arguments**:

  - start: Start position of a window
  - end: End position of a window
  - variants: Subset of genotype file. Only variants that are in the same window as
  the gene of interest
  

**Returns**:

  - variants: Subset of genotype dataframe that is filtered and masked by segmentation and window.

<a id="cis.Cis.gene_variant_regressions_permutations"></a>

#### gene\_variant\_regressions\_permutations

```python
def gene_variant_regressions_permutations(gene_index: int,
                                          transf_variants: pd.DataFrame,
                                          variant: str,
                                          regression_data: pd.DataFrame)
```

Perform permutations to obtain adjusted p-values. In case of 0 permutations,
do only nominal pass.

**Arguments**:

  - gene_index: Index of a gene of interest on the quantification file.
  - transf_variants: Dataframe of transformed variants that are processed for
  window and segmentation.
  - variant: Variant id
  - regression_data: Dataframe with current gene expression levels, genotypes,
  and covariates
  

**Returns**:

  - actual_associations: Dataframe of association testing results for a gene.
  When > 0 permutations are used, also adjusted p-values are provided.

<a id="cis.Cis.permutation_data"></a>

#### permutation\_data

```python
def permutation_data(gene_index: int, perm_index: int,
                     transf_variants: pd.DataFrame, variant: str)
```

Find data for association testing for permutations. In this case all
dependent variable values are fixed, only the phenotype levels are
permuted.

**Arguments**:

  - gene_index: Index of the actual gene on the quantification file.
  - perm_index: Index of a gene on the quantification file that is used for permutation.
  - transf_variants: Dataframe of transformed variants that are processed for
  window and segmentation.
  - variant: Variant ID
  

**Returns**:

  - perm_data: Dataframe of data linked with the fixed variant and permuted gene

<a id="cis.Cis.check_grouping"></a>

#### check\_grouping

```python
def check_grouping(cur_genotypes_filtered: np.ndarray)
```

Find if tail and middle values have adequate representation in data.

**Arguments**:

  - cur_genotypes_filtered: Array of genotype dosages
  

**Returns**:

  - Boolean value showing if there are enough instances in the different genotype groups.

<a id="cis.Cis.filter_arrays"></a>

#### filter\_arrays

```python
def filter_arrays(GEX: np.ndarray, CN: np.ndarray, cur_genotypes: np.ndarray,
                  cov_values: np.ndarray)
```

Filter data arrays and do validity checks.

**Arguments**:

  - GEX: Gene expression levels
  - CN: Gene copy numbers
  - cur_genotypes: Genotype dosages
  - cov_values: All other covariate values
  - group_check: Whether to check representation of values in the middle
  and in the tails in genotypes
  

**Returns**:

  Tuple of:
  - GEX_filtered: Filtered gene expression values
  - CN_filtered: Filtered copy numbers
  - cur_genotypes_filtered: Filtered genotypes dosages
  - cov_values_filtered: Filtered covariate values

<a id="cis.Cis.best_variant_data"></a>

#### best\_variant\_data

```python
def best_variant_data(gene_index: int, transf_variants: pd.DataFrame,
                      quantifications: pd.DataFrame)
```

Find variant and linked data for a gene that has strongest Pearson
correlation with the independent variable.

**Arguments**:

  - gene_index: Index of a gene of interest on the quantification file.
  - transf_variants: Dataframe of transformed variants that are processed for
  window and segmentation.
  - quantifications: Dataframe of quantifications.
  

**Returns**:

  - best_variant: Id of the variant with strongest correlation
  - data_best_corr: Dataframe of data linked with the chosen variant

<a id="cis.Cis.data_all_variants"></a>

#### data\_all\_variants

```python
def data_all_variants(GEX: np.ndarray, CN: np.ndarray, cov_values: np.ndarray,
                      cur_genotypes: np.ndarray)
```

Process data for association testing when in all variants mode.

**Arguments**:

  - GEX: Gene expression levels.
  - CN: Gene copy numbers
  - cov_values: All other covariate values
  - cur_genotypes: Genotype dosages
  

**Returns**:

  - Dataframe of filtered regression data.

<a id="cis.Cis.process_all_variants"></a>

#### process\_all\_variants

```python
def process_all_variants(gene_index: int, transf_variants: pd.DataFrame)
```

Conduct association testing for all variants in a window instead of selecting
only best correlated variant. Construct regression data and then run the
regressions.

**Arguments**:

  - gene_index: Index of a gene of interest on the quantification file.
  - transf_variants: Dataframe of transformed variants that are processed for
  window and segmentation.
  

**Returns**:

  - Dataframe with all association testing results for a gene.

<a id="cis.Cis.gene_variant_regressions"></a>

#### gene\_variant\_regressions

```python
def gene_variant_regressions(gene_index: int, quantifications: pd.DataFrame,
                             variant: str, regression_data: pd.DataFrame)
```

Find associations between the gene expression values of a gene and variants
by performing regressions. Using ordinary least square regression,
log-likelihood calculations, and likelihood ratio test to pinpoint the effect of genotypes.

**Arguments**:

  - gene_index: Index of a gene of interest on the quantification file.
  - quantifications: Dataframe of quantifications.
  - variant: Variant ID
  - regression_data: Regression data for current gene variant pair including covariates
  

**Returns**:

  - associations dataframe with statistics of the strenghts of associations

<a id="cis.Cis.calculate_associations"></a>

#### calculate\_associations

```python
def calculate_associations()
```

Calculate associations for gene indices using multiprocessing.

Steps:
1. Initializes the multiprocessing pool with the specified number of cores.
2. Maps gene indices to the helper function using the pool.
3. Closes the pool and waits for the processes to complete.
4. Concatenates the resulting DataFrames from each process into one DataFrame.

**Returns**:

  - full_associations: A concatenated dataframe containing the association results
  for all gene indices.

<a id="cis.Cis.calculate_associations_helper"></a>

#### calculate\_associations\_helper

```python
def calculate_associations_helper(gene_index: int)
```

Helper function to calculate associations for a single gene index.

This function performs several steps to calculate the associations for a
specific gene index:
1. Prints the current progress of the calculation.
2. Determines the start and end positions for the gene window.
3. Retrieves the variants within the gene window.
4. Transforms the variants based on a common segment.
5. Performs regressions to calculate associations.

**Arguments**:

  - gene_index (int): The index of the gene for which associations are being calculated.
  

**Returns**:

  - A dataframe containing the association results for the specified gene index.

<a id="fdr_correction"></a>

# fdr\_correction

<a id="fdr_correction.combine_chromosome"></a>

#### combine\_chromosome

```python
def combine_chromosome(outdir: str)
```

Combine all csv files fro the given directory.

**Arguments**:

  - outdir: Directory to which the mapping results have been saved.
  

**Returns**:

  - combined_df: Dataframe with data from all csv files from the folder.

<a id="fdr_correction.fdr"></a>

#### fdr

```python
def fdr(outdir: str, threshold: float)
```

Perform Benjamini Hochberg false discovery rate correction to mapping results.

**Arguments**:

  - outdir: Directory to which the mapping results have been saved.
  - threshold: Cutoff value for fdr correction.
  

**Returns**:

  - full_res: Dataframe with all mapping results including a column for fdr corrected p-values.

<a id="__init__"></a>

# plotting\_utils

<a id="plotting_utils.box_and_whisker"></a>

#### box\_and\_whisker

```python
def box_and_whisker(df: pd.DataFrame, gene_name: str, variant: str,
                    output_folder: str)
```

Create a box-and-whisker plot with significance bars and Kruskal-Wallis test for grouped data.

**Arguments**:

  - df: Dataframe containing 'GEX' and 'cur_genotypes' columns.
  - gene_name: Name of the gene that is used for the plot title and file name.
  - output_folder: Path to the folder where the plot should be saved.

<a id="segmentqtl"></a>

# statistical\_utils

<a id="statistical_utils.residualize"></a>

#### residualize

```python
def residualize(regression_data: pd.DataFrame)
```

Residualize the GEX and cur_genotypes columns by removing the variance explained by covariates.

**Arguments**:

  - regression_data: The input dataframe with GEX, cur_genotypes, and covariates.
  

**Returns**:

  - residualized_df: A dataframe with residualized GEX and cur_genotypes.

<a id="statistical_utils.get_tstat2"></a>

#### get\_tstat2

```python
def get_tstat2(corr: float, df: int)
```

Calculate t-statistic squared from correlation and degrees of freedom.

**Arguments**:

  - corr: Pearson correlation
  - df: Degrees of freedom
  

**Returns**:

  - t-statistic squared

<a id="statistical_utils.get_pvalue_from_tstat2"></a>

#### get\_pvalue\_from\_tstat2

```python
def get_pvalue_from_tstat2(tstat2: float, df: int)
```

Calculate the p-value from the t-statistic and degrees of freedom.

**Arguments**:

  - tstat2: t-statistic squared
  - df: Degrees of freedom
  

**Returns**:

  - p-value

<a id="statistical_utils.get_slope"></a>

#### get\_slope

```python
def get_slope(corr: float, phenotype_sd: np.ndarray, genotype_sd: np.ndarray)
```

Calculate the slope.

**Arguments**:

  - corr: Pearson correlation
  - phenotype_sd: Standard deviation of phenotypes
  - genotype_sd: Standard deviation of genotypes
  

**Returns**:

  - slope

<a id="statistical_utils.calculate_slope_and_se"></a>

#### calculate\_slope\_and\_se

```python
def calculate_slope_and_se(regression_data: pd.DataFrame, corr: float)
```

Calculate the slope and its standard error.

**Arguments**:

- `regression_data` - A dataframe with residualized "GEX" and "cur_genotypes" columns.
- `corr` - The correlation between residualized "GEX" and "cur_genotypes".
  

**Returns**:

- `slope` - The slope of the linear relationship.
- `slope_se` - The standard error of the slope.

<a id="statistical_utils.calculate_pvalue"></a>

#### calculate\_pvalue

```python
def calculate_pvalue(df: pd.DataFrame, corr: float)
```

Calculate the p-value using the residualized data and correlation.

**Arguments**:

- `df` - A dataframe with residualized "GEX" and "cur_genotypes" columns.
- `corr` - The correlation between residualized "GEX" and "cur_genotypes".
  

**Returns**:

- `pval` - The p-value for testing whether the slope is different from 0.

<a id="statistical_utils.calculate_beta_parameters"></a>

#### calculate\_beta\_parameters

```python
def calculate_beta_parameters(perm_p_values: np.ndarray)
```

Calculate beta parameters for the array of p-value obtained from permutations.

**Arguments**:

  - perm_p_values: Array of permutation p-values
  

**Returns**:

  Tuple of
  - Beta parameter 1
  - Beta parameter 2

<a id="statistical_utils.adjust_p_values"></a>

#### adjust\_p\_values

```python
def adjust_p_values(nominal_p_value: float, beta_shape1: float,
                    beta_shape2: float)
```

Adjust p-values for multiple comparisons.

**Arguments**:

  - nominal_p_value: The p-value from nominal pass
  - beta_shape1: Beta parameter 1
  - beta_shape2: Beta parameter 2
  

**Returns**:

  - Adjusted p-value
