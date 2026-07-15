# 5hmC2Stage

**5hmC2Stage** is a two-stage framework for cell-group-specific
5-hydroxymethylcytosine (5hmC) prediction from paired single-cell
methylome data.

Stage 1 learns a continuous site-level Propensity Score from
CpG-centered sequence, genomic context and transcription-factor motif
features. Stage 2 integrates this score with cell-group-specific 5mC
features to estimate the probability of a high-5hmC state for each CpG
site within each cell group.

## Repository structure

```text
5hmC2Stage/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── P0_data_prepare.py
│   ├── P1_build_long_table_and_labels.py
│   ├── P1b_export_continuous_signal.py
│   ├── P2_extract_cpg_sequences.py
│   ├── P2a_build_sequence_features.py
│   ├── P2b_build_genomic_context.py
│   ├── P2c_build_motif_features.py
│   ├── P3a_build_site_5mc_features.py
│   ├── P3b_build_promoter_5mc_features.py
│   ├── P4_build_stage1_dataset.py
│   ├── P4_build_stage2_dataset.py
│   ├── P5_train_stage1.py
│   ├── P5_generate_stage1_oof_scores.py
│   ├── P5_train_stage2.py
│   ├── P5_train_matched_baselines.py
│   ├── P6_train_deep5hmc_inspired_comparator.py
│   ├── P6_evaluate_regional_comparison.py
│   └── summarize_cell_group_performance.py
├── examples/
│   ├── stage1_input_example.csv
│   └── stage2_input_example.csv
└── results/
    ├── Table 1/
    ├── Table 2/
    ├── Table 3/
    └── Table 4/
```

The repository contains the analysis scripts, small example inputs and
compact result summaries supporting the manuscript tables. Raw
sequencing data, reference genomes, complete feature matrices, trained
models and large prediction files are not included.

## Data

The paired single-cell 5mC and 5hmC datasets analyzed in the manuscript
were obtained from the Gene Expression Omnibus under accession
**GSE197740**.

Reference genome assemblies:

- Mouse Brain: mm10 / GRCm38
- Human PBMC: hg38 / GRCh38

The required reference genomes, gene annotations and motif resources
must be downloaded separately. Local file paths are supplied through
the corresponding script arguments or dataset-specific settings.

## Installation

The code was developed with **Python 3.9.1**.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

The primary manuscript models use CatBoost 1.2.8 for Stage 1 and
LightGBM 4.6.0 for Stage 2. PyTorch is required for the
Deep5hmC-inspired regional comparator.

## Workflow

The scripts are organized in execution order.

### Data preparation and label construction

```text
P0_data_prepare.py
P1_build_long_table_and_labels.py
P1b_export_continuous_signal.py
```

These scripts prepare cell-group-level 5mC/5hmC tables, construct
cell-group-specific labels and export continuous signals.

### Feature construction

```text
P2_extract_cpg_sequences.py
P2a_build_sequence_features.py
P2b_build_genomic_context.py
P2c_build_motif_features.py
P3a_build_site_5mc_features.py
P3b_build_promoter_5mc_features.py
```

These scripts generate CpG-centered sequence features, genomic-context
features, motif-derived features, site-level 5mC features and
promoter-level 5mC features.

### Model input construction

```text
P4_build_stage1_dataset.py
P4_build_stage2_dataset.py
```

These scripts construct the final Stage-1 and Stage-2 input tables.

### Stage 1 and out-of-fold Propensity Scores

```text
P5_train_stage1.py
P5_generate_stage1_oof_scores.py
```

The manuscript uses CatBoost as the primary Stage-1 learner. Genomic
splits are defined using non-overlapping 10-kb blocks. Propensity Scores
used by Stage 2 are generated through block-level out-of-fold
prediction.

### Stage 2

```text
P5_train_stage2.py
```

Stage 2 integrates the Propensity Score with site-level 5mC,
promoter-level 5mC, relative and coupling features, and cell-group
information.

### Matched baselines and regional comparison

```text
P5_train_matched_baselines.py
P6_train_deep5hmc_inspired_comparator.py
P6_evaluate_regional_comparison.py
```

These scripts reproduce the matched one-stage baseline comparison and
the regional-scale comparison reported in the manuscript.

Run any script with `--help` to inspect its available arguments, for
example:

```bash
python src/P5_train_stage1.py --help
```

## Example

The `examples/` directory contains small subsets of the Brain input
tables. These files demonstrate the expected input format and execution
process and are not intended to reproduce the manuscript performance.

Generate example block-level out-of-fold Propensity Scores with:

```bash
python src/P5_generate_stage1_oof_scores.py --stage1_csv examples/stage1_input_example.csv --score_csv examples/stage1_input_example.csv --out_csv examples/stage1_oof_scores_example.csv --out_dir examples/oof_output --models cat --score_output cat --folds 5 --seed 42 --block_bp 10000
```

The command generates:

```text
examples/stage1_oof_scores_example.csv
examples/oof_output/stage1_oof_metrics.csv
examples/oof_output/stage1_oof_summary.csv
```

Because the example dataset is small, its AUC and AP values should not
be compared with the manuscript results.

## Results

The `results/` directory contains compact CSV files corresponding to the
main manuscript tables:

- `Table 1/`: Stage-2 baseline and full 5hmC2Stage performance
- `Table 2/`: cell-group-level performance
- `Table 3/`: matched one-stage baseline comparison
- `Table 4/`: regional comparison after 1-kb aggregation

Complete predictions, trained models and large intermediate files are
stored separately from this GitHub repository.

## Reproducibility notes

- Data splitting is performed using non-overlapping 10-kb genomic
  blocks.
- Stage-1 Propensity Scores used by Stage 2 are generated with
  block-level out-of-fold prediction.
- Repeated experiments use the random seeds described in the manuscript
  and supplementary materials.
- Local absolute paths should be replaced with valid local paths or
  command-line arguments before execution.
- External reference genomes, annotations and motif databases are not
  distributed in this repository.

## Citation

Citation information and the archived Zenodo DOI will be added when the
manuscript record and software archive are available.
