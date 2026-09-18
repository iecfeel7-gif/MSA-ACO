# MSA-ACO supplementary material

This archive contains the additional controlled experiments supporting the supplementary analyses. The 35-instance main comparison and the main solver package are distributed separately through the public repository identified in the manuscript.

## Contents

- `data/`: run-level records for the seven additional experiment groups.
- `tables_csv/`: supplementary tables and their numbering map.
- `protocols/EXPERIMENT_PROTOCOLS.md`: experiment designs, stopping rules, statistical families, and interpretation notes.
- `protocols/execution_environment.json`: detailed Windows workstation record used for the sequential timing experiment.
- `protocols/experiment_suite_environments.csv`: experiment-by-experiment environment mapping, including Python-version differences across batches.
- `protocols/configs/`: canonical JSON configurations referenced by `config_hash` in the run-level files.
- `protocols/config_hash_mapping.csv`: mapping from recorded configuration identifiers to the canonical content hashes of the included JSON files. Twelve equal-budget identifiers are retained as legacy run identifiers and are mapped explicitly in this file.
- `analysis/reproduce_supplementary_statistics.py`: deterministic implementation of the reported Friedman, Wilcoxon, Holm, bootstrap, and switch-quality analyses.
- `validate_package.py`: package-wide record-count, CSV/JSON, configuration, statistical-table, and checksum validation.
- `MANIFEST.sha256`: SHA-256 checksums for every other file in this archive.

## Reproduce the statistical tables

Python 3.12 or later and NumPy are required. From the archive root, run:

```bash
python analysis/reproduce_supplementary_statistics.py --check
python validate_package.py
```

Use `--write` only when intentionally regenerating the statistical CSV tables from the submitted run-level records.

## Data notes

- `MSA-ACO` is the final name of the frozen `hybrid_v4_2` implementation. The historical `GSH-ACO` label retained in the segmented-budget source is documented in `tables_csv/algorithm_name_mapping.csv`.
- The recorded equal-budget configuration identifiers were produced by the original budget-run driver and do not equal the canonical content hashes of the archived JSON parameter objects. Both identifiers are preserved in `protocols/config_hash_mapping.csv`; the mapping does not alter any solution-quality or effort value.
- The exact environment of the historical segmented-versus-fixed-1000 experiment was not retained. No Windows or Ubuntu timing claim is assigned to that suite.

## Statistical conventions

- Paired tests use identical run indices and seeds within each instance.
- Two-sided Wilcoxon tests use an exact conditional sign-permutation distribution. Zero differences are removed and tied absolute differences receive average ranks.
- Rank-biserial effects use `reference minus comparator`; negative values favor the reference because shorter tours are better.
- Holm correction families are enumerated in `protocols/EXPERIMENT_PROTOCOLS.md`.
- Mean confidence intervals use 5,000 percentile-bootstrap resamples with a deterministic SHA-256-derived seed based on the instance and configuration labels.
