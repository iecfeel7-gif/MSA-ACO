# MSA-ACO main-experiment reproducibility package

This package contains the implementation, TSPLIB benchmark files, and result records used for the 35-instance main comparison of ACS, MMAS, and MSA-ACO. The additional controlled experiments are distributed separately in the supplementary archive.

## Scope

- 35 symmetric TSPLIB instances, in the same order as the manuscript.
- 29 `EUC_2D` instances, four `GEO` instances, and two `EXPLICIT` instances.
- ACS, MMAS, and MSA-ACO.
- 10 independent runs per instance and algorithm in the archived main comparison.
- Sequential CPU execution is the default for new runs.

## Contents

- `run_main_experiments.py`: main experiment driver.
- `summarize_results.py`: descriptive summary generator.
- `source/`: algorithm implementation and parameter rules.
- `instances/`: the 35 TSPLIB instance files.
- `data/reported_main_results_35.csv`: authoritative values reported in the manuscript.
- `data/reported_main_results_long.csv`: normalized long form of the reported table.
- `data/main_runs_reconciled.csv`: 1,050 run records after reconciliation to the manuscript.
- `data/derived_summary_from_reconciled_runs.csv`: summary recalculated from the reconciled run records.
- `data/rerun_seed_manifest.csv`: deterministic seeds reconstructed from the original experiment scripts.
- `data/instance_manifest.csv`: instance dimensions, edge-weight categories and formats, BKS values, and SHA-256 checksums.
- `provenance/correction_log.csv`: the two recovered observations and their archived values.
- `provenance/reconciliation_report.csv`: group-level comparison between archived, reconciled, and reported best values.
- `provenance/explicit_instance_validation.csv`: independent verification of the two `EXPLICIT` instances against the archived best paths.
- `validate_package.py`: structural and consistency checks for the submitted package.
- `MANIFEST.sha256`: checksums for all submitted files.

## Instance and distance conventions

The implementation follows the edge-weight type declared inside each submitted problem file. For `EUC_2D`, nonnegative Euclidean distances are converted to integer edge weights as `int(distance + 0.5)`, which is the TSPLIB nearest-integer rule. For `GEO`, coordinates are converted from the TSPLIB `DDD.MM` representation before the standard spherical distance formula with radius 6378.388 is applied. For `EXPLICIT`, the matrix is loaded directly from `EDGE_WEIGHT_SECTION`; `bayg29` uses `UPPER_ROW` and `swiss42` uses `FULL_MATRIX`. The distance matrix is constructed once before optimization and reused throughout the run.

## Data reconciliation

The manuscript's 35-instance main-results table is the authoritative report of the completed experiments. The archived long-form export contained all 1,050 expected rows, but two MSA-ACO best observations were not preserved correctly:

- `ch130.tsp`: archived minimum 6131; reported observed best 6115.
- `ts225.tsp`: archived minimum 126726; reported observed best 126643.

In `main_runs_reconciled.csv`, each recovered value is assigned to the archived row that previously held the group minimum. The original archived value remains in `archived_best_length`, and `record_provenance` identifies the recovery. The original run identifier of each recovered observation was unavailable, so the assigned run number must not be interpreted as newly recovered provenance. No other solution-quality value was changed.

Archived MSA-ACO timing values were unavailable and remain blank. New executions record initialization, solver, and end-to-end times separately; no timing values were inferred.

## Environment

Python 3.10 or later and NumPy are required. Install the dependency with:

```bash
python -m pip install -r requirements.txt
```

## Reproduce the main comparison

Run the full experiment sequentially:

```bash
python run_main_experiments.py --instances all --algorithms ACS,MMAS,MSA-ACO --runs 10
```

The full experiment is computationally intensive. Validate the installation without running the solver:

```bash
python run_main_experiments.py --dry-run
```

Run a short functional check that is not suitable for reporting:

```bash
python run_main_experiments.py --instances burma14 --algorithms MSA-ACO --runs 1 --quick
```

Summarize a completed run file:

```bash
python summarize_results.py results/rerun/runs.csv results/rerun/summary.csv
```

## Reproducibility notes

- Formal runs use the parameter rules in `source/parameters.py`.
- Baseline ACS and MMAS use uniform pheromone initialization and no local search, matching the original main-comparison protocol.
- MSA-ACO uses GA initialization, the entropy-aligned MMAS-to-ACS transition, rank-based reinforcement, and scale-dependent 2-opt/3-opt settings.
- Each output row records the seed, configuration hash, engine mode, verified tour length, and timing components.
- `EXPLICIT` instances are read directly from `EDGE_WEIGHT_SECTION` using their declared `UPPER_ROW` or `FULL_MATRIX` format.
- Random seeds are independent of execution order. With base seed 20251110, the archived ACS and MMAS runs use `seed = base + 10000*instance_index + 1000*algorithm_index + run_index`, where the algorithm index is 1 for ACS and 2 for MMAS. The archived MSA-ACO runs use the original solver rule `seed = base + 1000*instance_index + run_index - 1`.
- Seed derivation preserves the instance indices used by the original experiment; the omitted `brg180.tsp` position remains reserved so later-instance seeds do not shift.

Run the package checks with:

```bash
python validate_package.py
```
