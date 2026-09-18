# Supplementary experiment protocols

## Component-wise ablation

Eight configurations (complete MSA-ACO plus seven ablations) were evaluated on ulysses22, st70, kroC100, and kroA150 using 15 paired independent runs per configuration and instance (480 records). The full configuration is the reference in the pairwise tests. Table S14 provides the numerical definitions of all ablated variants.

The two-sided complete-versus-ablation Wilcoxon tests form one Holm family containing all 28 instance-by-ablation comparisons. The four Friedman tests are reported separately and are not included in that Holm family. Reference-mean confidence intervals are 95% percentile-bootstrap intervals based on 5,000 resamples. The random generator is initialized deterministically from the first eight bytes of the SHA-256 digest of `instance|variant`.

## Focused interaction experiment

A 2×2×2 factorial design was evaluated on st70 for GA initialization, entropy-triggered switching, and rank-based reinforcement. Each factor cell used 10 paired runs (80 records). The analysis uses final tour length as the response, and Holm correction is applied across the seven main and interaction effects for this response. The experiment examines whether the selected mechanisms display detectable non-additive effects.

## Equal distance-query-budget comparison

ACS-based+LS, MMAS-based+LS, and MSA-ACO were compared on ulysses22, st70, kroC100, and kroA150 using 15 paired runs. Per-instance distance-query budgets were 6.5×10^6, 54×10^6, 300×10^6, and 2.85×10^9, respectively (180 records). The budget is nominal and is checked at the beginning of each iteration after the first. A run may therefore exceed the nominal budget only by the queries performed in that final completed iteration. Across all 180 runs, the mean overshoot was 0.305%, the median was 0.262%, and the largest was 2.539% (st70, MMAS-based+LS). The mean overshoot was below 1% in all 12 instance-method combinations.

The eight MSA-ACO-versus-comparator tests form one Holm family.

## Switching-control comparison

Entropy-triggered switching (complete MSA-ACO), fixed switching, random switching, and disabled switching were evaluated on the same four instances using 15 paired runs per strategy and instance (240 records). Table S13 reports the realized switching iteration and switching reason for each instance and strategy.

The switching-reason categories are defined as follows:

- `entropy_trigger`: the switch was activated by the entropy condition within the admissible switching window.
- `forced_at_window_end`: no entropy trigger occurred before the end of the switching window, and the switch was forced at the final admissible iteration.
- `fixed_control`: the switch was assigned to the fixed scale-specific center iteration.
- `random_control`: the switch iteration was sampled uniformly from the admissible switching window.
- `disabled_control`: stage switching was disabled.

Across the 15 paired runs, entropy-triggered switching occurred within iterations 47-56 on ulysses22, 47-58 on st70, and 77-97 on kroC100. On kroA150, all 15 MSA-ACO runs reached the end of the switching window and switched at iteration 135 through `forced_at_window_end`. Therefore, the kroA150 results for the complete method characterize the full switching policy rather than the entropy trigger specifically. After Holm correction, the entropy-triggered rule did not significantly outperform fixed switching on any instance.

Table S15 examines the association between the realized switch iteration and final tour length for the complete MSA-ACO policy. Only st70 varies in both quantities. Its Spearman correlation is -0.410 with an exact two-sided multiset-permutation p-value of 0.141, so the data do not establish a significant monotonic association. The association is not estimable for ulysses22 or kroC100 because final tour length is constant, or for kroA150 because switch iteration is constant. The 12 MSA-ACO-versus-control tests in Table S8 form one Holm family.

## Boundary sensitivity

The lower internal scale boundary was varied among 100, 120 (default), and 140 on lin105. The upper boundary was varied among 140, 170 (default), and 200 on ch150. Each setting used five paired runs (30 records in total).

The four default-versus-alternative comparisons form one Holm family. Because only one alternative per instance changes the active scale regime and each cell contains five paired runs, this experiment is interpreted as a local sensitivity check within the evaluated benchmark set. It does not externally validate the selected boundaries or establish general transferability to unseen instances.

## Sequential timing

The complete MSA-ACO and its seven ablated configurations were executed sequentially under the same frozen implementation, hardware environment, and single-thread setting. Each configuration was timed independently five times on ulysses22, st70, kroC100, and kroA150 (160 records). Wall-clock time was measured from initialization to termination. All timing runs were executed under the Windows environment recorded in `execution_environment.json`, with single-thread limits.

## Segmented-budget validation

The scale-segmented iteration limits (150, 300, 600, and 600 for eil76, ch150, kroA200, and ts225) were compared with a fixed 1000-iteration limit using 10 paired runs for each budget and instance (80 records).

These records predate the retained environment record. Their exact operating-system, processor, Python, and thread metadata were not retained, so the archived runtime values are not attributed to either the Ubuntu main-experiment platform or the Windows additional-experiment platform.

## Statistical implementation

All inferential tables are reproducible with `analysis/reproduce_supplementary_statistics.py`. Wilcoxon tests use a two-sided exact conditional sign-permutation distribution, discard zero differences before ranking, and assign average ranks to tied absolute differences. Rank-biserial correlation is `(W_plus - W_minus)/(W_plus + W_minus)`. Friedman tests use within-block average ranks and a tie-corrected chi-square statistic. Table S4 applies Holm correction across seven factorial effects; Tables S2, S6, S8, and S10 apply Holm correction across 28, 8, 12, and 4 comparisons, respectively.

The exact JSON parameter objects referenced by the run-level `config_hash` fields are included in `protocols/configs/`. `protocols/config_hash_mapping.csv` records both the original run identifier and the canonical SHA-256-derived content hash. The 12 equal-budget records use legacy identifiers that differ from the canonical hashes; the mapping preserves their association with the included configurations without changing any experimental outcome.
