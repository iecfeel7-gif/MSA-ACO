from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLES = ROOT / "tables_csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"No rows generated for {path.name}")
    names = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def gammaincc(a: float, x: float) -> float:
    if x <= 0:
        return 1.0
    eps, tiny, iterations = 3e-14, 1e-300, 500
    if x < a + 1.0:
        term = total = 1.0 / a
        ap = a
        for _ in range(iterations):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * eps:
                break
        lower = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return max(0.0, min(1.0, 1.0 - lower))
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, iterations + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return max(0.0, min(1.0, math.exp(-x + a * math.log(x) - math.lgamma(a)) * h))


def exact_wilcoxon(reference: Sequence[float], comparator: Sequence[float]) -> dict[str, float | int]:
    differences = np.asarray(reference, dtype=float) - np.asarray(comparator, dtype=float)
    differences = differences[np.abs(differences) > 1e-12]
    if not len(differences):
        return {"nonzero": 0, "w_plus": 0.0, "w_minus": 0.0, "p": 1.0, "effect": 0.0}
    ranks = rankdata(np.abs(differences))
    w_plus = float(ranks[differences > 0].sum())
    w_minus = float(ranks[differences < 0].sum())
    scaled = np.rint(2.0 * ranks).astype(int)
    distribution = np.zeros(int(scaled.sum()) + 1, dtype=np.int64)
    distribution[0] = 1
    current_max = 0
    for rank in scaled:
        distribution[rank:current_max + rank + 1] += distribution[:current_max + 1]
        current_max += int(rank)
    observed = int(round(2.0 * w_plus))
    total_assignments = float(2 ** len(differences))
    lower = float(distribution[:observed + 1].sum()) / total_assignments
    upper = float(distribution[observed:].sum()) / total_assignments
    p_value = min(1.0, 2.0 * min(lower, upper))
    total_rank = w_plus + w_minus
    effect = (w_plus - w_minus) / total_rank if total_rank else 0.0
    return {"nonzero": len(differences), "w_plus": w_plus, "w_minus": w_minus, "p": p_value, "effect": effect}


def holm(records: list[dict[str, object]], p_key: str = "Raw p") -> None:
    order = sorted(range(len(records)), key=lambda index: float(records[index][p_key]))
    running = 0.0
    count = len(records)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * float(records[index][p_key])))
        records[index]["Holm-adjusted p"] = running


def deterministic_bootstrap_ci(values: Sequence[float], label: str) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    seed_bytes = hashlib.sha256(label.encode("utf-8")).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
    means = rng.choice(array, size=(5000, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def grouped_values(rows: list[dict[str, str]], value: str = "best_length") -> dict[tuple[str, str], dict[int, float]]:
    result: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        result[(row["instance"], row["variant"])][int(row["run"])] = float(row[value])
    return result


def generate_s1(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    variants = ["MSA-ACO", "MSA-no-GA-init", "MSA-fixed-switch", "MSA-no-rank", "MSA-no-2opt", "MSA-no-3opt", "MSA-nonadaptive-3opt", "MSA-no-scale-adaptation"]
    values = grouped_values(rows)
    output = []
    for instance in ["ulysses22.tsp", "st70.tsp", "kroC100.tsp", "kroA150.tsp"]:
        run_ids = sorted(set.intersection(*(set(values[(instance, variant)]) for variant in variants)))
        matrix = np.asarray([[values[(instance, variant)][run] for variant in variants] for run in run_ids])
        if np.all(matrix == matrix[:, :1]):
            output.append({"Instance": instance.removesuffix(".tsp"), "Friedman chi-square": "", "df": "", "p-value": "", "Kendall's W": "", "Interpretation": "Not testable: all configurations are identical"})
            continue
        ranks = np.vstack([rankdata(row) for row in matrix])
        n, k = ranks.shape
        rank_sums = ranks.sum(axis=0)
        statistic = 12.0 / (n * k * (k + 1)) * float(np.sum(rank_sums ** 2)) - 3.0 * n * (k + 1)
        tie_sum = 0.0
        for row in matrix:
            _, counts = np.unique(row, return_counts=True)
            tie_sum += float(np.sum(counts ** 3 - counts))
        correction = 1.0 - tie_sum / (n * (k ** 3 - k))
        statistic /= correction
        p_value = gammaincc((k - 1) / 2.0, statistic / 2.0)
        output.append({"Instance": instance.removesuffix(".tsp"), "Friedman chi-square": statistic, "df": k - 1, "p-value": p_value, "Kendall's W": statistic / (n * (k - 1)), "Interpretation": "Significant omnibus difference" if p_value < 0.05 else "No significant omnibus difference"})
    return output


def paired_table(rows: list[dict[str, str]], reference: str, comparators: list[str], family_order: list[str] | None = None) -> list[dict[str, object]]:
    values = grouped_values(rows)
    instances = family_order or sorted({row["instance"] for row in rows})
    output: list[dict[str, object]] = []
    for instance in instances:
        for comparator in comparators:
            reference_runs = values[(instance, reference)]
            comparator_runs = values[(instance, comparator)]
            common = sorted(set(reference_runs) & set(comparator_runs))
            ref = [reference_runs[run] for run in common]
            cmp = [comparator_runs[run] for run in common]
            result = exact_wilcoxon(ref, cmp)
            differences = np.asarray(ref) - np.asarray(cmp)
            output.append({
                "Instance": instance.removesuffix(".tsp"),
                "Reference": reference,
                "Comparator": comparator,
                "Paired runs": len(common),
                "Reference better": int(np.sum(differences < -1e-12)),
                "Ties": int(np.sum(np.abs(differences) <= 1e-12)),
                "Reference worse": int(np.sum(differences > 1e-12)),
                "Raw p": result["p"],
                "Holm-adjusted p": 1.0,
                "Rank-biserial correlation": result["effect"],
            })
    holm(output)
    return output


def generate_s2(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    comparators = ["MSA-no-GA-init", "MSA-fixed-switch", "MSA-no-rank", "MSA-no-2opt", "MSA-no-3opt", "MSA-nonadaptive-3opt", "MSA-no-scale-adaptation"]
    values = grouped_values(rows)
    output: list[dict[str, object]] = []
    for instance in ["ulysses22.tsp", "st70.tsp", "kroC100.tsp", "kroA150.tsp"]:
        ref_map = values[(instance, "MSA-ACO")]
        ref_values = [ref_map[run] for run in sorted(ref_map)]
        ci_low, ci_high = deterministic_bootstrap_ci(ref_values, f"{instance}|MSA-ACO")
        for comparator in comparators:
            cmp_map = values[(instance, comparator)]
            common = sorted(set(ref_map) & set(cmp_map))
            ref = [ref_map[run] for run in common]
            cmp = [cmp_map[run] for run in common]
            result = exact_wilcoxon(ref, cmp)
            output.append({
                "Instance": instance.removesuffix(".tsp"),
                "Reference": "MSA-ACO",
                "Ablated configuration": comparator,
                "Paired runs": len(common),
                "Mean difference (reference-ablation)": float(np.mean(np.asarray(ref) - np.asarray(cmp))),
                "Wilcoxon p": result["p"],
                "Holm-adjusted p": 1.0,
                "Rank-biserial correlation": result["effect"],
                "Reference mean 95% CI": f"[{ci_low:.2f}, {ci_high:.2f}]",
            })
    holm(output, "Wilcoxon p")
    return output


def generate_s4(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    effects = {"GA": (0,), "SW": (1,), "RANK": (2,), "GA x SW": (0, 1), "GA x RANK": (0, 2), "SW x RANK": (1, 2), "GA x SW x RANK": (0, 1, 2)}
    by_run: dict[int, list[tuple[tuple[int, int, int], float]]] = defaultdict(list)
    for row in rows:
        variant = row["variant"]
        levels = tuple(1 if f"{name}1" in variant else -1 for name in ("GA", "SW", "RANK"))
        by_run[int(row["run"])].append((levels, float(row["best_length"])))
    contrasts: dict[str, list[float]] = defaultdict(list)
    for run, cells in sorted(by_run.items()):
        assert len(cells) == 8, f"Incomplete factorial run {run}"
        for effect, indices in effects.items():
            positive = [value for levels, value in cells if math.prod(levels[index] for index in indices) > 0]
            negative = [value for levels, value in cells if math.prod(levels[index] for index in indices) < 0]
            contrasts[effect].append(float(np.mean(positive) - np.mean(negative)))
    output = []
    for effect in effects:
        values = contrasts[effect]
        result = exact_wilcoxon(values, np.zeros(len(values)))
        mean_contrast = float(np.mean(values))
        output.append({"Effect": effect, "Final-length contrast": mean_contrast, "Direction": ("Negative main-effect contrast" if len(effects[effect]) == 1 and mean_contrast < 0 else "Positive main-effect contrast" if len(effects[effect]) == 1 else "Negative interaction contrast" if mean_contrast < 0 else "Positive interaction contrast"), "Paired runs": len(values), "Raw p": result["p"], "Holm-adjusted p": 1.0})
    holm(output)
    return output


def multiset_permutations(values: Sequence[float]) -> Iterable[tuple[float, ...]]:
    counts = Counter(values)
    keys = sorted(counts)
    current: list[float] = []
    def visit() -> Iterable[tuple[float, ...]]:
        if len(current) == len(values):
            yield tuple(current)
            return
        for key in keys:
            if counts[key] == 0:
                continue
            counts[key] -= 1
            current.append(key)
            yield from visit()
            current.pop()
            counts[key] += 1
    yield from visit()


def generate_s15(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output = []
    for instance in ["ulysses22.tsp", "st70.tsp", "kroC100.tsp", "kroA150.tsp"]:
        group = sorted((row for row in rows if row["instance"] == instance and row["variant"] == "MSA-ACO"), key=lambda row: int(row["run"]))
        switch = np.asarray([float(row["switch_iteration"]) for row in group])
        final = np.asarray([float(row["best_length"]) for row in group])
        if float(np.std(switch)) == 0.0:
            rho: float | str = ""
            p_value: float | str = ""
            interpretation = "Not estimable: switching iteration is constant"
        elif float(np.std(final)) == 0.0:
            rho = ""
            p_value = ""
            interpretation = "Not estimable: final tour length is constant"
        else:
            switch_ranks = rankdata(switch)
            final_ranks = rankdata(final)
            rho = float(np.corrcoef(switch_ranks, final_ranks)[0, 1])
            extreme = total = 0
            for permutation in multiset_permutations(final.tolist()):
                permuted_rho = float(np.corrcoef(switch_ranks, rankdata(permutation))[0, 1])
                total += 1
                if abs(permuted_rho) >= abs(rho) - 1e-12:
                    extreme += 1
            p_value = extreme / total
            interpretation = "No statistically significant monotonic association"
        output.append({"Instance": instance.removesuffix(".tsp"), "Runs": len(group), "Minimum switch iteration": int(np.min(switch)), "Median switch iteration": float(np.median(switch)), "Maximum switch iteration": int(np.max(switch)), "Unique final tour lengths": len(set(final.tolist())), "Spearman rho": rho, "Exact two-sided permutation p": p_value, "Interpretation": interpretation})
    return output


def normalized(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    return [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]


def compare_csv(path: Path, generated: list[dict[str, object]], tolerance: float = 1e-9) -> list[str]:
    existing = read_csv(path)
    wanted = normalized(generated)
    errors = []
    if len(existing) != len(wanted):
        return [f"{path.name}: row count {len(existing)} != {len(wanted)}"]
    if list(existing[0]) != list(wanted[0]):
        return [f"{path.name}: header mismatch"]
    for row_number, (left, right) in enumerate(zip(existing, wanted), start=2):
        for key in left:
            a, b = left[key], right[key]
            try:
                if a == "" and b == "":
                    continue
                if not math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance):
                    errors.append(f"{path.name}:{row_number}:{key}: {a} != {b}")
            except ValueError:
                if a != b:
                    errors.append(f"{path.name}:{row_number}:{key}: {a!r} != {b!r}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate or check the inferential supplementary tables.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate the statistical CSV tables")
    mode.add_argument("--check", action="store_true", help="verify the submitted statistical CSV tables")
    args = parser.parse_args()

    component = read_csv(DATA / "component_ablation_runs.csv")
    interaction = read_csv(DATA / "interaction_runs.csv")
    equal_budget = read_csv(DATA / "equal_distance_budget_runs.csv")
    switching = read_csv(DATA / "switching_control_runs.csv")
    boundary = read_csv(DATA / "boundary_lower_runs.csv") + read_csv(DATA / "boundary_upper_runs.csv")
    instance_order = ["ulysses22.tsp", "st70.tsp", "kroC100.tsp", "kroA150.tsp"]

    generated = {
        "S1_component_friedman.csv": generate_s1(component),
        "S2_component_wilcoxon.csv": generate_s2(component),
        "S4_interaction_effects.csv": generate_s4(interaction),
        "S6_equal_budget_tests.csv": paired_table(equal_budget, "MSA-ACO", ["ACS-based+LS", "MMAS-based+LS"], instance_order),
        "S8_switching_control_tests.csv": paired_table(switching, "MSA-ACO", ["MSA-fixed-switch", "MSA-random-switch", "MSA-no-switch"], instance_order),
        "S10_boundary_tests.csv": paired_table(boundary, "MSA-boundaries-120-170", ["MSA-boundaries-100-170", "MSA-boundaries-140-170"], ["lin105.tsp"]) + paired_table(boundary, "MSA-boundaries-120-170", ["MSA-boundaries-120-140", "MSA-boundaries-120-200"], ["ch150.tsp"]),
        "S15_switch_quality_association.csv": generate_s15(switching),
    }
    holm(generated["S10_boundary_tests.csv"])

    if args.write:
        for name, rows in generated.items():
            write_csv(TABLES / name, rows)
    else:
        errors = []
        for name, rows in generated.items():
            errors.extend(compare_csv(TABLES / name, rows))
        if errors:
            print("\n".join(errors))
            raise SystemExit(1)

    metadata = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "wilcoxon": "two-sided exact conditional sign permutation; zero differences removed; tied absolute differences use average ranks",
        "friedman": "custom rank implementation with within-block tie correction; chi-square tail from a deterministic incomplete-gamma routine",
        "holm_families": {"S2": 28, "S4": 7, "S6": 8, "S8": 12, "S10": 4},
        "confidence_intervals": "percentile bootstrap of the mean; 5000 resamples; SHA-256-derived deterministic seed from instance|variant",
        "switch_quality": "Spearman rho; exact two-sided permutation over the observed multiset of final tour lengths when both variables vary",
    }
    metadata_path = ROOT / "protocols" / "statistical_implementation.json"
    if args.write:
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        submitted_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if submitted_metadata != metadata:
            print("statistical_implementation.json does not match the active analysis environment")
            raise SystemExit(1)
    print("PASS" if args.check else "WROTE", ", ".join(generated))


if __name__ == "__main__":
    main()
