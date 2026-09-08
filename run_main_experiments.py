from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np


PACKAGE = Path(__file__).resolve().parent
SOURCE = PACKAGE / "source"
sys.path.insert(0, str(SOURCE))

from ga_initialization import ga_init_pheromone
from msa_aco import optimized_mmas_algorithm
from parameters import BKS_MAP, apply_common_overrides, apply_segment_overrides, build_base_params, size_segment
from tsp_io import read_tsp_file
from utils import calculate_path_length


ALGORITHMS = ("ACS", "MMAS", "MSA-ACO")
ALGORITHM_INDEX = {"ACS": 1, "MMAS": 2, "MSA-ACO": 3}
BASE_SEED = 20251110
ORIGINAL_INSTANCE_INDEX = {
    name: index + (1 if index >= 25 else 0)
    for index, name in enumerate(BKS_MAP, start=1)
}


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def select_instances(value: str) -> list[str]:
    if value.lower() == "all":
        return list(BKS_MAP)
    result = []
    for item in parse_list(value):
        result.append(item if item.lower().endswith(".tsp") else f"{item}.tsp")
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


def run_seed(base_seed: int, instance_index: int, algorithm: str, run_index: int) -> tuple[int, str]:
    if algorithm == "MSA-ACO":
        return base_seed + instance_index * 1000 + run_index - 1, "base+instance_index*1000+run_index-1"
    seed = base_seed + instance_index * 10000 + ALGORITHM_INDEX[algorithm] * 1000 + run_index
    return seed, "base+instance_index*10000+algorithm_index*1000+run_index"


def uniform_pheromone(n: int, tau_min: float) -> np.ndarray:
    pheromone = np.full((n, n), float(tau_min), dtype=float)
    np.fill_diagonal(pheromone, 0.0)
    return pheromone


def quick_overrides(params: dict[str, Any]) -> None:
    params.update(
        {
            "numAnts": min(10, int(params["numAnts"])),
            "maxIterations": 8,
            "gaPopSize": 20,
            "gaEarlyStopPatience": 3,
            "gaMaxGensSafety": 8,
            "twoOptMaxIter": 8,
            "threeOptMaxIter": 5,
            "ls_threeopt_topk": 1,
            "hybrid_min_iter": 2,
            "hybrid_force_iter": 4,
            "hybrid_stall_iters": 0,
            "hybrid_align_use_stall": False,
        }
    )


def configure(n: int, algorithm: str, quick: bool) -> dict[str, Any]:
    params = build_base_params(n)
    if algorithm == "MSA-ACO":
        apply_common_overrides(params, num_cities=n)
        apply_segment_overrides(params, num_cities=n)
        params.update(
            {
                "mode": "hybrid_v4_2",
                "gaInitializationEnabled": True,
                "localSearchEnabled": True,
                "localSearchPolicy": "matched",
                "rankUpdateEnabled": True,
                "twoOptEnabled": True,
                "threeOptEnabled": True,
                "adaptiveThreeOptEnabled": True,
                "scaleAwareEnabled": True,
                "hybrid_switch_rule": "aligned_tau_entropy",
            }
        )
    else:
        params.update(
            {
                "mode": "acs_end_seq" if algorithm == "ACS" else "mmas",
                "gaInitializationEnabled": False,
                "localSearchEnabled": False,
                "threeOptMaxIter": 20,
            }
        )
    if quick:
        quick_overrides(params)
    return params


def config_hash(params: dict[str, Any]) -> str:
    portable = {key: value for key, value in params.items() if not key.startswith("_")}
    payload = json.dumps(portable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def existing_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {(row["instance"], row["algorithm"], int(row["run"])) for row in csv.DictReader(handle)}


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_one(
    instance: str,
    instance_index: int,
    algorithm: str,
    run_index: int,
    base_seed: int,
    output: Path,
    quick: bool,
    verbose: bool,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    distance, n, distance_type, _ = read_tsp_file(PACKAGE / "instances" / instance, prefer_gpu=False)
    params = configure(n, algorithm, quick)
    seed, seed_scheme = run_seed(base_seed, instance_index, algorithm, run_index)
    set_seed(seed)

    initialization_start = time.perf_counter()
    if algorithm == "MSA-ACO":
        pheromone, ga_best, _ = ga_init_pheromone(distance, dict(params))
    else:
        pheromone = uniform_pheromone(n, float(params["tauMin"]))
        ga_best = ""
    initialization_seconds = time.perf_counter() - initialization_start

    solver_start = time.perf_counter()
    output_stream = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with output_stream:
        best_path, best_length, _ = optimized_mmas_algorithm(distance, pheromone.copy(), params)
    solver_seconds = time.perf_counter() - solver_start

    verified_length = float(calculate_path_length(best_path, distance))
    if len(best_path) != n or len(set(best_path)) != n or abs(verified_length - float(best_length)) > 1e-6:
        raise RuntimeError(f"Invalid result for {instance}, {algorithm}, run {run_index}.")

    path_dir = output / "best_paths" / Path(instance).stem
    path_dir.mkdir(parents=True, exist_ok=True)
    path_file = path_dir / f"{algorithm}_run{run_index:02d}.csv"
    path_file.write_text(",".join(str(int(city)) for city in best_path) + "\n", encoding="utf-8")
    bks = BKS_MAP[instance]
    return {
        "instance": instance,
        "n": n,
        "distance_type": distance_type,
        "algorithm": algorithm,
        "run": run_index,
        "seed": seed,
        "seed_scheme": seed_scheme,
        "segment": size_segment(n),
        "engine_mode": params["mode"],
        "config_hash": config_hash(params),
        "best_length": float(best_length),
        "bks": bks,
        "deviation_percent": (float(best_length) - bks) / bks * 100.0,
        "ga_best_length": ga_best,
        "initialization_seconds": initialization_seconds,
        "solver_seconds": solver_seconds,
        "end_to_end_seconds": time.perf_counter() - total_start,
        "best_path_file": path_file.relative_to(output).as_posix(),
        "quick_validation_only": quick,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 35-instance main comparison for ACS, MMAS, and MSA-ACO.")
    parser.add_argument("--instances", default="all", help="all or a comma-separated list")
    parser.add_argument("--algorithms", default=",".join(ALGORITHMS), help="comma-separated list")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--output", default=str(PACKAGE / "results" / "rerun"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true", help="short validation run; not a formal experiment")
    parser.add_argument("--dry-run", action="store_true", help="validate selections without running the solver")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be positive")
    instances = select_instances(args.instances)
    algorithms = parse_list(args.algorithms)
    unknown_algorithms = [algorithm for algorithm in algorithms if algorithm not in ALGORITHMS]
    missing_instances = [instance for instance in instances if instance not in BKS_MAP or not (PACKAGE / "instances" / instance).is_file()]
    if unknown_algorithms:
        raise ValueError(f"Unknown algorithms: {', '.join(unknown_algorithms)}")
    if missing_instances:
        raise FileNotFoundError(f"Unknown or missing instances: {', '.join(missing_instances)}")

    output = Path(args.output).resolve()
    plan = {
        "instances": instances,
        "algorithms": algorithms,
        "runs_per_instance_algorithm": args.runs,
        "base_seed": args.base_seed,
        "execution": "sequential single-process CPU",
        "quick_validation_only": bool(args.quick),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    output.mkdir(parents=True, exist_ok=True)
    (output / "run_manifest.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    runs_file = output / "runs.csv"
    completed = existing_keys(runs_file) if args.resume else set()
    for instance in instances:
        instance_index = ORIGINAL_INSTANCE_INDEX[instance]
        for run_index in range(1, args.runs + 1):
            for algorithm in algorithms:
                key = (instance, algorithm, run_index)
                if key in completed:
                    continue
                row = run_one(instance, instance_index, algorithm, run_index, args.base_seed, output, args.quick, args.verbose)
                append_row(runs_file, row)
                print(f"{instance} | {algorithm} | run {run_index} | best={row['best_length']:.0f}")


if __name__ == "__main__":
    main()
