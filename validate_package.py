from __future__ import annotations

import ast
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_csv(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    instance_files = sorted((ROOT / "instances").glob("*.tsp"))
    assert len(instance_files) == 35, f"Expected 35 problem files, found {len(instance_files)}"

    reported = read_csv("data/reported_main_results_long.csv")
    runs = read_csv("data/main_runs_archived.csv")
    summary = read_csv("data/derived_summary_from_archived_runs.csv")
    seeds = read_csv("data/rerun_seed_manifest.csv")

    assert len(reported) == 105
    assert len(runs) == 1050
    assert len(summary) == 105
    assert len(seeds) == 1050
    assert Counter(row["algorithm"] for row in runs) == {"ACS": 350, "MMAS": 350, "MSA-ACO": 350}

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in runs:
        groups[(row["instance"], row["algorithm"])].append(row)
    assert all(len(group) == 10 for group in groups.values())
    summary_lookup = {(row["instance"], row["algorithm"]): row for row in summary}
    for key, group in groups.items():
        values = [float(row["best_length"]) for row in group]
        archived_summary = summary_lookup[key]
        assert min(values) == float(archived_summary["best"])
        assert abs(sum(values) / len(values) - float(archived_summary["mean"])) < 1e-9
    seed_lookup = {(row["instance"], row["algorithm"], int(row["run"])): int(row["derived_seed"]) for row in seeds}
    for row in seeds:
        algorithm = row["algorithm"]
        scheme = row["seed_scheme"]
        if algorithm == "MSA-ACO":
            assert scheme == "base+instance_index*1000+run_index-1"
        else:
            assert scheme == "base+instance_index*10000+algorithm_index*1000+run_index"
    assert len(seed_lookup) == 1050

    for path in [ROOT / "run_main_experiments.py", ROOT / "summarize_results.py", *(ROOT / "source").glob("*.py")]:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    manifest = {}
    with (ROOT / "MANIFEST.sha256").open("r", encoding="utf-8") as handle:
        for line in handle:
            digest, relative_path = line.rstrip("\n").split("  ", 1)
            manifest[relative_path] = digest
    expected_files = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256" and "__pycache__" not in path.parts
    )
    assert sorted(manifest) == expected_files, "Manifest file list does not match package contents"
    for relative_path, expected_digest in manifest.items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected_digest, f"Checksum mismatch: {relative_path}"

    print("PASS: package structure, 1,050 archived runs, derived summaries, seed schemes, source syntax, and checksums verified.")


if __name__ == "__main__":
    main()
