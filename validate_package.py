from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent


EXPECTED_ROWS = {
    "component_ablation_runs.csv": 480,
    "interaction_runs.csv": 80,
    "equal_distance_budget_runs.csv": 180,
    "switching_control_runs.csv": 240,
    "boundary_lower_runs.csv": 15,
    "boundary_upper_runs.csv": 15,
    "sequential_timing_runs.csv": 160,
    "segmented_vs_fixed1000_runs.csv": 80,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path.relative_to(ROOT)}")
    return rows


def config_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def main() -> None:
    all_rows = []
    for name, expected in EXPECTED_ROWS.items():
        rows = read_csv(ROOT / "data" / name)
        assert len(rows) == expected, f"{name}: expected {expected} rows, found {len(rows)}"
        all_rows.extend((name, row) for row in rows)
        keys = [(row.get("instance", row.get("TSP", "")), row.get("variant", row.get("Budget", "")), row.get("run", row.get("Run", ""))) for row in rows]
        assert len(keys) == len(set(keys)), f"Duplicate run keys in {name}"

    assert sum(EXPECTED_ROWS.values()) == 1250

    for path in sorted(ROOT.rglob("*.json")):
        json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(ROOT.rglob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            parsed = list(csv.reader(handle))
        widths = Counter(len(row) for row in parsed)
        assert len(widths) == 1, f"Inconsistent CSV width in {path.relative_to(ROOT)}: {dict(widths)}"

    mapping_rows = read_csv(ROOT / "protocols" / "config_hash_mapping.csv")
    mapping = {row["recorded_config_hash"]: row for row in mapping_rows}
    referenced = {row.get("config_hash", "") for _, row in all_rows if row.get("config_hash", "")}
    assert referenced == set(mapping), "Configuration mapping does not cover every recorded config_hash"
    for row in mapping_rows:
        path = ROOT / "protocols" / "configs" / f"{row['canonical_config_hash']}.json"
        assert path.is_file(), f"Missing canonical configuration: {path.name}"
        actual = config_hash(json.loads(path.read_text(encoding="utf-8")))
        assert actual == row["canonical_config_hash"], f"Canonical configuration hash mismatch: {path.name}"

    completed = subprocess.run(
        [sys.executable, str(ROOT / "analysis" / "reproduce_supplementary_statistics.py"), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)

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

    print("PASS: 1,250 run-level records, CSV/JSON structure, configuration mapping, statistical tables, and checksums verified.")


if __name__ == "__main__":
    main()
