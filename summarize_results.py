from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize main-experiment run records.")
    parser.add_argument("runs_csv")
    parser.add_argument("summary_csv")
    args = parser.parse_args()

    with Path(args.runs_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["instance"], row["algorithm"])].append(row)

    summary = []
    for (instance, algorithm), group in sorted(groups.items()):
        values = np.asarray([float(row["best_length"]) for row in group], dtype=float)
        bks = float(group[0]["bks"])
        best = float(np.min(values))
        mean = float(np.mean(values))
        summary.append(
            {
                "instance": instance,
                "algorithm": algorithm,
                "runs": len(group),
                "bks": bks,
                "best": best,
                "mean": mean,
                "std_sample": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "best_deviation_percent": (best - bks) / bks * 100.0,
                "mean_error_percent": (mean - bks) / bks * 100.0,
            }
        )

    destination = Path(args.summary_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
