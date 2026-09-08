"""Tour-length accounting and candidate-restricted 2-opt local search."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def three_opt_delta(
    path: List[int], i: int, j: int, k: int, variant: int, distance_matrix: np.ndarray
) -> float:
    """Return the O(1) length change for one symmetric 3-opt reconnection."""
    a, b = int(path[i]), int(path[i + 1])
    c, d = int(path[j]), int(path[j + 1])
    e, f = int(path[k]), int(path[k + 1])
    matrix = distance_matrix
    removed = matrix[a, b] + matrix[c, d] + matrix[e, f]
    if variant == 1:
        added = matrix[a, c] + matrix[b, d] + matrix[e, f]
    elif variant == 2:
        added = matrix[a, b] + matrix[c, e] + matrix[d, f]
    elif variant == 3:
        added = matrix[a, c] + matrix[b, e] + matrix[d, f]
    elif variant == 4:
        added = matrix[a, d] + matrix[e, b] + matrix[c, f]
    elif variant == 5:
        added = matrix[a, d] + matrix[e, c] + matrix[b, f]
    elif variant == 6:
        added = matrix[a, e] + matrix[d, b] + matrix[c, f]
    elif variant == 7:
        added = matrix[a, e] + matrix[d, c] + matrix[b, f]
    else:
        raise ValueError(f"Unknown 3-opt variant: {variant}")
    return float(added - removed)


def calculate_path_length(
    path: List[int],
    distance_matrix: np.ndarray,
    metrics: Optional[dict] = None,
    category: Optional[str] = None,
) -> float:
    """Calculate a closed-tour length and optionally update evaluation counters."""
    n = len(path)
    if n < 2:
        return 0.0
    if metrics is not None:
        metrics["path_length_evaluations"] = int(metrics.get("path_length_evaluations", 0)) + 1
        metrics["distance_lookups"] = int(metrics.get("distance_lookups", 0)) + n
        if category:
            key = f"{category}_path_evaluations"
            metrics[key] = int(metrics.get(key, 0)) + 1
    total = sum(float(distance_matrix[path[i], path[i + 1]]) for i in range(n - 1))
    total += float(distance_matrix[path[-1], path[0]])
    return max(total, 0.0)


def candidate_two_opt(
    path: List[int],
    path_length: float,
    distance_matrix: np.ndarray,
    params: Optional[dict] = None,
) -> Tuple[List[int], float]:
    """Run candidate-restricted, first-improvement 2-opt with O(1) deltas."""
    params = {} if params is None else params
    metrics = params.get("_metrics")
    if metrics is not None:
        metrics["two_opt_calls"] = int(metrics.get("two_opt_calls", 0)) + 1
    if not bool(params.get("twoOptEnabled", True)):
        return path, float(path_length)
    n = len(path)
    if n < 5:
        return path, float(path_length)

    max_iterations = max(1, int(params.get("twoOptMaxIter", 120)))
    candidate_count = int(params.get("twoOptCandK", params.get("kCandidates", 25)))
    candidate_count = max(2, min(candidate_count, n - 1))
    candidate_lists = np.argsort(distance_matrix, axis=1)[:, 1:candidate_count + 1]
    best_path = list(path)
    best_length = float(path_length)

    for _ in range(max_iterations):
        improved = False
        position = np.empty(n, dtype=int)
        for index, city in enumerate(best_path):
            position[city] = index
        for i in range(n - 1):
            a = int(best_path[i])
            b = int(best_path[(i + 1) % n])
            for candidate_city in candidate_lists[a]:
                j = int(position[int(candidate_city)])
                if j <= i + 1 or (i == 0 and j == n - 1):
                    continue
                c = int(best_path[j])
                d = int(best_path[(j + 1) % n])
                if metrics is not None:
                    metrics["two_opt_move_evaluations"] = int(metrics.get("two_opt_move_evaluations", 0)) + 1
                    metrics["distance_lookups"] = int(metrics.get("distance_lookups", 0)) + 4
                delta = ((distance_matrix[a, c] + distance_matrix[b, d])
                         - (distance_matrix[a, b] + distance_matrix[c, d]))
                if delta < -1e-9:
                    best_path = (best_path[:i + 1] + list(reversed(best_path[i + 1:j + 1]))
                                 + best_path[j + 1:])
                    best_length += float(delta)
                    improved = True
                    if metrics is not None:
                        metrics["two_opt_improvements"] = int(metrics.get("two_opt_improvements", 0)) + 1
                    break
            if improved:
                break
        if not improved:
            break
    return best_path, float(best_length)
