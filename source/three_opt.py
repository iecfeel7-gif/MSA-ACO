"""Three-opt local search used by MSA-ACO."""
from __future__ import annotations

from typing import List, Tuple, Optional, Dict

import numpy as np

from utils import calculate_path_length, three_opt_delta


def enhanced_three_opt(
    path: List[int],
    path_length: float,
    distance_matrix: np.ndarray,
    params: Optional[Dict] = None
) -> Tuple[List[int], float]:
    """
    Candidate-restricted enhanced 3-opt (0-based).

    The candidate list contains the nearest k cities, with k configurable and
    capped at n-1. Candidate positions j and k are drawn from these lists, so
    the bounded search is approximately O(n*k^2). The first improving move is
    accepted, and ``threeOptMaxIter`` limits improvement iterations.

    params:
      - threeOptMaxIter: int, default 20
      - threeOptCandK : int, default params.get("kCandidates", 25) or 25
    """
    if path is None:
        return [], 0.0

    if (not isinstance(path, (list, tuple, np.ndarray))) or len(path) < 4:
        p = list(path) if isinstance(path, (list, tuple)) else [int(x) for x in np.array(path).tolist()]
        return p, float(path_length) if path_length is not None else 0.0

    if params is None:
        params = {}
    metrics = params.get("_metrics")
    if metrics is not None:
        metrics["three_opt_calls"] = int(metrics.get("three_opt_calls", 0)) + 1

    best_path = list(map(int, list(path)))
    n = len(best_path)
    best_length = (float(path_length) if path_length is not None
                   else float(calculate_path_length(best_path, distance_matrix, metrics, "three_opt")))

    max_iterations = int(params.get("threeOptMaxIter", 20))

    # Reuse the construction candidate count unless a 3-opt value is supplied.
    k_default = 25
    k = int(params.get("threeOptCandK", params.get("kCandidates", k_default)))
    k = max(2, min(k, n - 1))

    # Precompute nearest-neighbor lists without self entries.
    nn_idx = np.argsort(distance_matrix, axis=1)  # includes self at 0
    cand_lists = nn_idx[:, 1:k + 1]

    def _rotate_tour(tour: List[int], shift: int) -> List[int]:
        shift %= len(tour)
        return tour[shift:] + tour[:shift]

    improved = True
    iter_count = 0

    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1

        # Deterministically rotate the cycle to reduce linear-cut bias.
        if n > 6:
            best_path = _rotate_tour(best_path, (iter_count * 7) % n)

        # city -> position in current tour
        pos = np.empty(n, dtype=int)
        for idx, city in enumerate(best_path):
            pos[city] = idx

        # Fix i and obtain j and k from candidate neighborhoods.
        for i in range(0, n - 5):
            a_city = best_path[i]

            # Map candidates of a_city to positions in the current tour.
            cand_js: List[int] = []
            for c in cand_lists[a_city]:
                j = int(pos[int(c)])
                # Enforce non-adjacent 3-opt cuts.
                if j < i + 2:
                    continue
                # Leave room for the third cut.
                if j > n - 4:
                    continue
                cand_js.append(j)

            if not cand_js:
                continue
            for j in cand_js:
                c_city = best_path[j]

                # Map candidates of c_city to positions for the third cut.
                cand_ks: List[int] = []
                for e in cand_lists[c_city]:
                    kk = int(pos[int(e)])
                    if kk < j + 2:
                        continue
                    if kk > n - 2:
                        continue
                    cand_ks.append(kk)

                if not cand_ks:
                    continue
                for kk in cand_ks:
                    for variant in [1, 2, 4, 5, 3, 6, 7]:
                        if metrics is not None:
                            metrics["three_opt_candidate_evaluations"] = int(metrics.get("three_opt_candidate_evaluations", 0)) + 1
                            metrics["distance_lookups"] = int(metrics.get("distance_lookups", 0)) + 6
                        delta = three_opt_delta(best_path, i, j, kk, variant, distance_matrix)
                        new_length = best_length + delta
                        if new_length < best_length - 1e-9:
                            new_path = _perform_3opt(best_path, i, j, kk, variant)
                            if not _is_valid_path(new_path, n):
                                continue
                            best_path = new_path
                            best_length = new_length
                            improved = True
                            if metrics is not None:
                                metrics["three_opt_improvements"] = int(metrics.get("three_opt_improvements", 0)) + 1
                            break
                    if improved:
                        break

                if improved:
                    break
            if improved:
                break

    return best_path, float(best_length)


def _perform_3opt(path: List[int], i: int, j: int, k: int, variant: int) -> List[int]:
    """Apply one of the seven standard symmetric 3-opt reconnections."""
    A, B = path[:i + 1], path[i + 1:j + 1]
    C, D = path[j + 1:k + 1], path[k + 1:]
    variants = {
        1: A + B[::-1] + C + D,
        2: A + B + C[::-1] + D,
        3: A + B[::-1] + C[::-1] + D,
        4: A + C + B + D,
        5: A + C + B[::-1] + D,
        6: A + C[::-1] + B + D,
        7: A + C[::-1] + B[::-1] + D,
    }
    return variants.get(int(variant), list(path))


def _is_valid_path(path: List[int], n: int) -> bool:
    if path is None or len(path) != n:
        return False
    if len(set(path)) != n:
        return False
    mn = min(path)
    mx = max(path)
    return (mn >= 0) and (mx <= n - 1)
