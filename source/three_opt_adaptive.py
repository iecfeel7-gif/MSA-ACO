"""Entropy-aware three-opt local search used by MSA-ACO."""
from __future__ import annotations

import random
from typing import List, Optional, Dict

import numpy as np

from utils import three_opt_delta


def enhanced_three_opt(
    path: List[int],
    path_length: float,
    distance_matrix: np.ndarray,
    params: Optional[Dict] = None
):
    """
    Candidate-restricted 3-opt with optional deterministic search order (for lower variance).

    Params (optional):
      - threeOptMaxIter (int): maximum improvement iterations (default 80)
      - threeOptCandK / kCandidates (int): candidate list size (default min(25,n-1))
      - threeOptDeterministic (bool): if True, disable randomness for better stability (default True)
      - threeOptVariantOrder (list[int]): order to try 3-opt variants in deterministic mode
    """
    if params is None:
        params = {}
    metrics = params.get("_metrics")
    if metrics is not None:
        metrics["three_opt_calls"] = int(metrics.get("three_opt_calls", 0)) + 1

    n = len(path)
    if n < 6:
        return path, float(path_length)

    max_iterations = int(params.get("threeOptMaxIter", 80))
    max_iterations = max(1, max_iterations)

    # candidate size
    k_default = 25
    k = int(params.get("threeOptCandK", params.get("kCandidates", k_default)))
    k = max(2, min(k, n - 1))

    deterministic = bool(params.get("threeOptDeterministic", True))
    variant_order = params.get("threeOptVariantOrder", [1, 2, 4, 5, 3, 6, 7])
    try:
        variant_order = [int(v) for v in variant_order]
    except Exception:
        variant_order = [1, 2, 4, 5, 3, 6, 7]

    # nearest-neighbor candidate list
    nn_idx = np.argsort(distance_matrix, axis=1)
    cand_lists = nn_idx[:, 1:k + 1]

    def _rotate_tour(tour: List[int], shift: int) -> List[int]:
        shift %= len(tour)
        return tour[shift:] + tour[:shift]

    best_path = list(path)
    best_length = float(path_length)

    def _is_valid_path(p: List[int]) -> bool:
        return len(p) == n and len(set(p)) == n

    improved = True
    iter_count = 0

    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1

        # rotate to avoid bias on cut position
        if n > 6:
            if deterministic:
                best_path = _rotate_tour(best_path, (iter_count * 7) % n)
            else:
                best_path = _rotate_tour(best_path, random.randrange(n))

        # city -> position
        pos = np.empty(n, dtype=int)
        for idx, city in enumerate(best_path):
            pos[city] = idx

        # fix i; search j,k within candidate neighborhood
        for i in range(n - 5):
            a_city = int(best_path[i])

            cand_js: List[int] = []
            for c in cand_lists[a_city]:
                j = int(pos[int(c)])
                if j < i + 2:
                    continue
                if j > n - 4:
                    continue
                cand_js.append(j)

            if not cand_js:
                continue
            if not deterministic:
                random.shuffle(cand_js)

            for j in cand_js:
                b_city = int(best_path[j])

                cand_ks: List[int] = []
                for c2 in cand_lists[b_city]:
                    kk = int(pos[int(c2)])
                    if kk < j + 2:
                        continue
                    if kk > n - 2:
                        continue
                    cand_ks.append(kk)

                if not cand_ks:
                    continue
                if not deterministic:
                    random.shuffle(cand_ks)

                for kk in cand_ks:
                    if deterministic:
                        variants = variant_order
                    else:
                        variants = [random.randint(1, 7)]

                    for variant in variants:
                        if metrics is not None:
                            metrics["three_opt_candidate_evaluations"] = int(metrics.get("three_opt_candidate_evaluations", 0)) + 1
                            metrics["distance_lookups"] = int(metrics.get("distance_lookups", 0)) + 6
                        delta = three_opt_delta(best_path, i, j, kk, int(variant), distance_matrix)
                        new_length = best_length + delta
                        if new_length < best_length - 1e-9:
                            new_path = _perform_3opt(best_path, i, j, kk, int(variant))
                            if not _is_valid_path(new_path):
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
    """0-based 3-opt variant application."""
    A = path[0:i+1]
    B = path[i+1:j+1]
    C = path[j+1:k+1]
    D = path[k+1:]

    if variant == 1:
        return A + B[::-1] + C + D
    elif variant == 2:
        return A + B + C[::-1] + D
    elif variant == 3:
        return A + B[::-1] + C[::-1] + D
    elif variant == 4:
        return A + C + B + D
    elif variant == 5:
        return A + C + B[::-1] + D
    elif variant == 6:
        return A + C[::-1] + B + D
    elif variant == 7:
        return A + C[::-1] + B[::-1] + D
    else:
        return path[:]  # no-op
