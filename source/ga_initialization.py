"""Genetic-algorithm initialization for the MSA-ACO pheromone matrix."""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple, Optional

import numpy as np

from utils import calculate_path_length


def ga_init_pheromone(distance_matrix: np.ndarray, params: Dict) -> Tuple[np.ndarray, float, int]:
    """
    Python version of MATLAB GA_init_pheromone.m (0-based).

    Parameters
    ----------
    distance_matrix : (N,N) ndarray
    params : dict
        required:
          - numCities
          - gaPopSize
          - gaCrossoverRate
          - gaMutationRate
          - tauMin
          - tauMax
        optional:
          - gaEarlyStopPatience (default 15)
          - gaEarlyStopTolRel   (default 1e-10)
          - gaMaxGensSafety     (optional safety cap)

    Returns
    -------
    pheromone_matrix : (N,N) ndarray
    ga_best_len : float
    ga_gens_used : int
    """
    num_cities = int(params["numCities"])
    pop_size = int(params["gaPopSize"])
    crossover_rate = float(params["gaCrossoverRate"])
    mutation_rate = float(params["gaMutationRate"])

    tau_min = float(params["tauMin"])
    tau_max = float(params["tauMax"])

    patience = int(params.get("gaEarlyStopPatience", 15))
    tol_rel = float(params.get("gaEarlyStopTolRel", 1e-10))
    max_gens_safety = params.get("gaMaxGensSafety", None)
    if max_gens_safety is not None:
        max_gens_safety = int(max_gens_safety)

    stall_cnt = 0
    ga_gens_used = 0
    ga_best_len = float("inf")
    metrics = params.get("_metrics")

    # Initialize the population: 50% random and 50% nearest-neighbor tours.
    population = _initialize_population(pop_size, num_cities, distance_matrix)

    # Evaluate the initial population.
    fitness = np.zeros(pop_size, dtype=float)
    for i in range(pop_size):
        L = calculate_path_length(population[i].tolist(), distance_matrix, metrics, "ga")
        fitness[i] = 1.0 / max(L, 1e-12)

    best_fit = float(np.max(fitness))
    ga_best_len = 1.0 / max(best_fit, 1e-12)

    # Evolve until early stopping or the safety generation limit.
    while True:
        ga_gens_used += 1

        new_population = np.zeros((pop_size, num_cities), dtype=int)

        # Create the next population by crossover and mutation.
        i = 0
        while i < pop_size:
            parent1 = _tournament_selection(population, fitness, k=3)
            parent2 = _tournament_selection(population, fitness, k=3)

            if random.random() < crossover_rate:
                child1, child2 = _order_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = _insert_mutation(child1)
            if random.random() < mutation_rate:
                child2 = _insert_mutation(child2)

            new_population[i, :] = child1
            if i + 1 < pop_size:
                new_population[i + 1, :] = child2
            i += 2

        # Evaluate the new population.
        new_fitness = np.zeros(pop_size, dtype=float)
        for i in range(pop_size):
            L = calculate_path_length(new_population[i].tolist(), distance_matrix, metrics, "ga")
            new_fitness[i] = 1.0 / max(L, 1e-12)

        # Elitism: replace the worst offspring with the previous best tour.
        best_idx_old = int(np.argmax(fitness))
        worst_idx_new = int(np.argmin(new_fitness))
        new_population[worst_idx_new, :] = population[best_idx_old, :]
        new_fitness[worst_idx_new] = fitness[best_idx_old]

        # Advance to the new population.
        population = new_population
        fitness = new_fitness

        # Apply the relative-improvement early-stopping criterion.
        best_fit_now = float(np.max(fitness))
        best_len_now = 1.0 / max(best_fit_now, 1e-12)

        if best_len_now < ga_best_len * (1.0 - tol_rel):
            ga_best_len = best_len_now
            stall_cnt = 0
        else:
            stall_cnt += 1

        if stall_cnt >= patience:
            break

        # Optional safety limit.
        if max_gens_safety is not None and ga_gens_used >= max_gens_safety:
            break

    if metrics is not None:
        metrics["ga_generations"] = int(ga_gens_used)

    # =========================================================
    # Map the weighted top-K elite pool to the initial pheromone matrix.
    # =========================================================
    sorted_idx = np.argsort(-fitness)  # desc

    # Limit the elite pool to 5--30 tours and at most the population size.
    K = int(round(0.10 * pop_size))
    K = min(max(5, min(30, K)), pop_size)

    if K <= 1:
        lam = 0.0
    else:
        # This decay makes the Kth weight approximately 10% of the first.
        lam = math.log(10.0) / (K - 1)

    w = np.exp(-lam * np.arange(K, dtype=float))
    w = w / max(w.sum(), 1e-12)

    pheromone = np.zeros((num_cities, num_cities), dtype=float)

    for r in range(K):
        idx = int(sorted_idx[r])
        path = population[idx, :].tolist()
        L = 1.0 / max(float(fitness[idx]), 1e-12)
        gain = float(w[r]) * (1.0 / max(L, 1e-12))

        # add edges along tour
        for j in range(num_cities - 1):
            c1 = path[j]
            c2 = path[j + 1]
            pheromone[c1, c2] += gain
            pheromone[c2, c1] += gain
        c1 = path[-1]
        c2 = path[0]
        pheromone[c1, c2] += gain
        pheromone[c2, c1] += gain

    # =========================================================
    # Normalize and clip to [tauMin, tauMax].
    # =========================================================
    max_val = float(np.max(pheromone))
    if max_val > 0:
        pheromone = pheromone / max_val
        pheromone = tau_min + pheromone * (tau_max - tau_min)
    else:
        pheromone = np.full((num_cities, num_cities), tau_min, dtype=float)

    # =========================================================
    # Add a smoothing floor to avoid excessive initial concentration.
    # =========================================================
    eps_frac = 0.02  # MATLAB: epsFrac=0.02
    pheromone = (1.0 - eps_frac) * pheromone + eps_frac * tau_min

    # Clip again after smoothing.
    pheromone = np.clip(pheromone, tau_min, tau_max)

    # Self-loop pheromone is unused.
    np.fill_diagonal(pheromone, 0.0)

    return pheromone, float(ga_best_len), int(ga_gens_used)


# ==========================
# GA subroutines
# ==========================

def _initialize_population(pop_size: int, num_cities: int, distance_matrix: np.ndarray) -> np.ndarray:
    """
    MATLAB initializePopulation:
      - first half: randperm
      - second half: greedy nearest neighbor from random start
    """
    pop = np.zeros((pop_size, num_cities), dtype=int)

    half = pop_size // 2
    for i in range(pop_size):
        if i < half:
            pop[i, :] = np.random.permutation(num_cities)
        else:
            start = random.randrange(num_cities)
            path = [start]
            visited = np.zeros(num_cities, dtype=bool)
            visited[start] = True

            for _ in range(1, num_cities):
                last = path[-1]
                dists = distance_matrix[last, :].copy()
                dists[visited] = np.inf
                nxt = int(np.argmin(dists))
                path.append(nxt)
                visited[nxt] = True

            pop[i, :] = np.array(path, dtype=int)

    return pop


def _tournament_selection(population: np.ndarray, fitness: np.ndarray, k: int = 3) -> np.ndarray:
    pop_size = population.shape[0]
    selected = np.random.randint(0, pop_size, size=k)
    best_local = selected[int(np.argmax(fitness[selected]))]
    return population[best_local, :].copy()


def _order_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    MATLAB orderCrossover + fillCrossoverChild equivalent.
    """
    n = parent1.shape[0]
    child1 = np.full(n, -1, dtype=int)
    child2 = np.full(n, -1, dtype=int)

    a = random.randrange(0, n - 1)
    b = random.randrange(0, n - 1)
    if a > b:
        a, b = b, a

    child1[a:b + 1] = parent1[a:b + 1]
    child2[a:b + 1] = parent2[a:b + 1]

    child1 = _fill_crossover_child(child1, parent2, a, b)
    child2 = _fill_crossover_child(child2, parent1, a, b)

    return child1, child2


def _fill_crossover_child(child: np.ndarray, parent: np.ndarray, a: int, b: int) -> np.ndarray:
    n = child.shape[0]
    pos = b + 1
    if pos >= n:
        pos = 0

    # iterate: [b+1:n-1, 0:b]
    for i in list(range(b + 1, n)) + list(range(0, b + 1)):
        gene = int(parent[i])
        if gene not in child:
            child[pos] = gene
            pos += 1
            if pos >= n:
                pos = 0

    # safety: if any -1 remains, fill with missing
    if np.any(child < 0):
        missing = [g for g in range(n) if g not in child]
        for idx in np.where(child < 0)[0]:
            child[idx] = missing.pop(0)

    return child


def _insert_mutation(chromosome: np.ndarray) -> np.ndarray:
    """
    MATLAB insertMutation:
      - pick i,j; ensure i<j
      - take gene at j and insert after i (MATLAB uses [1:i, gene, i+1:j-1, j+1:end])
    """
    n = chromosome.shape[0]
    i = random.randrange(n)
    j = random.randrange(n)
    while j == i:
        j = random.randrange(n)
    if i > j:
        i, j = j, i

    gene = chromosome[j]
    # build new array
    mutated = np.concatenate([
        chromosome[:i + 1],
        np.array([gene], dtype=int),
        chromosome[i + 1:j],
        chromosome[j + 1:]
    ])
    # length remains n
    return mutated
