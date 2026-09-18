"""Core implementation of Multi-Strategy Adaptive Ant Colony Optimization."""
# Pheromone concentration is measured with normalized entropy.
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from utils import calculate_path_length, candidate_two_opt
from three_opt import enhanced_three_opt
try:
    # Optional deterministic 3-opt implementation.
    from three_opt_adaptive import enhanced_three_opt as enhanced_three_opt_v2
except Exception:
    enhanced_three_opt_v2 = None



@dataclass
class _EffFlags:
    mmas: bool
    acs_seq: bool
    acs_par: bool
    acs_end_seq: bool
    acs_end_par: bool


def optimized_mmas_algorithm(
    distance_matrix: np.ndarray,
    pheromone_matrix: np.ndarray,
    params: Dict
) -> Tuple[List[int], float, np.ndarray]:
    """
    modes: {'mmas','acs_seq','acs_par','acs_end_seq','acs_end_par','hybrid','hybrid_v4','hybrid_v4_2'}

    Hybrid:
      - Stage I : MMAS
      - Stage II: ACS_END_SEQ

    Entropy-aligned switching policy:
      hybrid_switch_rule = "aligned_tau_entropy"
        - r: center ratio (hybrid_center_ratio or hybrid_switch_ratio or piecewise default)
        - w: window half width (hybrid_window_half_width)
        - Tmin = round((r-w)T), Tmax = round((r+w)T)
        - switching is disabled before Tmin;
        - within [Tmin, Tmax), Hema and theta(t) determine the switch;
        - at Tmax, switching is forced to keep the budget bounded.

    The threshold schedule uses exponent gamma inside the switching window:
        theta(t) = th_lo + (th_hi - th_lo) * frac^gamma
        frac = (t - Tmin) / (Tmax - Tmin)
    """

    # -------------------- base params --------------------
    mode = str(params.get("mode", "mmas")).lower()

    num_cities = int(params["numCities"])
    num_ants = int(params["numAnts"])
    max_iters = int(params["maxIterations"])
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    rho = float(params["rho"])

    tau_min = float(params.get("tauMin", 0.01))
    tau_max = float(params.get("tauMax", 10.0))

    xi = float(params.get("xi", 0.05))
    tau0_mode = str(params.get("tau0_mode", "tauMin"))
    tau0_const = float(params.get("tau0_const", 0.1 * tau_max))

    q0_start = float(params.get("q0_start", params.get("q0", 0.9)))
    q0_end = float(params.get("q0_end", params.get("q0", 0.9)))

    k_candidates = int(params.get("kCandidates", min(20, num_cities - 1)))
    metrics = params.get("_metrics")
    collect_trace = bool(params.get("_collect_trace", False))
    trace = {
        "entropy": [], "entropy_ema": [], "switch_threshold": [],
        "q0": [], "rho": [], "beta": [],
        "population_diversity": [], "elite_diversity": [],
    } if collect_trace else None
    scale_aware_enabled = bool(params.get("scaleAwareEnabled", True))
    rank_update_enabled = bool(params.get("rankUpdateEnabled", True))
    two_opt_enabled = bool(params.get("twoOptEnabled", True))
    three_opt_enabled = bool(params.get("threeOptEnabled", True))
    adaptive_three_opt_enabled = bool(params.get("adaptiveThreeOptEnabled", True))

    # -------------------- hybrid switch config --------------------
    # v4 engine is shared by hybrid_v4 and hybrid_v4_2 (params may differ externally)
    is_hybrid = (mode in ("hybrid", "hybrid_v4", "hybrid_v4_2"))
    is_hybrid_v4 = (mode in ("hybrid_v4", "hybrid_v4_2"))
    # Local search is enabled by default only for the complete MSA-ACO mode.
    local_search_enabled = bool(params.get("localSearchEnabled", is_hybrid_v4))
    track_entropy_state = (
        (is_hybrid_v4 and scale_aware_enabled)
        or (local_search_enabled and adaptive_three_opt_enabled)
        or collect_trace
    )
    hybrid_switch_rule = str(params.get("hybrid_switch_rule", "iter")).lower()

    # entropy config
    entropy_k = int(params.get("hybrid_entropy_k", 20))
    Hn_patience = int(params.get("hybrid_entropy_patience", 2))
    Hn_patience = max(1, Hn_patience)

    th_hi = float(params.get("hybrid_entropy_th", 0.85))
    # A distinct lower threshold prevents immediate switching after Tmin.
    th_lo = float(params.get("hybrid_entropy_th_lo", th_hi - 0.10))
    th_hi = float(min(0.99, max(0.50, th_hi)))
    th_lo = float(min(th_hi, max(0.30, th_lo)))

    # gamma > 1 shifts likely triggering toward the middle of the window.
    gamma = float(params.get("hybrid_entropy_sched_gamma", 2.0))
    gamma = float(min(6.0, max(1.0, gamma)))

    # The optional stagnation gate is disabled by default.
    stall_iters = int(params.get("hybrid_stall_iters", 0))
    align_use_stall = bool(params.get("hybrid_align_use_stall", False))

    # EMA smoothing
    ema_lambda = float(params.get("hybrid_entropy_ema", 0.90))
    if int(params.get("entropyEmaUpdatesPerIteration", 1)) != 1:
        raise ValueError("entropyEmaUpdatesPerIteration must be 1 in the maintained protocol")
    ema_lambda = float(min(0.99, max(0.0, ema_lambda)))
    Hn_ema: Optional[float] = None

    entropy_hit = 0
    force_iter = max_iters
    min_iter = 1

    hybrid_switched = True  # latch
    hybrid_switch_iter = -1

    # -------------------- choose switching policy --------------------
    switch_reason = "not_applicable"
    if is_hybrid and hybrid_switch_rule == "no_switch":
        hybrid_switch_iter = max_iters
        hybrid_switched = True
        switch_reason = "disabled_control"
        print("[Switch] Disabled: MMAS is used for the complete run.")

    elif is_hybrid and hybrid_switch_rule == "random":
        lo = int(params.get("hybrid_random_min_iter", max(1, round(0.20 * max_iters))))
        hi = int(params.get("hybrid_random_max_iter", max(lo, round(0.50 * max_iters))))
        lo = max(1, min(max_iters - 1, lo))
        hi = max(lo, min(max_iters - 1, hi))
        hybrid_switch_iter = random.randint(lo, hi)
        hybrid_switched = True
        switch_reason = "random_control"
        print(f"[Switch] Random control: switch at {hybrid_switch_iter}/{max_iters}.")

    elif is_hybrid and hybrid_switch_rule == "iter":
        if "hybrid_switch_iter" in params:
            hybrid_switch_iter = int(round(params["hybrid_switch_iter"]))
            hybrid_switch_iter = max(1, min(max_iters - 1, hybrid_switch_iter))
        else:
            r = float(params.get("hybrid_switch_ratio", 0.35))
            hybrid_switch_iter = int(round(r * max_iters))
            hybrid_switch_iter = max(1, min(max_iters - 1, hybrid_switch_iter))
        print(f"[Switch] Fixed schedule: MMAS for {hybrid_switch_iter}/{max_iters} iterations, then ACS_END_SEQ.")
        hybrid_switched = True
        switch_reason = "fixed_control"

    elif is_hybrid and hybrid_switch_rule == "aligned_tau_entropy":
        # start as "no switch" until triggered / forced
        hybrid_switch_iter = max_iters
        hybrid_switched = False
        entropy_hit = 0

        # r center
        if "hybrid_center_ratio" in params:
            r = float(params["hybrid_center_ratio"])
        elif "hybrid_switch_ratio" in params:
            r = float(params["hybrid_switch_ratio"])
        else:
            # Default scale-specific center ratios.
            if num_cities <= 80:
                r = 0.40
            elif num_cities <= 150:
                r = 0.35
            else:
                r = 0.30

        w = float(params.get("hybrid_window_half_width", 0.10))
        w = float(min(0.30, max(0.01, w)))

        Tmin = int(round((r - w) * max_iters))
        Tmax = int(round((r + w) * max_iters))
        Tmin = max(1, min(max_iters - 1, Tmin))
        Tmax = max(Tmin + 1, min(max_iters - 1, Tmax))

        # allow user override but keep the alignment envelope
        min_iter = int(params.get("hybrid_min_iter", Tmin))
        min_iter = max(min_iter, Tmin)

        force_iter = int(params.get("hybrid_force_iter", Tmax))
        force_iter = max(min_iter + 1, min(max_iters - 1, force_iter))

        print(
            "[Switch] Entropy-aligned schedule: "
            f"r={r:.2f}, w={w:.2f}, Tmin={min_iter}, Tmax={force_iter}, "
            f"th_lo={th_lo:.3f}, th_hi={th_hi:.3f}, gamma={gamma:.2f}, "
            f"patience={Hn_patience}, k={entropy_k}, ema={ema_lambda:.2f}, "
            f"stall_iters={stall_iters}, use_stall={align_use_stall}"
        )

    elif is_hybrid:
        # fallback: keep close to fixed ratio
        r = float(params.get("hybrid_switch_ratio", 0.35))
        hybrid_switch_iter = int(round(r * max_iters))
        hybrid_switch_iter = max(1, min(max_iters - 1, hybrid_switch_iter))
        print(f"[Warning] Unknown hybrid_switch_rule={hybrid_switch_rule}; using fixed switch at {hybrid_switch_iter}/{max_iters}.")
        hybrid_switched = True

    else:
        hybrid_switch_iter = -1
        hybrid_switched = True

    # tau0 runtime
    tau0_runtime = tau0_const if tau0_mode.lower() == "const" else tau_min

    # -------------------- heuristic matrix --------------------
    epsv = 1e-6
    D = distance_matrix.astype(float).copy()
    D[D <= 0] = np.inf
    H = 1.0 / np.maximum(D, epsv)
    H[~np.isfinite(H)] = epsv
    heuristic = H

    # -------------------- candidate lists --------------------
    nn_idx = np.argsort(distance_matrix, axis=1)  # includes self
    k_take = min(k_candidates + 1, num_cities)
    cand_lists = nn_idx[:, 1:k_take]  # drop self

    # -------------------- init best --------------------
    best_path = np.random.permutation(num_cities).tolist()
    best_length = float(calculate_path_length(best_path, distance_matrix, metrics, "solver"))
    convergence = np.zeros(max_iters, dtype=float)
    max_distance_lookups = max(0, int(params.get("maxDistanceLookups", 0)))
    completed_iterations = 0

    stagnation_count = 0
    max_stagnation = 20  # half-reset threshold

    # ==================== main loop ====================
    for it in range(1, max_iters + 1):
        if it > 1 and max_distance_lookups > 0 and metrics is not None:
            if int(metrics.get("distance_lookups", 0)) >= max_distance_lookups:
                break
        theta_current = float("nan")
        Hn_trace = float("nan")
        Hn_ema_trace = float("nan")
        ant_paths = np.zeros((num_ants, num_cities), dtype=int)
        ant_lengths = np.zeros(num_ants, dtype=float)
        edge_packs: List[List[Tuple[int, int]]] = [[] for _ in range(num_ants)]

        eff = _effective_mode_flags(mode, is_hybrid, it, hybrid_switch_iter)

        # ---------- entropy closed-loop control (v4) ----------
        # Use previous EMA entropy as a feedback signal to adapt q0/beta/rho.
        H_ctrl = float(Hn_ema) if Hn_ema is not None else float(params.get("_v4_H_ema", 0.80))
        q0_start_use = q0_start
        q0_end_use = q0_end
        beta_use = beta
        rho_use = rho
        if is_hybrid_v4 and scale_aware_enabled:
            entropy_target = float(params.get("entropy_target", 0.78))
            rho_gain = float(params.get("entropy_rho_gain", 0.80))
            beta_gain = float(params.get("entropy_beta_gain", 0.20))

            # more evaporation when entropy is too low (too concentrated) -> encourage exploration
            rho_use = rho * (1.0 + rho_gain * max(0.0, entropy_target - H_ctrl))
            rho_use = float(min(0.50, max(0.02, rho_use)))

            # slightly increase beta when entropy is low -> stronger heuristic guidance
            beta_use = beta * (1.0 + beta_gain * (1.0 - H_ctrl))

            # mild q0 boost when entropy is low -> more exploitation in late stage
            q0_end_use = float(min(0.995, q0_end + 0.02 * (1.0 - H_ctrl)))
            q0_start_use = float(min(q0_end_use - 0.01, max(0.50, q0_start)))

        params["_v4_H_ema"] = float(H_ctrl)

        # ---------- construct ants ----------
        if eff.acs_end_par:
            for a in range(num_ants):
                p, L, edges = _build_path_record_edges(
                    num_cities, it, max_iters, alpha, beta_use,
                    pheromone_matrix, heuristic, distance_matrix, cand_lists,
                    q0_start_use, q0_end_use, metrics
                )
                ant_paths[a, :] = np.array(p, dtype=int)
                ant_lengths[a] = float(L)
                edge_packs[a] = edges
            pheromone_matrix = _apply_local_update_batch(pheromone_matrix, edge_packs, xi, tau0_runtime)

        else:
            for a in range(num_ants):
                if eff.acs_seq:
                    p, L, pheromone_matrix = _build_path_seq_acs(
                        num_cities, it, max_iters, alpha, beta_use,
                        pheromone_matrix, heuristic, distance_matrix, cand_lists,
                        tau0_runtime, xi, q0_start_use, q0_end_use, metrics
                    )
                elif eff.acs_end_seq:
                    p, L, edges = _build_path_record_edges(
                        num_cities, it, max_iters, alpha, beta_use,
                        pheromone_matrix, heuristic, distance_matrix, cand_lists,
                        q0_start_use, q0_end_use, metrics
                    )
                    pheromone_matrix = _apply_local_update_once(pheromone_matrix, edges, xi, tau0_runtime)
                else:
                    mode_tag = "acs_par" if eff.acs_par else "mmas"
                    p, L = _build_path_modes(
                        num_cities, it, max_iters, alpha, beta_use,
                        pheromone_matrix, heuristic, distance_matrix, cand_lists,
                        mode_tag, tau0_runtime, xi, q0_start_use, q0_end_use, metrics
                    )

                ant_paths[a, :] = np.array(p, dtype=int)
                ant_lengths[a] = float(L)

        # ---------- ranking / local search ----------
        order = np.argsort(ant_lengths)
        if local_search_enabled:
            ls_started = time.perf_counter()

            matched_policy = str(params.get("localSearchPolicy", "matched" if is_hybrid_v4 else "legacy")).lower() == "matched"
            if not matched_policy:
                # Standard non-adaptive local-search path.
                n_local = max(5, int(round(0.05 * num_ants)))
                n_local = min(n_local, num_ants)
                for kk in range(n_local):
                    idx = int(order[kk])
                    opt_p, opt_L = enhanced_three_opt(
                        ant_paths[idx, :].tolist(),
                        float(ant_lengths[idx]),
                        distance_matrix,
                        params
                    )
                    ant_paths[idx, :] = np.array(opt_p, dtype=int)
                    ant_lengths[idx] = float(opt_L)

            else:
                # v4: 2-opt on top group + deterministic 3-opt on top few (lower variance, better mean)
                ls_top_frac = float(params.get("ls_top_frac", 0.10))
                ls_top_frac = max(0.02, min(0.50, ls_top_frac))
                n_ls = max(3, int(round(ls_top_frac * num_ants)))
                n_ls = min(n_ls, num_ants)

                top3 = int(params.get("ls_threeopt_topk", max(1, int(round(0.02 * num_ants)))))
                top3 = max(1, min(top3, n_ls))

                # entropy-guided local search budget (needs H from control stage; fallback 0.8)
                H_for_ls = float(params.get("_v4_H_ema", 0.80))
                base_3opt = int(params.get("threeOptMaxIter", 80))
                gain_3opt = float(params.get("ls_threeopt_iter_gain", 1.5))
                if adaptive_three_opt_enabled:
                    three_opt_iters = int(round(base_3opt * (1.0 + gain_3opt * (1.0 - H_for_ls))))
                else:
                    three_opt_iters = base_3opt
                three_opt_iters = max(20, min(400, three_opt_iters))

                # The same 2-opt budget can be assigned to every algorithm.
                if two_opt_enabled:
                    for kk in range(n_ls):
                        idx = int(order[kk])
                        p0 = ant_paths[idx, :].tolist()
                        L0 = float(ant_lengths[idx])
                        p1, L1 = candidate_two_opt(p0, L0, distance_matrix, params)
                        ant_paths[idx, :] = np.array(p1, dtype=int)
                        ant_lengths[idx] = float(L1)

                # deterministic 3-opt for top3 (optionally)
                if enhanced_three_opt_v2 is None or not adaptive_three_opt_enabled:
                    three_opt_fn = enhanced_three_opt
                else:
                    three_opt_fn = enhanced_three_opt_v2

                # ensure deterministic for v4
                params_ls = dict(params)
                params_ls["threeOptDeterministic"] = bool(params.get("threeOptDeterministic", True))
                params_ls["threeOptMaxIter"] = three_opt_iters

                if three_opt_enabled:
                    for kk in range(top3):
                        idx = int(order[kk])
                        p0 = ant_paths[idx, :].tolist()
                        L0 = float(ant_lengths[idx])
                        p2, L2 = three_opt_fn(p0, L0, distance_matrix, params_ls)
                        ant_paths[idx, :] = np.array(p2, dtype=int)
                        ant_lengths[idx] = float(L2)

            if metrics is not None:
                metrics["local_search_seconds"] = float(metrics.get("local_search_seconds", 0.0)) + (time.perf_counter() - ls_started)

            # Rank pheromone deposits using the post-local-search tour lengths.
            order = np.argsort(ant_lengths)

        # ---------- update best ----------
        idx_best = int(np.argmin(ant_lengths))
        cur_best = float(ant_lengths[idx_best])

        if cur_best < best_length - 1e-9:
            best_length = cur_best
            best_path = ant_paths[idx_best, :].tolist()
            stagnation_count = 0
        else:
            stagnation_count += 1

        # ---------- global pheromone update ----------
        pheromone_matrix = (1.0 - rho_use) * pheromone_matrix

        if not (is_hybrid_v4 and rank_update_enabled):
            # legacy: deposit top 20% elites equally
            n_elite = max(1, int(round(0.20 * num_ants)))
            n_elite = min(n_elite, num_ants)
            elite_indices = order[:n_elite]
            for idx in elite_indices:
                elite_path = ant_paths[int(idx), :].tolist()
                delta = 1.0 / max(float(ant_lengths[int(idx)]), 1e-12)
                for k in range(num_cities - 1):
                    i = elite_path[k]
                    j = elite_path[k + 1]
                    pheromone_matrix[i, j] += delta
                    pheromone_matrix[j, i] = pheromone_matrix[i, j]
                i = elite_path[-1]
                j = elite_path[0]
                pheromone_matrix[i, j] += delta
                pheromone_matrix[j, i] = pheromone_matrix[i, j]

        else:
            # v4: rank-based update + global-best reinforcement (stronger and standard)
            m_rank = int(params.get("rank_m", max(5, int(round(0.10 * num_ants)))))
            m_rank = max(2, min(m_rank, num_ants))
            w_gb = float(params.get("rank_w_gb", float(m_rank)))

            def _deposit(tour, delta):
                for k in range(num_cities - 1):
                    i = int(tour[k]); j = int(tour[k + 1])
                    pheromone_matrix[i, j] += delta
                    pheromone_matrix[j, i] = pheromone_matrix[i, j]
                i = int(tour[-1]); j = int(tour[0])
                pheromone_matrix[i, j] += delta
                pheromone_matrix[j, i] = pheromone_matrix[i, j]

            # ranked ants
            for r in range(m_rank):
                idx = int(order[r])
                w = float(m_rank - r)
                delta = w / max(float(ant_lengths[idx]), 1e-12)
                _deposit(ant_paths[idx, :], delta)

            # global-best
            delta_gb = w_gb / max(float(best_length), 1e-12)
            _deposit(best_path, delta_gb)

# dynamic tau bounds (MMAS style)
        p_mmas = 0.05
        avg = max(2, int(round(num_cities / 2)))
        tau_max = 1.0 / max(rho_use * best_length, 1e-12)
        tau_min = tau_max * (1.0 - (p_mmas ** (1.0 / num_cities))) / max(
            (avg - 1) * (p_mmas ** (1.0 / num_cities)),
            1e-12
        )

        tau0_runtime = tau_min if tau0_mode.lower() == "taumin" else tau0_const
        pheromone_matrix = np.clip(pheromone_matrix, tau_min, tau_max)

        # Measure and update the entropy state exactly once per iteration.
        # The same observation is used by the trigger, feedback state, and trace.
        if track_entropy_state:
            Hn_trace = float(_normalized_tau_entropy(
                pheromone_matrix, cand_lists, num_cities, entropy_k
            ))
            if Hn_ema is None:
                Hn_ema = Hn_trace
            else:
                Hn_ema = ema_lambda * Hn_ema + (1.0 - ema_lambda) * Hn_trace
            Hn_ema_trace = float(Hn_ema)
            params["_v4_H_ema"] = Hn_ema_trace

        # ---------- aligned entropy trigger (v3 schedule) ----------
        if is_hybrid and (not hybrid_switched) and hybrid_switch_rule == "aligned_tau_entropy" and it < max_iters:
            # forced switch at Tmax
            if it >= force_iter:
                hybrid_switch_iter = it
                hybrid_switched = True
                switch_reason = "forced_at_window_end"
                print(f"[Switch] Forced at iteration {it} (Tmax={force_iter}); ACS_END_SEQ starts next iteration.")
            elif it >= min_iter:
                Hn = Hn_trace
                H_use = Hn_ema_trace

                # The threshold relaxes gradually across the switching window.
                denom = max(1, (force_iter - min_iter))
                frac = float((it - min_iter) / denom)
                frac = max(0.0, min(1.0, frac))
                theta_t = th_lo + (th_hi - th_lo) * (frac ** gamma)
                theta_current = float(theta_t)

                if (it % 10 == 0) or (it == min_iter):
                    print(f"[Entropy] it={it}, Hn={Hn:.3f}, Hema={H_use:.3f}, theta={theta_t:.3f}, stall={stagnation_count}")

                entropy_ok = (H_use <= theta_t)
                if entropy_ok:
                    entropy_hit += 1
                else:
                    entropy_hit = 0

                entropy_pat_ok = (entropy_hit >= Hn_patience)
                stall_ok = (stagnation_count >= stall_iters) if stall_iters > 0 else True

                trigger = entropy_pat_ok and (stall_ok if align_use_stall else True)
                if trigger:
                    hybrid_switch_iter = it
                    hybrid_switched = True
                    switch_reason = "entropy_trigger"
                    print(
                        f"[Switch] Entropy trigger at iteration {it}: theta={theta_t:.3f}, "
                        f"Hema={H_use:.3f}, Hn={Hn:.3f}, stall={stagnation_count}; ACS_END_SEQ starts next iteration."
                    )

        # ---------- stagnation half-reset ----------
        if stagnation_count >= max_stagnation:
            print("[Stagnation] Applying a half reset to the pheromone matrix.")
            pheromone_matrix = 0.5 * pheromone_matrix + 0.5 * tau_min * np.ones((num_cities, num_cities), dtype=float)
            stagnation_count = 0
            entropy_hit = 0
            Hn_ema = None  # reset EMA after half-reset
            params["_v4_H_ema"] = 0.80

        if collect_trace and trace is not None:
            if is_hybrid and hybrid_switch_rule == "aligned_tau_entropy" and math.isnan(theta_current):
                frac = max(0.0, min(1.0, float((it - min_iter) / max(1, force_iter - min_iter))))
                theta_current = float(th_lo + (th_hi - th_lo) * (frac ** gamma))
            elite_count = max(2, min(num_ants, int(round(float(params.get("ls_top_frac", 0.10)) * num_ants))))
            trace["entropy"].append(Hn_trace)
            trace["entropy_ema"].append(Hn_ema_trace)
            trace["switch_threshold"].append(theta_current)
            trace["q0"].append(float(_q0_dynamic(it, max_iters, q0_start_use, q0_end_use)))
            trace["rho"].append(float(rho_use))
            trace["beta"].append(float(beta_use))
            trace["population_diversity"].append(_edge_set_diversity(ant_paths))
            trace["elite_diversity"].append(_edge_set_diversity(ant_paths[order[:elite_count], :]))
        convergence[it - 1] = best_length
        completed_iterations = it
        if (it % 10 == 0) or (it == max_iters):
            print(f"Iteration {it:3d}/{max_iters} complete; current best: {best_length:.2f}")

    if metrics is not None:
        metrics["switch_iteration"] = int(hybrid_switch_iter)
        metrics["switch_reason"] = str(switch_reason)
        metrics["completed_iterations"] = int(completed_iterations)
    if trace is not None:
        params["_trace"] = trace
    return best_path, float(best_length), convergence[:completed_iterations]


# ========================= helpers =========================

def _effective_mode_flags(mode: str, is_hybrid: bool, iter_i: int, switch_iter: int) -> _EffFlags:
    m = mode.lower()
    if is_hybrid:
        if iter_i <= switch_iter:
            return _EffFlags(True, False, False, False, False)
        return _EffFlags(False, False, False, True, False)
    return _EffFlags(
        mmas=(m == "mmas"),
        acs_seq=(m == "acs_seq"),
        acs_par=(m == "acs_par"),
        acs_end_seq=(m == "acs_end_seq"),
        acs_end_par=(m == "acs_end_par"),
    )


def _q0_dynamic(iter_i: int, max_iters: int, q0_start: float, q0_end: float) -> float:
    progress = iter_i / max_iters
    return q0_start + (q0_end - q0_start) * progress


def _choose_start_city(iter_i: int, max_iters: int, pheromone: np.ndarray, num_cities: int) -> int:
    if iter_i < max_iters * 0.3:
        return random.randrange(num_cities)
    denom = float(np.sum(pheromone))
    if denom <= 0:
        return random.randrange(num_cities)
    start_probs = (np.sum(pheromone, axis=1) / denom).tolist()
    return _roulette_select(start_probs)


def _build_path_modes(
    num_cities: int,
    iter_i: int,
    max_iters: int,
    alpha: float,
    beta: float,
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    distance: np.ndarray,
    cand_lists: np.ndarray,
    mode_tag: str,  # "mmas" or "acs_par"
    tau0_runtime: float,
    xi: float,
    q0_start: float,
    q0_end: float,
    metrics: Optional[Dict] = None,
) -> Tuple[List[int], float]:
    q0 = _q0_dynamic(iter_i, max_iters, q0_start, q0_end)
    start_city = _choose_start_city(iter_i, max_iters, pheromone, num_cities)

    visited = np.zeros(num_cities, dtype=bool)
    visited[start_city] = True
    path = [start_city]
    current = start_city

    use_private_tau = (mode_tag.lower() == "acs_par")
    pher_work = pheromone.copy() if use_private_tau else None

    for _ in range(2, num_cities + 1):
        cands = cand_lists[current, :].tolist()
        cands = [c for c in cands if not visited[c]]
        if not cands:
            cands = [i for i in range(num_cities) if not visited[i]]

        tau_prob = pher_work if use_private_tau else pheromone
        dyn_beta = beta * (1.0 + iter_i / max_iters)

        weights = []
        for city in cands:
            w = (tau_prob[current, city] ** alpha) * (heuristic[current, city] ** dyn_beta)
            weights.append(float(w))

        s = float(sum(weights))
        probs = [w / s for w in weights] if s > 0 else [1.0 / len(cands)] * len(cands)

        if random.random() < q0:
            pick = int(np.argmax(probs))
            if probs[pick] == 0:
                pick = _roulette_idx(probs)
        else:
            pick = _roulette_idx(probs)

        next_city = cands[pick]
        path.append(next_city)

        if use_private_tau:
            i, j = current, next_city
            t = float(pher_work[i, j])
            tn = (1.0 - xi) * t + xi * tau0_runtime
            pher_work[i, j] = tn
            pher_work[j, i] = tn

        visited[next_city] = True
        current = next_city

    if metrics is not None:
        metrics["tour_constructions"] = int(metrics.get("tour_constructions", 0)) + 1
    L = float(calculate_path_length(path, distance, metrics, "solver"))
    return path, L


def _build_path_seq_acs(
    num_cities: int,
    iter_i: int,
    max_iters: int,
    alpha: float,
    beta: float,
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    distance: np.ndarray,
    cand_lists: np.ndarray,
    tau0_runtime: float,
    xi: float,
    q0_start: float,
    q0_end: float,
    metrics: Optional[Dict] = None,
) -> Tuple[List[int], float, np.ndarray]:
    q0 = _q0_dynamic(iter_i, max_iters, q0_start, q0_end)
    start_city = _choose_start_city(iter_i, max_iters, pheromone, num_cities)

    visited = np.zeros(num_cities, dtype=bool)
    visited[start_city] = True
    path = [start_city]
    current = start_city

    for _ in range(2, num_cities + 1):
        cands = cand_lists[current, :].tolist()
        cands = [c for c in cands if not visited[c]]
        if not cands:
            cands = [i for i in range(num_cities) if not visited[i]]

        dyn_beta = beta * (1.0 + iter_i / max_iters)

        weights = []
        for city in cands:
            w = (pheromone[current, city] ** alpha) * (heuristic[current, city] ** dyn_beta)
            weights.append(float(w))
        s = float(sum(weights))
        probs = [w / s for w in weights] if s > 0 else [1.0 / len(cands)] * len(cands)

        if random.random() < q0:
            pick = int(np.argmax(probs))
            if probs[pick] == 0:
                pick = _roulette_idx(probs)
        else:
            pick = _roulette_idx(probs)

        next_city = cands[pick]
        path.append(next_city)

        i, j = current, next_city
        t = float(pheromone[i, j])
        tn = (1.0 - xi) * t + xi * tau0_runtime
        pheromone[i, j] = tn
        pheromone[j, i] = tn

        visited[next_city] = True
        current = next_city

    if metrics is not None:
        metrics["tour_constructions"] = int(metrics.get("tour_constructions", 0)) + 1
    L = float(calculate_path_length(path, distance, metrics, "solver"))
    return path, L, pheromone


def _build_path_record_edges(
    num_cities: int,
    iter_i: int,
    max_iters: int,
    alpha: float,
    beta: float,
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    distance: np.ndarray,
    cand_lists: np.ndarray,
    q0_start: float,
    q0_end: float,
    metrics: Optional[Dict] = None,
) -> Tuple[List[int], float, List[Tuple[int, int]]]:
    q0 = _q0_dynamic(iter_i, max_iters, q0_start, q0_end)
    start_city = _choose_start_city(iter_i, max_iters, pheromone, num_cities)

    visited = np.zeros(num_cities, dtype=bool)
    visited[start_city] = True
    path = [start_city]
    edges: List[Tuple[int, int]] = []
    current = start_city

    for _ in range(2, num_cities + 1):
        cands = cand_lists[current, :].tolist()
        cands = [c for c in cands if not visited[c]]
        if not cands:
            cands = [i for i in range(num_cities) if not visited[i]]

        dyn_beta = beta * (1.0 + iter_i / max_iters)

        weights = []
        for city in cands:
            w = (pheromone[current, city] ** alpha) * (heuristic[current, city] ** dyn_beta)
            weights.append(float(w))
        s = float(sum(weights))
        probs = [w / s for w in weights] if s > 0 else [1.0 / len(cands)] * len(cands)

        if random.random() < q0:
            pick = int(np.argmax(probs))
            if probs[pick] == 0:
                pick = _roulette_idx(probs)
        else:
            pick = _roulette_idx(probs)

        next_city = cands[pick]
        edges.append((current, next_city))
        path.append(next_city)

        visited[next_city] = True
        current = next_city

    edges.append((path[-1], path[0]))
    if metrics is not None:
        metrics["tour_constructions"] = int(metrics.get("tour_constructions", 0)) + 1
    L = float(calculate_path_length(path, distance, metrics, "solver"))
    return path, L, edges


def _apply_local_update_once(tau: np.ndarray, edges: List[Tuple[int, int]], xi: float, tau0: float) -> np.ndarray:
    for i, j in edges:
        t = float(tau[i, j])
        tn = (1.0 - xi) * t + xi * tau0
        tau[i, j] = tn
        tau[j, i] = tn
    return tau


def _apply_local_update_batch(tau: np.ndarray, edge_packs: List[List[Tuple[int, int]]], xi: float, tau0: float) -> np.ndarray:
    for edges in edge_packs:
        if not edges:
            continue
        tau = _apply_local_update_once(tau, edges, xi, tau0)
    return tau


def _edge_set_diversity(paths: np.ndarray) -> float:
    """Mean pairwise edge-set difference, computed from edge frequencies.

    For m tours with n undirected edges each, the mean number of shared edges
    is sum_e C(count_e, 2) / C(m, 2). The normalized diversity is one minus
    this shared-edge fraction. This is algebraically identical to enumerating
    every tour pair but requires only O(m*n) edge counting.
    """
    tours = np.asarray(paths, dtype=int)
    if tours.ndim != 2 or tours.shape[0] < 2 or tours.shape[1] < 2:
        return 0.0
    m, n = tours.shape
    counts: Dict[Tuple[int, int], int] = {}
    for tour in tours:
        for index in range(n):
            u = int(tour[index])
            v = int(tour[(index + 1) % n])
            edge = (u, v) if u < v else (v, u)
            counts[edge] = counts.get(edge, 0) + 1
    shared_pairs = sum(count * (count - 1) / 2.0 for count in counts.values())
    tour_pairs = m * (m - 1) / 2.0
    shared_fraction = shared_pairs / max(tour_pairs * n, 1.0)
    return float(max(0.0, min(1.0, 1.0 - shared_fraction)))


def _normalized_tau_entropy(
    tau: np.ndarray,
    cand_lists: np.ndarray,
    n: int,
    k: int
) -> float:
    """Normalized entropy on sampled edges (i -> top-k candidates)."""
    if n <= 1:
        return 1.0
    k_eff = int(max(1, min(k, cand_lists.shape[1])))
    idx_i = np.arange(n)[:, None]
    idx_j = cand_lists[:, :k_eff]
    vals = tau[idx_i, idx_j].astype(float).reshape(-1)
    vals = np.maximum(vals, 1e-300)
    s = float(np.sum(vals))
    if s <= 0.0 or not np.isfinite(s):
        return 1.0
    p = vals / s
    H = -float(np.sum(p * np.log(p)))
    M = float(len(vals))
    if M <= 1:
        return 1.0
    Hn = H / math.log(M)
    if not np.isfinite(Hn):
        return 1.0
    return float(max(0.0, min(1.0, Hn)))


def _roulette_idx(probs: List[float]) -> int:
    s = float(sum(probs))
    if s <= 0:
        return random.randrange(len(probs))
    r = random.random() * s
    cum = 0.0
    for i, p in enumerate(probs):
        cum += float(p)
        if cum >= r:
            return i
    return int(np.argmax(probs))


def _roulette_select(probs: List[float]) -> int:
    s = float(sum(probs))
    if s <= 0:
        return random.randrange(len(probs))
    r = random.random() * s
    cum = 0.0
    for i, p in enumerate(probs):
        cum += float(p)
        if cum >= r:
            return i
    return int(np.argmax(probs))
