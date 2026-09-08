"""Published MSA-ACO parameters and TSPLIB best-known solution values."""
from __future__ import annotations

from typing import Any, Dict


BKS_MAP = {
    "burma14.tsp": 3323, "ulysses22.tsp": 7013, "bayg29.tsp": 1610,
    "swiss42.tsp": 1273, "att48.tsp": 33522, "eil51.tsp": 426,
    "berlin52.tsp": 7542, "st70.tsp": 675, "eil76.tsp": 538,
    "rat99.tsp": 1211, "kroA100.tsp": 21282, "kroB100.tsp": 22141,
    "kroC100.tsp": 20749, "kroD100.tsp": 21294, "kroE100.tsp": 22068,
    "eil101.tsp": 629, "lin105.tsp": 14379, "pr107.tsp": 44303,
    "bier127.tsp": 118282, "ch130.tsp": 6110, "gr137.tsp": 69853,
    "ch150.tsp": 6528, "kroB150.tsp": 26130, "kroA150.tsp": 26524,
    "d198.tsp": 15780, "kroA200.tsp": 29368,
    "kroB200.tsp": 29437, "gr202.tsp": 40160, "ts225.tsp": 126643,
    "pr226.tsp": 80369, "gil262.tsp": 2378, "pr264.tsp": 49135,
    "a280.tsp": 2579, "pr299.tsp": 48191, "lin318.tsp": 42029,
}


def size_segment(num_cities: int, small_max: int = 120, medium_max: int = 170) -> str:
    if small_max >= medium_max:
        raise ValueError("small_max must be less than medium_max")
    if num_cities <= small_max:
        return "small"
    if num_cities <= medium_max:
        return "medium"
    return "large"


def build_base_params(num_cities: int) -> Dict[str, Any]:
    """Return the scale-independent base configuration."""
    iterations = 150 if num_cities <= 80 else (300 if num_cities <= 150 else (600 if num_cities <= 250 else 1000))
    center, half_width = ((0.40, 0.10) if num_cities <= 80 else
                          ((0.35, 0.10) if num_cities <= 150 else (0.30, 0.08)))
    entropy_threshold = 0.88 if num_cities <= 80 else (0.85 if num_cities <= 150 else 0.82)
    return {
        "numCities": int(num_cities),
        "numAnts": int(max(25, round(0.8 * num_cities))),
        "maxIterations": iterations,
        "alpha": 1.8, "beta": 2.8, "rho": 0.10,
        "tauMin": 0.01, "tauMax": 10.0, "xi": 0.04,
        "tau0_mode": "tauMin", "tau0_const": 1.0,
        "q0_start": 0.80, "q0_end": 0.98,
        "kCandidates": int(min(25, num_cities - 1)),
        "threeOptMaxIter": 80,
        "gaPopSize": 150, "gaCrossoverRate": 0.8, "gaMutationRate": 0.06,
        "gaEarlyStopPatience": 15, "gaEarlyStopTolRel": 1e-10,
        "gaMaxGensSafety": 5000,
        "hybrid_switch_rule": "aligned_tau_entropy",
        "hybrid_center_ratio": center,
        "hybrid_window_half_width": half_width,
        "hybrid_entropy_k": 20,
        "hybrid_entropy_patience": 2,
        "hybrid_entropy_ema": 0.90,
        "entropyEmaUpdatesPerIteration": 1,
        "hybrid_entropy_th": entropy_threshold,
        "hybrid_entropy_th_lo": entropy_threshold - (0.18 if num_cities <= 80 else (0.11 if num_cities <= 150 else 0.10)),
        "hybrid_entropy_sched_gamma": 2.0,
        "hybrid_stall_iters": max(15, int(round(0.08 * iterations))),
        "hybrid_align_use_stall": False,
    }


def apply_common_overrides(params: Dict[str, Any], *, num_cities: int) -> None:
    """Apply the common adaptive, rank-update, and local-search settings."""
    params.update({
        "hybrid_entropy_th_lo": float(params["hybrid_entropy_th"]) - 0.05,
        "hybrid_entropy_sched_gamma": 1.0,
        "hybrid_align_use_stall": True,
        "hybrid_stall_iters": max(10, int(round(0.05 * params["maxIterations"]))),
        "hybrid_entropy_patience": 3,
        "rank_m": max(6, int(round(0.10 * params["numAnts"]))),
        "entropy_target": 0.78, "entropy_rho_gain": 0.80, "entropy_beta_gain": 0.20,
        "twoOptEnabled": True, "twoOptMaxIter": 120,
        "twoOptCandK": int(params["kCandidates"]), "ls_top_frac": 0.10,
        "ls_threeopt_topk": max(1, int(round(0.02 * params["numAnts"]))),
        "ls_threeopt_iter_gain": 1.5, "threeOptDeterministic": True,
    })
    params["rank_w_gb"] = float(params["rank_m"])
    _ = num_cities


def apply_segment_overrides(params: Dict[str, Any], *, num_cities: int) -> None:
    """Apply the three prespecified instance-size configurations."""
    num_ants = int(params["numAnts"])
    segment = size_segment(
        num_cities,
        int(params.get("segmentSmallMax", 120)),
        int(params.get("segmentMediumMax", 170)),
    )
    params["_size_segment"] = segment
    if segment == "small":
        params.update({
            "entropy_target": 0.77, "entropy_rho_gain": 0.65, "entropy_beta_gain": 0.15,
            "rank_m": max(6, int(round(0.10 * num_ants))), "q0_start": 0.79, "q0_end": 0.98,
            "ls_top_frac": 0.10, "twoOptMaxIter": 110,
            "twoOptCandK": int(params["kCandidates"]),
            "ls_threeopt_topk": max(1, int(round(0.025 * num_ants))),
            "ls_threeopt_iter_gain": 1.40,
        })
        params["rank_w_gb"] = float(params["rank_m"])
        return
    if segment == "medium":
        params.update({
            "entropy_target": 0.80, "entropy_rho_gain": 0.35, "entropy_beta_gain": 0.10,
            "rank_m": max(6, int(round(0.12 * num_ants))), "q0_start": 0.82, "q0_end": 0.985,
            "ls_top_frac": 0.08, "twoOptMaxIter": 80,
            "twoOptCandK": int(params["kCandidates"]),
            "ls_threeopt_topk": max(1, int(round(0.03 * num_ants))),
            "ls_threeopt_iter_gain": 1.40,
        })
        params["rank_w_gb"] = 1.5 * float(params["rank_m"])
        if num_cities <= 140:
            params.update({"q0_start": 0.80, "q0_end": 0.98,
                           "ls_threeopt_topk": max(1, int(round(0.04 * num_ants)))})
            params["rank_w_gb"] = 1.3 * float(params["rank_m"])
        return
    params.update({
        "entropy_target": 0.81, "entropy_rho_gain": 1.05, "entropy_beta_gain": 0.08,
        "rank_m": max(6, int(round(0.08 * num_ants))), "q0_start": 0.75, "q0_end": 0.965,
        "ls_top_frac": 0.06, "twoOptMaxIter": 60,
        "twoOptCandK": max(12, min(20, int(params["kCandidates"]))),
        "ls_threeopt_topk": max(1, int(round(0.015 * num_ants))),
        "ls_threeopt_iter_gain": 1.15,
    })
    params["rank_w_gb"] = 0.85 * float(params["rank_m"])
