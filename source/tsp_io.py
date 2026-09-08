# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union, Optional

import math
import numpy as np


def _gpu_info() -> Optional[str]:
    """
    Return a human-readable GPU info string if CuPy+CUDA is available; otherwise None.
    """
    try:
        import cupy as cp  # type: ignore
        try:
            n = cp.cuda.runtime.getDeviceCount()
        except Exception:
            return None
        if n <= 0:
            return None
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props.get("name", b"").decode("utf-8", "ignore") if isinstance(props.get("name", b""), (bytes, bytearray)) else str(props.get("name"))
        cc = f'{props.get("major", "?")}.{props.get("minor", "?")}'
        mem = props.get("totalGlobalMem", 0)
        mem_gb = mem / (1024**3) if isinstance(mem, (int, float)) else 0.0
        return f"CuPy CUDA OK | GPU#{dev.id}: {name} | CC {cc} | VRAM {mem_gb:.1f} GB"
    except Exception:
        return None


def _can_use_gpu() -> bool:
    """Return whether both CuPy and a CUDA device are available."""
    return _gpu_info() is not None


def read_tsp_file(
    file_name: Union[str, Path],
    *,
    prefer_gpu: bool = True,
    verbose: bool = False
) -> Tuple[np.ndarray, int, str, np.ndarray]:
    """Read a coordinate-based TSPLIB instance and construct its distance matrix.

    CuPy may construct EUC_2D, CEIL_2D, ATT, and MAN_2D matrices when explicitly
    requested and available. The solver itself remains a NumPy/CPU program. GEO
    distances follow the TSPLIB formula on the CPU.
    """
    file_path = Path(file_name)
    if not file_path.is_file():
        raise FileNotFoundError(f"TSP file not found: {file_path}")

    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    num_cities = 0
    distance_type = "EUC_2D"
    coords: Optional[np.ndarray] = None

    section = "header"
    for line in lines:
        line = line.strip()
        if not line:
            continue

        u = line.upper()
        if u == "NODE_COORD_SECTION":
            section = "coords"
            continue
        if u == "EOF":
            break

        if section == "header":
            if "DIMENSION" in u:
                parts = line.split(":")
                if len(parts) >= 2:
                    num_cities = int(parts[1].strip())
                    coords = np.zeros((num_cities, 2), dtype=np.float64)
            elif "EDGE_WEIGHT_TYPE" in u:
                parts = line.split(":")
                if len(parts) >= 2:
                    distance_type = parts[1].strip().upper()

        elif section == "coords":
            parts = line.split()
            if len(parts) >= 3 and coords is not None:
                try:
                    idx = int(float(parts[0])) - 1  # TSPLIB is 1-based
                    x = float(parts[1])
                    y = float(parts[2])
                except ValueError:
                    continue
                if 0 <= idx < num_cities:
                    coords[idx, 0] = x
                    coords[idx, 1] = y

    if coords is None or num_cities <= 0:
        raise ValueError(f"Failed to parse DIMENSION/NODE_COORD_SECTION from {file_path}")

    use_gpu = bool(prefer_gpu) and _can_use_gpu()
    if verbose:
        gi = _gpu_info()
        if gi:
            print(f"[tsp_io] {gi}")
        else:
            print("[tsp_io] CuPy/CUDA not available -> CPU (NumPy)")
        print(f"[tsp_io] EDGE_WEIGHT_TYPE={distance_type} | prefer_gpu={prefer_gpu}")

    if distance_type == "EUC_2D":
        D = _build_euc(coords, use_gpu=use_gpu)
    elif distance_type == "CEIL_2D":
        D = _build_ceil(coords, use_gpu=use_gpu)
    elif distance_type == "ATT":
        D = _build_att(coords, use_gpu=use_gpu)
    elif distance_type == "MAN_2D":
        D = _build_man(coords, use_gpu=use_gpu)
    elif distance_type == "GEO":
        if verbose and use_gpu:
            print("[tsp_io] GEO distance uses CPU implementation (TSPLIB standard).")
        D = _build_geo(coords)
    else:
        # fallback: EUC_2D round
        if verbose:
            print(f"[tsp_io] Unknown EDGE_WEIGHT_TYPE={distance_type}; fallback to EUC_2D.")
        distance_type = "EUC_2D"
        D = _build_euc(coords, use_gpu=use_gpu)

    return D, int(num_cities), str(distance_type), coords


# =========================
# EUC_2D (round)
# =========================
def _build_euc(coords: np.ndarray, use_gpu: bool) -> np.ndarray:
    n = coords.shape[0]
    if use_gpu:
        try:
            import cupy as cp  # type: ignore
            x = cp.asarray(coords[:, 0], dtype=cp.float32)
            y = cp.asarray(coords[:, 1], dtype=cp.float32)
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = cp.sqrt(dx * dx + dy * dy)
            dist = cp.rint(dist)  # round
            dist = cp.maximum(dist, 0.0)
            return cp.asnumpy(dist).astype(np.float64)
        except Exception:
            pass

    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dist = round(math.sqrt(dx * dx + dy * dy))
            dist = max(dist, 0.0)
            D[i, j] = dist
            D[j, i] = dist
    return D


# =========================
# CEIL_2D (ceil)
# =========================
def _build_ceil(coords: np.ndarray, use_gpu: bool) -> np.ndarray:
    n = coords.shape[0]
    if use_gpu:
        try:
            import cupy as cp  # type: ignore
            x = cp.asarray(coords[:, 0], dtype=cp.float32)
            y = cp.asarray(coords[:, 1], dtype=cp.float32)
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = cp.sqrt(dx * dx + dy * dy)
            dist = cp.ceil(dist)
            dist = cp.maximum(dist, 0.0)
            return cp.asnumpy(dist).astype(np.float64)
        except Exception:
            pass

    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dist = math.ceil(math.sqrt(dx * dx + dy * dy))
            dist = max(dist, 0.0)
            D[i, j] = dist
            D[j, i] = dist
    return D


# =========================
# ATT (TSPLIB)
# =========================
def _build_att(coords: np.ndarray, use_gpu: bool) -> np.ndarray:
    n = coords.shape[0]
    if use_gpu:
        try:
            import cupy as cp  # type: ignore
            x = cp.asarray(coords[:, 0], dtype=cp.float32)
            y = cp.asarray(coords[:, 1], dtype=cp.float32)
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            rij = cp.sqrt((dx * dx + dy * dy) / 10.0)
            tij = cp.rint(rij)
            dist = cp.where(tij < rij, tij + 1.0, tij)
            dist = cp.maximum(dist, 0.0)
            return cp.asnumpy(dist).astype(np.float64)
        except Exception:
            pass

    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            rij = math.sqrt((dx * dx + dy * dy) / 10.0)
            tij = round(rij)
            dist = tij + 1 if tij < rij else tij
            dist = max(dist, 0.0)
            D[i, j] = dist
            D[j, i] = dist
    return D


# =========================
# MAN_2D
# =========================
def _build_man(coords: np.ndarray, use_gpu: bool) -> np.ndarray:
    n = coords.shape[0]
    if use_gpu:
        try:
            import cupy as cp  # type: ignore
            x = cp.asarray(coords[:, 0], dtype=cp.float32)
            y = cp.asarray(coords[:, 1], dtype=cp.float32)
            dx = cp.abs(x[:, None] - x[None, :])
            dy = cp.abs(y[:, None] - y[None, :])
            dist = dx + dy
            dist = cp.maximum(dist, 0.0)
            return cp.asnumpy(dist).astype(np.float64)
        except Exception:
            pass

    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(coords[i, 0] - coords[j, 0])
            dy = abs(coords[i, 1] - coords[j, 1])
            dist = dx + dy
            dist = max(dist, 0.0)
            D[i, j] = dist
            D[j, i] = dist
    return D


# =========================
# GEO (TSPLIB)
# =========================
def _geo_to_deg(x: float) -> float:
    # TSPLIB uses integer truncation toward zero, which differs from floor for
    # negative coordinates (for example, -5.21). Using floor understates GEO
    # tour lengths for instances such as ulysses22.
    deg = int(x)
    minutes = x - deg
    return math.pi * (deg + 5.0 * minutes / 3.0) / 180.0


def _calculate_geo_distance(x1: float, y1: float, x2: float, y2: float) -> int:
    # TSPLIB standard GEO
    RRR = 6378.388
    lat1 = _geo_to_deg(x1)
    lon1 = _geo_to_deg(y1)
    lat2 = _geo_to_deg(x2)
    lon2 = _geo_to_deg(y2)

    q1 = math.cos(lon1 - lon2)
    q2 = math.cos(lat1 - lat2)
    q3 = math.cos(lat1 + lat2)

    arg = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
    # numeric clamp
    arg = max(-1.0, min(1.0, arg))
    dist = RRR * math.acos(arg) + 1.0
    return int(math.floor(dist))


def _build_geo(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dist = _calculate_geo_distance(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
            dist = max(dist, 0.0)
            D[i, j] = dist
            D[j, i] = dist
    return D
