# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import math
import re
import numpy as np


_HEADER_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_ ]*?)\s*(?::\s*(.*))?$")
_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_SECTION_NAMES = {
    "NODE_COORD_SECTION",
    "EDGE_WEIGHT_SECTION",
    "DISPLAY_DATA_SECTION",
    "TOUR_SECTION",
    "EDGE_DATA_SECTION",
    "FIXED_EDGES_SECTION",
    "DEMAND_SECTION",
    "DEPOT_SECTION",
    "EOF",
}


def _gpu_info() -> Optional[str]:
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
        name = props.get("name", b"")
        if isinstance(name, (bytes, bytearray)):
            name = name.decode("utf-8", "ignore")
        else:
            name = str(name)
        cc = f'{props.get("major", "?")}.{props.get("minor", "?")}'
        mem = props.get("totalGlobalMem", 0)
        mem_gb = float(mem) / (1024 ** 3) if isinstance(mem, (int, float)) else 0.0
        return f"CuPy CUDA OK | GPU#{dev.id}: {name} | CC {cc} | VRAM {mem_gb:.1f} GB"
    except Exception:
        return None


def _can_use_gpu() -> bool:
    return _gpu_info() is not None


def _nint(x: float) -> int:
    # TSPLIB uses nint for nonnegative distances.
    return int(x + 0.5)


def _geo_to_deg(x: float) -> float:
    # TSPLIB GEO encoding: DDD.MM where MM are minutes; truncation toward zero is required.
    deg = int(x)
    minutes = x - deg
    return deg + 5.0 * minutes / 3.0


def _parse_header_value(line: str) -> Optional[Tuple[str, str]]:
    m = _HEADER_RE.match(line.strip())
    if not m:
        return None
    key = m.group(1).strip().upper().replace(" ", "_")
    value = (m.group(2) or "").strip()
    if key in _SECTION_NAMES:
        return None
    return key, value


def _parse_problem_text(text: str) -> Tuple[Dict[str, str], List[str], List[str], List[str]]:
    header: Dict[str, str] = {}
    node_lines: List[str] = []
    edge_lines: List[str] = []
    display_lines: List[str] = []

    current_section: Optional[str] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "EOF":
            current_section = None
            break
        if upper in _SECTION_NAMES:
            current_section = upper
            continue

        if current_section is None:
            kv = _parse_header_value(line)
            if kv is not None:
                k, v = kv
                header[k] = v
            continue

        if current_section == "NODE_COORD_SECTION":
            node_lines.append(line)
        elif current_section == "EDGE_WEIGHT_SECTION":
            edge_lines.append(line)
        elif current_section == "DISPLAY_DATA_SECTION":
            display_lines.append(line)
        else:
            # Other sections are irrelevant for this distance-matrix reader.
            continue

    return header, node_lines, edge_lines, display_lines


def _parse_coord_section(lines: List[str], dimension: int, coord_dim: int) -> np.ndarray:
    coords = np.zeros((dimension, coord_dim), dtype=np.float64)
    seen = np.zeros(dimension, dtype=bool)
    for line in lines:
        parts = line.split()
        if len(parts) < coord_dim + 1:
            continue
        try:
            idx = int(float(parts[0])) - 1
        except Exception:
            continue
        if not (0 <= idx < dimension):
            continue
        vals = []
        ok = True
        for tok in parts[1:1 + coord_dim]:
            try:
                vals.append(float(tok))
            except Exception:
                ok = False
                break
        if not ok:
            continue
        coords[idx, :] = np.asarray(vals, dtype=np.float64)
        seen[idx] = True
    if not np.all(seen):
        raise ValueError("NODE_COORD_SECTION is incomplete or malformed.")
    return coords


def _parse_display_section(lines: List[str], dimension: int) -> Optional[np.ndarray]:
    if not lines:
        return None
    coords = np.zeros((dimension, 2), dtype=np.float64)
    seen = np.zeros(dimension, dtype=bool)
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            idx = int(float(parts[0])) - 1
            x = float(parts[1])
            y = float(parts[2])
        except Exception:
            continue
        if 0 <= idx < dimension:
            coords[idx, 0] = x
            coords[idx, 1] = y
            seen[idx] = True
    if np.any(seen):
        if not np.all(seen):
            raise ValueError("DISPLAY_DATA_SECTION is incomplete or malformed.")
        return coords
    return None


def _parse_edge_numbers(lines: List[str]) -> List[float]:
    numbers: List[float] = []
    for line in lines:
        for tok in _FLOAT_RE.findall(line):
            numbers.append(float(tok))
    return numbers


def _build_explicit_matrix(numbers: List[float], n: int, fmt: str) -> np.ndarray:
    fmt = fmt.upper().strip()
    D = np.zeros((n, n), dtype=np.float64)
    k = 0

    def take() -> float:
        nonlocal k
        if k >= len(numbers):
            raise ValueError(f"EDGE_WEIGHT_SECTION has too few numbers for format {fmt}.")
        v = float(numbers[k])
        k += 1
        return v

    if fmt == "FULL_MATRIX":
        need = n * n
        if len(numbers) < need:
            raise ValueError(f"EDGE_WEIGHT_SECTION has too few numbers for FULL_MATRIX: need {need}, got {len(numbers)}.")
        arr = np.asarray(numbers[:need], dtype=np.float64).reshape((n, n))
        return arr

    if fmt == "UPPER_ROW":
        for i in range(n):
            for j in range(i + 1, n):
                v = take()
                D[i, j] = v
                D[j, i] = v
    elif fmt == "LOWER_ROW":
        for i in range(n):
            for j in range(0, i):
                v = take()
                D[i, j] = v
                D[j, i] = v
    elif fmt == "UPPER_DIAG_ROW":
        for i in range(n):
            for j in range(i, n):
                v = take()
                D[i, j] = v
                D[j, i] = v
    elif fmt == "LOWER_DIAG_ROW":
        for i in range(n):
            for j in range(0, i + 1):
                v = take()
                D[i, j] = v
                D[j, i] = v
    elif fmt == "UPPER_COL":
        for j in range(n):
            for i in range(0, j):
                v = take()
                D[i, j] = v
                D[j, i] = v
    elif fmt == "LOWER_COL":
        for j in range(n):
            for i in range(j + 1, n):
                v = take()
                D[i, j] = v
                D[j, i] = v
    elif fmt == "UPPER_DIAG_COL":
        for j in range(n):
            for i in range(0, j + 1):
                v = take()
                D[i, j] = v
                D[j, i] = v
    elif fmt == "LOWER_DIAG_COL":
        for j in range(n):
            for i in range(j, n):
                v = take()
                D[i, j] = v
                D[j, i] = v
    else:
        raise NotImplementedError(f"Unsupported EDGE_WEIGHT_FORMAT: {fmt}")

    return D


def _build_pairwise(coords: np.ndarray, *, metric: str, use_gpu: bool) -> np.ndarray:
    metric = metric.upper()
    n, dim = coords.shape

    if use_gpu and metric in {"EUC", "CEIL", "MAN", "MAX"}:
        try:
            import cupy as cp  # type: ignore
            X = cp.asarray(coords, dtype=cp.float64)
            dif = X[:, None, :] - X[None, :, :]
            adif = cp.abs(dif)
            if metric in {"EUC", "CEIL"}:
                dist = cp.sqrt(cp.sum(dif * dif, axis=2))
                if metric == "EUC":
                    dist = cp.floor(dist + 0.5)
                else:
                    dist = cp.ceil(dist)
            elif metric == "MAN":
                dist = cp.floor(cp.sum(adif, axis=2) + 0.5)
            else:  # MAX
                dist = cp.floor(cp.max(adif, axis=2) + 0.5)
            dist = cp.maximum(dist, 0.0)
            return cp.asnumpy(dist).astype(np.float64)
        except Exception:
            pass

    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dif = coords[i] - coords[j]
            adif = np.abs(dif)
            if metric == "EUC":
                v = float(_nint(float(np.sqrt(np.sum(dif * dif)))))
            elif metric == "CEIL":
                v = float(math.ceil(float(np.sqrt(np.sum(dif * dif)))))
            elif metric == "MAN":
                v = float(_nint(float(np.sum(adif))))
            elif metric == "MAX":
                v = float(_nint(float(np.max(adif))))
            else:
                raise ValueError(metric)
            D[i, j] = v
            D[j, i] = v
    return D


def _build_att(coords: np.ndarray, use_gpu: bool) -> np.ndarray:
    n = coords.shape[0]
    if use_gpu:
        try:
            import cupy as cp  # type: ignore
            X = cp.asarray(coords, dtype=cp.float64)
            dif = X[:, None, :] - X[None, :, :]
            rij = cp.sqrt(cp.sum(dif * dif, axis=2) / 10.0)
            tij = cp.floor(rij + 0.5)
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
            tij = _nint(rij)
            dij = tij + 1 if tij < rij else tij
            v = float(max(dij, 0))
            D[i, j] = v
            D[j, i] = v
    return D


def _build_geo(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    lat = np.array([math.radians(_geo_to_deg(v)) for v in coords[:, 0]], dtype=np.float64)
    lon = np.array([math.radians(_geo_to_deg(v)) for v in coords[:, 1]], dtype=np.float64)
    RRR = 6378.388
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            q1 = math.cos(lon[i] - lon[j])
            q2 = math.cos(lat[i] - lat[j])
            q3 = math.cos(lat[i] + lat[j])
            arg = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
            arg = max(-1.0, min(1.0, arg))
            dij = int(RRR * math.acos(arg) + 1.0)
            v = float(max(dij, 0))
            D[i, j] = v
            D[j, i] = v
    return D


def _build_xray(coords: np.ndarray, *, sx: float, sy: float, sz: float) -> np.ndarray:
    if coords.shape[1] < 3:
        raise ValueError("XRAY1/XRAY2 require 3-dimensional coordinates.")
    n = coords.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dx = sx * abs(coords[i, 0] - coords[j, 0])
            dy = sy * abs(coords[i, 1] - coords[j, 1])
            dz = sz * abs(coords[i, 2] - coords[j, 2])
            v = float(_nint(math.sqrt(dx * dx + dy * dy + dz * dz)))
            D[i, j] = v
            D[j, i] = v
    return D


def read_tsp_file(
    file_name: Union[str, Path],
    *,
    prefer_gpu: bool = True,
    verbose: bool = False,
    special_weight_fn: Optional[Callable[[int, int, Dict[str, str]], float]] = None,
) -> Tuple[np.ndarray, int, str, np.ndarray]:
    """
    Read a TSPLIB problem file and return a distance matrix.

    Supported EDGE_WEIGHT_TYPE:
      EXPLICIT, EUC_2D, EUC_3D, MAX_2D, MAX_3D, MAN_2D, MAN_3D,
      CEIL_2D, GEO, ATT, XRAY1, XRAY2.

    Supported EDGE_WEIGHT_FORMAT for EXPLICIT:
      FULL_MATRIX, UPPER_ROW, LOWER_ROW, UPPER_DIAG_ROW, LOWER_DIAG_ROW,
      UPPER_COL, LOWER_COL, UPPER_DIAG_COL, LOWER_DIAG_COL.

    SPECIAL can only be handled if `special_weight_fn` is provided.
    """
    file_path = Path(file_name)
    if not file_path.is_file():
        raise FileNotFoundError(f"TSP file not found: {file_path}")

    text = file_path.read_text(encoding="utf-8", errors="ignore")
    header, node_lines, edge_lines, display_lines = _parse_problem_text(text)

    if "DIMENSION" not in header:
        raise ValueError(f"Failed to parse DIMENSION from {file_path}")
    dimension = int(float(header["DIMENSION"]))
    edge_weight_type = header.get("EDGE_WEIGHT_TYPE", "EUC_2D").strip().upper()
    edge_weight_format = header.get("EDGE_WEIGHT_FORMAT", "").strip().upper()

    use_gpu = bool(prefer_gpu) and _can_use_gpu()
    if verbose:
        gi = _gpu_info()
        if gi:
            print(f"[readTSPFile] {gi}")
        else:
            print("[readTSPFile] CuPy/CUDA not available -> CPU (NumPy)")
        print(f"[readTSPFile] EDGE_WEIGHT_TYPE={edge_weight_type} | EDGE_WEIGHT_FORMAT={edge_weight_format or 'N/A'} | prefer_gpu={prefer_gpu}")

    coords: Optional[np.ndarray] = None
    display_coords = _parse_display_section(display_lines, dimension)

    if edge_weight_type in {"EUC_2D", "MAX_2D", "MAN_2D", "CEIL_2D", "ATT", "GEO"}:
        coords = _parse_coord_section(node_lines, dimension, 2)
    elif edge_weight_type in {"EUC_3D", "MAX_3D", "MAN_3D", "XRAY1", "XRAY2"}:
        coords = _parse_coord_section(node_lines, dimension, 3)
    elif edge_weight_type == "EXPLICIT":
        coords = display_coords if display_coords is not None else np.zeros((dimension, 2), dtype=np.float64)
    elif edge_weight_type == "SPECIAL":
        coords = display_coords if display_coords is not None else np.zeros((dimension, 2), dtype=np.float64)
    else:
        # Unknown types should not silently fall back to EUC_2D.
        raise NotImplementedError(f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}")

    if edge_weight_type == "EXPLICIT":
        if not edge_lines:
            raise ValueError("EXPLICIT problem missing EDGE_WEIGHT_SECTION.")
        if not edge_weight_format:
            raise ValueError("EXPLICIT problem missing EDGE_WEIGHT_FORMAT.")
        numbers = _parse_edge_numbers(edge_lines)
        D = _build_explicit_matrix(numbers, dimension, edge_weight_format)
    elif edge_weight_type == "EUC_2D":
        D = _build_pairwise(coords, metric="EUC", use_gpu=use_gpu)
    elif edge_weight_type == "EUC_3D":
        D = _build_pairwise(coords, metric="EUC", use_gpu=use_gpu)
    elif edge_weight_type == "MAX_2D":
        D = _build_pairwise(coords, metric="MAX", use_gpu=use_gpu)
    elif edge_weight_type == "MAX_3D":
        D = _build_pairwise(coords, metric="MAX", use_gpu=use_gpu)
    elif edge_weight_type == "MAN_2D":
        D = _build_pairwise(coords, metric="MAN", use_gpu=use_gpu)
    elif edge_weight_type == "MAN_3D":
        D = _build_pairwise(coords, metric="MAN", use_gpu=use_gpu)
    elif edge_weight_type == "CEIL_2D":
        D = _build_pairwise(coords, metric="CEIL", use_gpu=use_gpu)
    elif edge_weight_type == "GEO":
        if verbose and use_gpu:
            print("[readTSPFile] GEO distance uses CPU implementation (TSPLIB standard).")
        D = _build_geo(coords)
    elif edge_weight_type == "ATT":
        D = _build_att(coords, use_gpu=use_gpu)
    elif edge_weight_type == "XRAY1":
        D = _build_xray(coords, sx=1.0, sy=1.0, sz=1.0)
    elif edge_weight_type == "XRAY2":
        D = _build_xray(coords, sx=1.25, sy=1.5, sz=1.15)
    elif edge_weight_type == "SPECIAL":
        if special_weight_fn is None:
            raise NotImplementedError(
                "EDGE_WEIGHT_TYPE=SPECIAL requires a problem-specific special_weight_fn; "
                "it cannot be reconstructed from the TSPLIB file alone."
            )
        D = np.zeros((dimension, dimension), dtype=np.float64)
        for i in range(dimension):
            for j in range(dimension):
                if i == j:
                    continue
                D[i, j] = float(special_weight_fn(i, j, header))
    else:
        raise NotImplementedError(f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}")

    if D.shape != (dimension, dimension):
        raise ValueError(f"Distance matrix shape mismatch: got {D.shape}, expected ({dimension}, {dimension}).")

    D = np.asarray(D, dtype=np.float64)
    D[~np.isfinite(D)] = np.inf
    np.fill_diagonal(D, 0.0)

    # Prefer display coords for EXPLICIT problems when available; otherwise keep parsed coords.
    if edge_weight_type == "EXPLICIT" and display_coords is not None:
        coords_out = display_coords
    else:
        coords_out = np.asarray(coords, dtype=np.float64)

    return D, int(dimension), str(edge_weight_type), coords_out
