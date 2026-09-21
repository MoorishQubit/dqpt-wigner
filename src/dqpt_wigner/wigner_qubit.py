"""Qubit discrete Wigner frames used only as representation-robust controls.

Unlike odd-dimensional Gross phase space, qubit Wigner negativity is not
unique.  The project therefore reports its spread over more than one valid
phase-point frame and does not base the principal theorem on qubit negativity.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import itertools
import multiprocessing as mp
from typing import Literal

import numpy as np
from tqdm.auto import tqdm

from .operators import pauli_matrices

FrameName = Literal["standard", "reflected_y"]


def tetrahedral_vectors(frame: FrameName = "standard") -> np.ndarray:
    standard = np.array(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
        dtype=float,
    )
    if frame == "standard":
        return standard
    if frame == "reflected_y":
        reflected = standard.copy()
        reflected[:, 1] *= -1.0
        return reflected
    raise ValueError(f"unknown qubit Wigner frame: {frame}")


def local_phase_point_operators(frame: FrameName = "standard") -> list[np.ndarray]:
    identity, x, y, z = pauli_matrices(sparse=False)
    paulis = (x, y, z)
    return [
        0.5 * (identity + sum(vector[index] * paulis[index] for index in range(3)))
        for vector in tetrahedral_vectors(frame)
    ]


def local_wigner_from_density(rho: np.ndarray, frame: FrameName = "standard") -> np.ndarray:
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (2, 2):
        raise ValueError("rho must be a one-qubit density matrix")
    return np.array([0.5 * np.trace(operator @ rho).real for operator in local_phase_point_operators(frame)])


def local_wigner_from_bloch(mx: float, my: float, mz: float, frame: FrameName = "standard") -> np.ndarray:
    vector = np.array([mx, my, mz], dtype=float)
    return 0.25 * (1.0 + tetrahedral_vectors(frame) @ vector)


def l1_norm(wigner: np.ndarray) -> float:
    return float(np.sum(np.abs(wigner)))


def mana(wigner: np.ndarray) -> float:
    return float(np.log(max(l1_norm(wigner), 1e-300)))


def negativity(wigner: np.ndarray) -> float:
    return float(0.5 * (l1_norm(wigner) - 1.0))


def frame_robust_local_metrics(mx: float, my: float, mz: float) -> dict[str, float]:
    values = {}
    negativities = []
    manas = []
    for frame in ("standard", "reflected_y"):
        wigner = local_wigner_from_bloch(mx, my, mz, frame=frame)
        values[f"negativity_{frame}"] = negativity(wigner)
        values[f"mana_{frame}"] = mana(wigner)
        negativities.append(values[f"negativity_{frame}"])
        manas.append(values[f"mana_{frame}"])
    values["negativity_min"] = float(min(negativities))
    values["negativity_max"] = float(max(negativities))
    values["negativity_spread"] = float(max(negativities) - min(negativities))
    values["mana_min"] = float(min(manas))
    values["mana_max"] = float(max(manas))
    return values


_QUBIT_WORKER: dict[str, object] = {}


def _init_qubit_worker(psi: np.ndarray, n_sites: int, frame: FrameName) -> None:
    _QUBIT_WORKER.clear()
    _QUBIT_WORKER.update(psi=np.asarray(psi, dtype=complex), n_sites=n_sites, ops=local_phase_point_operators(frame))


def _qubit_point(index_tuple: tuple[int, ...]) -> tuple[tuple[int, ...], float]:
    psi = np.asarray(_QUBIT_WORKER["psi"])
    operators = _QUBIT_WORKER["ops"]
    operator = operators[index_tuple[0]]
    for index in index_tuple[1:]:
        operator = np.kron(operator, operators[index])
    dimension = psi.size
    value = float(np.vdot(psi, operator @ psi).real / dimension)
    return index_tuple, value


def full_wigner_pure(
    psi: np.ndarray,
    n_sites: int,
    *,
    frame: FrameName = "standard",
    jobs: int = 1,
    progress: bool = True,
) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex)
    if psi.shape != (2**n_sites,):
        raise ValueError("psi has the wrong dimension")
    points = list(itertools.product(range(4), repeat=n_sites))
    output = np.empty((4,) * n_sites, dtype=float)
    if jobs <= 1:
        _init_qubit_worker(psi, n_sites, frame)
        for point in tqdm(points, desc=f"qubit Wigner ({frame})", disable=not progress):
            key, value = _qubit_point(point)
            output[key] = value
        return output
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=context,
        initializer=_init_qubit_worker,
        initargs=(psi, n_sites, frame),
    ) as executor:
        for key, value in tqdm(
            executor.map(_qubit_point, points),
            total=len(points),
            desc=f"qubit Wigner ({frame})",
            disable=not progress,
        ):
            output[key] = value
    return output


def validate_frame(frame: FrameName = "standard", tolerance: float = 1e-11) -> dict[str, float | bool]:
    operators = local_phase_point_operators(frame)
    trace_error = max(abs(np.trace(operator) - 1.0) for operator in operators)
    gram_error = 0.0
    for i, first in enumerate(operators):
        for j, second in enumerate(operators):
            target = 2.0 if i == j else 0.0
            gram_error = max(gram_error, float(abs(np.trace(first @ second) - target)))
    return {
        "trace_error": float(trace_error),
        "gram_error": float(gram_error),
        "passed": bool(max(trace_error, gram_error) < tolerance),
    }
