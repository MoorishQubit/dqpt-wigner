"""Constructive examples showing the local/global information separation."""
from __future__ import annotations

import numpy as np

from .observables import partial_trace_pure


def ghz_state(n_sites: int, phase: float = 0.0) -> np.ndarray:
    if n_sites < 2:
        raise ValueError("the marginal counterexample requires at least two sites")
    state = np.zeros(2**n_sites, dtype=complex)
    state[0] = 1.0 / np.sqrt(2.0)
    state[-1] = np.exp(1.0j * phase) / np.sqrt(2.0)
    return state


def ghz_return_probability(n_sites: int, phase: float) -> float:
    reference = ghz_state(n_sites, 0.0)
    evolved = ghz_state(n_sites, phase)
    return float(abs(np.vdot(reference, evolved)) ** 2)


def ghz_proper_marginal_distance(n_sites: int, phase_a: float, phase_b: float, keep_sites: int = 1) -> float:
    if keep_sites >= n_sites:
        raise ValueError("keep_sites must define a proper subsystem")
    rho_a = partial_trace_pure(
        ghz_state(n_sites, phase_a),
        local_dimension=2,
        n_sites=n_sites,
        keep=tuple(range(keep_sites)),
    )
    rho_b = partial_trace_pure(
        ghz_state(n_sites, phase_b),
        local_dimension=2,
        n_sites=n_sites,
        keep=tuple(range(keep_sites)),
    )
    difference = rho_a - rho_b
    singular_values = np.linalg.svd(difference, compute_uv=False)
    return float(0.5 * np.sum(singular_values))
