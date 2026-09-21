"""Exact finite-MPS return supports using a split, memory-bounded contraction.

This enumerates the same canonical Wigner cells as ``streamed_wigner_line``.
It changes contraction order only, never takes absolute values of tensors or
partially contracted environments, and does not change the represented state.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .event_audit import positive_block_return, support_quantities


def _extend(frontier, tensor, branch, direction, *, resident_bytes, budget):
    """Append/prepend one momentum digit, retaining lexicographic order."""
    n, before, _ = frontier.shape
    after = tensor.shape[2] if direction == "left" else tensor.shape[0]
    output_bytes = 16 * 3 * n * after**2
    fixed = resident_bytes + frontier.nbytes + output_bytes
    # Two matrix-product temporaries and a Fourier-scaled update, with a
    # conservative extra product buffer. BLAS private packing is external.
    per_batch = 32 * (before * after + 2 * after**2)
    batch = min(n, 256, (budget - fixed) // max(1, per_batch))
    if batch < 1:
        raise MemoryError(f"split-support frontier needs at least {fixed + per_batch} bytes; cap {budget}")
    output = np.zeros((3*n, after, after), dtype=np.complex128)
    view = output.reshape(n, 3, after, after) if direction == "left" else output.reshape(3, n, after, after)
    omega = np.exp(2j * np.pi / 3)
    for offset in range(0, n, batch):
        stop = min(n, offset + batch)
        environments = frontier[offset:stop]
        for x in range(3):
            ar = tensor[:, (branch + 2*x) % 3, :]
            ac = tensor[:, (branch - 2*x) % 3, :].conj()
            if direction == "left":
                contracted = (ar.T @ environments) @ ac
            else:
                contracted = (ar @ environments) @ ac.T
            for momentum in range(3):
                target = view[offset:stop, momentum] if direction == "left" else view[momentum, offset:stop]
                target += (omega ** (-momentum*x) / 3) * contracted
    return output, fixed + batch * per_batch


def direct_support_split(state, start: int, length: int, branch: int,
                         max_workspace_bytes: int = 256 * 1024**2,
                         return_values: bool = False) -> dict:
    """Return independently checked support statistics for a finite open MPS.

    ``state`` is a qutrit ``VidalMPS``. The two half-support frontiers require
    O(3**ceil(length/2) * chi**2) storage, and their scalar joins cost
    O(3**length * chi**2). Frontiers and joins are chunked under the managed
    NumPy-workspace cap. The existing input state and BLAS-private workspace
    are excluded from that cap. No dense reduced state is constructed.

    With ``return_values=True``, ``wigner_values`` contains all normalized
    cells in ordinary base-three lexicographic momentum order. Its output
    storage is included in the cap. Without it, no full support is retained.
    Absolute/negative weights are accumulated only after each complete
    left/right/environment contraction. The projector probability is an
    independent contraction, not the signed sum used as its own validation.
    """
    began = time.perf_counter()
    if branch not in (0, 1, 2) or state.local_dim != 3:
        raise ValueError("a qutrit state and branch 0, 1, or 2 are required")
    if start < 0 or length < 1 or start + length > state.n_sites:
        raise ValueError("invalid block")
    budget = int(max_workspace_bytes)
    if budget <= 0:
        raise ValueError("max_workspace_bytes must be positive")
    split = length // 2
    n_left, n_right = 3**split, 3**(length-split)
    cells = n_left * n_right
    output_bytes = cells * 8 if return_values else 0
    chi = int(state.maximum_bond_dimension)
    # Conventional tensor materialization and boundary contraction occur
    # before the exponential frontiers. Reject even this preprocessing if
    # the supplied cap cannot accommodate it conservatively.
    preparation_bytes = 2 * sum(g.nbytes for g in state.gammas) + 128 * chi**2
    if preparation_bytes + output_bytes > budget:
        raise MemoryError(f"split-support preprocessing estimate {preparation_bytes + output_bytes} bytes exceeds cap {budget}")
    left, right, tensors = state._environments(start, length)
    norm = float(state.norm())
    if not math.isfinite(norm) or norm <= 0:
        raise FloatingPointError("state norm must be finite and positive")
    probability = positive_block_return(state, start, length, branch)
    tensor_bytes = sum(t.nbytes for t in tensors)
    base_bytes = tensor_bytes + left.nbytes + right.nbytes + output_bytes
    middle = tensors[split].shape[0] if split < length else tensors[-1].shape[2]
    frontier_bytes = 16 * (n_left+n_right) * middle**2
    if base_bytes + frontier_bytes + 48 > budget:
        raise MemoryError(f"split-support final frontiers need {base_bytes + frontier_bytes + 48} bytes; cap {budget}")
    peak = max(preparation_bytes + output_bytes, base_bytes + frontier_bytes)
    left_frontier = left[None, :, :].copy()
    for tensor in tensors[:split]:
        left_frontier, used = _extend(left_frontier, tensor, branch, "left",
                                      resident_bytes=base_bytes, budget=budget)
        peak = max(peak, used)
    right_frontier = right[None, :, :].copy()
    for tensor in reversed(tensors[split:]):
        right_frontier, used = _extend(right_frontier, tensor, branch, "right",
                                       resident_bytes=base_bytes+left_frontier.nbytes, budget=budget)
        peak = max(peak, used)
    left_flat = left_frontier.reshape(n_left, -1)
    right_flat = right_frontier.reshape(n_right, -1)
    # A join holds a complex tile plus conservative real/mask temporaries.
    fixed = base_bytes + left_frontier.nbytes + right_frontier.nbytes
    available_cells = (budget-fixed) // 48
    if available_cells < 1:
        raise MemoryError("no workspace remains for even one scalar Wigner join")
    column_batch = min(n_right, 1024, available_cells)
    row_batch = min(n_left, 256, max(1, available_cells//column_batch))
    peak = max(peak, fixed + 48*row_batch*column_batch)
    values = np.empty(cells, dtype=np.float64) if return_values else None
    values_matrix = values.reshape(n_left, n_right) if return_values else None
    signed_parts, unsigned_parts, negative_parts = [], [], []
    minimum, maximum, max_imaginary = math.inf, -math.inf, 0.0
    counted = 0
    for lo in range(0, n_left, row_batch):
        hi = min(n_left, lo+row_batch)
        for ro in range(0, n_right, column_batch):
            rh = min(n_right, ro+column_batch)
            # No conjugation: the ket/bra indices of both existing transfer
            # conventions join elementwise as sum_ab L_ab R_ab.
            complete = left_flat[lo:hi] @ right_flat[ro:rh].T
            complete /= norm
            real = complete.real
            max_imaginary = max(max_imaginary, float(np.max(np.abs(complete.imag))))
            signed_parts.append(float(np.sum(real, dtype=np.float64)))
            unsigned_parts.append(float(np.sum(np.abs(real), dtype=np.float64)))
            negative_parts.append(float(np.sum(-real, where=real < 0, dtype=np.float64)))
            minimum = min(minimum, float(np.min(real)))
            maximum = max(maximum, float(np.max(real)))
            counted += real.size
            if return_values:
                values_matrix[lo:hi, ro:rh] = real
    if max_imaginary > 5e-10:
        raise FloatingPointError(f"nonreal Wigner cell: imaginary residual {max_imaginary}")
    assert counted == cells
    signed = math.fsum(signed_parts)
    result = support_quantities(probability, math.fsum(unsigned_parts),
                                math.fsum(negative_parts), length)
    result.update(
        wigner_signed_sum=signed, signed_reconstruction_error=signed-probability,
        minimum_wigner=minimum, maximum_wigner=maximum,
        maximum_imaginary_residual=max_imaginary,
        maximum_phase_point_expectation=cells*max(abs(minimum), abs(maximum)),
        method="split_support_frontiers_chunked_scalar_join",
        primitive_source="independent positive projector; fully contracted Wigner absolute and negative cell sums",
        n_sites=int(state.n_sites), block_start=int(start), block_length=int(length), branch=int(branch),
        exact_cell_count=int(cells), contracted_cell_count=int(counted),
        checkpoint_mps_maximum_bond=chi, split_after_sites=int(split), split_bond_dimension=int(middle),
        block_tensor_shapes=[list(t.shape) for t in tensors], state_norm=norm,
        frontier_bytes=int(frontier_bytes), estimated_workspace_bytes=int(peak),
        max_workspace_bytes=budget, row_batch=int(row_batch), column_batch=int(column_batch),
        workspace_excludes="input state and BLAS-private workspace",
        elapsed_seconds=time.perf_counter()-began,
        scope="finite_reduced_block_of_fixed_finite_open_MPS", rigorous_certificate=False,
    )
    if return_values:
        result["wigner_values"] = values
    return result
