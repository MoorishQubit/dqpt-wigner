"""Independent checks of split finite-support contraction and its normalization."""
import numpy as np
import pytest

from dqpt_wigner.block_support_extension import direct_support_split
from dqpt_wigner.event_audit import streamed_wigner_line
from dqpt_wigner.potts_mps import VidalMPS


def complex_state(seed=71):
    rng = np.random.default_rng(seed)
    bonds = [1, 3, 4, 4, 4, 3, 1]
    gammas = [(rng.normal(size=(a, 3, b)) + 1j*rng.normal(size=(a, 3, b))) / np.sqrt(3*a)
              for a, b in zip(bonds[:-1], bonds[1:])]
    state = VidalMPS(gammas, [np.ones(b) for b in bonds[1:-1]])
    # Deliberately noncanonical, nonunit-normalized tensors exercise physical
    # environments and explicit norm division rather than Schmidt shortcuts.
    state.gammas[0] *= 1.7 / np.sqrt(state.norm())
    return state


def dense_support(state, start, length, branch):
    tensors = state.as_left_weighted_tensors()
    wave = tensors[0][0]
    for tensor in tensors[1:]:
        wave = np.tensordot(wave, tensor, axes=(-1, 0))
    wave = wave.reshape(3**start, 3**length, 3**(state.n_sites-start-length))
    density = np.einsum("aib,ajb->ij", wave, wave.conj())
    density /= np.trace(density)
    digits = np.indices((3,)*length).reshape(length, -1).T
    powers = 3**np.arange(length-1, -1, -1)
    rows = ((branch+2*digits) % 3) @ powers
    cols = ((branch-2*digits) % 3) @ powers
    wigner = np.fft.fftn(density[rows, cols].reshape((3,)*length)).reshape(-1) / 3**length
    reference = int(np.full(length, branch) @ powers)
    return wigner, float(density[reference, reference].real)


@pytest.mark.parametrize("start,length", [(0, 1), (1, 3), (1, 4), (0, 6)])
def test_all_sectors_odd_even_and_boundaries_against_dense_and_streamed(start, length):
    state = complex_state()
    for branch in range(3):
        result = direct_support_split(state, start, length, branch, return_values=True)
        dense, probability = dense_support(state, start, length, branch)
        streamed = np.concatenate(list(streamed_wigner_line(state, start, length, branch, batch_size=11)))
        np.testing.assert_allclose(result["wigner_values"], dense, atol=2e-15, rtol=2e-12)
        np.testing.assert_allclose(result["wigner_values"], streamed, atol=2e-15, rtol=2e-12)
        assert result["probability"] == pytest.approx(probability, abs=2e-15)
        assert abs(result["signed_reconstruction_error"]) < 3e-15
        assert abs(result["identity_A_minus_P_minus_2nu"]) < 3e-15
        assert result["contracted_cell_count"] == 3**length
        assert result["state_norm"] == pytest.approx(1.7**2)
        assert result["estimated_workspace_bytes"] <= result["max_workspace_bytes"]


def test_chunked_join_and_frontier_preserve_values_and_reject_small_budget():
    state = complex_state()
    full = direct_support_split(state, 0, 6, 1, return_values=True)
    bounded = direct_support_split(state, 0, 6, 1, max_workspace_bytes=36_000, return_values=True)
    np.testing.assert_allclose(bounded["wigner_values"], full["wigner_values"], atol=2e-15)
    assert bounded["row_batch"]*bounded["column_batch"] < bounded["exact_cell_count"]
    assert bounded["estimated_workspace_bytes"] <= 36_000
    stats_only = direct_support_split(state, 0, 6, 1, max_workspace_bytes=36_000)
    assert "wigner_values" not in stats_only
    for key in ("probability", "unsigned_weight", "negative_mass"):
        assert stats_only[key] == pytest.approx(full[key], abs=2e-15)
    with pytest.raises(MemoryError):
        direct_support_split(state, 0, 6, 1, max_workspace_bytes=100)


def test_complex_gauge_and_product_zero_domains():
    state = complex_state()
    reference = direct_support_split(state, 1, 4, 2, return_values=True)
    rng = np.random.default_rng(931)
    gauge = np.eye(4) + .1*(rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4)))
    state.gammas[2] = state.gammas[2] @ gauge
    state.gammas[3] = np.einsum("ab,bsr->asr", np.linalg.inv(gauge), state.gammas[3])
    changed = direct_support_split(state, 1, 4, 2, return_values=True)
    np.testing.assert_allclose(changed["wigner_values"], reference["wigner_values"], atol=2e-15)
    assert reference["negative_mass"] > 0
    product = VidalMPS.product(8)
    for branch in range(3):
        result = direct_support_split(product, 1, 5, branch)
        assert result["probability"] == (1 if branch == 0 else 0)
        assert result["unsigned_weight"] == pytest.approx(1 if branch == 0 else 0, abs=2e-15)
        assert result["negative_mass"] == 0
        assert result["zero_probability"] == (branch != 0)
        if branch:
            assert result["sign_cost"] is None
