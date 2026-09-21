"""Independent controls for two-site iTEBD and return-sector bookkeeping."""
import json
import numpy as np
import pytest
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import expm_multiply

from dqpt_wigner.itebd_thermo import (
    InfiniteMPS, branch_spectrum, bulk_return_asymptotic_check, finite_open_return, finite_periodic_state,
    measure_state, transfer_fixed_points,
)


def _periodic_exact(n, h, tau):
    """Construct the Hamiltonian from computational strings, not TEBD gates."""
    dim = 3**n
    digits = np.array(np.unravel_index(np.arange(dim), (3,) * n)).T
    diagonal = np.zeros(dim)
    rows, columns, values = [], [], []
    powers = 3 ** np.arange(n - 1, -1, -1)
    for j in range(n):
        diagonal -= 2 * np.cos(2 * np.pi * (digits[:, j] - digits[:, (j + 1) % n]) / 3)
        for direction in (-1, 1):
            target = digits.copy()
            target[:, j] = (target[:, j] + direction) % 3
            rows.extend(target @ powers)
            columns.extend(range(dim))
            values.extend([-h] * dim)
    hamiltonian = diags(diagonal) + coo_matrix((values, (rows, columns)), shape=(dim, dim)).tocsr()
    psi = np.zeros(dim, complex)
    psi[0] = 1
    return expm_multiply(-1j * tau * hamiltonian, psi)


def test_product_transfer_unit_cell_and_exact_zeros():
    state = InfiniteMPS.product(vector=np.array([np.sqrt(.5), .5j, .5]))
    result = measure_state(state, finite_cells=(1, 7))
    expected = -np.log([.5, .25, .25])
    np.testing.assert_allclose([b["rate"] for b in result["branches"]], expected, atol=1e-14)
    np.testing.assert_allclose(result["local_populations_a"], [.5, .25, .25], atol=1e-14)
    for symbol in range(3):
        np.testing.assert_allclose([p["rate"] for p in result["finite_open"][str(symbol)]], expected[symbol], atol=1e-14)
    zero = measure_state(InfiniteMPS.product())
    assert zero["branches"][1]["rate"] == float("inf")
    assert zero["branches"][1]["radius"] == 0


def test_rotation_reaches_known_crossing_and_slopes():
    critical = np.arccos(1 / np.sqrt(3))
    state = InfiniteMPS.product()
    for _ in range(47):
        state.step(.02, 2, model="rotation")
    state.step(critical - state.tau, 2, model="rotation")
    result = measure_state(state)
    np.testing.assert_allclose([b["rate"] for b in result["branches"]], np.log(3), atol=3e-13)
    np.testing.assert_allclose(result["local_populations_a"], np.full(3, 1/3), atol=3e-13)
    assert state.bonds == (1, 1)
    h = 1e-5
    plus = state.copy()
    plus.step(h, 2, model="rotation")
    slope = (measure_state(plus)["delta_rate"] - result["delta_rate"]) / h
    assert slope == pytest.approx(3 * np.sqrt(2), rel=2e-5)


def test_periodic_exact_dynamics_and_second_order_convergence():
    exact = _periodic_exact(4, 1.5, .12)
    errors = []
    for dt in (.02, .01):
        state = InfiniteMPS.product()
        for _ in range(round(.12 / dt)):
            state.step(dt, 32, cutoff=1e-14)
        actual = finite_periodic_state(*state.cell_tensors(), 2)
        actual /= np.linalg.norm(actual)
        phase = np.vdot(exact, actual)
        actual *= np.exp(-1j * np.angle(phase))
        errors.append(np.linalg.norm(actual - exact))
    assert errors[1] < 1.5e-4
    assert 3.7 < errors[0] / errors[1] < 4.3


def test_return_gauge_invariance_and_explicit_finite_contraction():
    state = InfiniteMPS.product()
    for _ in range(10):
        state.step(.02, 8)
    a, b = state.cell_tensors()
    fixed = transfer_fixed_points(a, b)
    rng = np.random.default_rng(802)
    g = np.eye(a.shape[0]) + .08 * (rng.normal(size=(a.shape[0], a.shape[0])) + 1j * rng.normal(size=(a.shape[0], a.shape[0])))
    h = np.eye(b.shape[0]) + .08 * (rng.normal(size=(b.shape[0], b.shape[0])) + 1j * rng.normal(size=(b.shape[0], b.shape[0])))
    ag = np.einsum("ij,jsk,kl->isl", np.linalg.inv(g), a, h)
    bg = np.einsum("ij,jsk,kl->isl", np.linalg.inv(h), b, g)
    fg = transfer_fixed_points(ag, bg)
    assert fg["eta"] == pytest.approx(fixed["eta"], abs=1e-11)
    v = np.ones(a.shape[0], complex) / np.sqrt(a.shape[0])
    w = v.copy()
    vg = g.conj().T @ v
    wg = np.linalg.solve(g, w)
    for symbol in range(3):
        r = branch_spectrum(a, b, symbol, fixed["eta"])
        rg = branch_spectrum(ag, bg, symbol, fg["eta"])
        assert rg["rate"] == pytest.approx(r["rate"], abs=1e-11)
        ordinary = finite_open_return(a, b, symbol, 4, v, w)
        gauged = finite_open_return(ag, bg, symbol, 4, vg, wg)
        assert ordinary["rate"] == pytest.approx(gauged["rate"], abs=1e-11)
        # Direct full finite-state contraction, independently summed norm.
        amplitudes = []
        for string in np.ndindex(*(3,) * 4):
            matrix = np.eye(a.shape[0], dtype=complex)
            for tensor, s in zip([a, b, a, b], string):
                matrix = matrix @ tensor[:, s, :]
            amplitudes.append(np.vdot(v, matrix @ w))
        amplitudes = np.asarray(amplitudes)
        target = sum(symbol * 3**i for i in range(4))
        probability = abs(amplitudes[target])**2 / np.vdot(amplitudes, amplitudes).real
        finite = finite_open_return(a, b, symbol, 2, v, w)
        assert finite["log_probability"] == pytest.approx(np.log(probability), abs=1e-11)


def test_projected_leading_sector_and_peripheral_degeneracy():
    a = np.zeros((2, 3, 2), complex)
    b = np.zeros_like(a)
    a[:, 0, :] = np.diag([.8, .3])
    b[:, 0, :] = np.eye(2)
    selected = branch_spectrum(a, b, 0, left_boundary=[0, 1], right_boundary=[0, 1])
    assert selected["open_leading_coefficient_abs"] == 0
    assert selected["open_selected_index"] == 1
    assert selected["open_selected_rate"] == pytest.approx(-np.log(.3))
    a[:, 0, :] = np.diag([.8, -.8])
    degenerate = branch_spectrum(a, b, 0)
    assert not degenerate["unique_leading_modulus_numerically"]
    # Odd powers cancel in the periodic amplitude, despite nonzero radius.
    assert abs(np.trace((a[:, 0, :] @ b[:, 0, :])**1)) == 0


def test_checkpoint_is_normalized_and_resumes_exactly(tmp_path):
    state = InfiniteMPS.product()
    for _ in range(8):
        state.step(.02, 12)
    path = tmp_path / "state.npz"
    state.save(path, {"dtau": .02, "max_bond": 12})
    with np.load(path) as checkpoint:
        fixed = transfer_fixed_points(checkpoint["A"], checkpoint["B"])
        assert fixed["eta"] == pytest.approx(1, abs=1e-12)
        assert json.loads(str(checkpoint["metadata"]))["unit_cell_length"] == 2
    resumed, _ = InfiniteMPS.load(path)
    for current in (state, resumed):
        current.step(.02, 12)
    np.testing.assert_array_equal(state.gamma_a, resumed.gamma_a)
    assert measure_state(state)["delta_rate"] == measure_state(resumed)["delta_rate"]


def test_signed_bulk_boundary_selected_rate_matches_global_transfer():
    state = InfiniteMPS.product()
    for _ in range(32):
        state.step(.02, 12)
    for symbol in range(3):
        audit = bulk_return_asymptotic_check(*state.cell_tensors(), symbol, (2, 8, 32))
        assert audit["coefficient_positive_numerically"]
        # The direct positive block contraction and the eigenpair/prefactor
        # prediction are independent computational paths.
        assert abs(audit["finite_bulk_block_returns"][-1]["prediction_residual"]) < 1e-10
