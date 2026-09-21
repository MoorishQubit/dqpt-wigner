import numpy as np
import pandas as pd

from dqpt_wigner.potts_mps import (
    VidalMPS,
    combine_support_diagnostics,
    diagnose_wigner_line,
    exact_product_probabilities,
    evolve_to_times,
)
from dqpt_wigner.finite_size_scaling import block_crossing_table, first_linear_crossing, polynomial_limit


def test_block_crossings_without_crossing_keep_diagnostic_columns():
    frame = pd.DataFrame({
        "n_sites": [8, 8], "block_length": [2, 2], "time": [0.05, 0.10],
        "delta_physical_rate": [-2., -1.], "delta_support_rate": [-2., -1.],
    })
    crossings = block_crossing_table(frame)
    assert crossings["physical_crossing"].isna().all()
    assert crossings["support_crossing"].isna().all()
    assert crossings["critical_time_shift"].isna().all()
    assert crossings["sign_cost_competing_at_physical"].isna().all()
    assert crossings["surviving_fraction_competing_at_physical"].isna().all()


def test_tebd_matches_small_exact_product_returns():
    N, J, h, t = 3, 1.0, 1.2, 0.08
    state = list(evolve_to_times(N, [t], J, h, 0.01, 27, 1e-13))[-1][1]
    tebd = np.asarray([state.product_probability(a) for a in range(3)])
    exact = np.asarray(exact_product_probabilities(N, J, h, t))
    assert np.max(np.abs(tebd - exact)) < 2e-5
    assert abs(state.norm() - 1.0) < 1e-10


def test_full_block_wigner_line_reconstructs_branch_return():
    N = 3
    state = list(evolve_to_times(N, [0.11], 1.0, 1.5, 0.01, 27, 1e-13))[-1][1]
    diagnostics = []
    for a in range(3):
        line = state.block_wigner_line(0, N, a)
        d = diagnose_wigner_line(line, N, f"b{a}")
        diagnostics.append(d)
        assert abs(d.probability - state.product_probability(a)) < 2e-8
        assert abs(d.unsigned_weight - d.probability - 2 * d.negative_mass) < 1e-10
        assert abs(d.physical_rate - d.support_rate - d.sign_cost) < 1e-10
    group = combine_support_diagnostics(diagnostics[1:])
    assert abs(group.probability - diagnostics[1].probability - diagnostics[2].probability) < 1e-12
    assert group.sign_cost >= -1e-12


def test_product_state_has_positive_affine_wigner_support():
    state = VidalMPS.product(5, 0, 3)
    line0 = state.block_wigner_line(1, 3, 0)
    line1 = state.block_wigner_line(1, 3, 1)
    assert np.all(line0 >= -1e-13)
    assert abs(np.sum(line0) - 1.0) < 1e-12
    assert abs(np.sum(line1)) < 1e-12


def test_crossing_and_limit_helpers():
    assert abs(first_linear_crossing([0, 1], [-1, 1]) - 0.5) < 1e-12
    sizes = np.asarray([20, 40, 80, 160], float)
    values = 0.63 + 0.2 / sizes
    result = polynomial_limit(sizes, values, degree=1)
    assert abs(result["limit"] - 0.63) < 1e-12
