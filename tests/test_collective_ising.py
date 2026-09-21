import numpy as np
import pytest
from dqpt_wigner.collective_ising import (
    exact_collective_trajectory,
    mean_field_trajectory,
    late_time_order,
)


def test_late_time_order_signed_and_absolute_nonuniform_samples():
    time = np.array([0., 1., 2., 4.])
    mz = np.array([100., -1., 0., 2.])
    signed, absolute = late_time_order(time, mz, start=1.)
    assert signed == pytest.approx(0.5)
    assert absolute == pytest.approx(5. / 6.)


def test_late_time_order_requires_two_samples():
    with pytest.raises(ValueError, match="fewer than two samples"):
        late_time_order(np.array([0., 1.]), np.ones(2), start=1.)


def test_collective_initial_conditions_and_probability_bounds():
    times = np.linspace(0.0, 0.5, 21)
    data = exact_collective_trajectory(20, 1.0, 0.7, times)
    assert abs(data["prob_up"][0] - 1.0) < 1e-12
    assert abs(data["prob_down"][0]) < 1e-12
    assert abs(data["magnetization_z"][0] - 1.0) < 1e-12
    assert np.all(data["prob_up"] >= -1e-13)
    assert np.all(data["prob_down"] >= -1e-13)
    assert np.all(data["return_echo"] <= 1.0 + 1e-10)


def test_mean_field_norm_and_energy_are_conserved():
    J, h = 1.0, 0.7
    times = np.linspace(0.0, 10.0, 1001)
    data = mean_field_trajectory(J, h, times)
    norm = data["mx"]**2 + data["my"]**2 + data["mz"]**2
    energy = -0.5 * J * data["mz"]**2 - h * data["mx"]
    assert np.max(np.abs(norm - 1.0)) < 2e-8
    assert np.max(np.abs(energy - energy[0])) < 2e-8
