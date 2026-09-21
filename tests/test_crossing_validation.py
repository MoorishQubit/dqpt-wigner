from __future__ import annotations

import numpy as np
import pandas as pd

from dqpt_wigner.crossing_validation import (
    find_branch_crossings,
    first_resolved_branch_crossing,
)


def test_probability_floor_rejects_spurious_zero() -> None:
    frame = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0, 3.0, 4.0],
            "delta_rate": [-1.0, 1.0, -1.0, -0.5, 0.5],
            "prob_up": [1e-30, 1e-30, 0.2, 0.1, 0.08],
            "prob_down": [1e-30, 1e-30, 0.2, 0.09, 0.07],
        }
    )
    raw = find_branch_crossings(frame, minimum_probability=0.0, direction="up")
    assert len(raw) == 2
    resolved = first_resolved_branch_crossing(
        frame, minimum_probability=1e-12, direction="up"
    )
    assert resolved is not None
    assert 3.0 < resolved.time < 4.0
    assert resolved.minimum_probability > 1e-12


def test_linear_interpolation() -> None:
    frame = pd.DataFrame(
        {
            "time": [0.0, 2.0],
            "delta_rate": [-1.0, 3.0],
            "prob_up": [0.5, 0.4],
            "prob_down": [0.5, 0.3],
        }
    )
    crossing = first_resolved_branch_crossing(
        frame, minimum_probability=1e-6, direction="up"
    )
    assert crossing is not None
    assert np.isclose(crossing.time, 0.5)
