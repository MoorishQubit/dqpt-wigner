"""Numerically reliable branch-crossing utilities.

A sign change of a rate difference is meaningful only while the underlying
branch probabilities are numerically resolved. These helpers reject zeros
that occur after both overlaps have fallen below a predeclared probability
floor.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BranchCrossing:
    time: float
    left_index: int
    right_index: int
    minimum_probability: float
    direction: str
    resolved: bool


def _linear_zero(t0: float, t1: float, y0: float, y1: float) -> float:
    if y1 == y0:
        return float(t0)
    return float(t0 - y0 * (t1 - t0) / (y1 - y0))


def find_branch_crossings(
    frame: pd.DataFrame,
    *,
    time_column: str = "time",
    difference_column: str = "delta_rate",
    probability_columns: tuple[str, str] = ("prob_up", "prob_down"),
    minimum_probability: float = 0.0,
    direction: str = "up",
) -> list[BranchCrossing]:
    """Return interpolated branch crossings that pass a probability floor.

    Both branch probabilities must exceed ``minimum_probability`` at both
    endpoints bracketing the interpolation. ``direction`` may be ``up``,
    ``down``, or ``either``.
    """
    required = {time_column, difference_column, *probability_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    if direction not in {"up", "down", "either"}:
        raise ValueError("direction must be 'up', 'down', or 'either'")
    if minimum_probability < 0.0:
        raise ValueError("minimum_probability must be nonnegative")

    data = frame.sort_values(time_column).reset_index(drop=True)
    t = data[time_column].to_numpy(float)
    d = data[difference_column].to_numpy(float)
    p0 = data[probability_columns[0]].to_numpy(float)
    p1 = data[probability_columns[1]].to_numpy(float)
    out: list[BranchCrossing] = []

    for i in range(len(data) - 1):
        upward = d[i] <= 0.0 < d[i + 1]
        downward = d[i] >= 0.0 > d[i + 1]
        accepted_direction = (
            (direction == "up" and upward)
            or (direction == "down" and downward)
            or (direction == "either" and (upward or downward))
        )
        if not accepted_direction:
            continue
        resolved = bool(
            min(p0[i], p1[i], p0[i + 1], p1[i + 1])
            >= minimum_probability
        )
        if minimum_probability > 0.0 and not resolved:
            continue
        tc = _linear_zero(t[i], t[i + 1], d[i], d[i + 1])
        fraction = 0.0 if t[i + 1] == t[i] else (tc - t[i]) / (t[i + 1] - t[i])
        pp0 = p0[i] + fraction * (p0[i + 1] - p0[i])
        pp1 = p1[i] + fraction * (p1[i + 1] - p1[i])
        out.append(
            BranchCrossing(
                time=float(tc),
                left_index=i,
                right_index=i + 1,
                minimum_probability=float(min(pp0, pp1)),
                direction="up" if upward else "down",
                resolved=resolved,
            )
        )
    return out


def first_resolved_branch_crossing(
    frame: pd.DataFrame,
    *,
    minimum_probability: float,
    direction: str = "up",
    **kwargs: object,
) -> BranchCrossing | None:
    crossings = find_branch_crossings(
        frame,
        minimum_probability=minimum_probability,
        direction=direction,
        **kwargs,
    )
    return crossings[0] if crossings else None
