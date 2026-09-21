"""Crossing, extrapolation, and numerical convergence helpers."""
from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, Sequence
import math
import numpy as np
import pandas as pd


def first_linear_crossing(
    x: Sequence[float],
    y: Sequence[float],
    direction: str = "negative_to_positive",
) -> float | None:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if xx.size != yy.size or xx.size < 2:
        raise ValueError("x and y must have equal length >= 2")
    for i in range(xx.size - 1):
        a, b = yy[i], yy[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if direction == "negative_to_positive":
            hit = a <= 0.0 and b >= 0.0 and (a < 0.0 or b > 0.0)
        elif direction == "positive_to_negative":
            hit = a >= 0.0 and b <= 0.0 and (a > 0.0 or b < 0.0)
        elif direction == "either":
            hit = a == 0.0 or b == 0.0 or a * b < 0.0
        else:
            raise ValueError(f"unknown direction: {direction}")
        if not hit:
            continue
        if a == b:
            return float(0.5 * (xx[i] + xx[i + 1]))
        return float(xx[i] - a * (xx[i + 1] - xx[i]) / (b - a))
    return None


def interpolate(x: Sequence[float], y: Sequence[float], point: float) -> float:
    return float(np.interp(float(point), np.asarray(x, float), np.asarray(y, float)))


def polynomial_limit(
    sizes: Sequence[float],
    values: Sequence[float],
    degree: int = 1,
) -> dict[str, object]:
    L = np.asarray(sizes, dtype=float)
    val = np.asarray(values, dtype=float)
    mask = np.isfinite(L) & np.isfinite(val) & (L > 0)
    L, val = L[mask], val[mask]
    if L.size < degree + 1:
        raise ValueError("not enough points for requested fit")
    x = 1.0 / L
    coeff, cov = np.polyfit(x, val, degree, cov=True)
    predicted = np.polyval(coeff, x)
    residual = val - predicted
    rss = float(np.sum(residual**2))
    n = int(L.size)
    k = int(degree + 1)
    aic = n * math.log(max(rss / n, np.finfo(float).tiny)) + 2 * k
    aicc = aic + (2 * k * (k + 1) / (n - k - 1) if n > k + 1 else float("inf"))
    return {
        "degree": degree,
        "sizes": L.tolist(),
        "coefficients_descending": coeff.tolist(),
        "covariance": cov.tolist(),
        "limit": float(coeff[-1]),
        "limit_stderr": float(math.sqrt(max(cov[-1, -1], 0.0))),
        "rss": rss,
        "aic": float(aic),
        "aicc": float(aicc),
    }


def global_crossing_summary(frame: pd.DataFrame) -> dict[str, float | None]:
    t = frame["time"].to_numpy(float)
    return {
        "individual_crossing": first_linear_crossing(t, frame["delta_rate_individual"], "negative_to_positive"),
        "grouped_crossing": first_linear_crossing(t, frame["delta_rate_grouped"], "negative_to_positive"),
    }


def support_rows(
    time: float,
    n_sites: int,
    block_length: int,
    start: int,
    diagnostics: Iterable[object],
    cumulative_discarded: float,
    maximum_bond: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for diagnostic in diagnostics:
        data = asdict(diagnostic)
        data.update(
            {
                "time": float(time),
                "n_sites": int(n_sites),
                "block_start": int(start),
                "block_length": int(block_length),
                "cumulative_discarded_weight": float(cumulative_discarded),
                "maximum_bond": int(maximum_bond),
                "identity_A_minus_P_minus_2nu": float(
                    data["unsigned_weight"] - data["probability"] - 2.0 * data["negative_mass"]
                ),
                "identity_r_minus_s_minus_q": float(
                    data["physical_rate"] - data["support_rate"] - data["sign_cost"]
                ),
            }
        )
        rows.append(data)
    return rows


def wide_block_table(long_frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["n_sites", "time", "block_start", "block_length"]
    value_cols = [
        "probability",
        "unsigned_weight",
        "negative_mass",
        "physical_rate",
        "support_rate",
        "sign_cost",
        "surviving_fraction",
        "minimum_wigner",
        "maximum_wigner",
    ]
    parts = []
    for branch in ("initial", "competing"):
        piece = long_frame.loc[long_frame["branch"] == branch, keys + value_cols].copy()
        piece = piece.rename(columns={c: f"{c}_{branch}" for c in value_cols})
        parts.append(piece)
    wide = parts[0].merge(parts[1], on=keys, how="inner")
    for base in ("physical_rate", "support_rate", "sign_cost"):
        wide[f"delta_{base}"] = wide[f"{base}_initial"] - wide[f"{base}_competing"]
    wide["decomposition_error"] = (
        wide["delta_physical_rate"] - wide["delta_support_rate"] - wide["delta_sign_cost"]
    )
    wide["crossing_shift_integrand"] = -wide["delta_sign_cost"]
    return wide.sort_values(keys).reset_index(drop=True)


def block_crossing_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (N, ell), group in wide.groupby(["n_sites", "block_length"], sort=True):
        g = group.sort_values("time")
        t = g["time"].to_numpy(float)
        tc = first_linear_crossing(t, g["delta_physical_rate"], "negative_to_positive")
        ts = first_linear_crossing(t, g["delta_support_rate"], "negative_to_positive")
        row: dict[str, object] = {
            "n_sites": int(N),
            "block_length": int(ell),
            "physical_crossing": tc,
            "support_crossing": ts,
            "critical_time_shift": None if tc is None or ts is None else float(tc - ts),
        }
        for name in (
            "delta_support_rate",
            "delta_sign_cost",
            "sign_cost_initial",
            "sign_cost_competing",
            "surviving_fraction_initial",
            "surviving_fraction_competing",
        ):
            row[f"{name}_at_physical"] = float("nan") if tc is None else interpolate(t, g[name], tc)
        rows.append(row)
    return pd.DataFrame(rows)


def finite_difference_jump(
    time: Sequence[float], values: Sequence[float], crossing: float, window: int = 2
) -> float:
    t = np.asarray(time, float)
    y = np.asarray(values, float)
    idx = int(np.searchsorted(t, crossing))
    left_lo = max(0, idx - window)
    left_hi = max(left_lo + 2, idx)
    right_lo = min(len(t) - 2, idx)
    right_hi = min(len(t), right_lo + window + 1)
    if left_hi - left_lo < 2 or right_hi - right_lo < 2:
        return float("nan")
    slope_l = np.polyfit(t[left_lo:left_hi], y[left_lo:left_hi], 1)[0]
    slope_r = np.polyfit(t[right_lo:right_hi], y[right_lo:right_hi], 1)[0]
    return float(slope_r - slope_l)
