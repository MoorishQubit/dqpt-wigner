"""Symmetry-resolved return branches and their Wigner support/sign anatomy."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import multiprocessing as mp
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .observables import partial_trace_pure, potts_complex_order, rate_from_probability
from .wigner_odd import (
    SupportMap,
    mana,
    sum_negativity,
    support_map,
    support_metrics_pure,
    wigner_from_density_matrix,
)


@dataclass(frozen=True)
class CrossingEvent:
    event_index: int
    from_group: str
    to_group: str
    physical_time: float
    support_time: float | None
    critical_time_shift: float | None
    jump_total: float
    jump_support: float
    jump_sign: float
    jump_closure_error: float
    sign_magnitude_fraction: float
    sign_signed_share: float | None
    sign_cost_from: float
    sign_cost_to: float
    sign_cost_difference: float
    mechanism: str


@dataclass(frozen=True)
class ReturnDefinition:
    branch_labels: Mapping[str, tuple[int, ...]]
    groups: Mapping[str, tuple[str, ...]]

    def validate(self) -> None:
        branch_names = set(self.branch_labels)
        used: list[str] = []
        for group, members in self.groups.items():
            if not members:
                raise ValueError(f"return group {group!r} is empty")
            missing = set(members) - branch_names
            if missing:
                raise ValueError(f"group {group!r} contains unknown branches: {sorted(missing)}")
            used.extend(members)
        if len(used) != len(set(used)):
            raise ValueError("a return branch occurs in more than one group")


def default_potts_return_definition(n_sites: int) -> ReturnDefinition:
    branches = {f"sector_{sector}": (sector,) * n_sites for sector in range(3)}
    # Charge conjugation makes sectors 1 and 2 equivalent for the standard quench.
    groups = {"initial": ("sector_0",), "other": ("sector_1", "sector_2")}
    definition = ReturnDefinition(branches, groups)
    definition.validate()
    return definition


def _rates(probability: float, absolute_weight: float, n_sites: int, floor: float) -> tuple[float, float, float]:
    branch_rate = float(rate_from_probability(probability, n_sites, floor=floor))
    support_rate = float(rate_from_probability(absolute_weight, n_sites, floor=floor))
    if probability > floor and absolute_weight > floor:
        sign_rate = float(np.log(absolute_weight / probability) / n_sites)
    elif probability <= floor and absolute_weight <= floor:
        sign_rate = 0.0
    else:
        sign_rate = float(np.log(absolute_weight / floor) / n_sites)
    return branch_rate, support_rate, sign_rate


_ANALYZER: dict[str, object] = {}


def _init_state_analyzer(
    mappings: Mapping[str, SupportMap],
    groups: Mapping[str, tuple[str, ...]],
    n_sites: int,
    local_sites: int,
    floor: float,
) -> None:
    _ANALYZER.clear()
    _ANALYZER.update(
        mappings=dict(mappings),
        groups=dict(groups),
        n_sites=int(n_sites),
        local_sites=int(local_sites),
        floor=float(floor),
    )


def _analyze_state_task(task: tuple[int, float, np.ndarray]) -> tuple[int, dict[str, float | str]]:
    index, time, psi = task
    mappings: dict[str, SupportMap] = _ANALYZER["mappings"]  # type: ignore[assignment]
    groups: dict[str, tuple[str, ...]] = _ANALYZER["groups"]  # type: ignore[assignment]
    n_sites = int(_ANALYZER["n_sites"])
    local_sites = int(_ANALYZER["local_sites"])
    floor = float(_ANALYZER["floor"])

    row: dict[str, float | str] = {"time": float(time)}
    order = potts_complex_order(psi, n_sites)
    row.update(
        order_real=float(order.real),
        order_imag=float(order.imag),
        order_magnitude=float(abs(order)),
        order_parallel=float(order.real),
    )

    branch_values: dict[str, dict[str, float]] = {}
    max_reconstruction_error = 0.0
    max_imaginary_residual = 0.0
    for name, mapping in mappings.items():
        metrics = support_metrics_pure(psi, mapping)
        branch_values[name] = {
            "P": metrics.probability,
            "A": metrics.absolute_weight,
            "nu": metrics.negative_mass,
        }
        max_reconstruction_error = max(max_reconstruction_error, metrics.reconstruction_error)
        max_imaginary_residual = max(max_imaginary_residual, metrics.imaginary_residual)
        row[f"P_branch_{name}"] = metrics.probability
        row[f"A_branch_{name}"] = metrics.absolute_weight
        row[f"nu_branch_{name}"] = metrics.negative_mass
        row[f"Wmin_branch_{name}"] = metrics.minimum

    group_rates: dict[str, float] = {}
    group_support_rates: dict[str, float] = {}
    for group, members in groups.items():
        probability = float(sum(branch_values[member]["P"] for member in members))
        absolute_weight = float(sum(branch_values[member]["A"] for member in members))
        negative_mass = float(sum(branch_values[member]["nu"] for member in members))
        branch_rate, support_rate, sign_rate = _rates(probability, absolute_weight, n_sites, floor)
        row[f"P_{group}"] = probability
        row[f"A_{group}"] = absolute_weight
        row[f"nu_{group}"] = negative_mass
        row[f"r_{group}"] = branch_rate
        row[f"s_{group}"] = support_rate
        row[f"q_{group}"] = sign_rate
        group_rates[group] = branch_rate
        group_support_rates[group] = support_rate

    total_probability = float(sum(float(row[f"P_{group}"]) for group in groups))
    total_absolute_weight = float(sum(float(row[f"A_{group}"]) for group in groups))
    row["return_probability"] = total_probability
    row["return_rate"] = float(rate_from_probability(total_probability, n_sites, floor=floor))
    row["unsigned_return_weight"] = total_absolute_weight
    row["unsigned_return_rate"] = float(rate_from_probability(total_absolute_weight, n_sites, floor=floor))
    row["envelope_rate"] = float(min(group_rates.values()))
    row["support_envelope_rate"] = float(min(group_support_rates.values()))
    row["dominant_group"] = min(group_rates, key=group_rates.get)
    row["dominant_support_group"] = min(group_support_rates, key=group_support_rates.get)
    row["wigner_reconstruction_error"] = max_reconstruction_error
    row["wigner_imaginary_residual"] = max_imaginary_residual

    if local_sites > 0:
        keep = tuple(range(min(local_sites, n_sites)))
        rho_local = partial_trace_pure(
            psi, local_dimension=3, n_sites=n_sites, keep=keep
        )
        local_wigner = wigner_from_density_matrix(rho_local, 3, len(keep))
        row["local_wigner_mana"] = mana(local_wigner)
        row["local_wigner_negativity"] = sum_negativity(local_wigner)
    return index, row


def analyze_potts_states(
    states: np.ndarray,
    times: np.ndarray,
    definition: ReturnDefinition,
    *,
    jobs: int = 1,
    local_sites: int = 0,
    progress: bool = True,
    floor: float = 1e-300,
) -> pd.DataFrame:
    """Analyze a state batch, optionally with multiprocessing."""
    definition.validate()
    states = np.asarray(states, dtype=complex)
    times = np.asarray(times, dtype=float)
    if states.ndim != 2 or states.shape[0] != len(times):
        raise ValueError("states must have shape (len(times), Hilbert dimension)")
    n_sites = len(next(iter(definition.branch_labels.values())))
    mappings = {name: support_map(label, 3) for name, label in definition.branch_labels.items()}
    tasks = [(index, float(time), states[index]) for index, time in enumerate(times)]
    rows: list[dict[str, float | str] | None] = [None] * len(tasks)

    if jobs <= 1:
        _init_state_analyzer(mappings, definition.groups, n_sites, local_sites, floor)
        for task in tqdm(tasks, desc="Wigner return supports", disable=not progress):
            index, row = _analyze_state_task(task)
            rows[index] = row
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=context,
            initializer=_init_state_analyzer,
            initargs=(mappings, definition.groups, n_sites, local_sites, floor),
        ) as executor:
            futures = [executor.submit(_analyze_state_task, task) for task in tasks]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Wigner return supports", disable=not progress
            ):
                index, row = future.result()
                rows[index] = row
    return pd.DataFrame([row for row in rows if row is not None]).sort_values("time").reset_index(drop=True)


def _linear_root(t0: float, t1: float, y0: float, y1: float) -> float:
    if y1 == y0:
        return 0.5 * (t0 + t1)
    return float(t0 - y0 * (t1 - t0) / (y1 - y0))


def _interpolate(times: np.ndarray, values: np.ndarray, time: float) -> float:
    return float(np.interp(time, times, values))


def _polynomial_derivative(
    times: np.ndarray,
    values: np.ndarray,
    time: float,
    *,
    half_window_points: int = 6,
    degree: int = 3,
) -> float:
    nearest = int(np.argmin(np.abs(times - time)))
    start = max(0, nearest - half_window_points)
    stop = min(len(times), nearest + half_window_points + 1)
    x = times[start:stop] - time
    y = values[start:stop]
    actual_degree = min(degree, len(x) - 1)
    if actual_degree < 1:
        return 0.0
    coefficients = np.polyfit(x, y, actual_degree)
    return float(coefficients[-2])


def _directional_crossings(
    times: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    from_first_to_second: bool,
    minimum_time: float,
) -> list[float]:
    difference = first - second
    output: list[float] = []
    for index in range(len(times) - 1):
        if times[index + 1] < minimum_time:
            continue
        left, right = difference[index], difference[index + 1]
        correct = (left <= 0.0 and right > 0.0) if from_first_to_second else (left >= 0.0 and right < 0.0)
        if correct:
            output.append(_linear_root(times[index], times[index + 1], left, right))
    return output


def find_envelope_crossings(
    frame: pd.DataFrame,
    group_names: Sequence[str],
    *,
    minimum_time: float = 1e-9,
    derivative_half_window_points: int = 6,
    support_match_max_distance: float | None = None,
) -> list[CrossingEvent]:
    """Locate dominant-branch exchanges and decompose their cusp slopes.

    The unsigned crossing is matched using the same branch pair and exchange
    direction, rather than merely choosing the nearest crossing in time.
    """
    times = frame["time"].to_numpy(float)
    rates = np.column_stack([frame[f"r_{name}"].to_numpy(float) for name in group_names])
    dominant = np.argmin(rates, axis=1)
    events: list[CrossingEvent] = []
    last_pair_time: dict[tuple[int, int], float] = {}

    for index in range(len(times) - 1):
        before = int(dominant[index])
        after = int(dominant[index + 1])
        if before == after or times[index + 1] < minimum_time:
            continue
        from_name = group_names[before]
        to_name = group_names[after]
        difference = rates[:, before] - rates[:, after]
        if not (difference[index] <= 0.0 and difference[index + 1] >= 0.0):
            continue
        physical_time = _linear_root(
            times[index], times[index + 1], difference[index], difference[index + 1]
        )
        pair = (before, after)
        if physical_time - last_pair_time.get(pair, -np.inf) < 0.5 * np.min(np.diff(times)):
            continue
        last_pair_time[pair] = physical_time

        support_from = frame[f"s_{from_name}"].to_numpy(float)
        support_to = frame[f"s_{to_name}"].to_numpy(float)
        candidates = _directional_crossings(
            times,
            support_from,
            support_to,
            from_first_to_second=True,
            minimum_time=minimum_time,
        )
        if support_match_max_distance is None:
            support_match_max_distance = max(10.0 * float(np.median(np.diff(times))), 0.5)
        nearby = [candidate for candidate in candidates if abs(candidate - physical_time) <= support_match_max_distance]
        support_time = min(nearby, key=lambda candidate: abs(candidate - physical_time)) if nearby else None

        derivatives: dict[str, tuple[float, float]] = {}
        for component in ("r", "s", "q"):
            from_values = frame[f"{component}_{from_name}"].to_numpy(float)
            to_values = frame[f"{component}_{to_name}"].to_numpy(float)
            derivatives[component] = (
                _polynomial_derivative(
                    times,
                    from_values,
                    physical_time,
                    half_window_points=derivative_half_window_points,
                ),
                _polynomial_derivative(
                    times,
                    to_values,
                    physical_time,
                    half_window_points=derivative_half_window_points,
                ),
            )
        jump_total = derivatives["r"][1] - derivatives["r"][0]
        jump_support = derivatives["s"][1] - derivatives["s"][0]
        jump_sign = derivatives["q"][1] - derivatives["q"][0]
        denominator = abs(jump_support) + abs(jump_sign)
        magnitude_fraction = abs(jump_sign) / denominator if denominator > 0.0 else 0.0
        signed_share = jump_sign / jump_total if abs(jump_total) > 1e-14 else None
        q_from = _interpolate(times, frame[f"q_{from_name}"].to_numpy(float), physical_time)
        q_to = _interpolate(times, frame[f"q_{to_name}"].to_numpy(float), physical_time)
        shift = physical_time - support_time if support_time is not None else None

        dt = float(np.median(np.diff(times)))
        if support_time is None:
            mechanism = "sign-enabled"
        elif abs(shift) <= 2.0 * dt and magnitude_fraction < 0.2:
            mechanism = "support-driven"
        elif magnitude_fraction >= 0.8:
            mechanism = "sign-dominated"
        else:
            mechanism = "mixed/sign-shifted"

        events.append(
            CrossingEvent(
                event_index=len(events),
                from_group=from_name,
                to_group=to_name,
                physical_time=physical_time,
                support_time=support_time,
                critical_time_shift=shift,
                jump_total=jump_total,
                jump_support=jump_support,
                jump_sign=jump_sign,
                jump_closure_error=jump_total - jump_support - jump_sign,
                sign_magnitude_fraction=magnitude_fraction,
                sign_signed_share=signed_share,
                sign_cost_from=q_from,
                sign_cost_to=q_to,
                sign_cost_difference=q_to - q_from,
                mechanism=mechanism,
            )
        )
    return events


def events_to_frame(events: Sequence[CrossingEvent]) -> pd.DataFrame:
    return pd.DataFrame([asdict(event) for event in events])
