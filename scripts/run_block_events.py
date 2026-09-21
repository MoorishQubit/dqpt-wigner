#!/usr/bin/env python3
"""Recompute finite Potts block events and common-primitive interpolated tables.

The default calculation directly evolves each trial state for signed and unsigned
crossings. Use --interpolation-only to derive the sampled block series, or
--sizes 50 --dt .01 --skip-unsigned for the finer-step signed-root comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq

from dqpt_wigner.event_audit import (
    assemble_event,
    direct_support,
    downward_primitive_root,
    grouped_support,
    positive_block_return,
    support_quantities,
)
from dqpt_wigner.potts_mps import evolve_to_times


def repository_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def event_row(record: dict) -> dict:
    event = record["physical"]
    competitor = record["competitor"]
    unsigned = record.get("unsigned")
    row = {
        "n_sites": record["n_sites"],
        "block_length": record["block_length"],
        "competitor": competitor,
        "physical_crossing": event["tau"],
        "support_crossing": unsigned["tau"] if unsigned else None,
        "critical_time_shift": event["tau"] - unsigned["tau"] if unsigned else None,
        "delta_support_rate_at_physical": event["delta_s"],
        "delta_sign_cost_at_physical": event["delta_q"],
        "delta_physical_rate_at_physical": event["delta_r"],
        "sign_cost_initial_at_physical": event["branches"]["0"]["sign_cost"],
        "sign_cost_competing_at_physical": event["branches"][competitor]["sign_cost"],
        "surviving_fraction_initial_at_physical": event["branches"]["0"]["surviving_fraction"],
        "surviving_fraction_competing_at_physical": event["branches"][competitor]["surviving_fraction"],
    }
    row.update({key: event[key] for key in ("dt", "requested_chi", "cutoff") if key in event})
    return row


def interpolated_events(source: Path, output: Path) -> list[dict]:
    """Apply a common linear interpolation to P, A, and negative mass."""
    frame = pd.read_csv(source)
    records = []
    for (size, length), group in frame.groupby(["n_sites", "block_length"]):
        sectors = {
            key: group.loc[group.branch == name].sort_values("time")
            for key, name in (("0", "initial"), ("1", "branch1"), ("2", "branch2"))
        }
        times = sectors["0"].time.to_numpy()
        for sector in sectors.values():
            if not np.array_equal(times, sector.time.to_numpy()):
                raise ValueError("All sectors must share the same sampled time grid")

        def at(tau: float) -> dict:
            branches = {
                key: support_quantities(
                    *(float(np.interp(tau, sector.time, sector[column]))
                      for column in ("probability", "unsigned_weight", "negative_mass")),
                    int(length),
                )
                for key, sector in sectors.items()
            }
            branches["1+2"] = grouped_support(branches["1"], branches["2"], int(length))
            return branches

        for competitor in ("1", "1+2"):
            events = {}
            for kind, column in (("physical", "probability"), ("unsigned", "unsigned_weight")):
                other = sectors["1"][column].to_numpy().copy()
                if competitor == "1+2":
                    other += sectors["2"][column].to_numpy()
                tau, index = downward_primitive_root(times, sectors["0"][column].to_numpy() - other)
                event = assemble_event(tau, int(length), at(tau), competitor)
                event.update({
                    "n_sites": int(size), "kind": kind,
                    "root_bracket": times[index:index + 2].tolist(),
                    "event_evaluation": "piecewise-linear interpolation of P, A and independently accumulated negative mass",
                    "probability_source": "Wigner signed sum; direct projector contraction is checked by evolved-state events",
                })
                events[kind] = event
            records.append({
                "n_sites": int(size), "block_length": int(length), "competitor": competitor,
                **events, "tau_displacement": events["physical"]["tau"] - events["unsigned"]["tau"],
            })
    pd.DataFrame([event_row(record) for record in records]).to_csv(
        output / "potts_block_crossings_consistent.csv", index=False
    )
    write_json(output / "potts_interpolated_events.json", {
        "source": repository_path(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "records": records,
    })
    return records


def save_checkpoint(state, path: Path) -> None:
    arrays = {f"gamma_{i}": tensor for i, tensor in enumerate(state.gammas)}
    arrays.update({f"lambda_{i}": tensor for i, tensor in enumerate(state.lambdas)})
    arrays["norm_before_final_normalization"] = np.array(state.norm())
    arrays["gamma_0"] = arrays["gamma_0"] / np.sqrt(state.norm())
    np.savez_compressed(path, **arrays)


def direct_events(args, config: dict, estimates: list[dict]) -> list[dict]:
    J = float(config["model"]["J"])
    h = float(config["model"]["final_field"])
    cutoff = float(config["block_hierarchy"]["cutoff"])
    records = []
    for size in args.sizes:
        started = time.perf_counter()
        checkpoints = {
            round(tau, 12): (state, discarded)
            for tau, state, discarded in evolve_to_times(
                size, np.arange(0, args.tmax + 1e-7, args.dt),
                J, h, args.dt, args.max_bond, cutoff,
            )
        }
        length, start = args.block_length, (size - args.block_length) // 2

        def at(tau: float):
            base = min(round(float(np.floor((tau + 1e-12) / args.dt) * args.dt), 12), max(checkpoints))
            state, discarded = checkpoints[base]
            state = state.copy()
            if tau - base > 1e-12:
                discarded += state.step(J, h, tau - base, args.max_bond, cutoff).discarded_weight
            return state, discarded

        for competitor in ("1+2", "1"):
            estimate = next((record for record in estimates if record["n_sites"] == size
                             and record["block_length"] == length
                             and record["competitor"] == competitor), None)
            events = {}
            for kind in ("physical", "unsigned"):
                if kind == "unsigned" and args.skip_unsigned:
                    continue
                evaluations, support_cache = [], {}

                def difference(tau: float) -> float:
                    state, _ = at(tau)
                    if kind == "physical":
                        values = [positive_block_return(state, start, length, a) for a in range(3)]
                    else:
                        supports = {str(a): direct_support(state, start, length, a, args.batch_size)
                                    for a in range(3)}
                        support_cache[tau] = supports
                        values = [supports[str(a)]["unsigned_weight"] for a in range(3)]
                    delta = values[0] - values[1] - (values[2] if competitor == "1+2" else 0)
                    evaluations.append({"tau": float(tau), "primitive_difference": float(delta)})
                    return float(delta)

                bracket = estimate[kind]["root_bracket"] if estimate else None
                if bracket is None or not difference(bracket[0]) >= 0 >= difference(bracket[1]):
                    times = np.array(sorted(checkpoints))
                    values = np.array([difference(tau) for tau in times])
                    _, index = downward_primitive_root(times, values)
                    bracket = times[index:index + 2].tolist()
                root = brentq(difference, *bracket, xtol=args.root_tolerance, rtol=1e-13)
                state, discarded = at(root)
                branches = support_cache.get(root) or {
                    str(a): direct_support(state, start, length, a, args.batch_size) for a in range(3)
                }
                branches["1+2"] = grouped_support(branches["1"], branches["2"], length)
                event = assemble_event(J * root, length, branches, competitor)
                stem = f"N{size}_ell{length}_{competitor.replace('+', 'plus')}_{kind}_dt{args.dt}_chi{args.max_bond}"
                checkpoint = args.output_dir / f"potts_{stem}.npz"
                save_checkpoint(state, checkpoint)
                event.update({
                    "n_sites": size, "kind": kind,
                    "event_evaluation": "direct finite MPS evolved at refined root",
                    "dt": args.dt, "J": J, "h": h, "requested_chi": args.max_bond,
                    "realized_chi": state.maximum_bond_dimension, "cutoff": cutoff,
                    "root_bracket": bracket, "root_tolerance_tau": J * args.root_tolerance,
                    "root_iterations": evaluations, "mps_norm": state.norm(),
                    "cumulative_discarded_weight": discarded,
                    "checkpoint": repository_path(checkpoint),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "error_status": "Root tolerance and reconstruction errors are diagnostics; evolution step and bond refinements are separate checks.",
                })
                events[kind] = event
                write_json(args.output_dir / f"event_{stem}.json", event)
                print(json.dumps({"N": size, "ell": length, "competitor": competitor,
                                  "kind": kind, "tau": event["tau"], "delta_s": event["delta_s"],
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
            records.append({"n_sites": size, "block_length": length, "competitor": competitor, **events})
        pd.DataFrame([event_row(record) for record in records]).to_csv(
            args.output_dir / "direct_block_events.csv", index=False
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/potts_finite.yaml")
    parser.add_argument("--input", type=Path, default=ROOT / "results/potts/finite_chain/potts_block_wigner_long.csv")
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--block-length", type=int, default=9)
    parser.add_argument("--dt", type=float, default=.02)
    parser.add_argument("--max-bond", type=int, default=32)
    parser.add_argument("--tmax", type=float, default=.7)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--root-tolerance", type=float, default=2e-9)
    parser.add_argument("--skip-unsigned", action="store_true")
    parser.add_argument("--interpolation-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/reproduction/potts/events")
    args = parser.parse_args()
    if min(args.sizes) < args.block_length or args.block_length < 1:
        parser.error("Block length must lie between one and every chain size")
    if min(args.dt, args.tmax, args.root_tolerance, args.max_bond, args.batch_size) <= 0:
        parser.error("Evolution, root, and contraction parameters must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    estimates = interpolated_events(args.input, args.output_dir) if args.input.is_file() else []
    if args.interpolation_only:
        if not estimates:
            parser.error(f"Interpolation requires the sampled block table: {args.input}")
        print(f"Interpolated {len(estimates)} block comparisons in {args.output_dir}")
        return
    config = yaml.safe_load(args.config.read_text())
    direct_events(args, config, estimates)


if __name__ == "__main__":
    main()
