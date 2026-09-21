#!/usr/bin/env python3
"""Derive compact supplemental plot and table inputs from numerical run records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def signed_data(base: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = []
    for path in sorted((base / "potts/infinite_chain").glob("*/*/event_report.json")):
        run = read_json(path)
        config = run["configuration"]
        if config["model"] == "potts" and config["max_bond"] >= 48 and run.get("event_found"):
            runs.append((path.parent, run))
    if not runs:
        raise ValueError("No completed infinite-chain Potts signed events found")
    runs.sort(key=lambda pair: (-pair[1]["configuration"]["dtau"],
                               pair[1]["configuration"]["max_bond"],
                               -pair[1]["configuration"]["cutoff"]))
    refinements = pd.DataFrame([{
        "run_id": run["run_id"], "dtau": run["configuration"]["dtau"],
        "cap": run["configuration"]["max_bond"], "cutoff": run["configuration"]["cutoff"],
        "bond_a": run["actual_max_bonds"][0], "bond_b": run["actual_max_bonds"][1],
        "root": run["root_tau"], "slope": run["slope_audit"][-1]["difference_slope"],
    } for _, run in runs])
    directory, _ = min(runs, key=lambda pair: (pair[1]["configuration"]["dtau"],
                        pair[1]["configuration"]["cutoff"], -pair[1]["configuration"]["max_bond"]))
    measurements = [json.loads(line) for line in (directory / "measurements.jsonl").read_text().splitlines()]
    trajectory = pd.DataFrame([{"tau": row["tau"], **{
        f"r{index}": branch["rate"] for index, branch in enumerate(row["branches"])
    }} for row in measurements])
    return trajectory, refinements


def block_data(base: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = base / "potts/events"
    frame = pd.read_csv(events / "potts_block_crossings_consistent.csv")
    grouped = frame.loc[(frame.n_sites == 100) & (frame.competitor == "1+2")].sort_values("block_length")
    long = pd.read_csv(base / "potts/finite_chain/potts_block_wigner_long.csv")
    long = long.loc[(long.n_sites == 100) & (long.block_length == 9)]
    zero = long.loc[long.branch == "initial"].set_index("time")
    one = long.loc[long.branch == "branch1"].set_index("time")
    trace = pd.DataFrame([{
        "tau": time,
        "delta_r": float(zero.loc[time, "physical_rate"] - one.loc[time, "physical_rate"]),
        "delta_s": float(zero.loc[time, "support_rate"] - one.loc[time, "support_rate"]),
        "delta_q": float(zero.loc[time, "sign_cost"] - one.loc[time, "sign_cost"]),
    } for time in zero.index])
    direct = [read_json(path) for path in sorted(events.glob("event_*.json"))]
    physical = sorted((event for event in direct if event["block_length"] == 9
                      and event["kind"] == "physical" and event["dt"] == .02),
                      key=lambda event: (event["n_sites"], event["competitor"]))
    rows = []
    for event in physical:
        unsigned = next(item for item in direct if item["kind"] == "unsigned"
                        and item["n_sites"] == event["n_sites"]
                        and item["block_length"] == event["block_length"]
                        and item["competitor"] == event["competitor"] and item["dt"] == .02)
        branch = event["branches"][event["competitor"]]
        rows.append({"N": event["n_sites"], "ell": 9, "comparator": event["competitor"],
            "dtau": event["dt"], "cap": event["requested_chi"], "tau_c": event["tau"],
            "tau_s": unsigned["tau"], "delta_s": event["delta_s"], "delta_q": event["delta_q"],
            "q_comp": branch["sign_cost"], "survival_comp": branch["surviving_fraction"]})
    if grouped.empty or trace.empty or not rows:
        raise ValueError("Supplemental block data require sampled and directly reevaluated events")
    return trace, grouped, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/reproduction/supplement")
    args = parser.parse_args()
    base = (ROOT / args.results_root).resolve()
    out = (ROOT / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    trajectory, refinements = signed_data(base)
    trace, grouped, direct = block_data(base)
    for name, frame in {
        "potts_signed_trajectory": trajectory,
        "potts_signed_refinements": refinements,
        "potts_individual_block_trace": trace,
        "potts_grouped_block_events": grouped,
        "potts_direct_blocks": direct,
    }.items():
        frame.to_csv(out / f"{name}.csv", index=False, float_format="%.17g")
    print(f"Five supplemental data tables written to {out}")


if __name__ == "__main__":
    main()
