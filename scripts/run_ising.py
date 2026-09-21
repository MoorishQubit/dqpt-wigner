#!/usr/bin/env python3
"""Ising benchmark: symmetry-resolved return branches, echo, and DQPT-I order."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dqpt_wigner.collective_ising import (  # noqa: E402
    exact_collective_trajectory,
    late_time_order,
    mean_field_trajectory,
)
from dqpt_wigner.publication import plot_ising_benchmark  # noqa: E402
from dqpt_wigner.finite_size_scaling import first_linear_crossing  # noqa: E402

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")


def _branch_worker(payload: dict[str, Any]) -> dict[str, Any]:
    N, J, h = int(payload["N"]), float(payload["J"]), float(payload["h"])
    times = np.linspace(0.0, float(payload["tmax"]), int(payload["points"]))
    data = exact_collective_trajectory(N, J, h, times)
    frame = pd.DataFrame(data)
    frame.insert(1, "n_sites", N)
    frame.insert(2, "final_field", h)
    path = Path(payload["output_dir"]) / f"ising_branches_N{N}_h{h:.3f}.csv"
    frame.to_csv(path, index=False)
    crossing = first_linear_crossing(frame["time"], frame["delta_rate"], "negative_to_positive")
    return {
        "n_sites": N,
        "final_field": h,
        "first_branch_crossing": crossing,
        "minimum_echo": float(frame["return_echo"].min()),
        "maximum_rate": float(frame["rate_total"].max()),
        "trajectory": path.name,
    }


def _field_worker(payload: dict[str, Any]) -> dict[str, float]:
    times = np.linspace(0.0, float(payload["tmax"]), int(payload["points"]))
    traj = mean_field_trajectory(float(payload["J"]), float(payload["h"]), times)
    signed, absolute = late_time_order(times, traj["mz"], float(payload["average_start"]))
    return {
        "final_field": float(payload["h"]),
        "late_signed_magnetization": signed,
        "late_absolute_magnetization": absolute,
        "minimum_magnetization": float(np.min(traj["mz"])),
        "maximum_magnetization": float(np.max(traj["mz"])),
    }


def _map(worker, payloads: list[dict[str, Any]], jobs: int, description: str) -> list[dict[str, Any]]:
    if jobs <= 1:
        return [worker(p) for p in tqdm(payloads, desc=description)]
    context = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
        futures = [pool.submit(worker, p) for p in payloads]
        for f in tqdm(as_completed(futures), total=len(futures), desc=description):
            results.append(f.result())
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ising.yaml")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    out = ROOT / cfg["output_dir"]
    if args.smoke:
        out = out / "smoke"
        cfg["branch_dynamics"].update({"sizes": [20, 30], "final_fields": [0.7], "tmax": 1.0, "points": 101})
        cfg["mean_field_dqpt_i"].update({"field_count": 4, "tmax": 4.0, "points": 201, "average_start": 2.0})
    out.mkdir(parents=True, exist_ok=True)
    J = float(cfg["model"]["J"])
    bd = cfg["branch_dynamics"]
    branch_payloads = [
        {"N": N, "J": J, "h": h, "tmax": bd["tmax"], "points": bd["points"], "output_dir": str(out)}
        for N in bd["sizes"] for h in bd["final_fields"]
    ]
    branch_summary = pd.DataFrame(_map(_branch_worker, branch_payloads, args.jobs, "Ising branches"))
    branch_summary = branch_summary.sort_values(["final_field", "n_sites"])
    branch_summary.to_csv(out / "ising_branch_summary.csv", index=False)
    frames = [pd.read_csv(out / name) for name in branch_summary["trajectory"]]
    branch_frame = pd.concat(frames, ignore_index=True)
    branch_frame.to_csv(out / "ising_branch_trajectories.csv", index=False)

    mf = cfg["mean_field_dqpt_i"]
    fields = np.linspace(float(mf["field_min"]), float(mf["field_max"]), int(mf["field_count"]))
    field_payloads = [
        {"J": J, "h": h, "tmax": mf["tmax"], "points": mf["points"], "average_start": mf["average_start"]}
        for h in fields
    ]
    scan = pd.DataFrame(_map(_field_worker, field_payloads, args.jobs, "Ising DQPT-I scan")).sort_values("final_field")
    scan.to_csv(out / "ising_dqpt_i_scan.csv", index=False)
    high_field = float(max(bd["final_fields"]))
    figure_frame = branch_frame.loc[branch_frame["final_field"] == high_field]
    if cfg.get("plots", {}).get("enabled", True):
        plot_ising_benchmark(scan, figure_frame, out / "ising_branch_and_order_benchmark")
    summary = {
        "model": cfg["model"],
        "branch_summary": branch_summary.to_dict(orient="records"),
        "dqpt_i_zero_estimate": first_linear_crossing(
            scan["final_field"], scan["late_signed_magnetization"], "positive_to_negative"
        ),
        "files": {
            "branch_summary": "ising_branch_summary.csv",
            "branch_trajectories": "ising_branch_trajectories.csv",
            "dqpt_i_scan": "ising_dqpt_i_scan.csv",
        },
    }
    (out / "ising_benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
