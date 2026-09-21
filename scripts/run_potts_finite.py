#!/usr/bin/env python3
"""High-N Potts TEBD, bulk-block Wigner hierarchy, convergence, and claim gates."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dqpt_wigner.potts_mps import (  # noqa: E402
    combine_support_diagnostics,
    diagnose_wigner_line,
    evolve_to_times,
)
from dqpt_wigner.finite_size_scaling import (  # noqa: E402
    block_crossing_table,
    first_linear_crossing,
    global_crossing_summary,
    polynomial_limit,
    support_rows,
    wide_block_table,
)
from dqpt_wigner.publication import (  # noqa: E402
    plot_block_sign_cost,
    plot_potts_convergence,
    plot_potts_thermodynamic,
)

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")


def _times(tmax: float, dt: float) -> np.ndarray:
    count = int(round(tmax / dt))
    return np.linspace(0.0, count * dt, count + 1)


def _global_worker(payload: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(payload["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    N = int(payload["n_sites"])
    J = float(payload["J"])
    h = float(payload["h"])
    dt = float(payload["dt"])
    tmax = float(payload["tmax"])
    chi = int(payload["max_bond"])
    cutoff = float(payload["cutoff"])
    label = str(payload["label"])
    times = _times(tmax, dt)
    rows: list[dict[str, float | int]] = []
    for time, state, discarded in evolve_to_times(N, times, J, h, dt, chi, cutoff):
        rates = [state.product_rate(a) for a in range(3)]
        probs = [float(np.exp(-N * r)) if np.isfinite(r) else 0.0 for r in rates]
        grouped = probs[1] + probs[2]
        grouped_rate = -math.log(max(grouped, np.finfo(float).tiny)) / N
        total = sum(probs)
        total_rate = -math.log(max(total, np.finfo(float).tiny)) / N
        symmetry_den = max(0.5 * (probs[1] + probs[2]), np.finfo(float).tiny)
        rows.append(
            {
                "time": time,
                "n_sites": N,
                "dt": dt,
                "max_bond": chi,
                "cutoff": cutoff,
                "prob_initial": probs[0],
                "prob_branch1": probs[1],
                "prob_branch2": probs[2],
                "prob_competing_group": grouped,
                "return_echo": total,
                "rate_initial": rates[0],
                "rate_branch1": rates[1],
                "rate_branch2": rates[2],
                "rate_competing_group": grouped_rate,
                "rate_total": total_rate,
                "rate_lower_envelope": min(rates),
                "delta_rate_individual": rates[0] - 0.5 * (rates[1] + rates[2]),
                "delta_rate_grouped": rates[0] - grouped_rate,
                "branch_symmetry_relative_error": abs(probs[1] - probs[2]) / symmetry_den,
                "local_ordered_moment": state.ordered_local_moment(),
                "mps_norm": state.norm(),
                "cumulative_discarded_weight": discarded,
                "realized_maximum_bond": state.maximum_bond_dimension,
            }
        )
    frame = pd.DataFrame(rows)
    csv_path = out_dir / f"{label}_trajectory.csv"
    frame.to_csv(csv_path, index=False)
    crossings = global_crossing_summary(frame)
    summary = {
        "label": label,
        "n_sites": N,
        "dt": dt,
        "max_bond": chi,
        "cutoff": cutoff,
        **crossings,
        "maximum_symmetry_relative_error": float(frame["branch_symmetry_relative_error"].replace([np.inf], np.nan).dropna().max()),
        "maximum_norm_error": float(np.max(np.abs(frame["mps_norm"] - 1.0))),
        "maximum_cumulative_discarded_weight": float(frame["cumulative_discarded_weight"].max()),
        "trajectory": csv_path.name,
    }
    (out_dir / f"{label}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _parallel_global(cases: list[dict[str, Any]], jobs: int, description: str) -> list[dict[str, Any]]:
    if jobs <= 1:
        return [_global_worker(case) for case in tqdm(cases, desc=description)]
    context = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
        futures = {pool.submit(_global_worker, case): case for case in cases}
        for future in tqdm(as_completed(futures), total=len(futures), desc=description):
            results.append(future.result())
    return sorted(results, key=lambda x: (int(x["n_sites"]), float(x["dt"]), int(x["max_bond"])))


def _run_blocks(config: dict[str, Any], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = config["model"]
    block = config["block_hierarchy"]
    times = np.linspace(float(block["probe_time_min"]), float(block["probe_time_max"]), int(block["probe_count"]))
    long_rows: list[dict[str, object]] = []
    for N in tqdm(block["sizes"], desc="Potts bulk sizes"):
        N = int(N)
        iterator = evolve_to_times(
            N,
            times,
            float(model["J"]),
            float(model["final_field"]),
            float(block["dt"]),
            int(block["max_bond"]),
            float(block["cutoff"]),
            int(model.get("initial_symbol", 0)),
        )
        for time, state, discarded in tqdm(iterator, total=len(times), leave=False, desc=f"N={N} time"):
            for ell in block["block_lengths"]:
                ell = int(ell)
                start = (N - ell) // 2
                diagnostics = []
                for branch, name in ((0, "initial"), (1, "branch1"), (2, "branch2")):
                    values = state.block_wigner_line(start, ell, branch)
                    diagnostics.append(diagnose_wigner_line(values, ell, name))
                competing = combine_support_diagnostics(diagnostics[1:], "competing")
                long_rows.extend(
                    support_rows(
                        time,
                        N,
                        ell,
                        start,
                        [diagnostics[0], diagnostics[1], diagnostics[2], competing],
                        discarded,
                        state.maximum_bond_dimension,
                    )
                )
    long_frame = pd.DataFrame(long_rows)
    long_path = out_dir / "potts_block_wigner_long.csv"
    long_frame.to_csv(long_path, index=False)
    analysis_long = long_frame.loc[long_frame["branch"].isin(["initial", "competing"])].copy()
    wide = wide_block_table(analysis_long)
    wide_path = out_dir / "potts_block_wigner_wide.csv"
    wide.to_csv(wide_path, index=False)
    crossings = block_crossing_table(wide)
    crossings.to_csv(out_dir / "potts_block_crossings.csv", index=False)
    return long_frame, wide, crossings


def _fit_reports(global_summary: pd.DataFrame, block_crossings: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    reports: dict[str, Any] = {"global": [], "blocks": {}}
    for degree in config["fits"]["global_degrees"]:
        try:
            reports["global"].append(
                polynomial_limit(global_summary["n_sites"], global_summary["individual_crossing"], int(degree))
            )
        except ValueError:
            continue
    min_ell = int(config["block_hierarchy"].get("minimum_fit_block", 4))
    for N, group in block_crossings.groupby("n_sites"):
        g = group.loc[group["block_length"] >= min_ell].dropna(subset=["physical_crossing", "support_crossing"])
        entry: dict[str, Any] = {}
        for observable in ("physical_crossing", "support_crossing", "critical_time_shift"):
            entry[observable] = []
            for degree in config["fits"]["block_degrees"]:
                try:
                    entry[observable].append(polynomial_limit(g["block_length"], g[observable], int(degree)))
                except ValueError:
                    continue
        reports["blocks"][str(int(N))] = entry
    return reports


def _evaluate_gates(
    primary: pd.DataFrame,
    convergence: pd.DataFrame,
    long_frame: pd.DataFrame,
    block_crossings: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    gates = config["claim_gates"]
    maxN = int(primary["n_sites"].max())
    # Compare N=100 dt and bond controls against the closest matched cases.
    conv100 = convergence.loc[convergence["n_sites"] == maxN]
    dt_shift = float("nan")
    bond_shift = float("nan")
    try:
        a = conv100.loc[(conv100["dt"] == 0.04) & (conv100["max_bond"] == 32), "individual_crossing"].iloc[0]
        b = conv100.loc[(conv100["dt"] == 0.02) & (conv100["max_bond"] == 32), "individual_crossing"].iloc[0]
        dt_shift = abs(float(a - b))
    except Exception:
        pass
    try:
        a = conv100.loc[(conv100["dt"] == 0.04) & (conv100["max_bond"] == 32), "individual_crossing"].iloc[0]
        b = conv100.loc[(conv100["dt"] == 0.04) & (conv100["max_bond"] == 48), "individual_crossing"].iloc[0]
        bond_shift = abs(float(a - b))
    except Exception:
        pass
    largest = block_crossings.sort_values("block_length").groupby("n_sites").tail(1)
    largest_shift = float(largest["critical_time_shift"].min())
    size_spread = float(largest["critical_time_shift"].max() - largest["critical_time_shift"].min())
    comp_q = float(largest["sign_cost_competing_at_physical"].min())
    sym = long_frame.loc[long_frame["branch"].isin(["branch1", "branch2"])].pivot_table(
        index=["n_sites", "time", "block_length"], columns="branch", values="probability"
    ).dropna()
    sym_error = float(np.max(np.abs(sym["branch1"] - sym["branch2"]) / np.maximum(0.5 * (sym["branch1"] + sym["branch2"]), np.finfo(float).tiny)))
    decomp = float(np.max(np.abs(long_frame["identity_r_minus_s_minus_q"])))
    norm = float(primary["maximum_norm_error"].max())
    values = {
        "maximum_size": maxN,
        "dt_crossing_shift_Nmax": dt_shift,
        "bond_crossing_shift_Nmax": bond_shift,
        "bulk_size_shift_difference": size_spread,
        "largest_block_sign_shift_minimum": largest_shift,
        "competing_sign_cost_at_physical_minimum": comp_q,
        "branch_symmetry_relative_error": sym_error,
        "decomposition_error": decomp,
        "norm_error": norm,
    }
    passed = {
        "reaches_N100": maxN >= int(gates["required_maximum_size"]),
        "time_step_converged": np.isfinite(dt_shift) and dt_shift <= float(gates["maximum_dt_crossing_shift"]),
        "bond_dimension_converged": np.isfinite(bond_shift) and bond_shift <= float(gates["maximum_bond_crossing_shift"]),
        "bulk_size_consistent": size_spread <= float(gates["maximum_bulk_size_shift_difference"]),
        "sign_shift_resolved": largest_shift >= float(gates["minimum_largest_block_sign_shift"]),
        "sign_cost_resolved": comp_q >= float(gates["minimum_competing_sign_cost_at_crossing"]),
        "branch_symmetry_control": sym_error <= float(gates["maximum_branch_symmetry_relative_error"]),
        "exact_decomposition": decomp <= float(gates["maximum_decomposition_error"]),
        "mps_norm_control": norm <= float(gates["maximum_norm_error"]),
    }
    return {"values": values, "passed": {key: bool(value) for key, value in passed.items()},
            "all_passed": bool(all(passed.values()))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/potts_finite.yaml")
    parser.add_argument("--jobs", type=int, default=1, help="parallel independent global cases")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-blocks", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text())
    out_dir = ROOT / config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        config["primary"].update({"sizes": [8, 12], "tmax": 0.20, "dt": 0.05, "max_bond": 12})
        config["convergence_cases"] = [{"n_sites": 8, "dt": 0.05, "max_bond": 12, "cutoff": 1e-9}]
        config["block_hierarchy"].update(
            {"sizes": [8], "block_lengths": [2, 3], "probe_time_min": 0.05, "probe_time_max": 0.15, "probe_count": 3, "dt": 0.05, "max_bond": 12}
        )
        out_dir = out_dir / "smoke"
        out_dir.mkdir(parents=True, exist_ok=True)
    model = config["model"]
    primary_cases = [
        {
            "output_dir": str(out_dir),
            "label": f"primary_N{int(N)}",
            "n_sites": int(N),
            "J": model["J"],
            "h": model["final_field"],
            "tmax": config["primary"]["tmax"],
            "dt": config["primary"]["dt"],
            "max_bond": config["primary"]["max_bond"],
            "cutoff": config["primary"]["cutoff"],
        }
        for N in config["primary"]["sizes"]
    ]
    primary_results = _parallel_global(primary_cases, args.jobs, "Potts primary")
    primary = pd.DataFrame(primary_results)
    if not args.smoke:
        fit = polynomial_limit(primary["n_sites"], primary["individual_crossing"], 1)
        primary["fit_individual_limit"] = fit["limit"]
    primary.to_csv(out_dir / "potts_global_summary.csv", index=False)
    trajectories = pd.concat(
        [pd.read_csv(out_dir / item["trajectory"]) for item in primary_results], ignore_index=True
    )
    trajectories.to_csv(out_dir / "potts_global_trajectories.csv", index=False)

    convergence_cases = []
    for i, case in enumerate(config["convergence_cases"]):
        convergence_cases.append(
            {
                "output_dir": str(out_dir),
                "label": f"convergence_{i}_N{int(case['n_sites'])}_dt{case['dt']}_chi{int(case['max_bond'])}",
                "n_sites": int(case["n_sites"]),
                "J": model["J"],
                "h": model["final_field"],
                "tmax": config["primary"]["tmax"],
                "dt": float(case["dt"]),
                "max_bond": int(case["max_bond"]),
                "cutoff": float(case["cutoff"]),
            }
        )
    convergence_results = _parallel_global(convergence_cases, args.jobs, "Potts convergence")
    convergence = pd.DataFrame(convergence_results)
    convergence.to_csv(out_dir / "potts_convergence_summary.csv", index=False)

    if args.skip_blocks:
        print(primary.to_string(index=False))
        return
    long_frame, wide, crossings = _run_blocks(config, out_dir)
    fits = _fit_reports(primary, crossings, config)
    (out_dir / "potts_scaling_fits.json").write_text(json.dumps(fits, indent=2) + "\n")
    gates = _evaluate_gates(primary, convergence, long_frame, crossings, config)
    (out_dir / "claim_gates.json").write_text(json.dumps(gates, indent=2) + "\n")

    if config.get("plots", {}).get("enabled", True):
        plot_potts_thermodynamic(trajectories, primary, crossings, out_dir / "potts_thermodynamic_scaling")
        plot_potts_convergence(convergence, out_dir / "potts_tebd_convergence")
        plot_block_sign_cost(wide, out_dir / "potts_block_sign_cost", int(max(config["block_hierarchy"]["sizes"])))
    summary = {
        "model": model,
        "global": primary_results,
        "fits": fits,
        "claim_gates": gates,
        "files": {
            "global_summary": "potts_global_summary.csv",
            "global_trajectories": "potts_global_trajectories.csv",
            "convergence": "potts_convergence_summary.csv",
            "block_long": "potts_block_wigner_long.csv",
            "block_wide": "potts_block_wigner_wide.csv",
            "block_crossings": "potts_block_crossings.csv",
        },
    }
    (out_dir / "potts_tn_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
