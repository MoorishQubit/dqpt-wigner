#!/usr/bin/env python3
"""Run and resume infinite-MPS signed Potts branch-exchange calculations."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scipy.optimize import brentq

from dqpt_wigner.itebd_thermo import (
    InfiniteMPS, boundary_asymptotic_check, bulk_return_asymptotic_check, measure_state,
    norm_spectrum_diagnostics, transfer_fixed_points,
)
from dqpt_wigner.paths import repository_path


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_provenance() -> dict:
    source_paths = [Path(__file__), ROOT / "src/dqpt_wigner/itebd_thermo.py",
                    ROOT / "src/dqpt_wigner/potts_mps.py"]
    return {"recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(),
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                          capture_output=True, text=True).stdout.strip(),
            "sources": {repository_path(p): sha256(p) for p in source_paths if p.exists()},
            "command": sys.argv}


def _evolve_from(state: InfiniteMPS, target: float, cfg: dict) -> InfiniteMPS:
    current = state.copy()
    if target < current.tau - 1e-12:
        raise ValueError("cannot evolve backwards from checkpoint")
    while target - current.tau > 2e-14:
        step = min(cfg["dtau"], target - current.tau)
        current.step(step, cfg["max_bond"], cfg["cutoff"], cfg["model"], cfg["field"])
    return current


def _checkpoint_for(target: float, out: Path) -> tuple[InfiniteMPS, Path]:
    candidates = []
    for path in (out / "checkpoints").glob("tau_*.npz"):
        stored_time = float(path.stem.split("_", 1)[1])
        if stored_time <= target + 1e-10:
            candidates.append((stored_time, path))
    if not candidates:
        raise RuntimeError(f"no preceding checkpoint for tau={target}")
    path = max(candidates)[1]
    state, _ = InfiniteMPS.load(path)
    return state, path


def analyze_run(cfg: dict, out: Path) -> dict:
    records = [json.loads(line) for line in (out / "measurements.jsonl").read_text().splitlines()]
    records = sorted({record["tau"]: record for record in records}.values(), key=lambda x: x["tau"])
    bracket = None
    for first, second in zip(records[:-1], records[1:]):
        if first["delta_rate"] is not None and second["delta_rate"] is not None and first["delta_rate"] <= 0 < second["delta_rate"]:
            bracket = (first, second)
            break
    result = {"run_id": cfg["run_id"], "configuration": cfg,
              "analysis_provenance": source_provenance(),
              "classification": "direct iMPS numerical evidence, not a rigorous Hamiltonian-dynamics certificate",
              "time_variable": "Omega*t" if cfg["model"] == "rotation" else "tau=J*t",
              "signed_global_limit": "N=2L periodic represented-MPS family, L->infinity; unique contributing transfer sector required",
              "event_found": bracket is not None,
              "measurement_count": len(records),
              "actual_max_bonds": [max(r["bonds"][k] for r in records) for k in range(2)],
              "maximum_branch_symmetry_error": max(r["branch_symmetry_error"] or 0 for r in records),
              "maximum_norm_radius_deviation": max(abs(r["norm"]["eta"] - 1) for r in records),
              "uncertainty_interpretation": "refinement differences are empirical sensitivities, not statistical confidence intervals or rigorous bounds"}
    if bracket is None:
        write_json(out / "event_report.json", result)
        return result
    first, second = bracket
    left_time, right_time = first["tau"], second["tau"]
    base, base_path = _checkpoint_for(left_time, out)
    evaluations = []

    def difference(target):
        trial = _evolve_from(base, float(target), cfg)
        measured = measure_state(trial)
        evaluations.append({"tau": target, "delta_rate": measured["delta_rate"],
                            "rates": [b["rate"] for b in measured["branches"]]})
        return measured["delta_rate"]

    root_tolerance = 2e-11
    root = brentq(difference, left_time, right_time, xtol=root_tolerance, rtol=1e-13)
    root_state = _evolve_from(base, root, cfg)
    event = measure_state(root_state)
    root_state.save(out / "event_state.npz", {**cfg, "event": "first resolved upward r0-r1 exchange in requested window",
                                              "root_base_checkpoint": repository_path(base_path),
                                              "root_tolerance": root_tolerance})
    linear_root = left_time - first["delta_rate"] * (right_time - left_time) / (second["delta_rate"] - first["delta_rate"])
    step = min(1e-4, .2 * (root - left_time), .2 * (right_time - root))
    derivative_rows = []
    for delta in (step, step / 2):
        minus = measure_state(_evolve_from(base, root - delta, cfg))
        plus = measure_state(_evolve_from(base, root + delta, cfg))
        slopes = [(plus["branches"][j]["rate"] - minus["branches"][j]["rate"]) / (2 * delta) for j in range(3)]
        derivative_rows.append({"delta_tau": delta, "branch_slopes": slopes,
                                "difference_slope": slopes[0] - slopes[1],
                                "cusp_derivative_jump_post_minus_pre": slopes[1] - slopes[0]})
    a, b = root_state.cell_tensors()
    spectral = norm_spectrum_diagnostics(a, b)
    boundary = {str(symbol): boundary_asymptotic_check(a, b, symbol, tuple(cfg.get("finite_cells", [8, 32, 64])))
                for symbol in range(3)}
    bulk_boundary = {str(symbol): bulk_return_asymptotic_check(a, b, symbol, (1, 2, 4, 8, 16, 32, 64))
                     for symbol in range(3)}
    checkpoints = []
    for target in cfg.get("evaluation_times", []):
        parent, path = _checkpoint_for(target, out)
        measured_state = _evolve_from(parent, target, cfg)
        measured = measure_state(measured_state)
        checkpoints.append({"base_checkpoint": repository_path(path), **measured})
    unique = all(branch["unique_leading_modulus_numerically"] for branch in event["branches"])
    finite_check = max(abs(row["prediction_residual"]) for group in boundary.values()
                       for row in group["finite_contractions"][-1:])
    result.update({"root_tau": root, "event": event, "event_checkpoint": repository_path(out / "event_state.npz"),
                   "root_bracket": [left_time, right_time], "linear_grid_root": linear_root,
                   "grid_linear_vs_replayed_root": root - linear_root,
                   "root_absolute_tolerance": root_tolerance, "root_evaluations": evaluations,
                   "slope_audit": derivative_rows,
                   "slope_step_sensitivity": abs(derivative_rows[0]["difference_slope"] - derivative_rows[1]["difference_slope"]),
                   "norm_peripheral_spectrum": spectral, "boundary_selection_audit": boundary,
                   "signed_bulk_global_comparison": bulk_boundary,
                   "largest_finite_length_prefactor_residual": finite_check,
                   "common_time_evaluations": checkpoints,
                   "gates": {"transverse_signed_exchange_numerically": bool(unique and derivative_rows[-1]["difference_slope"] > 0),
                             "sector2_independently_contracted": True,
                             "sector2_symmetry_error": event["branch_symmetry_error"],
                             "unsigned_interference_gap": "not assessed by this signed-transfer runner",
                             "unsigned_root_displacement": "not assessed",
                             "exact_state_error_bound": False,
                             "numerical_leading_sector_criteria": unique,
                             "open_boundary_leading_coefficients_nonzero": all(x["return_boundary_coefficient_abs"] > 1e-10 for x in boundary.values())},
                   "limitations": ["Schmidt cutoff and bond cap errors are checked by refinements, not bounded rigorously.",
                                   "Nonnormal eigenvalue condition and residual are diagnostics, not enclosures.",
                                   "Finite open tensor contractions use specified virtual boundaries, not independently evolved open-chain boundaries.",
                                   "Global signed periodic rates and bulk-block unsigned rates remain distinct limits unless separately justified.",
                                   "Fractional final steps define event replay; grid/root sensitivity is reported separately from uniform evolution-step sensitivity."]})
    write_json(out / "event_report.json", result)
    return result


def run_one(cfg: dict, base_output: Path, resume: bool, max_seconds: float | None) -> dict:
    out = base_output / cfg["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "configuration.json"
    canonical = clean(cfg)
    if config_path.exists() and json.loads(config_path.read_text()) != canonical:
        raise RuntimeError(f"configuration differs from existing run {out}; choose a new run_id")
    write_json(config_path, canonical)
    config_hash = sha256(config_path)
    latest = out / "latest.npz"
    event_path = out / "event_report.json"
    run_path = out / "run_report.json"
    if resume and event_path.exists() and run_path.exists():
        old_run = json.loads(run_path.read_text())
        if old_run.get("complete"):
            return json.loads(event_path.read_text())
    if latest.exists():
        if not resume:
            raise RuntimeError(f"existing checkpoint {latest}; use --resume or a fresh run_id")
        state, metadata = InfiniteMPS.load(latest)
        if metadata["configuration_sha256"] != config_hash:
            raise RuntimeError("checkpoint configuration hash mismatch")
    else:
        state = InfiniteMPS.product()
        if (out / "measurements.jsonl").exists():
            raise RuntimeError("measurements exist without resume checkpoint")
    started = time.perf_counter()
    provenance = source_provenance()
    start_tau = state.tau
    metadata = {**cfg, "configuration_sha256": config_hash}
    complete = False
    with (out / "measurements.jsonl").open("a") as stream:
        while state.tau < cfg["tmax"] - 1e-12:
            if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                break
            delta = min(cfg["dtau"], cfg["tmax"] - state.tau)
            state.step(delta, cfg["max_bond"], cfg["cutoff"], cfg["model"], cfg["field"])
            if state.tau >= cfg["measure_start"] - 1e-12:
                record = measure_state(state)
                record["elapsed_this_invocation_seconds"] = time.perf_counter() - started
                stream.write(json.dumps(clean(record), allow_nan=False) + "\n")
                stream.flush()
                state.save(out / "checkpoints" / f"tau_{state.tau:.10f}.npz", metadata)
            # The lightweight state is checkpointed at each measured event step,
            # and at declared coarse intervals before the measurement window.
            if state.tau >= cfg["measure_start"] - 1e-12 or state.steps % 10 == 0:
                state.save(latest, metadata)
        complete = state.tau >= cfg["tmax"] - 1e-12
    state.save(latest, metadata)
    elapsed = time.perf_counter() - started
    run_record = {"run_id": cfg["run_id"], "complete": complete, "start_tau": start_tau,
                  "last_tau": state.tau, "elapsed_seconds_this_invocation": elapsed,
                  "actual_bonds": list(state.bonds), "configuration_sha256": config_hash,
                  "provenance": provenance}
    if run_path.exists():
        previous = json.loads(run_path.read_text())
        run_record["prior_invocations"] = previous.get("prior_invocations", []) + [{k: v for k, v in previous.items() if k != "prior_invocations"}]
    write_json(run_path, run_record)
    print(json.dumps({"run": cfg["run_id"], "completed": complete, "seconds": elapsed, "bonds": state.bonds}), flush=True)
    if not complete:
        return {"run_id": cfg["run_id"], "complete": False, "resumable_checkpoint": repository_path(latest)}
    result = analyze_run(cfg, out)
    run_record["analysis_elapsed_seconds"] = time.perf_counter() - started - elapsed
    write_json(run_path, run_record)
    print(json.dumps({"run": cfg["run_id"], "root": result.get("root_tau"),
                      "slope": result.get("slope_audit", [{}])[-1].get("difference_slope")}), flush=True)
    return result


def campaign_summary(reports: list[dict]) -> dict:
    successful = [r for r in reports if r.get("event_found")]
    rows = [{"run_id": r["run_id"], "model": r["configuration"]["model"],
             "dtau": r["configuration"]["dtau"], "max_bond": r["configuration"]["max_bond"],
             "cutoff": r["configuration"]["cutoff"], "actual_max_bonds": r["actual_max_bonds"],
             "root_tau": r["root_tau"], "slope": r["slope_audit"][-1]["difference_slope"],
             "branch_symmetry_error": r["event"]["branch_symmetry_error"],
             "root_replay_minus_linear_grid": r["grid_linear_vs_replayed_root"],
             "slope_step_sensitivity": r["slope_step_sensitivity"],
             "maximum_discarded_weight": r["event"]["maximum_discarded_weight"],
             "accumulated_discarded_weight": r["event"]["accumulated_discarded_weight"],
             "event_rates": [b["rate"] for b in r["event"]["branches"]]} for r in successful]
    pairs = []
    for i, first in enumerate(rows):
        for second in rows[i+1:]:
            if first["model"] != second["model"]:
                continue
            changes = [key for key in ("dtau", "max_bond", "cutoff") if first[key] != second[key]]
            if len(changes) == 1:
                pairs.append({"first": first["run_id"], "second": second["run_id"],
                              "isolated_changed_parameter": changes[0],
                              "absolute_root_difference": abs(first["root_tau"] - second["root_tau"]),
                              "absolute_slope_difference": abs(first["slope"] - second["slope"])})
    finite_path = ROOT / "results/potts/finite_chain/potts_global_summary.csv"
    finite_comparison = None
    matching = [r for r in successful if r["configuration"]["model"] == "potts"
                and abs(r["configuration"]["dtau"] - .02) < 1e-14]
    if matching and finite_path.exists():
        reference = max(matching, key=lambda r: r["configuration"]["max_bond"])
        with finite_path.open() as stream:
            finite_rows = list(csv.DictReader(stream))
        finite_comparison = {"immutable_source": repository_path(finite_path),
                             "source_sha256": sha256(finite_path),
                             "matched_evolution_dtau": .02,
                             "direct_thermodynamic_root": reference["root_tau"],
                             "rows": [{"n_sites": int(r["n_sites"]),
                                       "linear_grid_root": float(r["individual_crossing"]),
                                       "direct_minus_finite_root": reference["root_tau"] - float(r["individual_crossing"])}
                                      for r in finite_rows],
                             "interpretation": "separately evolved finite OBC data approach the direct infinite result; finite root grid interpolation and boundary shifts remain; no finite-size fit used to define the iMPS rate"}
    return {"rows": rows, "one_parameter_refinements": pairs,
            "finite_open_comparison": finite_comparison,
            "classification": "empirical direct thermodynamic signed-return convergence for represented iMPS states",
            "rigorous_Hamiltonian_state_error_bound": False,
            "unsigned_thermodynamic_selection_gate": "requires separate moment/bound report",
            "reporting_rule": "any claimed nonzero interference gap must exceed five times its observed refinement envelope plus separately reported root/spectral sensitivities; not a confidence interval"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="fresh campaign destination; defaults to build/reproduction/potts/infinite_chain/<profile>")
    parser.add_argument("--run-id", help="execute only one named run in the profile")
    parser.add_argument("--max-seconds", type=float, help="bounded wall seconds per run, saving resumable checkpoints")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    profile = json.loads(args.config.read_text())
    destination = args.output_dir or Path("build/reproduction/potts/infinite_chain") / Path(profile["output_dir"]).name
    base_output = (ROOT / destination).resolve()
    reports = []
    for item in profile["runs"]:
        cfg = {"model": "potts", "field": 1.5, "cutoff": 1e-13, **profile.get("defaults", {}), **item}
        if args.run_id and cfg["run_id"] != args.run_id:
            continue
        if args.analyze_only:
            result = analyze_run(cfg, base_output / cfg["run_id"])
        else:
            result = run_one(cfg, base_output, args.resume, args.max_seconds)
        reports.append(result)
    if args.run_id and not reports:
        raise ValueError(f"run_id {args.run_id} not in profile")
    # Re-read all completed reports so selective resumption preserves the full
    # campaign summary instead of replacing it with one selected run.
    all_reports = [json.loads(path.read_text()) for path in base_output.glob("*/event_report.json")]
    summary = campaign_summary(all_reports)
    summary.update({"profile": str(args.config), "profile_sha256": sha256(args.config)})
    write_json(base_output / "campaign_summary.json", summary)
    print(json.dumps({"summary": repository_path(base_output / "campaign_summary.json")}), flush=True)


if __name__ == "__main__":
    main()
