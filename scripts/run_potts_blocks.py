#!/usr/bin/env python3
"""Direct finite-block events for the Potts return-support comparison.

Uniform TEBD checkpoints define the trajectory; a fractional final step
replays each root. All three sectors are contracted independently. Both
the physical and unsigned comparisons use the grouped competitor 1+2.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from dqpt_wigner.block_support_extension import direct_support_split
from dqpt_wigner.event_audit import assemble_event, grouped_support, positive_block_return
from dqpt_wigner.potts_mps import evolve_to_times
from dqpt_wigner.paths import repository_path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint(state, path):
    arrays = {f"gamma_{i}": a for i, a in enumerate(state.gammas)}
    arrays.update({f"lambda_{i}": a for i, a in enumerate(state.lambdas)})
    np.savez_compressed(path, **arrays)


def run_size(payload):
    n, options = payload
    started = time.perf_counter()
    out = Path(options["out"])
    dt, chi, cutoff = options["dt"], options["max_bond"], options["cutoff"]
    grids = {}
    for tau, state, discarded in evolve_to_times(
            n, np.arange(0., .700001, dt), 1., 1.5, dt, chi, cutoff):
        grids[round(tau, 12)] = state, discarded
    print(f"N={n}: evolved checkpoints, {time.perf_counter()-started:.1f}s", flush=True)

    def at(tau):
        base = round(float(np.floor((tau + 1e-12) / dt) * dt), 12)
        state, discarded = grids[base]
        state = state.copy()
        if tau - base > 1e-12:
            discarded += state.step(1., 1.5, tau-base, chi, cutoff).discarded_weight
        return state, discarded

    records = []
    previous_unsigned = .37
    for length in options["lengths"]:
        path = out / f"N{n}_ell{length}.json"
        if path.exists():
            record = json.loads(path.read_text())
            for key in ("dt", "max_bond", "cutoff", "root_tolerance"):
                if record["configuration"][key] != options[key]:
                    raise ValueError(f"Configuration mismatch in {path}: {key}")
            records.append(record)
            previous_unsigned = record["unsigned"]["tau"]
            continue
        block_start = (n-length)//2
        case_started = time.perf_counter()
        events = {}
        for kind in ("physical", "unsigned"):
            cache = {}
            evaluations = []

            def difference(tau):
                tau = float(tau)
                if tau in cache:
                    return cache[tau][0]
                state, _ = at(tau)
                if kind == "physical":
                    weights = [positive_block_return(state, block_start, length, a)
                               for a in range(3)]
                    branches = None
                else:
                    branches = {str(a): direct_support_split(
                        state, block_start, length, a,
                        max_workspace_bytes=options["workspace_mib"]*1024**2)
                        for a in range(3)}
                    weights = [branches[str(a)]["unsigned_weight"] for a in range(3)]
                # Scaling by the total avoids a length-dependent root residual.
                delta = float((weights[0]-weights[1]-weights[2])/sum(weights))
                cache[tau] = delta, branches
                evaluations.append({"tau": tau, "normalized_primitive_difference": delta})
                return delta

            if kind == "physical":
                bracket = [.50, .66]
            else:
                # Neighboring lengths supply a seed only, never a missing value.
                bracket = [max(.30, previous_unsigned-.025),
                           min(.66, previous_unsigned+.04)]
            while difference(bracket[0]) <= 0 and bracket[0] > .300001:
                bracket[0] = max(.30, bracket[0]-.02)
            while difference(bracket[1]) >= 0 and bracket[1] < .659999:
                bracket[1] = min(.66, bracket[1]+.02)
            if not difference(bracket[0]) > 0 > difference(bracket[1]):
                raise RuntimeError(f"No correctly oriented {kind} bracket N={n}, ell={length}")
            root = float(brentq(difference, *bracket,
                               xtol=options["root_tolerance"], rtol=1e-13))
            state, discarded = at(root)
            branches = cache[root][1] or {str(a): direct_support_split(
                state, block_start, length, a,
                max_workspace_bytes=options["workspace_mib"]*1024**2)
                for a in range(3)}
            branches["1+2"] = grouped_support(branches["1"], branches["2"], length)
            event = assemble_event(root, length, branches, "1+2")
            residual = event["delta_r" if kind == "physical" else "delta_s"]
            if abs(residual) > 1e-7:
                raise RuntimeError(f"Root residual too large: {residual}")
            for a in ("0", "1", "2"):
                b = branches[a]
                scale = max(b["unsigned_weight"], b["probability"], 1e-300)
                if max(abs(b["signed_reconstruction_error"]),
                       abs(b["identity_A_minus_P_minus_2nu"])) / scale > 1e-8:
                    raise RuntimeError("Independent Wigner/projector consistency failed")
            state_path = out / f"N{n}_ell{length}_{kind}.npz"
            checkpoint(state, state_path)
            event.update({"n_sites": n, "block_start": block_start, "kind": kind,
                          "root_bracket": bracket, "root_evaluations": evaluations,
                          "normalized_primitive_residual": difference(root),
                          "mps_norm": state.norm(), "realized_maximum_bond": state.maximum_bond_dimension,
                          "cumulative_discarded_weight": discarded,
                          "checkpoint": repository_path(state_path),
                          "checkpoint_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest()})
            events[kind] = event
            print(f"N={n} ell={length} {kind}: t={root:.10f}, "
                  f"residual={residual:.2e}, case elapsed={time.perf_counter()-case_started:.1f}s", flush=True)
        record = {"n_sites": n, "block_length": length,
                  "configuration": options, **events,
                  "critical_time_shift": events["physical"]["tau"]-events["unsigned"]["tau"],
                  "elapsed_seconds": time.perf_counter()-case_started,
                  "scope": "finite central block in an independently evolved finite open chain",
                  "event_method": "direct fractional-step TEBD replay; grouped competitor 1+2",
                  "error_status": "Root and contraction checks do not bound evolution or extrapolation error."}
        write_json(path, record)
        records.append(record)
        previous_unsigned = events["unsigned"]["tau"]
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--lengths", type=int, nargs="+", default=list(range(4, 17)))
    parser.add_argument("--dt", type=float, default=.02)
    parser.add_argument("--max-bond", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=1e-9)
    parser.add_argument("--root-tolerance", type=float, default=2e-9)
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--out", default="build/reproduction/potts/blocks/production")
    args = parser.parse_args()
    options = vars(args).copy()
    options["out"] = str((ROOT / args.out).resolve())
    out = Path(options["out"])
    out.mkdir(parents=True, exist_ok=True)
    code_paths = [Path(__file__), ROOT / "src/dqpt_wigner/block_support_extension.py",
                  ROOT / "src/dqpt_wigner/event_audit.py", ROOT / "src/dqpt_wigner/potts_mps.py"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    write_json(out / "run_configuration.json", {"options": options, "code_sha256": hashes})
    payloads = [(n, options) for n in args.sizes]
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(payloads))) as pool:
            all_records = list(pool.map(run_size, payloads))
    else:
        all_records = [run_size(p) for p in payloads]
    rows = []
    for records in all_records:
        for record in records:
            physical, unsigned = record["physical"], record["unsigned"]
            a, b = physical["branches"]["0"], physical["branches"]["1+2"]
            rows.append({"n_sites": record["n_sites"], "block_length": record["block_length"],
                         "physical_crossing": physical["tau"], "support_crossing": unsigned["tau"],
                         "critical_time_shift": record["critical_time_shift"],
                         "delta_support_rate_at_physical": physical["delta_s"],
                         "delta_sign_cost_at_physical": physical["delta_q"],
                         "sign_cost_initial_at_physical": a["sign_cost"],
                         "sign_cost_competing_at_physical": b["sign_cost"],
                         "surviving_fraction_initial_at_physical": a["surviving_fraction"],
                         "surviving_fraction_competing_at_physical": b["surviving_fraction"]})
    frame = pd.DataFrame(rows).sort_values(["n_sites", "block_length"])
    frame.to_csv(out / "potts_block_crossings.csv", index=False)
    records = [record for group in all_records for record in group]
    events = [record[kind] for record in records for kind in ("physical", "unsigned")]
    summary = {
        "number_of_blocks": len(rows), "sizes": sorted(frame.n_sites.unique().tolist()),
        "block_lengths": sorted(frame.block_length.unique().tolist()),
        "maximum_norm_error": max(abs(e["mps_norm"]-1) for e in events),
        "maximum_event_rate_residual": max(abs(e["delta_r" if e["kind"] == "physical" else "delta_s"]) for e in events),
        "maximum_projector_vs_wigner_error": max(abs(e["branches"][a]["signed_reconstruction_error"]) for e in events for a in ("0", "1", "2")),
        "maximum_sector_unsigned_relative_error": max(e["independent_sector_symmetry"]["unsigned_weight_relative_error"] for e in events),
        "code_unchanged_during_run": all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest for p,digest in hashes.items()),
    }
    fit_rows = []
    for n, group in frame.groupby("n_sites"):
        for observable in ("critical_time_shift", "sign_cost_competing_at_physical"):
            for degree in (1, 2):
                for minimum in (4, 5, 6):
                    data = group.loc[group.block_length >= minimum]
                    if len(data) <= degree:
                        continue
                    coefficients = np.polyfit(1/data.block_length.to_numpy(float), data[observable], degree)
                    fit_rows.append({"n_sites": int(n), "observable": observable,
                                     "degree": degree, "minimum_block_length": minimum,
                                     "maximum_block_length": int(data.block_length.max()),
                                     "intercept": float(coefficients[-1]),
                                     "coefficients_descending": coefficients.tolist()})
    summary["fits"] = fit_rows
    summary["fit_status"] = "Sensitivity of finite-sequence polynomial fits; no statistical confidence or thermodynamic error bound."
    write_json(out / "summary.json", summary)
    if fit_rows and len(summary["sizes"]) == 2:
        largest = frame.loc[frame.block_length == frame.block_length.max()].sort_values("n_sites")
        high = largest.iloc[-1]
        limits = {observable: [r["intercept"] for r in fit_rows if r["observable"] == observable]
                  for observable in ("critical_time_shift", "sign_cost_competing_at_physical")}
        macros = {
            "FigTwoEllMax": str(int(high.block_length)),
            "FigTwoShiftLowN": f"{largest.iloc[0].critical_time_shift:.4f}",
            "FigTwoShiftHighN": f"{high.critical_time_shift:.4f}",
            "FigTwoQComp": f"{high.sign_cost_competing_at_physical:.4f}",
            "FigTwoSurvivalComp": f"{high.surviving_fraction_competing_at_physical:.3f}",
            "FigTwoSurvivalFirst": f"{frame.iloc[0].surviving_fraction_competing_at_physical:.3f}",
            "FigTwoShiftFitLow": f"{min(limits['critical_time_shift']):.4f}",
            "FigTwoShiftFitHigh": f"{max(limits['critical_time_shift']):.4f}",
            "FigTwoQFitLow": f"{min(limits['sign_cost_competing_at_physical']):.4f}",
            "FigTwoQFitHigh": f"{max(limits['sign_cost_competing_at_physical']):.4f}",
        }
        (out / "figure_numbers.tex").write_text(
            "% Direct finite-block events from scripts/extend_fig2_blocks.py\n" +
            "".join("\\newcommand{\\"+key+"}{"+value+"}\n" for key,value in macros.items()))
    print(f"Completed {len(rows)} direct block records in {out}", flush=True)


if __name__ == "__main__":
    main()
