#!/usr/bin/env python3
"""Validate direct finite-block events and regenerate the refinement table.

The production records contain blocks 4 through 16 for N=50 and N=100.
Independent step and bond/cutoff checks quantify evolution sensitivity.
"""
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def repository_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=Path("build/reproduction/tables"))
    args = parser.parse_args()
    base = ROOT / args.results_root / "potts/blocks"
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(base / "production/potts_block_crossings.csv")
    assert len(frame) == 26 and set(frame.n_sites) == {50, 100}
    for n, group in frame.groupby("n_sites"):
        assert list(group.block_length) == list(range(4, 17))
        assert (group.critical_time_shift > 0).all()
        for column in ("critical_time_shift", "sign_cost_competing_at_physical",
                       "surviving_fraction_competing_at_physical"):
            assert (np.diff(group[column]) < 0).all(), (n, column)
    np.testing.assert_allclose(frame.surviving_fraction_competing_at_physical,
        np.exp(-frame.block_length * frame.sign_cost_competing_at_physical), atol=3e-15)
    spread = {column: float(np.max(np.abs(
        frame.loc[frame.n_sites == 50, column].to_numpy() -
        frame.loc[frame.n_sites == 100, column].to_numpy())))
        for column in frame.columns if column not in ("n_sites", "block_length")}
    refinements = []
    for directory, label in [("production", "production"),
                             ("refinement_step", "smaller time step"),
                             ("refinement_bond", "larger cap and tighter cutoff")]:
        record = json.loads((base / directory / "N50_ell16.json").read_text())
        event = record["physical"]
        branch = event["branches"]["1+2"]
        config = record["configuration"]
        refinements.append({"case": label, "dt": config["dt"],
            "max_bond": config["max_bond"], "cutoff": config["cutoff"],
            "tc": event["tau"], "ts": record["unsigned"]["tau"],
            "delay": record["critical_time_shift"], "q": branch["sign_cost"],
            "survival": branch["surviving_fraction"],
            "realized_bond": event["realized_maximum_bond"]})
    lines = [r"\begin{tabular}{ccccccc}", r"\toprule",
        r"$\delta\tau$ & $\chi_{\max}$ & $\epsilon$ & $\tau_c$ & $\tau_s$ & $\tau_c-\tau_s$ & $q_\perp(\tau_c)$ \\",
        r"\midrule"]
    for row in refinements:
        cutoff = r"$10^{-9}$" if row["cutoff"] == 1e-9 else r"$10^{-10}$"
        lines.append(f"{row['dt']:.2f} & {row['max_bond']} & {cutoff} & "
            f"{row['tc']:.9f} & {row['ts']:.9f} & {row['delay']:.9f} & {row['q']:.9f}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "block_refinement.tex").write_text("\n".join(lines) + "\n")
    report = {
        "block_lengths": list(range(4, 17)),
        "all_plotted_blocks_recomputed_directly": 26,
        "maximum_chain_difference": spread, "refinement_rows": refinements,
        "refinement_absolute_changes": {row["case"]: {
            key: abs(row[key] - refinements[0][key])
            for key in ("tc", "ts", "delay", "q", "survival")}
            for row in refinements[1:]},
        "source_directory": repository_path(base),
        "source_sha256": {str(path.relative_to(base)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [base / "production/potts_block_crossings.csv",
                *[base / directory / "N50_ell16.json" for directory in
                    ("production", "refinement_step", "refinement_bond")]]},
        "scope": "Finite central blocks; polynomial intercept ranges are fit sensitivity, not additional computed points or a thermodynamic certificate.",
    }
    pd.DataFrame(refinements).to_csv(out / "block_refinement.csv", index=False)
    (out / "numerical_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
