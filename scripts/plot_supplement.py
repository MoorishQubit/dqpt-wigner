#!/usr/bin/env python3
"""Reproduce the three supplemental figures and their numerical tables.

Inputs are compact numerical records and the same Ising data as the main figures.
The plotting commands perform no state evolution. Use derive_supplement.py to
regenerate the compact records from numerical run outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dqpt_supplement_mpl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dqpt_wigner.plotting import configure_publication_style

INPUT = ROOT / "results/supplement"
ISING = ROOT / "results/ising"
FIG = ROOT / "figures/supplement"
TABLES = ROOT / "build/reproduction/tables"
FONT = None


def repository_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def read_frame(name: str) -> pd.DataFrame:
    return pd.read_csv(INPUT / f"{name}.csv")


def save_figure(fig, stem: str, inputs: list[Path], panels: list[str]) -> None:
    assert all(not ax.get_title() for ax in fig.axes)
    outputs = []
    for suffix in ("pdf", "png"):
        path = FIG / f"{stem}.{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        outputs.append(path.name)
    plt.close(fig)
    manifest = {
        "figure": stem,
        "panels": panels,
        "inputs": {repository_path(path): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in inputs},
        "outputs": outputs,
        "requested_font": "Times New Roman",
        "resolved_font": FONT,
        "script": repository_path(Path(__file__)),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (FIG / f"{stem}_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_table(name: str, columns: str, header: str, rows: list[str]) -> None:
    text = "\\begin{tabular}{" + columns + "}\n\\toprule\n" + header + "\\\\\n\\midrule\n"
    text += "\n".join(row + r"\\" for row in rows)
    text += "\n\\bottomrule\n\\end{tabular}\n"
    (TABLES / f"{name}.tex").write_text(text)


def panels(axes) -> None:
    for index, ax in enumerate(np.ravel(axes)):
        ax.text(0, 1.025, f"({chr(97+index)})", transform=ax.transAxes,
                va="bottom", ha="left", fontweight="bold", clip_on=False)
        ax.spines[["top", "right"]].set_visible(False)

def scientific(value: float, digits: int = 3) -> str:
    mantissa, exponent = f"{float(value):.{digits}e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"

def signed_exchange() -> None:
    table = read_frame("potts_signed_refinements")
    sequence = table[table.cap == 128].sort_values(["dtau", "cutoff"]).drop_duplicates("dtau")
    trajectory = read_frame("potts_signed_trajectory")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), layout="constrained")
    axes[0].plot(trajectory.tau, trajectory.r0, label=r"$r_0$")
    axes[0].plot(trajectory.tau, trajectory.r1, label=r"$r_1=r_2$")
    axes[0].plot(trajectory.tau, trajectory[["r0", "r1", "r2"]].min(axis=1), "k:", label=r"$\min_a r_a$")
    axes[0].axvline(sequence.iloc[0]["root"], color="gray", ls="--", lw=.7)
    axes[0].set(xlabel=r"$\tau=Jt$", ylabel="signed rate")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].plot(sequence.dtau, sequence.root, "o-")
    axes[1].set(xlabel=r"$\delta\tau$", ylabel=r"measured $\tau_c$")
    axes[2].plot(sequence.dtau, sequence.slope, "o-")
    axes[2].set(xlabel=r"$\delta\tau$", ylabel=r"$\partial_\tau\Delta r$")
    panels(axes)
    save_figure(fig, "potts_signed_exchange",
        [INPUT / "potts_signed_trajectory.csv", INPUT / "potts_signed_refinements.csv"],
        ["Individually contracted signed rates and their lower envelope.",
         "Crossing times under evolution-step refinement at cap 128.",
         "Transverse slope differences at the corresponding crossings."])


def block_own_events() -> None:
    trace = read_frame("potts_individual_block_trace")
    group = read_frame("potts_grouped_block_events").sort_values("block_length")
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.9), layout="constrained")
    axes[0,0].plot(trace.tau, trace.delta_r, "o-", ms=3, label=r"$\Delta r$")
    axes[0,0].plot(trace.tau, trace.delta_s, "s-", ms=3, label=r"$\Delta s$")
    mask = (trace.delta_r < 0) & (trace.delta_s > 0)
    axes[0,0].fill_between(trace.tau, 0, 1, where=mask,
        transform=axes[0,0].get_xaxis_transform(), color="green", alpha=.10)
    axes[0,0].axhline(0, color="gray", lw=.6)
    axes[0,0].set(xlabel=r"$\tau=Jt$", ylabel=r"individual difference, $\ell=9$")
    axes[0,0].legend(frameon=False)
    axes[0,1].plot(group.block_length, group.critical_time_shift, "o-")
    axes[0,1].set(xlabel=r"block length $\ell$", ylabel=r"$\tau_{c,\mathrm{grp}}-\tau_{s,\mathrm{grp}}$")
    for key, label in (("initial", "sector 0"), ("competing", "group 1+2")):
        axes[1,0].plot(group.block_length, group[f"sign_cost_{key}_at_physical"], "o-", label=label)
        axes[1,1].plot(group.block_length, group[f"surviving_fraction_{key}_at_physical"], "o-", label=label)
    axes[1,0].set(xlabel=r"block length $\ell$", ylabel=r"$q$ at grouped block event")
    axes[1,1].set(xlabel=r"block length $\ell$", ylabel=r"$P/A$ at grouped block event")
    axes[1,0].legend(frameon=False)
    for ax in (axes[0,1], axes[1,0], axes[1,1]): ax.set_xticks(group.block_length)
    panels(axes)
    save_figure(fig, "potts_block_own_events",
        [INPUT / "potts_individual_block_trace.csv", INPUT / "potts_grouped_block_events.csv"],
        ["Individual-sector differences for the nine-site block of the N=100 chain.",
         "Grouped signed/unsigned crossing displacement.",
         "Branch sign costs at each interpolated grouped block crossing.",
         "Surviving fractions from the same interpolated primitive weights."])


def ising_information_scales() -> None:
    trajectory_path = ISING / "ising_branches_N400_h0.700.csv"
    scan_path = ISING / "ising_dqpt_i_scan.csv"
    config_path = ROOT / "configs/ising.yaml"
    trajectory = pd.read_csv(trajectory_path)
    scan = pd.read_csv(scan_path)
    config = yaml.safe_load(config_path.read_text())
    coupling = float(config["model"]["J"])
    start = float(config["mean_field_dqpt_i"]["average_start"])
    end = float(config["mean_field_dqpt_i"]["tmax"])
    critical = np.isclose(scan.final_field, coupling / 2)
    if critical.sum() != 1 or not 0 <= start < end or coupling <= 0:
        raise ValueError("Expected one separatrix point and a valid averaging window")
    scan.loc[critical, "late_signed_magnetization"] = (
        2 * (np.arctan(np.exp(-coupling * start)) - np.arctan(np.exp(-coupling * end)))
        / (coupling * (end - start)))
    late = trajectory[(trajectory.time > 6.65) & (trajectory.time < 6.9)]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), layout="constrained")
    axes[0].plot(scan.final_field, scan.late_signed_magnetization, ".-")
    critical = scan[np.isclose(scan.final_field,.5)]
    axes[0].scatter(critical.final_field, critical.late_signed_magnetization, marker="D", s=20,
                    facecolor="white", edgecolor="black", zorder=4)
    axes[0].axvline(.5, color="black", ls=":", lw=.7)
    axes[0].set(xlabel=r"$h/J$", ylabel=r"signed late-time $\overline{m_z}$")
    axes[1].plot(trajectory.time, trajectory.magnetization_z)
    axes[1].set(xlabel=r"$\tau=Jt$", ylabel=r"local order $m_z$", xlim=(0,8))
    for key, label in (("rate_up", r"$r_\uparrow$"), ("rate_down", r"$r_\downarrow$"), ("rate_total", r"$R_N$")):
        axes[2].plot(late.time, late[key], label=label)
    axes[2].set(xlabel=r"$\tau=Jt$", ylabel="ordered return rate")
    axes[2].legend(frameon=False, fontsize=7)
    panels(axes)
    save_figure(fig, "ising_information_scales", [scan_path, trajectory_path, config_path],
        ["Signed finite-window mean-field magnetization; the critical field uses the exact separatrix mean.",
         "Local magnetization in the N=400, h/J=0.7 trajectory.",
         "Ordered return branches near the selected late global exchange in the same trajectory."])


def tables() -> None:
    frame = read_frame("potts_signed_refinements")
    rows = [f"{r.dtau:g} & {r.cap} & {r.bond_a}/{r.bond_b} & ${scientific(r.cutoff)}$ & {r.root:.9f} & {r.slope:.7f}"
            for r in frame.itertuples()]
    write_table("signed_refinement", "cccccc",
        r"$\delta\tau$ & cap & realized bonds & $\epsilon$ & $\tau_c$ & $\partial_\tau\Delta r$", rows)
    frame = read_frame("potts_direct_blocks")
    rows = []
    for row in frame.itertuples():
        comparator = "individual" if str(row.comparator) == "1" else "grouped"
        rows.append(f"{row.N} & {comparator} & {row.tau_c:.6f} & {row.tau_s:.6f} & "
                    f"{row.delta_s:.6f} & {row.q_comp:.6f} & {row.survival_comp:.6f}")
    write_table("direct_blocks", "clccccc",
        r"$N$ & comparison & $\tau_c$ & $\tau_s$ & $\Delta s(\tau_c)$ & $q_{\rm comp}$ & $P_{\rm comp}/A_{\rm comp}$", rows)


def main() -> None:
    global INPUT, ISING, FIG, TABLES, FONT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=FIG)
    parser.add_argument("--table-dir", type=Path, default=TABLES)
    args = parser.parse_args()
    base = (ROOT / args.results_root).resolve()
    INPUT, ISING = base / "supplement", base / "ising"
    FIG = (ROOT / args.output_dir).resolve()
    TABLES = (ROOT / args.table_dir).resolve()
    FIG.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FONT = configure_publication_style()
    plt.rcParams.update({"legend.fontsize": 7.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
    signed_exchange()
    block_own_events()
    ising_information_scales()
    tables()
    print(f"Three supplemental figures written to {FIG}; two tables written to {TABLES}")


if __name__ == "__main__":
    main()
