#!/usr/bin/env python3
"""Reproduce the paper's three main figures from the published numerical data.

The critical-field Ising sample uses the exact finite-window separatrix mean.
Figure manifests record inputs, fit ranges, fonts, and this analytic evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dqpt_main_mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import yaml

from scripts.summary_panel import draw_summary_panel

from dqpt_wigner.crossing_validation import (
    first_resolved_branch_crossing,
)

POTTS = ROOT / "results/potts/finite_chain"
ISING = ROOT / "results/ising"
BLOCK_RESULTS = ROOT / "results/potts/blocks/production"
FIG = ROOT / "figures" / "main"

FONT_CANDIDATES = [
    "Times New Roman",
    "Times",
    "Tinos",
    "Nimbus Roman",
    "Liberation Serif",
    "DejaVu Serif",
]


def repository_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def resolve_font() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    return next((name for name in FONT_CANDIDATES if name in available), "DejaVu Serif")


def configure_style() -> str:
    chosen = resolve_font()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": FONT_CANDIDATES,
            "mathtext.fontset": "stix",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.5,
            "legend.fontsize": 6.9,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 350,
            "savefig.bbox": "tight",
        }
    )
    return chosen


def save_figure(fig: plt.Figure, stem: str, inputs: list[Path]) -> None:
    outputs = []
    for suffix in (".pdf", ".png"):
        path = FIG / f"{stem}{suffix}"
        fig.savefig(path, dpi=350)
        outputs.append(path.name)
    manifest = {
        "figure": stem,
        "inputs": [repository_path(path) for path in inputs if path.exists()],
        "outputs": outputs,
        "requested_font": "Times New Roman",
        "resolved_font": resolve_font(),
        "formats": ["pdf", "png"],
        "dpi": 350,
        "script": repository_path(Path(__file__).resolve()),
    }
    (FIG / f"{stem}_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def first_column(frame: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise KeyError(f"None of {names!r} is present; columns={list(frame.columns)!r}")


def fit_intercept(sizes: np.ndarray, values: np.ndarray, degree: int) -> float:
    return float(np.polyfit(1.0 / sizes, values, degree)[-1])


def panel_labels(axes: list[plt.Axes] | np.ndarray, labels: str) -> None:
    for label, axis in zip(labels, np.asarray(axes).flat):
        axis.text(
            -0.16,
            1.08,
            f"({label})",
            transform=axis.transAxes,
            fontweight="bold",
            va="top",
        )


def make_main_figure_one() -> None:
    trajectory_path = POTTS / "potts_global_trajectories.csv"
    block_path = POTTS / "potts_block_crossings.csv"
    summary_path = POTTS / "potts_global_summary.csv"
    trajectories = pd.read_csv(trajectory_path)
    blocks = pd.read_csv(block_path)
    summary = pd.read_csv(summary_path)

    nmax = int(trajectories["n_sites"].max())
    global_data = trajectories.loc[trajectories["n_sites"] == nmax].sort_values("time")
    rate_initial = first_column(global_data, "rate_initial", "rate_0")
    rate_competing = first_column(global_data, "rate_branch1", "rate_competing", "rate_1")
    rate_total = first_column(global_data, "rate_total")
    rate_lower = first_column(global_data, "rate_lower_envelope")
    critical_time = float(
        summary.loc[summary["n_sites"] == nmax, "individual_crossing"].iloc[0]
    )
    window = global_data.loc[
        (global_data["time"] >= critical_time - 0.15)
        & (global_data["time"] <= critical_time + 0.15)
    ]

    block_row = blocks.loc[
        (blocks["n_sites"] == blocks["n_sites"].max())
        & (blocks["block_length"] == blocks["block_length"].max())
    ].iloc[0]
    delta_support = float(block_row["delta_support_rate_at_physical"])
    delta_sign = float(block_row["delta_sign_cost_at_physical"])
    survival_initial = float(block_row["surviving_fraction_initial_at_physical"])
    survival_competing = float(block_row["surviving_fraction_competing_at_physical"])
    physical_time = float(block_row["physical_crossing"])
    support_time = float(block_row["support_crossing"])

    fig = plt.figure(figsize=(7.1, 5.75), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.0])
    draw_summary_panel(fig, grid[0, :])
    fig.text(0.005, 0.995, "(a)", fontweight="bold", va="top")

    branch_axis = axis = fig.add_subplot(grid[1, 0])
    axis.plot(window["time"], window[rate_initial], label=r"$r_0$")
    axis.plot(window["time"], window[rate_competing], label=r"$r_1$")
    axis.plot(window["time"], window[rate_total], ls="--", label=r"$R_N$")
    axis.plot(window["time"], window[rate_lower], ls=":", label=r"$\min_a r_a$")
    axis.axvline(critical_time, ls="--", lw=0.85)
    axis.set(
        xlabel=r"$Jt$",
        ylabel="return rate density",
    )
    axis.legend(frameon=False, ncol=2)

    sign_axis = axis = fig.add_subplot(grid[1, 1])
    values = [delta_support, delta_sign, delta_support + delta_sign]
    axis.bar([r"$\Delta s$", r"$\Delta q$", r"$\Delta r$"], values, edgecolor="black", linewidth=0.6)
    axis.axhline(0, lw=0.7)
    axis.set(ylabel="branch difference at $t_c$", ylim=(-0.15, 0.22))
    axis.text(
        0.03,
        0.97,
        fr"$Jt_s={support_time:.3f}\rightarrow Jt_c={physical_time:.3f}$"
        + "\n"
        + fr"$P_0/A_0={survival_initial:.3f}$, $P_\perp/A_\perp={survival_competing:.3f}$",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=6.8,
    )
    panel_labels([branch_axis, sign_axis], "bc")
    save_figure(fig, "fig1_phase_space_framework", [trajectory_path, block_path, summary_path])
    manifest_path = FIG / "fig1_phase_space_framework_figure_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["summary_panel"] = {
        "builder": "scripts/summary_panel.py",
        "content": "Vector schematic; local order versus quench strength, smooth global branches, and an auxiliary crossing preceding the physical crossing.",
        "schematic_curves_are_data": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def make_main_figure_two() -> None:
    import hashlib

    global_path = POTTS / "potts_global_summary.csv"
    production_path = BLOCK_RESULTS / "potts_block_crossings.csv"
    block_path = production_path
    global_summary = pd.read_csv(global_path).sort_values("n_sites")
    blocks = pd.read_csv(block_path).sort_values(["n_sites", "block_length"])
    plotted_columns = ["physical_crossing", "support_crossing", "critical_time_shift",
                       "sign_cost_competing_at_physical"]
    if (blocks.empty or blocks.duplicated(["n_sites", "block_length"]).any()
            or (blocks["block_length"] <= 0).any()
            or not np.isfinite(blocks[plotted_columns].to_numpy(float)).all()):
        raise ValueError("Figure 2 requires unique, finite block-event records")

    def block_style(n_sites: int) -> dict:
        # Nested marker sizes reveal coincident chains without moving their data.
        if n_sites == 50:
            return {"color": "#1f77b4", "markerfacecolor": "none",
                    "markeredgecolor": "#1f77b4", "markersize": 6.0,
                    "markeredgewidth": 1.15, "zorder": 3}
        if n_sites == 100:
            return {"color": "#ff7f0e", "markerfacecolor": "#ff7f0e",
                    "markeredgecolor": "#ff7f0e", "markersize": 3.3,
                    "markeredgewidth": 0.7, "zorder": 4}
        return {"markerfacecolor": "none", "markersize": 5.0}

    fit_records = {"critical_time_shift": [], "sign_cost_competing_at_physical": []}

    def collect_intercepts(data: pd.DataFrame, column: str, n_sites: int) -> list[float]:
        intercepts = []
        for degree in (1, 2):
            for minimum_length in (4, 5, 6):
                fit_data = data.loc[data["block_length"] >= minimum_length]
                if len(fit_data) > degree:
                    intercept = fit_intercept(fit_data["block_length"].to_numpy(float),
                                              fit_data[column].to_numpy(float), degree)
                    intercepts.append(intercept)
                    fit_records[column].append({
                        "n_sites": int(n_sites), "degree": degree,
                        "minimum_block_length": minimum_length,
                        "block_lengths": [int(value) for value in fit_data["block_length"]],
                        "intercept": intercept,
                    })
        return intercepts

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.65), constrained_layout=True)

    axis = axes[0, 0]
    x = 1.0 / global_summary["n_sites"].to_numpy(float)
    individual = global_summary["individual_crossing"].to_numpy(float)
    grouped = global_summary["grouped_crossing"].to_numpy(float)
    axis.plot(x, individual, "o-", label="per sector")
    axis.plot(x, grouped, "s--", label="ordered manifold")
    fit_x = np.linspace(0.0, max(x) * 1.05, 160)
    limits = []
    for degree, line_style in ((1, ":"), (2, "-.")):
        fit_individual = np.polyfit(x, individual, degree)
        fit_grouped = np.polyfit(x, grouped, degree)
        axis.plot(fit_x, np.polyval(fit_individual, fit_x), ls=line_style, lw=0.9)
        axis.plot(fit_x, np.polyval(fit_grouped, fit_x), ls=line_style, lw=0.9)
        limits.append(float(fit_individual[-1]))
    axis.plot([0, 0], [min(limits), max(limits)], lw=5, solid_capstyle="butt")
    axis.set(xlabel=r"$1/N$", ylabel=r"crossing time $Jt_c$")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    for n_sites, group in blocks.groupby("n_sites"):
        data = group.sort_values("block_length")
        inverse_length = 1.0 / data["block_length"].to_numpy(float)
        axis.plot(inverse_length, data["physical_crossing"], "o-",
                  label=fr"$t_c$, $N={int(n_sites)}$", **block_style(int(n_sites)))
        axis.plot(inverse_length, data["support_crossing"], "s--",
                  label=fr"$t_s$, $N={int(n_sites)}$", **block_style(int(n_sites)))
    axis.set(xlabel=r"$1/\ell$", ylabel=r"crossing time $Jt$")
    axis.legend(frameon=False, ncol=2)

    axis = axes[1, 0]
    displacement_limits = []
    for n_sites, group in blocks.groupby("n_sites"):
        data = group.sort_values("block_length")
        axis.plot(1.0 / data["block_length"], data["critical_time_shift"], "o-",
                  label=fr"$N={int(n_sites)}$", **block_style(int(n_sites)))
        displacement_limits.extend(collect_intercepts(data, "critical_time_shift", int(n_sites)))
    if displacement_limits:
        axis.plot([0, 0], [min(displacement_limits), max(displacement_limits)],
                  color="#2ca02c", lw=5, solid_capstyle="butt",
                  label=r"$\ell\to\infty$ fit range")
    axis.axhline(0, lw=0.7)
    axis.set(xlabel=r"$1/\ell$", ylabel=r"$J(t_c-t_s)$")
    axis.legend(frameon=False, loc="lower right")

    axis = axes[1, 1]
    sign_limits = []
    for n_sites, group in blocks.groupby("n_sites"):
        data = group.sort_values("block_length")
        axis.plot(
            1.0 / data["block_length"],
            data["sign_cost_competing_at_physical"],
            "s--",
            label=fr"$N={int(n_sites)}$",
            **block_style(int(n_sites)),
        )
        sign_limits.extend(collect_intercepts(data, "sign_cost_competing_at_physical", int(n_sites)))
    if sign_limits:
        axis.plot([0, 0], [min(sign_limits), max(sign_limits)],
                  color="#2ca02c", lw=5, solid_capstyle="butt",
                  label=r"$\ell\to\infty$ fit range")
    axis.axhline(0, lw=0.7)
    axis.set(xlabel=r"$1/\ell$", ylabel=r"$q_\perp(t_c)$")
    axis.legend(frameon=False, loc="lower right")

    panel_labels(axes, "abcd")
    inputs = [global_path, block_path]
    if block_path == production_path:
        inputs.extend(path for path in [block_path.parent / "run_configuration.json",
            block_path.parent / "summary.json",
            *[block_path.parent / f"N{int(row.n_sites)}_ell{int(row.block_length)}.json"
              for row in blocks.itertuples()]] if path.exists())
    save_figure(fig, "fig2_potts_thermodynamic", inputs)
    manifest_path = FIG / "fig2_potts_thermodynamic_figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "block_event_method": (sorted(blocks["event_method"].dropna().astype(str).unique())
            if "event_method" in blocks else
            ["direct fractional-step TEBD replay; grouped competitor 1+2"]
),
        "block_comparison": "initial sector 0 versus grouped competitor 1+2",
        "block_lengths_by_chain": {str(int(n)): [int(value) for value in group["block_length"]]
                                   for n, group in blocks.groupby("n_sites")},
        "plotted_block_record_count": len(blocks),
        "block_markers": {"N50": "large open blue circles/squares",
                          "N100": "smaller filled orange circles/squares",
                          "x_coordinates": "exact 1/block_length, without jitter"},
        "fit_family": "degree 1 and 2 in inverse block length; minimum lengths 4, 5, 6; each chain separately",
        "fit_records": fit_records,
        "fit_ranges": {column: [min(row["intercept"] for row in rows),
                                max(row["intercept"] for row in rows)] if rows else None
                       for column, rows in fit_records.items()},
        "fit_scope": "Sensitivity of the selected finite-data fit family, not confidence intervals or a thermodynamic-limit proof.",
        "source_sha256": {repository_path(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in inputs},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")






def load_ising_branch() -> tuple[pd.DataFrame, Path]:
    path = ISING / "ising_branches_N400_h0.700.csv"
    return pd.read_csv(path), path


def make_ising_benchmark(stem: str = "fig3_ising_benchmark") -> None:
    scan_path = ISING / "ising_dqpt_i_scan.csv"
    scan = pd.read_csv(scan_path)
    data, branch_path = load_ising_branch()
    data = data.sort_values("time").reset_index(drop=True)
    resolved = first_resolved_branch_crossing(data, minimum_probability=1e-12, direction="up")
    if resolved is None:
        raise RuntimeError("No probability-resolved Ising crossing")
    critical_time = resolved.time
    moment_column = first_column(
        scan,
        "late_signed_magnetization",
        "late_time_signed_magnetization",
        "signed_late_magnetization",
    )
    field_column = first_column(scan, "final_field", "field", "h")
    inputs = [scan_path, branch_path]
    separatrix_marker = 0.497
    analytic_correction = None
    if stem == "fig3_ising_benchmark":
        config_path = ROOT / "configs" / "ising.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        coupling = float(config["model"]["J"])
        start = float(config["mean_field_dqpt_i"]["average_start"])
        end = float(config["mean_field_dqpt_i"]["tmax"])
        separatrix_marker = coupling / 2.0
        separatrix_rows = np.isclose(scan[field_column], separatrix_marker)
        if separatrix_rows.sum() != 1 or not 0 <= start < end or coupling <= 0:
            raise ValueError("Expected one separatrix point and a valid averaging window")
        raw_value = float(scan.loc[separatrix_rows, moment_column].iloc[0])
        # Integration drifts from the unstable separatrix; use its exact finite-window mean.
        analytic_mean = float(
            2.0 * (np.arctan(np.exp(-coupling * start)) - np.arctan(np.exp(-coupling * end)))
            / (coupling * (end - start))
        )
        scan.loc[separatrix_rows, moment_column] = analytic_mean
        inputs.append(config_path)
        analytic_correction = {
            "panel": "a",
            "reason": "Replace numerical drift at the unstable separatrix by the exact finite-window mean; raw CSV is unchanged.",
            "raw_source": repository_path(scan_path),
            "field_column": field_column,
            "moment_column": moment_column,
            "raw_field": float(scan.loc[separatrix_rows, field_column].iloc[0]),
            "raw_value": raw_value,
            "plotted_value": analytic_mean,
            "coupling_J": coupling,
            "time_window": [start, end],
            "dimensionless_time_window_Jt": [coupling * start, coupling * end],
            "exact_trajectory": "m_z(t) = sech(J*t)",
            "averaging_formula": "2*(atan(exp(-J*t_start))-atan(exp(-J*t_end)))/(J*(t_end-t_start))",
            "config": repository_path(config_path),
            "exact_separatrix_marker": {"h_f": separatrix_marker, "h_f_over_J": 0.5},
        }
    window = data.loc[
        (data["time"] >= critical_time - 0.8)
        & (data["time"] <= critical_time + 0.8)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5), constrained_layout=True)
    axis = axes[0]
    axis.plot(scan[field_column], scan[moment_column], "o-")
    axis.axhline(0, lw=0.7)
    axis.axvline(separatrix_marker, ls=":", lw=0.9)
    axis.set(
        xlabel=r"final field $h_f/J$",
        ylabel=r"signed late time order $\overline{m_z}$",
        title="DQPT I: local marginal",
    )

    axis = axes[1]
    axis.plot(window["time"], window["rate_up"], label=r"$r_\uparrow$")
    axis.plot(window["time"], window["rate_down"], label=r"$r_\downarrow$")
    axis.plot(window["time"], window["rate_total"], ls="--", label=r"$R_N$")
    axis.axvline(critical_time, ls=":", lw=0.9)
    axis.set(xlabel=r"$Jt$", ylabel="return rate density", title=r"DQPT II: global branches, $N=400$")
    axis.legend(frameon=False)

    axis = axes[2]
    magnetization_line = axis.plot(window["time"], window["magnetization_z"], label=r"$m_z(t)$")[0]
    axis.axhline(0, lw=0.7)
    axis.axvline(critical_time, ls=":", lw=0.9)
    axis.set(xlabel=r"$Jt$", ylabel=r"local order $m_z$", title="Different information, same quench")
    twin = axis.twinx()
    rate_line = twin.plot(window["time"], window["rate_total"], ls="--", label=r"$R_N(t)$")[0]
    twin.set_ylabel("ordered return rate")
    axis.legend(
        [magnetization_line, rate_line],
        [magnetization_line.get_label(), rate_line.get_label()],
        frameon=False,
    )
    if stem == "fig3_ising_benchmark":
        for axis in axes:
            axis.set_title("")
    panel_labels(axes, "abc")
    save_figure(fig, stem, inputs)
    if analytic_correction is not None:
        manifest_path = FIG / f"{stem}_figure_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["analytic_separatrix_correction"] = analytic_correction
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")




def main() -> None:
    global FIG, POTTS, ISING, BLOCK_RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIG)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--figure", type=int, choices=(1, 2, 3), help="build one main figure")
    args = parser.parse_args()
    base = (ROOT / args.results_root).resolve()
    POTTS = base / "potts/finite_chain"
    ISING = base / "ising"
    BLOCK_RESULTS = base / "potts/blocks/production"
    FIG = (ROOT / args.output_dir).resolve()
    FIG.mkdir(parents=True, exist_ok=True)
    configure_style()
    builders = {1: make_main_figure_one, 2: make_main_figure_two, 3: make_ising_benchmark}
    for number, builder in builders.items():
        if args.figure is None or args.figure == number:
            builder()
    print(f"Main figures written to {FIG}")


if __name__ == "__main__":
    main()
