"""Publication plotting and data-adjacent output helpers."""
from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

FONT_CANDIDATES = (
    "Times New Roman",
    "Times",
    "Tinos",
    "Nimbus Roman",
    "Liberation Serif",
    "DejaVu Serif",
)


def resolved_font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((x for x in FONT_CANDIDATES if x in names), "DejaVu Serif")


def apply_publication_style() -> str:
    chosen = resolved_font()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": list(FONT_CANDIDATES),
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.35,
        }
    )
    return chosen


def save_adjacent(
    fig: plt.Figure,
    stem: Path,
    inputs: Iterable[Path] = (),
    dpi: int = 300,
) -> list[str]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=dpi)
        outputs.append(path.name)
    manifest = {
        "figure_stem": stem.name,
        "outputs": outputs,
        "inputs": [str(Path(x).name) for x in inputs],
        "requested_font": "Times New Roman",
        "resolved_font": resolved_font(),
        "formats": ["png", "pdf"],
        "dpi": dpi,
    }
    stem.with_name(stem.name + "_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    plt.close(fig)
    return outputs


def plot_potts_thermodynamic(
    trajectories: pd.DataFrame,
    global_summary: pd.DataFrame,
    block_crossings: pd.DataFrame,
    output_stem: Path,
) -> None:
    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.25), constrained_layout=True)

    ax = axes[0, 0]
    max_n = int(trajectories["n_sites"].max())
    g = trajectories.loc[trajectories["n_sites"] == max_n].sort_values("time")
    ax.plot(g["time"], g["rate_initial"], label=r"$r_0$")
    ax.plot(g["time"], g["rate_branch1"], label=r"$r_1$")
    ax.plot(g["time"], g["rate_total"], linestyle="--", label=r"$R_N$")
    ax.plot(g["time"], g["rate_lower_envelope"], linestyle=":", label=r"$\min_a r_a$")
    row = global_summary.loc[global_summary["n_sites"] == max_n].iloc[0]
    if pd.notna(row["individual_crossing"]):
        ax.axvline(row["individual_crossing"], linewidth=0.9, linestyle="--")
    ax.set(xlabel=r"$Jt$", ylabel="return-rate density", title=fr"Potts return branches, $N={max_n}$")
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    s = global_summary.sort_values("n_sites")
    inv = 1.0 / s["n_sites"].to_numpy(float)
    ax.plot(inv, s["individual_crossing"], marker="o", label="per-sector crossing")
    ax.plot(inv, s["grouped_crossing"], marker="s", label="ordered-manifold crossing")
    if "fit_individual_limit" in s.columns:
        lim = float(s["fit_individual_limit"].dropna().iloc[0])
        xx = np.linspace(0, max(inv) * 1.05, 100)
        coeff = np.polyfit(inv, s["individual_crossing"], 1)
        ax.plot(xx, np.polyval(coeff, xx), linestyle="--", label=fr"$t_c(\infty)={lim:.4f}$")
    ax.set(xlabel=r"$1/N$", ylabel=r"critical time $Jt_c$", title="Thermodynamic branch crossing")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for N, group in block_crossings.groupby("n_sites"):
        q = group.sort_values("block_length")
        ax.plot(1.0 / q["block_length"], q["physical_crossing"], marker="o", label=fr"physical, $N={int(N)}$")
        ax.plot(1.0 / q["block_length"], q["support_crossing"], marker="s", linestyle="--", label=fr"sign-stripped, $N={int(N)}$")
    ax.set(xlabel=r"$1/\ell$", ylabel=r"crossing time $Jt$", title="Bulk-block Wigner hierarchy")
    ax.legend(frameon=False, ncol=2)

    ax = axes[1, 1]
    for N, group in block_crossings.groupby("n_sites"):
        q = group.sort_values("block_length")
        ax.plot(q["block_length"], q["critical_time_shift"], marker="o", label=fr"$N={int(N)}$")
    ax.axhline(0.0, linewidth=0.7)
    ax.set(xlabel=r"block length $\ell$", ylabel=r"$J(t_c-t_s)$", title="Sign-selected displacement")
    ax.legend(frameon=False)

    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.16, 1.05, f"({label})", transform=ax.transAxes, fontweight="bold", va="top")
    save_adjacent(fig, Path(output_stem))


def plot_potts_convergence(summary: pd.DataFrame, output_stem: Path) -> None:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    for N, group in summary.groupby("n_sites"):
        g = group.sort_values(["dt", "max_bond"])
        labels = [fr"$\Delta t={dt:g},\ \chi={int(chi)}$" for dt, chi in zip(g["dt"], g["max_bond"])]
        x = np.arange(len(g))
        ax.plot(x, g["individual_crossing"], marker="o", label=fr"$N={int(N)}$")
        if len(summary["n_sites"].unique()) == 1:
            ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel(r"per-sector crossing $Jt_c$")
    ax.set_title("TEBD convergence")
    ax.legend(frameon=False)
    save_adjacent(fig, Path(output_stem))


def plot_block_sign_cost(block_wide: pd.DataFrame, output_stem: Path, n_sites: int | None = None) -> None:
    apply_publication_style()
    N = int(block_wide["n_sites"].max() if n_sites is None else n_sites)
    data = block_wide.loc[block_wide["n_sites"] == N]
    lengths = sorted(data["block_length"].unique())
    select = [lengths[0], lengths[len(lengths) // 2], lengths[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), constrained_layout=True)
    for ell in select:
        g = data.loc[data["block_length"] == ell].sort_values("time")
        axes[0].plot(g["time"], g["delta_physical_rate"], label=fr"$\ell={ell}$")
        axes[1].plot(g["time"], g["delta_support_rate"], label=fr"$\ell={ell}$")
        axes[2].plot(g["time"], g["delta_sign_cost"], label=fr"$\ell={ell}$")
    axes[0].set(title=r"physical: $\Delta r$", xlabel=r"$Jt$", ylabel="branch difference")
    axes[1].set(title=r"sign stripped: $\Delta s$", xlabel=r"$Jt$")
    axes[2].set(title=r"interference cost: $\Delta q$", xlabel=r"$Jt$")
    for label, ax in zip("abc", axes):
        ax.axhline(0.0, linewidth=0.7)
        ax.text(-0.17, 1.06, f"({label})", transform=ax.transAxes, fontweight="bold", va="top")
        ax.legend(frameon=False)
    fig.suptitle(fr"Central-block Wigner branch anatomy, $N={N}$", y=1.03)
    save_adjacent(fig, Path(output_stem))


def plot_ising_benchmark(
    scan: pd.DataFrame,
    branch_frame: pd.DataFrame,
    output_stem: Path,
) -> None:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), constrained_layout=True)
    ax = axes[0]
    ax.plot(scan["final_field"], scan["late_signed_magnetization"], marker="o", label=r"$\overline{m_z}$")
    ax.plot(scan["final_field"], scan["late_absolute_magnetization"], marker="s", linestyle="--", label=r"$\overline{|m_z|}$")
    ax.axhline(0.0, linewidth=0.7)
    ax.set(xlabel=r"final field $h_f/J$", ylabel="late-time magnetization", title="DQPT-I: local phase space")
    ax.legend(frameon=False)

    max_n = int(branch_frame["n_sites"].max())
    g = branch_frame.loc[branch_frame["n_sites"] == max_n].sort_values("time")
    ax = axes[1]
    ax.plot(g["time"], g["rate_up"], label=r"$r_\uparrow$")
    ax.plot(g["time"], g["rate_down"], label=r"$r_\downarrow$")
    ax.plot(g["time"], g["rate_total"], linestyle="--", label=r"$R_N$")
    ax.set(xlabel=r"$Jt$", ylabel="return-rate density", title=fr"DQPT-II branches, $N={max_n}$")
    ax.legend(frameon=False)

    ax = axes[2]
    ax.plot(g["time"], g["magnetization_z"], label=r"$m_z(t)$")
    if "return_echo" in g:
        ax.plot(g["time"], g["return_echo"], linestyle="--", label=r"$P_\uparrow+P_\downarrow$")
    ax.axhline(0.0, linewidth=0.7)
    ax.set(xlabel=r"$Jt$", ylabel="observable", title="Local order versus global return")
    ax.legend(frameon=False)
    for label, ax in zip("abc", axes):
        ax.text(-0.18, 1.06, f"({label})", transform=ax.transAxes, fontweight="bold", va="top")
    save_adjacent(fig, Path(output_stem))


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.05, label, transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", va="top")


def plot_hierarchy_control(control: pd.DataFrame, wigner: np.ndarray,
                           output_stem: Path | str) -> list[str]:
    apply_publication_style()
    fig = plt.figure(figsize=(7.05, 2.35), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.05, 1.3, 1.0))
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Phase-space scale hierarchy.
    ax0.set_xlim(0, 1); ax0.set_ylim(0, 1); ax0.axis("off")
    y = 0.78
    for ell, width in [(1, 0.24), (2, 0.40), (4, 0.62), ("N", 0.88)]:
        x0 = 0.5 - width / 2
        rect = FancyBboxPatch(
            (x0, y - 0.06), width, 0.12,
            boxstyle="round,pad=0.012", fill=False, linewidth=1.0
        )
        ax0.add_patch(rect)
        ax0.text(0.5, y, rf"block $\ell={ell}$", ha="center", va="center")
        y -= 0.18
    ax0.annotate("local marginal\nDQPT-I", xy=(0.19, 0.77), xytext=(0.02, 0.95),
                 arrowprops=dict(arrowstyle="->", lw=0.8), ha="left", va="top")
    ax0.annotate("global return support\nDQPT-II", xy=(0.5, 0.22), xytext=(0.5, 0.02),
                 arrowprops=dict(arrowstyle="->", lw=0.8), ha="center", va="bottom")
    panel_label(ax0, "(a)")

    t = control["time"].to_numpy()
    for col, lab in [("r0", r"$r_0$"), ("r1", r"$r_1=r_2$"),
                     ("total_rate", r"$R_N$")]:
        ax1.plot(t, control[col], label=lab)
    tc = float(control["critical_time"].iloc[0])
    ax1.axvline(tc, linestyle="--", linewidth=0.9)
    ax1.set(xlabel=r"$\Omega t$", ylabel="return-rate density",
            title="Wigner-positive support crossing")
    ax1.legend(frameon=False, ncol=1)
    ax1.text(tc + 0.025, 0.95 * ax1.get_ylim()[1], r"$t_\star$", va="top")
    panel_label(ax1, "(b)")

    im = ax2.imshow(wigner, origin="lower", interpolation="nearest", aspect="equal")
    ax2.set_xticks(range(3)); ax2.set_yticks(range(3))
    ax2.set_xlabel(r"$p$"); ax2.set_ylabel(r"$q$")
    ax2.set_title(r"$W(q,p;t_\star)\geq0$")
    for q in range(3):
        for p in range(3):
            ax2.text(p, q, f"{wigner[q,p]:.2f}", ha="center", va="center",
                     fontsize=7)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    panel_label(ax2, "(c)")
    return save_adjacent(fig, Path(output_stem))
