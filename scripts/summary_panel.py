"""Vector schematic of local and global information scales.

The curves in this panel are schematic.  The discrete Wigner example is
the exact positive line of a single qutrit computational basis state.
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle


LOCAL = "#c77b43"
GLOBAL = "#476d87"
SUPPORT = "#a45125"
INK = "#252525"


def _chain(axis, *, local: bool) -> None:
    """Draw sites and the information retained by a local or global support."""
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    heading = "DQPT I: local information" if local else "DQPT II: global information"
    axis.text(0.5, 0.98, heading, ha="center", va="top", fontsize=10.2,
              fontweight="bold")
    left, width = (0.235, 0.36) if local else (0.07, 0.86)
    color = LOCAL if local else GLOBAL
    axis.add_patch(FancyBboxPatch(
        (left, 0.32), width, 0.35, boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor=color, alpha=0.16, edgecolor=color, linewidth=1.0,
    ))
    sites = np.linspace(0.09, 0.91, 8)
    axis.plot(sites, np.full_like(sites, 0.49), color=INK, linewidth=1.1,
              zorder=1)
    colors = [color if (not local or 2 <= index <= 4) else "#dedede"
              for index in range(len(sites))]
    axis.scatter(sites, np.full_like(sites, 0.49), s=67, c=colors,
                 edgecolors=INK, linewidths=0.8, zorder=2)
    description = r"fixed block, $\ell\ll N$" if local else r"full system, $\ell=N$"
    axis.text(0.5, 0.07, description, ha="center", va="bottom", fontsize=8.4)


def _wigner_example(axis) -> None:
    """Draw an exact positive qutrit line with vector rectangles."""
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title("Local Wigner marginal", fontsize=8.8, pad=4)
    left, bottom, dx, dy = 0.13, 0.15, 0.10, 0.22
    for q in range(3):
        for p in range(3):
            axis.add_patch(Rectangle(
                (left + p * dx, bottom + q * dy), dx, dy,
                facecolor=LOCAL if q == 0 else "#f5f3ef",
                edgecolor="white", linewidth=0.8,
            ))
        axis.text(left - 0.025, bottom + (q + 0.5) * dy, str(q),
                  ha="right", va="center", fontsize=8)
    for p in range(3):
        axis.text(left + (p + 0.5) * dx, bottom - 0.035, str(p),
                  ha="center", va="top", fontsize=8)
    axis.text(left - 0.09, bottom + 1.5 * dy, r"$q$", ha="center", va="center",
              fontsize=9)
    axis.text(left + 3.35 * dx, bottom - 0.02, r"$p$", ha="center", va="top",
              fontsize=9)
    axis.text(0.57, 0.72, r"Example: $|0\rangle$", ha="left", va="center", fontsize=8.8)
    axis.text(0.57, 0.45, r"$W=1/3$ on the line", ha="left", va="center", fontsize=8.4)
    axis.text(0.57, 0.21, r"$W=0$ elsewhere", ha="left", va="center", fontsize=8.4)


def _clean_plot(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color(INK)
    axis.set_yticks([])
    axis.tick_params(axis="both", labelsize=8, length=3)
    axis.xaxis.label.set_size(8.5)
    axis.yaxis.label.set_size(8.5)


def draw_summary_panel(fig, subspec) -> None:
    """Populate a 7.1 inch wide, approximately 3.7 inch tall figure region.

    ``subspec`` is a Matplotlib SubplotSpec.  The caller supplies the outer
    panel label and can place numerical panels in subsequent figure rows.
    Styling inherits the figure builder's Times compatible font setup.
    """
    columns = subspec.subgridspec(1, 2, wspace=0.04)
    local = columns[0, 0].subgridspec(3, 1, height_ratios=[0.8, 1.0, 1.0], hspace=0.015)
    global_ = columns[0, 1].subgridspec(3, 1, height_ratios=[0.8, 1.0, 1.0], hspace=0.015)

    _chain(fig.add_subplot(local[0, 0]), local=True)
    _chain(fig.add_subplot(global_[0, 0]), local=False)
    _wigner_example(fig.add_subplot(local[1, 0]))

    order = fig.add_subplot(local[2, 0])
    field = np.linspace(0, 1.6, 801)
    order.plot(field, np.sqrt(np.maximum(1.0 - field, 0.0)), color=GLOBAL, lw=1.8)
    order.axvline(1.0, color="#888888", ls=":", lw=0.8)
    order.set(xlim=(0, 1.6), ylim=(-0.06, 1.12), xlabel=r"quench parameter $g$",
              ylabel=r"late time order $\overline{m}$", xticks=[1.0], xticklabels=[r"$g_c$"])
    order.set_title("Local order (schematic)", fontsize=8.8, pad=4)
    _clean_plot(order)

    time = np.linspace(0, 1, 501)
    support_time, crossing_time = 0.37, 0.64
    rate_initial = 0.65 + 0.3 * (time - crossing_time)
    rate_competing = 0.65 - 0.3 * (time - crossing_time)
    rate = fig.add_subplot(global_[1, 0])
    rate.plot(time, rate_initial, color=GLOBAL, lw=1.5)
    rate.plot(time, rate_competing, color=SUPPORT, lw=1.5)
    rate.plot(time, np.minimum(rate_initial, rate_competing), color=INK, lw=2.0,
              ls=(0, (2.0, 1.6)))
    rate.axvline(crossing_time, color="#888888", ls=":", lw=0.8)
    rate.set(xlim=(0, 1), ylim=(0.40, 0.88), ylabel="return rates",
             xticks=[crossing_time], xticklabels=[r"$t_c$"])
    rate.set_title("Branch exchange (schematic)", fontsize=8.8, pad=4)
    rate.text(0.06, 0.85, r"$r_\perp$", color=SUPPORT, fontsize=10)
    rate.text(0.88, 0.79, r"$r_0$", color=GLOBAL, fontsize=10)
    rate.text(0.12, 0.58, r"$\min_a r_a$", color=INK, fontsize=9)
    rate.text(0.99, -0.1, r"time $t$", ha="right", va="top", fontsize=8.5,
              transform=rate.transAxes)
    _clean_plot(rate)

    difference = fig.add_subplot(global_[2, 0])
    delta_support = 0.6 * (time - support_time)
    delta_interference = 0.6 * (support_time - crossing_time)
    delta_physical = delta_support + delta_interference
    difference.axhline(0, color="#999999", lw=0.7)
    difference.axvline(support_time, color="#999999", ls=":", lw=0.8)
    difference.axvline(crossing_time, color="#999999", ls=":", lw=0.8)
    difference.plot(time, delta_support, color=SUPPORT, lw=1.5)
    difference.plot(time, delta_physical, color=GLOBAL, lw=1.5)
    difference.axhline(delta_interference, color="#666666", ls="--", lw=1.0)
    difference.scatter([support_time, crossing_time], [0, 0], s=16,
                       color=[SUPPORT, GLOBAL], zorder=3)
    difference.set(xlim=(0, 1), ylim=(-0.43, 0.46), xlabel=r"time $t$",
                   ylabel="branch differences", xticks=[support_time, crossing_time],
                   xticklabels=[r"$t_s$", r"$t_c$"])
    difference.set_title("Sign displacement (schematic)", fontsize=8.8, pad=4)
    difference.text(0.94, 0.41, r"$\Delta s$", ha="right", color=SUPPORT, fontsize=9)
    difference.text(0.97, 0.08, r"$\Delta r$", ha="right", color=GLOBAL, fontsize=9)
    difference.text(0.96, delta_interference - 0.03, r"$\Delta q$", ha="right", va="top",
                    color="#666666", fontsize=9)
    difference.annotate("", xy=(crossing_time, 0.34), xytext=(support_time, 0.34),
                        arrowprops={"arrowstyle": "<->", "color": INK, "lw": 0.9})
    _clean_plot(difference)
