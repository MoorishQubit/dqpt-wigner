from pathlib import Path

from dqpt_wigner.io import PlotSettings, override_plot_settings, plot_settings_from_mapping
import matplotlib.pyplot as plt

from dqpt_wigner.plotting import SELECTED_SERIF_FONT
from dqpt_wigner.publication import save_adjacent


def test_plot_settings_and_dual_format_output(tmp_path: Path):
    settings = plot_settings_from_mapping(
        {"plots": {"enabled": True, "formats": ["png", "pdf"], "dpi": 120}}
    )
    assert settings == PlotSettings(True, ("png", "pdf"), 120)
    settings = override_plot_settings(settings, dpi=144)
    assert settings.dpi == 144
    output = tmp_path / "hierarchy.png"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    save_adjacent(fig, output.with_suffix(""), dpi=settings.dpi)
    assert output.exists() and output.stat().st_size > 1000
    assert output.with_suffix(".pdf").exists()
    assert SELECTED_SERIF_FONT
