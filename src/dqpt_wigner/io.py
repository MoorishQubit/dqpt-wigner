"""Configuration, plotting, and reproducible output helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


@dataclass(frozen=True)
class PlotSettings:
    """Common figure-output settings shared by every command-line workflow."""

    enabled: bool = True
    formats: tuple[str, ...] = ("png", "pdf")
    dpi: int = 300

    def __post_init__(self) -> None:
        normalized = tuple(
            dict.fromkeys(str(value).strip().lower().lstrip(".") for value in self.formats if str(value).strip())
        )
        if not normalized:
            normalized = ("png",)
        unsupported = set(normalized) - {"png", "pdf", "svg", "eps", "ps"}
        if unsupported:
            raise ValueError(f"unsupported plot formats: {sorted(unsupported)}")
        if self.dpi < 72:
            raise ValueError("plot dpi must be at least 72")
        object.__setattr__(self, "formats", normalized)

    @property
    def primary_format(self) -> str:
        return self.formats[0]

    @property
    def extra_formats(self) -> tuple[str, ...]:
        return self.formats[1:]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return {} if data is None else dict(data)


def plot_settings_from_mapping(
    mapping: Mapping[str, Any] | None,
    *,
    default_enabled: bool = True,
) -> PlotSettings:
    """Read a top-level ``plots`` block from a YAML-like mapping.

    Every numerical script uses the same block::

        plots:
          enabled: true
          formats: [png, pdf]
          dpi: 300

    Missing blocks default to PNG and vector PDF output so that figures are
    produced beside their source data without additional commands.
    """

    root = {} if mapping is None else dict(mapping)
    raw = root.get("plots", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError("plots must be a mapping")
    formats = raw.get("formats", ("png", "pdf"))
    if isinstance(formats, str):
        formats = (formats,)
    return PlotSettings(
        enabled=bool(raw.get("enabled", default_enabled)),
        formats=tuple(str(value) for value in formats),
        dpi=int(raw.get("dpi", 300)),
    )



def override_plot_settings(
    settings: PlotSettings,
    *,
    enabled: bool | None = None,
    formats: tuple[str, ...] | list[str] | None = None,
    dpi: int | None = None,
) -> PlotSettings:
    """Return validated plotting settings after command-line overrides."""

    return PlotSettings(
        enabled=settings.enabled if enabled is None else bool(enabled),
        formats=settings.formats if formats is None else tuple(formats),
        dpi=settings.dpi if dpi is None else int(dpi),
    )

def _json_default(value: Any):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def save_json(path: str | Path, data: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    return output


def configuration_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, default=_json_default).encode("utf-8")
    return sha256(payload).hexdigest()[:16]


def ensure_directory(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output
