from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 20777
    track: str = "spa"
    output_root: Path = Path("data")
    window_seconds: float = 20.0
    reference_path: Path | None = None
    start_view: str = "drive"
    drive_trace_seconds: float = 3.0
    tyre_warning_c: float = 105.0
    tyre_critical_c: float = 115.0
    overlay_enabled: bool = True
    overlay_visible_on_start: bool = True
    overlay_width: int = 560
    overlay_height: int = 260
    overlay_x: int = 30
    overlay_y: int = 30
    overlay_opacity: float = 0.92
    overlay_click_through: bool = True
    overlay_trace_seconds: float = 7.0
    overlay_refresh_hz: float = 20.0


def _nested(mapping: dict[str, Any], section: str, key: str, default: Any) -> Any:
    value = mapping.get(section, {})
    if not isinstance(value, dict):
        return default
    return value.get(key, default)

def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def load_config(path: Path | None) -> AppConfig:
    if path is None or not path.exists():
        return AppConfig()

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("The YAML configuration root must be a mapping.")

    reference_value = _nested(raw, "reference", "path", None)
    return AppConfig(
        host=str(_nested(raw, "udp", "host", "0.0.0.0")),
        port=int(_nested(raw, "udp", "port", 20777)),
        track=str(_nested(raw, "app", "track", "spa")),
        output_root=Path(str(_nested(raw, "recording", "data_directory", "data"))),
        window_seconds=float(_nested(raw, "ui", "dashboard_window_seconds", 20.0)),
        reference_path=(Path(str(reference_value)) if reference_value else None),
        start_view=str(_nested(raw, "ui", "start_view", "drive")).lower(),
        drive_trace_seconds=float(_nested(raw, "ui", "drive_trace_seconds", 3.0)),
        tyre_warning_c=float(_nested(raw, "ui", "tyre_warning_c", 105.0)),
        tyre_critical_c=float(_nested(raw, "ui", "tyre_critical_c", 115.0)),
        overlay_enabled=_as_bool(
            _nested(raw, "overlay", "enabled", True), True
        ),
        overlay_visible_on_start=_as_bool(
            _nested(raw, "overlay", "visible_on_start", True), True
        ),
        overlay_width=int(_nested(raw, "overlay", "width", 560)),
        overlay_height=int(_nested(raw, "overlay", "height", 260)),
        overlay_x=int(_nested(raw, "overlay", "x", 30)),
        overlay_y=int(_nested(raw, "overlay", "y", 30)),
        overlay_opacity=float(_nested(raw, "overlay", "opacity", 0.92)),
        overlay_click_through=_as_bool(
            _nested(raw, "overlay", "click_through", True), True
        ),
        overlay_trace_seconds=float(
            _nested(raw, "overlay", "trace_seconds", 7.0)
        ),
        overlay_refresh_hz=float(
            _nested(raw, "overlay", "refresh_hz", 20.0)
        ),
    )
