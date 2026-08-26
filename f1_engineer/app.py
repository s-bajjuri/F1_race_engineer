from __future__ import annotations

from f1_engineer.config import AppConfig
from f1_engineer.ui.dashboard import build_app


def normalize_track(track: str) -> str:
    normalized = track.strip().lower().replace(" ", "_")
    if not normalized:
        raise ValueError("Track cannot be empty.")
    return normalized


def run(config: AppConfig) -> None:
    if config.window_seconds <= 0:
        raise ValueError("Dashboard window must be greater than zero.")
    if config.drive_trace_seconds <= 0:
        raise ValueError("Drive trace window must be greater than zero.")
    if not 1 <= config.port <= 65535:
        raise ValueError("UDP port must be between 1 and 65535.")
    if config.reference_path is not None and not config.reference_path.is_file():
        raise ValueError(f"Reference file does not exist: {config.reference_path}")
    if config.start_view not in {"drive", "analysis", "diagnostics"}:
        raise ValueError(
            "Start view must be drive, analysis, or diagnostics."
        )
    if config.tyre_critical_c <= config.tyre_warning_c:
        raise ValueError(
            "Critical tyre threshold must be above the warning threshold."
        )
    if config.overlay_width < 320 or config.overlay_height < 180:
        raise ValueError("Overlay must be at least 320×180 pixels.")
    if not 0.2 <= config.overlay_opacity <= 1.0:
        raise ValueError("Overlay opacity must be between 0.2 and 1.0.")
    if config.overlay_trace_seconds <= 0:
        raise ValueError("Overlay trace window must be greater than zero.")
    if not 1 <= config.overlay_refresh_hz <= 60:
        raise ValueError("Overlay refresh rate must be between 1 and 60 Hz.")

    build_app(
        host=config.host,
        port=config.port,
        track=normalize_track(config.track),
        output_root=config.output_root,
        window_seconds=config.window_seconds,
        reference_path=config.reference_path,
        start_view=config.start_view,
        drive_trace_seconds=config.drive_trace_seconds,
        tyre_warning_c=config.tyre_warning_c,
        tyre_critical_c=config.tyre_critical_c,
        overlay_enabled=config.overlay_enabled,
        overlay_visible_on_start=config.overlay_visible_on_start,
        overlay_width=config.overlay_width,
        overlay_height=config.overlay_height,
        overlay_x=config.overlay_x,
        overlay_y=config.overlay_y,
        overlay_opacity=config.overlay_opacity,
        overlay_click_through=config.overlay_click_through,
        overlay_trace_seconds=config.overlay_trace_seconds,
        overlay_refresh_hz=config.overlay_refresh_hz,
    )
