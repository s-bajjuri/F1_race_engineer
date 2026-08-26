from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import tkinter as tk

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from f1_engineer.runtime import AppRuntime
from f1_engineer.ui import theme
from f1_engineer.ui.analysis_view import AnalysisView
from f1_engineer.ui.diagnostics_view import DiagnosticsView
from f1_engineer.ui.drive_overlay import DriveOverlay, OverlayOptions
from f1_engineer.ui.drive_view import DriveView
from f1_engineer.ui.view_state import DashboardState


class DashboardController:
    def __init__(
        self,
        *,
        runtime: AppRuntime,
        track: str,
        port: int,
        analysis_window_seconds: float,
        drive_trace_seconds: float,
        tyre_warning_c: float,
        tyre_critical_c: float,
        start_view: str,
        overlay_enabled: bool,
        overlay_visible_on_start: bool,
        overlay_width: int,
        overlay_height: int,
        overlay_x: int,
        overlay_y: int,
        overlay_opacity: float,
        overlay_click_through: bool,
        overlay_trace_seconds: float,
        overlay_refresh_hz: float,
    ) -> None:
        self.runtime = runtime
        self.track = track
        self.port = port
        self.fig = plt.figure(figsize=(14.4, 8.5))
        theme.style_figure(self.fig)
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("F1 Race Engineer")

        self.title_text = self.fig.text(
            0.055,
            0.958,
            "F1 RACE ENGINEER",
            color=theme.TEXT,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="center",
        )
        self.subtitle_text = self.fig.text(
            0.055,
            0.925,
            f"{track.upper()} · UDP {port}",
            color=theme.MUTED,
            fontsize=9,
            ha="left",
            va="center",
        )
        self.status_text = self.fig.text(
            0.945,
            0.945,
            "WAITING FOR TELEMETRY",
            color=theme.MUTED,
            fontsize=9.5,
            fontweight="medium",
            ha="right",
            va="center",
        )
        self.hint_text = self.fig.text(
            0.945,
            0.065,
            "Keys: 1 Drive · 2 Analysis · 3 Diagnostics · O Overlay",
            color=theme.MUTED,
            fontsize=8,
            ha="right",
            va="center",
        )

        self.views = {
            "drive": DriveView(
                self.fig,
                trace_seconds=drive_trace_seconds,
                tyre_warning_c=tyre_warning_c,
                tyre_critical_c=tyre_critical_c,
            ),
            "analysis": AnalysisView(
                self.fig,
                window_seconds=analysis_window_seconds,
            ),
            "diagnostics": DiagnosticsView(
                self.fig,
                track=track,
                output_root=runtime.output_root,
            ),
        }

        self.overlay: DriveOverlay | None = None
        if overlay_enabled:
            manager_window = getattr(manager, "window", None)
            if isinstance(manager_window, tk.Misc):
                self.overlay = DriveOverlay(
                    parent=manager_window,
                    options=OverlayOptions(
                        width=overlay_width,
                        height=overlay_height,
                        x=overlay_x,
                        y=overlay_y,
                        opacity=overlay_opacity,
                        click_through=overlay_click_through,
                        trace_seconds=overlay_trace_seconds,
                        refresh_hz=overlay_refresh_hz,
                        visible_on_start=overlay_visible_on_start,
                        tyre_warning_c=tyre_warning_c,
                        tyre_critical_c=tyre_critical_c,
                    ),
                )
            else:
                print(
                    "Overlay disabled: the active Matplotlib backend is not Tk. "
                    "Use the TkAgg backend on Windows."
                )

        self.button_axes = {
            "drive": self.fig.add_axes([0.345, 0.92, 0.095, 0.05]),
            "analysis": self.fig.add_axes([0.447, 0.92, 0.115, 0.05]),
            "diagnostics": self.fig.add_axes([0.569, 0.92, 0.135, 0.05]),
            "overlay": self.fig.add_axes([0.711, 0.92, 0.105, 0.05]),
        }
        self.buttons: dict[str, Button] = {}
        labels = {
            "drive": "DRIVE",
            "analysis": "ANALYSIS",
            "diagnostics": "DIAGNOSTICS",
            "overlay": "OVERLAY",
        }
        for name, axis in self.button_axes.items():
            axis.set_facecolor(theme.PANEL)
            button = Button(
                axis,
                labels[name],
                color=theme.PANEL,
                hovercolor=theme.PANEL_ALT,
            )
            button.label.set_color(theme.MUTED)
            button.label.set_fontsize(8.5)
            if name == "overlay":
                button.on_clicked(self._overlay_callback)
            else:
                button.on_clicked(self._button_callback(name))
            self.buttons[name] = button

        self.active_view = "drive"
        self._latest_state: DashboardState | None = None
        self.set_view(start_view if start_view in self.views else "drive")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _button_callback(self, name: str) -> Callable[[object], None]:
        def callback(_event: object) -> None:
            self.set_view(name)
        return callback

    def _overlay_callback(self, _event: object) -> None:
        if self.overlay is not None:
            self.overlay.toggle()
            self._style_overlay_button()
            self.fig.canvas.draw_idle()

    def _style_overlay_button(self) -> None:
        button = self.buttons.get("overlay")
        if button is None:
            return
        enabled = self.overlay is not None and self.overlay.visible
        button.color = theme.CYAN if enabled else theme.PANEL
        button.hovercolor = theme.CYAN if enabled else theme.PANEL_ALT
        button.ax.set_facecolor(button.color)
        button.label.set_color(theme.BACKGROUND if enabled else theme.MUTED)
        button.label.set_fontweight("bold" if enabled else "normal")

    def set_view(self, name: str) -> None:
        self.active_view = name
        for view_name, view in self.views.items():
            view.set_visible(view_name == name)
        for button_name, button in self.buttons.items():
            if button_name == "overlay":
                continue
            selected = button_name == name
            button.color = theme.ACCENT if selected else theme.PANEL
            button.hovercolor = theme.ACCENT if selected else theme.PANEL_ALT
            button.ax.set_facecolor(button.color)
            button.label.set_color(theme.BACKGROUND if selected else theme.MUTED)
            button.label.set_fontweight("bold" if selected else "normal")
        self._style_overlay_button()
        if self._latest_state is not None:
            self.views[name].update(self._latest_state)
        self.fig.canvas.draw_idle()

    def _on_key(self, event: object) -> None:
        key = getattr(event, "key", None)
        if key == "o":
            self._overlay_callback(event)
            return
        mapping = {
            "1": "drive",
            "d": "drive",
            "2": "analysis",
            "a": "analysis",
            "3": "diagnostics",
            "g": "diagnostics",
        }
        if key in mapping:
            self.set_view(mapping[key])

    def _on_close(self, _event: object) -> None:
        if self.overlay is not None:
            self.overlay.close()
        self.runtime.request_stop()

    def update(self, _frame: int):
        state = self.runtime.snapshot()
        self._latest_state = state
        receiver = state.receiver
        recorder = state.recorder
        writer = state.writer

        packet_age = receiver.last_packet_age_s
        live = packet_age is not None and packet_age <= 1.5
        if receiver.error:
            self.status_text.set_text(f"RECEIVER ERROR · {receiver.error}")
            self.status_text.set_color(theme.RED)
        elif live:
            lap = recorder.active_lap_num
            lap_text = f"LAP {lap}" if lap is not None else "SESSION LIVE"
            self.status_text.set_text(
                f"LIVE · {receiver.telemetry_rate_hz:.1f} HZ · "
                f"{lap_text} · SAVED {writer.saved_laps}"
            )
            self.status_text.set_color(theme.GREEN)
        else:
            self.status_text.set_text("WAITING FOR TELEMETRY")
            self.status_text.set_color(theme.MUTED)

        if self.overlay is not None:
            self.overlay.update(state)
        artists = self.views[self.active_view].update(state)
        return [self.title_text, self.subtitle_text, self.status_text, *artists]


def build_app(
    *,
    host: str,
    port: int,
    track: str,
    output_root: Path,
    window_seconds: float,
    reference_path: Path | None,
    start_view: str,
    drive_trace_seconds: float,
    tyre_warning_c: float,
    tyre_critical_c: float,
    overlay_enabled: bool,
    overlay_visible_on_start: bool,
    overlay_width: int,
    overlay_height: int,
    overlay_x: int,
    overlay_y: int,
    overlay_opacity: float,
    overlay_click_through: bool,
    overlay_trace_seconds: float,
    overlay_refresh_hz: float,
) -> None:
    runtime = AppRuntime(
        host=host,
        port=port,
        track=track,
        output_root=output_root,
        buffer_seconds=max(
            window_seconds, drive_trace_seconds, overlay_trace_seconds
        ),
        reference_path=reference_path,
    )
    runtime.start()

    controller = DashboardController(
        runtime=runtime,
        track=track,
        port=port,
        analysis_window_seconds=window_seconds,
        drive_trace_seconds=drive_trace_seconds,
        tyre_warning_c=tyre_warning_c,
        tyre_critical_c=tyre_critical_c,
        start_view=start_view,
        overlay_enabled=overlay_enabled,
        overlay_visible_on_start=overlay_visible_on_start,
        overlay_width=overlay_width,
        overlay_height=overlay_height,
        overlay_x=overlay_x,
        overlay_y=overlay_y,
        overlay_opacity=overlay_opacity,
        overlay_click_through=overlay_click_through,
        overlay_trace_seconds=overlay_trace_seconds,
        overlay_refresh_hz=overlay_refresh_hz,
    )

    animation = FuncAnimation(
        controller.fig,
        controller.update,
        interval=50,
        blit=False,
        cache_frame_data=False,
    )

    print(f"Listening on UDP port {port}")
    print(f"Track label: {track}")
    print(f"Saving laps to: {runtime.raw_lap_dir}")
    print("Views: Drive, Analysis, Diagnostics")
    print(
        "Driver overlay: "
        + ("enabled (press O in the dashboard to toggle)" if overlay_enabled else "disabled")
        + "\n"
    )

    try:
        plt.show()
    finally:
        _ = animation
        runtime.stop()
        writer_state = runtime.writer.snapshot()
        print(
            "\nApplication stopped. "
            f"Completed lap files saved this run: {writer_state.saved_laps}"
        )
