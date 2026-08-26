from __future__ import annotations

import os
import time
import tkinter as tk
from dataclasses import dataclass
from typing import Final

from f1_engineer.ui import theme
from f1_engineer.ui.view_state import DashboardState


@dataclass(frozen=True, slots=True)
class OverlayOptions:
    width: int
    height: int
    x: int
    y: int
    opacity: float
    click_through: bool
    trace_seconds: float
    refresh_hz: float
    visible_on_start: bool
    tyre_warning_c: float
    tyre_critical_c: float


class DriveOverlay:
    """Small always-on-top, non-activating driver overlay.

    The overlay does not own telemetry or open a UDP socket. It only renders
    snapshots produced by the existing AppRuntime, so recording, comparison,
    analysis, and diagnostics continue to share one backend.
    """

    _PADDING: Final[int] = 18

    def __init__(
        self,
        *,
        parent: tk.Misc,
        options: OverlayOptions,
    ) -> None:
        self.options = options
        self._last_draw_time = 0.0
        self._visible = options.visible_on_start
        self._closed = False
        self._windows_styles_applied = False

        self.window = tk.Toplevel(parent)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.configure(bg=theme.BACKGROUND)
        self.window.attributes("-topmost", True)
        self.window.attributes("-alpha", options.opacity)
        self.window.geometry(
            f"{options.width}x{options.height}+{options.x}+{options.y}"
        )

        self.canvas = tk.Canvas(
            self.window,
            width=options.width,
            height=options.height,
            bg=theme.BACKGROUND,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        if self._visible:
            self.show()

    @property
    def visible(self) -> bool:
        return self._visible and not self._closed

    def show(self) -> None:
        if self._closed:
            return
        self._visible = True
        self.window.deiconify()
        self.window.lift()
        self.window.attributes("-topmost", True)
        self.window.update_idletasks()
        self._apply_windows_overlay_styles()

    def hide(self) -> None:
        if self._closed:
            return
        self._visible = False
        self.window.withdraw()

    def toggle(self) -> None:
        if self.visible:
            self.hide()
        else:
            self.show()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._visible = False
        try:
            self.window.destroy()
        except tk.TclError:
            pass

    def update(self, state: DashboardState) -> None:
        if not self.visible:
            return

        now = time.monotonic()
        minimum_interval = 1.0 / max(self.options.refresh_hz, 1.0)
        if now - self._last_draw_time < minimum_interval:
            return
        self._last_draw_time = now

        try:
            self._draw(state)
        except tk.TclError:
            self._closed = True
            self._visible = False

    def _draw(self, state: DashboardState) -> None:
        canvas = self.canvas
        width = max(canvas.winfo_width(), self.options.width)
        height = max(canvas.winfo_height(), self.options.height)
        canvas.delete("all")

        # Background and subtle border.
        canvas.create_rectangle(
            1,
            1,
            width - 2,
            height - 2,
            fill=theme.BACKGROUND,
            outline=theme.GRID,
            width=1,
        )

        comparison = state.comparison
        latest_delta = comparison.latest_delta_s
        if latest_delta is None:
            delta_text = "---.---"
            delta_color = theme.MUTED
        else:
            delta_text = f"{latest_delta:+.3f}"
            delta_color = theme.GREEN if latest_delta <= 0.0 else theme.RED

        canvas.create_text(
            self._PADDING,
            16,
            text="LIVE DELTA",
            anchor="nw",
            fill=theme.MUTED,
            font=("Segoe UI", 9, "bold"),
        )
        canvas.create_text(
            self._PADDING,
            38,
            text=delta_text,
            anchor="nw",
            fill=delta_color,
            font=("Segoe UI", 21, "bold"),
        )

        receiver = state.receiver
        recorder = state.recorder
        writer = state.writer
        packet_age = receiver.last_packet_age_s
        live = packet_age is not None and packet_age <= 1.5

        if recorder.active_lap_num is None:
            lap_text = "WAITING FOR LAP"
        elif recorder.recording_complete_lap:
            lap_text = f"LAP {recorder.active_lap_num} · REC"
        else:
            lap_text = f"LAP {recorder.active_lap_num} · ARMING"

        status_color = theme.GREEN if live else theme.MUTED
        status_text = (
            f"{'LIVE' if live else 'WAIT'}  ·  {lap_text}  ·  "
            f"SAVED {writer.saved_laps}"
        )
        canvas.create_oval(
            width - 232,
            19,
            width - 220,
            31,
            fill=status_color,
            outline="",
        )
        canvas.create_text(
            width - 210,
            16,
            text=status_text,
            anchor="nw",
            fill=theme.TEXT,
            font=("Segoe UI", 9, "bold"),
        )

        latest = receiver.latest_telemetry
        tyre_values: dict[str, float | None]
        if latest is None:
            tyre_values = {"FL": None, "FR": None, "RL": None, "RR": None}
        else:
            tyre_values = {
                "FL": float(latest.tyre_fl),
                "FR": float(latest.tyre_fr),
                "RL": float(latest.tyre_rl),
                "RR": float(latest.tyre_rr),
            }

        tyre_box_width = 72
        tyre_gap = 8
        total_tyre_width = 4 * tyre_box_width + 3 * tyre_gap
        tyre_start_x = width - self._PADDING - total_tyre_width
        tyre_y = 48
        for index, label in enumerate(("FL", "FR", "RL", "RR")):
            x1 = tyre_start_x + index * (tyre_box_width + tyre_gap)
            x2 = x1 + tyre_box_width
            value = tyre_values[label]
            status, border_color = self._tyre_status(value)
            canvas.create_rectangle(
                x1,
                tyre_y,
                x2,
                tyre_y + 50,
                fill=theme.PANEL,
                outline=border_color,
                width=1,
            )
            temperature = "--°" if value is None else f"{value:.0f}°"
            canvas.create_text(
                (x1 + x2) / 2,
                tyre_y + 13,
                text=f"{label}  {temperature}",
                fill=theme.TEXT,
                font=("Segoe UI", 10, "bold"),
            )
            canvas.create_text(
                (x1 + x2) / 2,
                tyre_y + 35,
                text=status,
                fill=border_color,
                font=("Segoe UI", 8, "bold"),
            )

        # Pedal trace region.
        graph_left = self._PADDING
        graph_right = width - self._PADDING
        graph_top = 112
        graph_bottom = height - self._PADDING
        canvas.create_rectangle(
            graph_left,
            graph_top,
            graph_right,
            graph_bottom,
            fill=theme.PANEL,
            outline=theme.GRID,
            width=1,
        )

        canvas.create_text(
            graph_left + 10,
            graph_top + 8,
            text=f"PEDALS · LAST {self.options.trace_seconds:g}s",
            anchor="nw",
            fill=theme.MUTED,
            font=("Segoe UI", 8, "bold"),
        )
        canvas.create_line(
            graph_left + 105,
            graph_top + 14,
            graph_left + 133,
            graph_top + 14,
            fill=theme.GREEN,
            width=3,
        )
        canvas.create_text(
            graph_left + 140,
            graph_top + 8,
            text="THROTTLE",
            anchor="nw",
            fill=theme.MUTED,
            font=("Segoe UI", 8),
        )
        canvas.create_line(
            graph_left + 208,
            graph_top + 14,
            graph_left + 236,
            graph_top + 14,
            fill=theme.RED,
            width=3,
        )
        canvas.create_text(
            graph_left + 243,
            graph_top + 8,
            text="BRAKE",
            anchor="nw",
            fill=theme.MUTED,
            font=("Segoe UI", 8),
        )

        plot_left = graph_left + 10
        plot_right = graph_right - 10
        plot_top = graph_top + 31
        plot_bottom = graph_bottom - 10

        # 0%, 50%, and 100% guides.
        for ratio in (0.0, 0.5, 1.0):
            y = plot_bottom - ratio * (plot_bottom - plot_top)
            canvas.create_line(
                plot_left,
                y,
                plot_right,
                y,
                fill=theme.GRID,
                width=1,
            )

        samples = state.samples
        if samples:
            newest_timestamp = samples[-1].timestamp
            visible_samples = [
                sample
                for sample in samples
                if newest_timestamp - sample.timestamp <= self.options.trace_seconds
            ]
            self._draw_trace(
                canvas,
                visible_samples,
                newest_timestamp,
                value_name="throttle",
                color=theme.GREEN,
                plot_left=plot_left,
                plot_right=plot_right,
                plot_top=plot_top,
                plot_bottom=plot_bottom,
            )
            self._draw_trace(
                canvas,
                visible_samples,
                newest_timestamp,
                value_name="brake",
                color=theme.RED,
                plot_left=plot_left,
                plot_right=plot_right,
                plot_top=plot_top,
                plot_bottom=plot_bottom,
            )

    def _draw_trace(
        self,
        canvas: tk.Canvas,
        samples: list[object],
        newest_timestamp: float,
        *,
        value_name: str,
        color: str,
        plot_left: float,
        plot_right: float,
        plot_top: float,
        plot_bottom: float,
    ) -> None:
        if len(samples) < 2:
            return

        trace_seconds = max(self.options.trace_seconds, 0.1)
        points: list[float] = []
        for sample in samples:
            age = newest_timestamp - float(getattr(sample, "timestamp"))
            x_ratio = 1.0 - min(max(age / trace_seconds, 0.0), 1.0)
            value = min(max(float(getattr(sample, value_name)), 0.0), 1.0)
            x = plot_left + x_ratio * (plot_right - plot_left)
            y = plot_bottom - value * (plot_bottom - plot_top)
            points.extend((x, y))

        if len(points) >= 4:
            canvas.create_line(
                *points,
                fill=color,
                width=2,
                smooth=False,
            )

    def _tyre_status(self, temperature: float | None) -> tuple[str, str]:
        if temperature is None:
            return "WAIT", theme.MUTED
        if temperature >= self.options.tyre_critical_c:
            return "CRITICAL", theme.RED
        if temperature >= self.options.tyre_warning_c:
            return "HIGH", theme.AMBER
        return "OK", theme.GREEN

    def _apply_windows_overlay_styles(self) -> None:
        if os.name != "nt" or self._windows_styles_applied:
            return

        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            hwnd = self.window.winfo_id()

            # Tk may expose a child HWND. Prefer its top-level parent when present.
            parent_hwnd = user32.GetParent(hwnd)
            if parent_hwnd:
                hwnd = parent_hwnd

            get_window_long = user32.GetWindowLongW
            set_window_long = user32.SetWindowLongW
            get_window_long.argtypes = [wintypes.HWND, ctypes.c_int]
            get_window_long.restype = ctypes.c_long
            set_window_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
            set_window_long.restype = ctypes.c_long

            gwl_exstyle = -20
            ws_ex_toolwindow = 0x00000080
            ws_ex_noactivate = 0x08000000
            ws_ex_layered = 0x00080000
            ws_ex_transparent = 0x00000020

            styles = get_window_long(hwnd, gwl_exstyle)
            styles |= ws_ex_toolwindow | ws_ex_noactivate | ws_ex_layered
            if self.options.click_through:
                styles |= ws_ex_transparent
            set_window_long(hwnd, gwl_exstyle, styles)

            hwnd_topmost = -1
            swp_nosize = 0x0001
            swp_nomove = 0x0002
            swp_noactivate = 0x0010
            swp_showwindow = 0x0040
            user32.SetWindowPos(
                hwnd,
                hwnd_topmost,
                0,
                0,
                0,
                0,
                swp_nosize | swp_nomove | swp_noactivate | swp_showwindow,
            )
            self._windows_styles_applied = True
        except Exception as exc:  # pragma: no cover - Windows/backend specific
            print(f"Overlay style warning: {exc}")
