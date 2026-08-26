from __future__ import annotations

import numpy as np
from matplotlib.artist import Artist
from matplotlib.figure import Figure

from f1_engineer.formatting import format_lap_time
from f1_engineer.ui import theme
from f1_engineer.ui.view_state import DashboardState


class AnalysisView:
    """Detailed engineering view for live and post-lap inspection."""

    name = "analysis"

    def __init__(self, fig: Figure, *, window_seconds: float) -> None:
        self.window_seconds = window_seconds

        self.ax_speed = fig.add_axes([0.065, 0.56, 0.41, 0.29])
        self.ax_inputs = fig.add_axes([0.54, 0.56, 0.41, 0.29])
        self.ax_delta = fig.add_axes([0.065, 0.13, 0.41, 0.30])
        self.ax_tyre = fig.add_axes([0.54, 0.13, 0.41, 0.30])
        self.axes = (
            self.ax_speed,
            self.ax_inputs,
            self.ax_delta,
            self.ax_tyre,
        )

        theme.style_axis(
            self.ax_speed,
            title="Speed comparison",
            ylabel="km/h",
        )
        theme.style_axis(
            self.ax_inputs,
            title="Throttle and brake",
            ylabel="Input",
        )
        theme.style_axis(
            self.ax_delta,
            title="Time delta",
            xlabel="Lap distance (%)",
            ylabel="Seconds",
        )
        theme.style_axis(
            self.ax_tyre,
            title="Tyre surface temperatures",
            xlabel="Seconds ago",
            ylabel="°C",
        )

        self.ref_speed, = self.ax_speed.plot(
            [], [], color=theme.MUTED, linestyle="--", linewidth=1.4,
            label="Reference"
        )
        self.cur_speed, = self.ax_speed.plot(
            [], [], color=theme.ACCENT, linewidth=2.0, label="Current"
        )

        self.ref_throttle, = self.ax_inputs.plot(
            [], [], color=theme.GREEN, linestyle="--", alpha=0.55,
            linewidth=1.2, label="Ref throttle"
        )
        self.cur_throttle, = self.ax_inputs.plot(
            [], [], color=theme.GREEN, linewidth=1.9, label="Throttle"
        )
        self.ref_brake, = self.ax_inputs.plot(
            [], [], color=theme.RED, linestyle=":", alpha=0.58,
            linewidth=1.2, label="Ref brake"
        )
        self.cur_brake, = self.ax_inputs.plot(
            [], [], color=theme.RED, linewidth=1.9, label="Brake"
        )

        self.delta_line, = self.ax_delta.plot(
            [], [], color=theme.CYAN, linewidth=2.0, label="Delta"
        )
        self.delta_zero = self.ax_delta.axhline(
            0.0, color=theme.MUTED, linewidth=0.9, alpha=0.8
        )

        self.tyre_rl, = self.ax_tyre.plot(
            [], [], color=theme.PURPLE, linewidth=1.5, label="RL"
        )
        self.tyre_rr, = self.ax_tyre.plot(
            [], [], color=theme.CYAN, linewidth=1.5, label="RR"
        )
        self.tyre_fl, = self.ax_tyre.plot(
            [], [], color=theme.AMBER, linewidth=1.5, label="FL"
        )
        self.tyre_fr, = self.ax_tyre.plot(
            [], [], color=theme.RED, linewidth=1.5, label="FR"
        )

        self.speed_marker = self.ax_speed.axvline(
            0.0, color=theme.WHITE, linestyle=":", linewidth=0.9, alpha=0.7
        )
        self.inputs_marker = self.ax_inputs.axvline(
            0.0, color=theme.WHITE, linestyle=":", linewidth=0.9, alpha=0.7
        )
        self.delta_marker = self.ax_delta.axvline(
            0.0, color=theme.WHITE, linestyle=":", linewidth=0.9, alpha=0.7
        )

        self.ax_speed.set_xlim(0.0, 100.0)
        self.ax_inputs.set_xlim(0.0, 100.0)
        self.ax_inputs.set_ylim(-0.04, 1.04)
        self.ax_delta.set_xlim(0.0, 100.0)
        self.ax_delta.set_ylim(-1.0, 1.0)
        self.ax_tyre.set_xlim(-window_seconds, 0.0)

        theme.style_legend(self.ax_speed, columns=2)
        theme.style_legend(self.ax_inputs, columns=2)
        theme.style_legend(self.ax_delta)
        theme.style_legend(self.ax_tyre, columns=4)

        self.delta_badge = self.ax_delta.text(
            0.98,
            0.93,
            "Delta ---.--- s",
            transform=self.ax_delta.transAxes,
            ha="right",
            va="top",
            color=theme.MUTED,
            fontsize=11,
            fontweight="bold",
        )
        self.reference_note = self.ax_delta.text(
            0.02,
            0.06,
            "Reference: none",
            transform=self.ax_delta.transAxes,
            ha="left",
            va="bottom",
            color=theme.MUTED,
            fontsize=8.5,
        )

        self._artists: list[Artist] = [
            self.ref_speed,
            self.cur_speed,
            self.ref_throttle,
            self.cur_throttle,
            self.ref_brake,
            self.cur_brake,
            self.delta_line,
            self.delta_zero,
            self.tyre_rl,
            self.tyre_rr,
            self.tyre_fl,
            self.tyre_fr,
            self.speed_marker,
            self.inputs_marker,
            self.delta_marker,
            self.delta_badge,
            self.reference_note,
        ]

    def set_visible(self, visible: bool) -> None:
        for axis in self.axes:
            axis.set_visible(visible)

    @staticmethod
    def _marker(line: Artist, value: float | None) -> None:
        marker = 0.0 if value is None else value
        line.set_xdata([marker, marker])

    def update(self, state: DashboardState) -> list[Artist]:
        comparison = state.comparison

        if comparison.has_reference:
            ref_x = comparison.reference_distance_pct * 100.0
            self.ref_speed.set_data(ref_x, comparison.reference_speed_kmh)
            self.ref_throttle.set_data(ref_x, comparison.reference_throttle)
            self.ref_brake.set_data(ref_x, comparison.reference_brake)
            self.reference_note.set_text(
                "Reference: "
                f"lap {comparison.reference_lap_num or '?'} · "
                f"{format_lap_time(comparison.reference_lap_time_ms)}"
            )
        else:
            self.ref_speed.set_data([], [])
            self.ref_throttle.set_data([], [])
            self.ref_brake.set_data([], [])
            self.reference_note.set_text("Reference: finish one valid lap")

        cur_x = comparison.current_distance_pct * 100.0
        self.cur_speed.set_data(cur_x, comparison.current_speed_kmh)
        self.cur_throttle.set_data(cur_x, comparison.current_throttle)
        self.cur_brake.set_data(cur_x, comparison.current_brake)
        self.delta_line.set_data(cur_x, comparison.current_delta_s)

        self._marker(self.speed_marker, comparison.current_progress_pct)
        self._marker(self.inputs_marker, comparison.current_progress_pct)
        self._marker(self.delta_marker, comparison.current_progress_pct)

        latest_delta = comparison.latest_delta_s
        if latest_delta is None:
            self.delta_badge.set_text("Delta ---.--- s")
            self.delta_badge.set_color(theme.MUTED)
        else:
            self.delta_badge.set_text(f"Delta {latest_delta:+.3f} s")
            self.delta_badge.set_color(
                theme.GREEN if latest_delta <= 0.0 else theme.RED
            )

        finite_delta = comparison.current_delta_s[
            np.isfinite(comparison.current_delta_s)
        ]
        if len(finite_delta):
            magnitude = float(np.max(np.abs(finite_delta)))
            limit = max(0.35, min(5.0, magnitude * 1.25))
            self.ax_delta.set_ylim(-limit, limit)

        speed_parts = [
            values[np.isfinite(values)]
            for values in (
                comparison.reference_speed_kmh,
                comparison.current_speed_kmh,
            )
            if len(values)
        ]
        if speed_parts:
            speed_max = max(float(np.max(values)) for values in speed_parts if len(values))
            self.ax_speed.set_ylim(0.0, max(350.0, speed_max * 1.08))

        samples = state.samples
        if samples:
            now = samples[-1].timestamp
            visible = [
                sample
                for sample in samples
                if now - sample.timestamp <= self.window_seconds
            ]
            x_time = [sample.timestamp - now for sample in visible]
            self.tyre_rl.set_data(x_time, [sample.tyre_rl for sample in visible])
            self.tyre_rr.set_data(x_time, [sample.tyre_rr for sample in visible])
            self.tyre_fl.set_data(x_time, [sample.tyre_fl for sample in visible])
            self.tyre_fr.set_data(x_time, [sample.tyre_fr for sample in visible])
            tyre_values = [
                value
                for sample in visible
                for value in (
                    sample.tyre_rl,
                    sample.tyre_rr,
                    sample.tyre_fl,
                    sample.tyre_fr,
                )
            ]
            if tyre_values:
                low = min(tyre_values)
                high = max(tyre_values)
                margin = max(3.0, (high - low) * 0.15)
                self.ax_tyre.set_ylim(low - margin, high + margin)
        else:
            for line in (self.tyre_rl, self.tyre_rr, self.tyre_fl, self.tyre_fr):
                line.set_data([], [])

        return self._artists
