from __future__ import annotations

from matplotlib.artist import Artist
from matplotlib.figure import Figure

from f1_engineer.formatting import format_gear
from f1_engineer.ui import theme
from f1_engineer.ui.view_state import DashboardState


class DriveView:
    """Minimal glanceable display intended for use while driving."""

    name = "drive"

    def __init__(
        self,
        fig: Figure,
        *,
        trace_seconds: float,
        tyre_warning_c: float,
        tyre_critical_c: float,
    ) -> None:
        self.trace_seconds = trace_seconds
        self.tyre_warning_c = tyre_warning_c
        self.tyre_critical_c = tyre_critical_c

        self.ax_summary = fig.add_axes([0.055, 0.51, 0.89, 0.34])
        self.ax_inputs = fig.add_axes([0.075, 0.13, 0.85, 0.29])
        self.axes = (self.ax_summary, self.ax_inputs)

        self.ax_summary.set_facecolor(theme.PANEL)
        self.ax_summary.set_xticks([])
        self.ax_summary.set_yticks([])
        for spine in self.ax_summary.spines.values():
            spine.set_color(theme.GRID)

        theme.style_axis(
            self.ax_inputs,
            title=f"Pedal history — last {trace_seconds:g} seconds",
            xlabel="Seconds ago",
            ylabel="Input",
        )
        self.ax_inputs.set_xlim(-trace_seconds, 0.0)
        self.ax_inputs.set_ylim(-0.04, 1.04)
        self.ax_inputs.set_yticks([0.0, 0.5, 1.0])

        self.throttle_line, = self.ax_inputs.plot(
            [], [], color=theme.GREEN, linewidth=2.4, label="Throttle"
        )
        self.brake_line, = self.ax_inputs.plot(
            [], [], color=theme.RED, linewidth=2.4, label="Brake"
        )
        theme.style_legend(self.ax_inputs, columns=2)

        self.mode_text = self.ax_summary.text(
            0.025,
            0.91,
            "DRIVE",
            transform=self.ax_summary.transAxes,
            color=theme.MUTED,
            fontsize=10,
            fontweight="medium",
            va="top",
        )
        self.delta_text = self.ax_summary.text(
            0.50,
            0.66,
            "---.---",
            transform=self.ax_summary.transAxes,
            color=theme.MUTED,
            fontsize=42,
            fontweight="bold",
            ha="center",
            va="center",
        )
        self.delta_caption = self.ax_summary.text(
            0.50,
            0.42,
            "LIVE DELTA",
            transform=self.ax_summary.transAxes,
            color=theme.MUTED,
            fontsize=9,
            ha="center",
            va="center",
        )
        self.vehicle_text = self.ax_summary.text(
            0.035,
            0.58,
            "--- km/h\nGear -\nDRS OFF",
            transform=self.ax_summary.transAxes,
            color=theme.TEXT,
            fontsize=16,
            linespacing=1.45,
            va="center",
        )
        self.session_text = self.ax_summary.text(
            0.035,
            0.14,
            "Waiting for telemetry",
            transform=self.ax_summary.transAxes,
            color=theme.MUTED,
            fontsize=9.5,
            va="bottom",
        )

        tyre_positions = {
            "FL": (0.73, 0.67),
            "FR": (0.88, 0.67),
            "RL": (0.73, 0.29),
            "RR": (0.88, 0.29),
        }
        self.tyre_texts = {
            label: self.ax_summary.text(
                x,
                y,
                f"{label}\n-- °C",
                transform=self.ax_summary.transAxes,
                color=theme.TEXT,
                fontsize=11,
                fontweight="medium",
                ha="center",
                va="center",
                linespacing=1.25,
                bbox={
                    "boxstyle": "round,pad=0.55",
                    "facecolor": theme.PANEL_ALT,
                    "edgecolor": theme.GRID,
                    "linewidth": 1.0,
                },
            )
            for label, (x, y) in tyre_positions.items()
        }

        self._artists: list[Artist] = [
            self.throttle_line,
            self.brake_line,
            self.mode_text,
            self.delta_text,
            self.delta_caption,
            self.vehicle_text,
            self.session_text,
            *self.tyre_texts.values(),
        ]

    def set_visible(self, visible: bool) -> None:
        for axis in self.axes:
            axis.set_visible(visible)

    def _tyre_style(self, temperature: float) -> tuple[str, str]:
        if temperature >= self.tyre_critical_c:
            return "CRITICAL", theme.RED
        if temperature >= self.tyre_warning_c:
            return "HIGH", theme.AMBER
        return "OK", theme.GREEN

    def update(self, state: DashboardState) -> list[Artist]:
        comparison = state.comparison
        latest_delta = comparison.latest_delta_s
        if latest_delta is None:
            self.delta_text.set_text("---.---")
            self.delta_text.set_color(theme.MUTED)
        else:
            self.delta_text.set_text(f"{latest_delta:+.3f}")
            self.delta_text.set_color(
                theme.GREEN if latest_delta <= 0.0 else theme.RED
            )

        latest = state.receiver.latest_telemetry
        if latest is None:
            self.vehicle_text.set_text("--- km/h\nGear -\nDRS OFF")
        else:
            self.vehicle_text.set_text(
                f"{latest.speed_kmh:3d} km/h\n"
                f"Gear {format_gear(latest.gear)}\n"
                f"DRS {'ON' if latest.drs else 'OFF'}"
            )
            tyre_values = {
                "FL": latest.tyre_fl,
                "FR": latest.tyre_fr,
                "RL": latest.tyre_rl,
                "RR": latest.tyre_rr,
            }
            for label, value in tyre_values.items():
                status, color = self._tyre_style(float(value))
                text = self.tyre_texts[label]
                text.set_text(f"{label}  {value:.0f} °C\n{status}")
                text.set_bbox(
                    {
                        "boxstyle": "round,pad=0.55",
                        "facecolor": theme.PANEL_ALT,
                        "edgecolor": color,
                        "linewidth": 1.3,
                    }
                )

        recorder = state.recorder
        writer = state.writer
        if recorder.active_lap_num is None:
            lap_label = "Waiting for lap data"
        elif recorder.recording_complete_lap:
            lap_label = (
                f"Recording lap {recorder.active_lap_num} · "
                f"{recorder.buffered_rows:,} rows"
            )
        else:
            lap_label = f"Lap {recorder.active_lap_num} · waiting for timing line"

        packet_age = state.receiver.last_packet_age_s
        live = packet_age is not None and packet_age <= 1.5
        self.session_text.set_text(
            f"{'LIVE' if live else 'WAIT'} · {lap_label} · "
            f"Saved {writer.saved_laps}"
        )
        self.session_text.set_color(theme.GREEN if live else theme.MUTED)

        samples = state.samples
        if samples:
            now = samples[-1].timestamp
            visible = [
                sample
                for sample in samples
                if now - sample.timestamp <= self.trace_seconds
            ]
            x_values = [sample.timestamp - now for sample in visible]
            self.throttle_line.set_data(x_values, [s.throttle for s in visible])
            self.brake_line.set_data(x_values, [s.brake for s in visible])
        else:
            self.throttle_line.set_data([], [])
            self.brake_line.set_data([], [])

        return self._artists
