from __future__ import annotations

from pathlib import Path

from matplotlib.artist import Artist
from matplotlib.figure import Figure

from f1_engineer.formatting import format_lap_time
from f1_engineer.health import format_health_panel
from f1_engineer.ui import theme
from f1_engineer.ui.view_state import DashboardState


class DiagnosticsView:
    """Developer-facing health and recorder diagnostics."""

    name = "diagnostics"

    def __init__(self, fig: Figure, *, track: str, output_root: Path) -> None:
        self.track = track
        self.output_root = output_root

        self.ax_health = fig.add_axes([0.055, 0.12, 0.43, 0.73])
        self.ax_details = fig.add_axes([0.515, 0.12, 0.43, 0.73])
        self.axes = (self.ax_health, self.ax_details)

        for axis, title in (
            (self.ax_health, "Data health"),
            (self.ax_details, "Runtime details"),
        ):
            axis.set_facecolor(theme.PANEL)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color(theme.GRID)
            axis.set_title(
                title,
                loc="left",
                color=theme.TEXT,
                fontsize=11,
                fontweight="medium",
                pad=12,
            )

        self.health_text = self.ax_health.text(
            0.035,
            0.965,
            "DATA HEALTH: WAIT\n\nWaiting for telemetry…",
            transform=self.ax_health.transAxes,
            ha="left",
            va="top",
            color=theme.TEXT,
            family="monospace",
            fontsize=9.3,
            linespacing=1.30,
        )
        self.details_text = self.ax_details.text(
            0.035,
            0.965,
            "Waiting for runtime snapshot…",
            transform=self.ax_details.transAxes,
            ha="left",
            va="top",
            color=theme.TEXT,
            family="monospace",
            fontsize=9.3,
            linespacing=1.38,
        )
        self._artists: list[Artist] = [self.health_text, self.details_text]

    def set_visible(self, visible: bool) -> None:
        for axis in self.axes:
            axis.set_visible(visible)

    @staticmethod
    def _short_path(value: str | None) -> str:
        if not value:
            return "—"
        path = Path(value)
        try:
            return str(path.relative_to(Path.cwd()))
        except ValueError:
            return str(path)

    def update(self, state: DashboardState) -> list[Artist]:
        self.health_text.set_text(
            format_health_panel(state.health_overall, state.health_items)
        )
        self.health_text.set_color(
            {
                "OK": theme.GREEN,
                "WARN": theme.AMBER,
                "ERROR": theme.RED,
                "WAIT": theme.MUTED,
            }.get(state.health_overall, theme.TEXT)
        )

        receiver = state.receiver
        recorder = state.recorder
        writer = state.writer
        comparison = state.comparison

        age = (
            "—"
            if receiver.last_packet_age_s is None
            else f"{receiver.last_packet_age_s:.2f} s"
        )
        reference_time = (
            format_lap_time(comparison.reference_lap_time_ms)
            if comparison.has_reference
            else "—"
        )
        missing = (
            ", ".join(recorder.missing_enrichment_fields)
            if recorder.missing_enrichment_fields
            else "none"
        )
        implausible = (
            ", ".join(recorder.implausible_fields)
            if recorder.implausible_fields
            else "none"
        )

        details = [
            "NETWORK",
            f"  UDP endpoint       {receiver.packet_rate_hz:7.1f} pkt/s",
            f"  Telemetry rate     {receiver.telemetry_rate_hz:7.1f} Hz",
            f"  Total packets      {receiver.packet_count:7,d}",
            f"  Malformed          {receiver.malformed_packet_count:7,d}",
            f"  Last packet age    {age}",
            f"  Receiver error     {receiver.error or 'none'}",
            "",
            "RECORDER",
            f"  Track              {self.track}",
            f"  Session UID        {recorder.session_uid or '—'}",
            f"  Active lap         {recorder.active_lap_num or '—'}",
            f"  Full-lap capture   {recorder.recording_complete_lap}",
            f"  Buffered rows      {recorder.buffered_rows:,}",
            f"  Incomplete frames  {recorder.dropped_incomplete_frames:,}",
            f"  Missing fields     {missing}",
            f"  Implausible fields {implausible}",
            "",
            "PARQUET WRITER",
            f"  Saved this run     {writer.saved_laps}",
            f"  Queue length       {writer.queued_jobs}",
            f"  Last error         {writer.last_error or 'none'}",
            f"  Last file          {self._short_path(writer.last_saved_file)}",
            "",
            "REFERENCE",
            f"  Available          {comparison.has_reference}",
            f"  Lap                {comparison.reference_lap_num or '—'}",
            f"  Time               {reference_time}",
            f"  Source             {self._short_path(comparison.reference_source)}",
            "",
            f"Data root: {self.output_root}",
        ]
        self.details_text.set_text("\n".join(details))
        return self._artists
