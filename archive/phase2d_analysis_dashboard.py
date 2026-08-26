from __future__ import annotations

import threading
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from f1_engineer.comparison.manager import LapComparisonManager
from f1_engineer.core.telemetry_buffer import TelemetryBuffer
from f1_engineer.formatting import format_gear, format_lap_time
from f1_engineer.health import evaluate_health, format_health_panel
from f1_engineer.recording.recorder import LapRecorder
from f1_engineer.recording.writer import LapWriter
from f1_engineer.telemetry.receiver import UDPReceiver

def build_app(
    host: str,
    port: int,
    track: str,
    output_root: Path,
    window_seconds: float,
    reference_path: Path | None,
) -> None:
    raw_lap_dir = output_root / "raw_laps" / track
    raw_lap_dir.mkdir(parents=True, exist_ok=True)

    comparison = LapComparisonManager(raw_lap_dir=raw_lap_dir)
    comparison.load_reference(reference_path)

    writer = LapWriter(raw_lap_dir=raw_lap_dir)
    writer.start()

    recorder = LapRecorder(
        track=track,
        output_root=output_root,
        writer=writer,
        comparison=comparison,
    )
    dashboard_buffer = TelemetryBuffer(
        maxlen=max(600, int(window_seconds * 75))
    )
    stop_event = threading.Event()
    receiver = UDPReceiver(
        host=host,
        port=port,
        dashboard_buffer=dashboard_buffer,
        recorder=recorder,
        stop_event=stop_event,
    )
    receiver.start()

    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(
        4,
        2,
        width_ratios=(4.4, 1.6),
        hspace=0.38,
        wspace=0.22,
    )
    ax_speed = fig.add_subplot(grid[0, 0])
    ax_inputs = fig.add_subplot(grid[1, 0])
    ax_delta = fig.add_subplot(grid[2, 0])
    ax_tyre = fig.add_subplot(grid[3, 0])
    ax_health = fig.add_subplot(grid[:, 1])
    axes = (ax_speed, ax_inputs, ax_delta, ax_tyre)
    ax_health.axis("off")
    fig.canvas.manager.set_window_title(
        "F1 25 Phase 2D — Lap Comparison, Recorder, Data Health"
    )

    reference_speed_line, = ax_speed.plot(
        [], [], linestyle="--", linewidth=1.5, label="Reference speed"
    )
    current_speed_line, = ax_speed.plot(
        [], [], linewidth=2.0, label="Current speed"
    )

    reference_throttle_line, = ax_inputs.plot(
        [], [], linestyle="--", linewidth=1.2, label="Reference throttle"
    )
    current_throttle_line, = ax_inputs.plot(
        [], [], linewidth=1.8, label="Current throttle"
    )
    reference_brake_line, = ax_inputs.plot(
        [], [], linestyle=":", linewidth=1.2, label="Reference brake"
    )
    current_brake_line, = ax_inputs.plot(
        [], [], linewidth=1.8, label="Current brake"
    )

    delta_line, = ax_delta.plot([], [], linewidth=2.0, label="Live delta")
    ax_delta.axhline(0.0, linewidth=1.0)

    tyre_rl_line, = ax_tyre.plot([], [], label="RL")
    tyre_rr_line, = ax_tyre.plot([], [], label="RR")
    tyre_fl_line, = ax_tyre.plot([], [], label="FL")
    tyre_fr_line, = ax_tyre.plot([], [], label="FR")

    speed_marker = ax_speed.axvline(0.0, linestyle=":", linewidth=1.0)
    inputs_marker = ax_inputs.axvline(0.0, linestyle=":", linewidth=1.0)
    delta_marker = ax_delta.axvline(0.0, linestyle=":", linewidth=1.0)

    ax_speed.set_ylabel("Speed (km/h)")
    ax_inputs.set_ylabel("Pedal input")
    ax_delta.set_ylabel("Delta (s)")
    ax_delta.set_xlabel("Lap distance (%)")
    ax_tyre.set_ylabel("Tyre surface °C")
    ax_tyre.set_xlabel("Seconds ago")

    ax_inputs.set_ylim(-0.05, 1.05)
    ax_speed.set_xlim(0.0, 100.0)
    ax_inputs.set_xlim(0.0, 100.0)
    ax_delta.set_xlim(0.0, 100.0)
    ax_delta.set_ylim(-2.0, 2.0)
    ax_tyre.set_xlim(-window_seconds, 0.0)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", ncols=2)

    live_text = fig.text(
        0.01,
        0.988,
        "Waiting for telemetry…",
        ha="left",
        va="top",
    )
    delta_text = fig.text(
        0.50,
        0.988,
        "Delta: ---.--- s",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    recorder_text = fig.text(
        0.50,
        0.963,
        f"Track: {track} | Waiting for lap data",
        ha="center",
        va="top",
    )
    network_text = fig.text(
        0.99,
        0.988,
        f"Listening on UDP {port}",
        ha="right",
        va="top",
    )
    reference_text = fig.text(
        0.99,
        0.963,
        "Reference: loading…",
        ha="right",
        va="top",
    )

    health_text = ax_health.text(
        0.02,
        0.98,
        "DATA HEALTH: WAIT\n\nWaiting for telemetry…",
        ha="left",
        va="top",
        transform=ax_health.transAxes,
        family="monospace",
        fontsize=9.5,
        linespacing=1.25,
    )

    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.07, right=0.98)

    artists = (
        reference_speed_line,
        current_speed_line,
        reference_throttle_line,
        current_throttle_line,
        reference_brake_line,
        current_brake_line,
        delta_line,
        tyre_rl_line,
        tyre_rr_line,
        tyre_fl_line,
        tyre_fr_line,
        speed_marker,
        inputs_marker,
        delta_marker,
        live_text,
        delta_text,
        recorder_text,
        network_text,
        reference_text,
        health_text,
    )

    def update(_frame: int):
        receiver_state = receiver.snapshot()
        recorder_state = recorder.snapshot()
        writer_state = writer.snapshot()
        comparison_state = comparison.snapshot()

        if receiver_state.error is not None:
            network_text.set_text(f"Receiver error: {receiver_state.error}")

        if recorder_state.active_lap_num is None:
            recording_label = "Waiting for lap data"
        elif recorder_state.recording_complete_lap:
            recording_label = (
                f"Recording lap {recorder_state.active_lap_num} | "
                f"Rows {recorder_state.buffered_rows:,}"
            )
        else:
            recording_label = (
                f"Lap {recorder_state.active_lap_num} partial | "
                "waiting for timing line"
            )

        recorder_text.set_text(
            f"Track: {track} | {recording_label} | "
            f"Saved {writer_state.saved_laps}"
        )
        if writer_state.last_error:
            recorder_text.set_text(f"Write error: {writer_state.last_error}")

        age = receiver_state.last_packet_age_s
        connection = "waiting" if age is None or age > 1.5 else "live"
        network_text.set_text(
            f"UDP {connection} | Packets {receiver_state.packet_count:,} | "
            f"Telemetry {receiver_state.telemetry_rate_hz:.1f} Hz | "
            f"Write queue {writer_state.queued_jobs}"
        )

        if comparison_state.has_reference:
            reference_label = (
                f"Reference lap {comparison_state.reference_lap_num or '?'} | "
                f"{format_lap_time(comparison_state.reference_lap_time_ms)}"
            )
            reference_text.set_text(reference_label)

            ref_x = comparison_state.reference_distance_pct * 100.0
            reference_speed_line.set_data(
                ref_x, comparison_state.reference_speed_kmh
            )
            reference_throttle_line.set_data(
                ref_x, comparison_state.reference_throttle
            )
            reference_brake_line.set_data(
                ref_x, comparison_state.reference_brake
            )
        else:
            reference_text.set_text(
                "Reference: none — finish one valid lap"
            )
            reference_speed_line.set_data([], [])
            reference_throttle_line.set_data([], [])
            reference_brake_line.set_data([], [])

        current_x = comparison_state.current_distance_pct * 100.0
        current_speed_line.set_data(
            current_x, comparison_state.current_speed_kmh
        )
        current_throttle_line.set_data(
            current_x, comparison_state.current_throttle
        )
        current_brake_line.set_data(
            current_x, comparison_state.current_brake
        )
        delta_line.set_data(current_x, comparison_state.current_delta_s)

        if comparison_state.current_progress_pct is not None:
            marker_x = comparison_state.current_progress_pct
            speed_marker.set_xdata([marker_x, marker_x])
            inputs_marker.set_xdata([marker_x, marker_x])
            delta_marker.set_xdata([marker_x, marker_x])
        else:
            speed_marker.set_xdata([0.0, 0.0])
            inputs_marker.set_xdata([0.0, 0.0])
            delta_marker.set_xdata([0.0, 0.0])

        latest_delta = comparison_state.latest_delta_s
        if latest_delta is None:
            delta_text.set_text("Delta: ---.--- s")
        else:
            delta_text.set_text(f"Delta: {latest_delta:+.3f} s")

        if len(comparison_state.current_delta_s):
            abs_delta = float(
                np.nanmax(np.abs(comparison_state.current_delta_s))
            )
            delta_limit = max(0.5, min(5.0, abs_delta * 1.25))
            ax_delta.set_ylim(-delta_limit, delta_limit)

        speed_values = []
        if len(comparison_state.reference_speed_kmh):
            speed_values.extend(comparison_state.reference_speed_kmh.tolist())
        if len(comparison_state.current_speed_kmh):
            speed_values.extend(comparison_state.current_speed_kmh.tolist())
        ax_speed.set_ylim(0, max(350, max(speed_values, default=320) * 1.08))

        overall_health, health_items = evaluate_health(
            receiver=receiver_state,
            recorder=recorder_state,
            writer=writer_state,
            comparison=comparison_state,
            receiver_alive=receiver.is_alive(),
            writer_alive=writer.is_alive(),
        )
        health_text.set_text(
            format_health_panel(overall_health, health_items)
        )

        samples = dashboard_buffer.snapshot()
        if samples:
            now = samples[-1].timestamp
            visible = [
                sample
                for sample in samples
                if now - sample.timestamp <= window_seconds
            ]
            x_time = [sample.timestamp - now for sample in visible]
            tyre_rl_line.set_data(x_time, [s.tyre_rl for s in visible])
            tyre_rr_line.set_data(x_time, [s.tyre_rr for s in visible])
            tyre_fl_line.set_data(x_time, [s.tyre_fl for s in visible])
            tyre_fr_line.set_data(x_time, [s.tyre_fr for s in visible])

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
            tyre_min = min(tyre_values)
            tyre_max = max(tyre_values)
            margin = max(3.0, (tyre_max - tyre_min) * 0.15)
            ax_tyre.set_ylim(tyre_min - margin, tyre_max + margin)

            latest = visible[-1]
            live_text.set_text(
                f"{latest.speed_kmh:3d} km/h | "
                f"Gear {format_gear(latest.gear)} | "
                f"RPM {latest.rpm:5d} | "
                f"DRS {'ON' if latest.drs else 'OFF'}"
            )

        return artists

    animation = FuncAnimation(
        fig,
        update,
        interval=50,
        blit=False,
        cache_frame_data=False,
    )

    def request_close(_event) -> None:
        stop_event.set()

    fig.canvas.mpl_connect("close_event", request_close)

    print(f"Listening on UDP port {port}")
    print(f"Track label: {track}")
    print(f"Saving laps to: {raw_lap_dir}")
    print(
        "Phase 2D aligns current and reference laps by lap distance, "
        "not wall-clock time.\n"
    )

    try:
        plt.show()
    finally:
        _ = animation
        stop_event.set()
        receiver.join(timeout=2.0)
        writer.finish()
        writer_state = writer.snapshot()
        print(
            "\nApplication stopped. "
            f"Completed lap files saved this run: {writer_state.saved_laps}"
        )
