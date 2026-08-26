from __future__ import annotations

from f1_engineer.formatting import format_lap_time
from f1_engineer.models import (
    ComparisonSnapshot,
    ReceiverSnapshot,
    RecorderSnapshot,
    WriterSnapshot,
)
from f1_engineer.telemetry.packets import HIGH_RATE_PACKET_IDS, PACKET_NAMES

HealthItem = tuple[str, str, str]

def evaluate_health(
    receiver: ReceiverSnapshot,
    recorder: RecorderSnapshot,
    writer: WriterSnapshot,
    comparison: ComparisonSnapshot,
    receiver_alive: bool,
    writer_alive: bool,
) -> tuple[str, list[HealthItem]]:
    items: list[HealthItem] = []

    if receiver.error is not None:
        items.append(("ERROR", "UDP receiver", receiver.error))
    elif not receiver_alive:
        items.append(("ERROR", "UDP receiver", "thread stopped"))
    elif receiver.last_packet_age_s is None:
        items.append(("WAIT", "UDP stream", "waiting for packets"))
    elif receiver.last_packet_age_s > 2.0:
        items.append(
            (
                "ERROR",
                "UDP stream",
                f"silent for {receiver.last_packet_age_s:.1f} s",
            )
        )
    elif receiver.stream_age_s < 2.5:
        items.append(("WAIT", "UDP stream", "measuring packet rate"))
    elif receiver.telemetry_rate_hz < 40.0:
        items.append(
            (
                "ERROR",
                "Telemetry rate",
                f"{receiver.telemetry_rate_hz:.1f} Hz; expected about 60",
            )
        )
    elif receiver.telemetry_rate_hz < 52.0:
        items.append(
            (
                "WARN",
                "Telemetry rate",
                f"{receiver.telemetry_rate_hz:.1f} Hz; below target",
            )
        )
    else:
        items.append(
            (
                "OK",
                "Telemetry rate",
                f"{receiver.telemetry_rate_hz:.1f} Hz",
            )
        )

    if recorder.session_uid is None:
        items.append(("WAIT", "Core packets", "no active session yet"))
    else:
        stale_packets: list[str] = []
        for packet_id in HIGH_RATE_PACKET_IDS:
            age = receiver.packet_age_by_id.get(packet_id)
            if age is None or age > 0.75:
                stale_packets.append(PACKET_NAMES[packet_id])
        if stale_packets:
            items.append(
                (
                    "WARN",
                    "Core packets",
                    "stale: " + ", ".join(stale_packets),
                )
            )
        else:
            items.append(("OK", "Core packets", "all required streams fresh"))

    malformed_ratio = (
        receiver.malformed_packet_count / receiver.packet_count
        if receiver.packet_count
        else 0.0
    )
    if malformed_ratio > 0.02:
        items.append(
            (
                "ERROR",
                "Packet parsing",
                f"{malformed_ratio * 100:.2f}% malformed",
            )
        )
    elif malformed_ratio > 0.005:
        items.append(
            (
                "WARN",
                "Packet parsing",
                f"{malformed_ratio * 100:.2f}% malformed",
            )
        )
    else:
        items.append(
            (
                "OK",
                "Packet parsing",
                f"{receiver.malformed_packet_count} malformed",
            )
        )

    if recorder.active_lap_num is None:
        items.append(("WAIT", "Recorder", "waiting for lap data"))
    elif recorder.recording_complete_lap:
        items.append(
            (
                "OK",
                "Recorder",
                f"lap {recorder.active_lap_num}; {recorder.buffered_rows:,} rows",
            )
        )
    else:
        items.append(
            (
                "WAIT",
                "Recorder",
                f"lap {recorder.active_lap_num}; waiting for timing line",
            )
        )

    if recorder.buffered_rows < 10:
        items.append(("WAIT", "Enrichment", "waiting for joined frames"))
    elif recorder.missing_enrichment_fields:
        missing = ", ".join(recorder.missing_enrichment_fields[:3])
        if len(recorder.missing_enrichment_fields) > 3:
            missing += ", …"
        items.append(("WARN", "Enrichment", f"missing: {missing}"))
    elif not recorder.latest_damage_available:
        items.append(("WARN", "Enrichment", "tyre-wear packet not seen"))
    else:
        items.append(("OK", "Enrichment", "all model fields populated"))

    if recorder.implausible_fields:
        fields = ", ".join(recorder.implausible_fields[:3])
        if len(recorder.implausible_fields) > 3:
            fields += ", …"
        items.append(("ERROR", "Value ranges", f"check: {fields}"))
    elif recorder.buffered_rows >= 10:
        items.append(("OK", "Value ranges", "latest frame plausible"))
    else:
        items.append(("WAIT", "Value ranges", "no complete frame yet"))

    joined_total = recorder.buffered_rows + recorder.dropped_incomplete_frames
    drop_ratio = (
        recorder.dropped_incomplete_frames / joined_total
        if joined_total
        else 0.0
    )
    if drop_ratio > 0.05:
        items.append(
            (
                "ERROR",
                "Frame joining",
                f"{drop_ratio * 100:.2f}% incomplete",
            )
        )
    elif drop_ratio > 0.01:
        items.append(
            (
                "WARN",
                "Frame joining",
                f"{drop_ratio * 100:.2f}% incomplete",
            )
        )
    elif joined_total:
        items.append(
            (
                "OK",
                "Frame joining",
                f"{recorder.dropped_incomplete_frames} dropped",
            )
        )
    else:
        items.append(("WAIT", "Frame joining", "no lap frames yet"))

    if not writer_alive:
        items.append(("ERROR", "Parquet writer", "thread stopped"))
    elif writer.last_error:
        items.append(("ERROR", "Parquet writer", writer.last_error))
    elif writer.queued_jobs > 2:
        items.append(
            (
                "WARN",
                "Parquet writer",
                f"save queue contains {writer.queued_jobs} jobs",
            )
        )
    else:
        items.append(
            (
                "OK",
                "Parquet writer",
                f"queue {writer.queued_jobs}; saved {writer.saved_laps}",
            )
        )

    if comparison.has_reference:
        items.append(
            (
                "OK",
                "Reference lap",
                format_lap_time(comparison.reference_lap_time_ms),
            )
        )
    else:
        items.append(("WAIT", "Reference lap", "finish one valid lap"))

    levels = {level for level, _label, _detail in items}
    if "ERROR" in levels:
        overall = "ERROR"
    elif "WARN" in levels:
        overall = "WARN"
    elif "OK" in levels:
        overall = "OK"
    else:
        overall = "WAIT"
    return overall, items

def format_health_panel(overall: str, items: list[HealthItem]) -> str:
    lines = [f"DATA HEALTH: {overall}", ""]
    for level, label, detail in items:
        lines.append(f"[{level}] {label}")
        lines.append(f"  {detail}")
    return "\n".join(lines)
