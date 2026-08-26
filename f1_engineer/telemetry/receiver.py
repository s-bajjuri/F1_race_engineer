from __future__ import annotations

import socket
import struct
import threading
import time
from collections import defaultdict, deque

from f1_engineer.core.telemetry_buffer import TelemetryBuffer
from f1_engineer.models import DashboardSample, ReceiverSnapshot
from f1_engineer.recording.recorder import LapRecorder
from f1_engineer.telemetry.packets import (
    CAR_DAMAGE_PACKET_ID,
    CAR_STATUS_PACKET_ID,
    CAR_TELEMETRY_PACKET_ID,
    F1_25_PACKET_FORMAT,
    LAP_DATA_PACKET_ID,
    MOTION_EX_PACKET_ID,
    MOTION_PACKET_ID,
    parse_header,
    parse_player_damage,
    parse_player_lap_data,
    parse_player_motion,
    parse_player_motion_ex,
    parse_player_status,
    parse_player_telemetry,
)

class UDPReceiver(threading.Thread):
    def __init__(
        self,
        host: str,
        port: int,
        dashboard_buffer: TelemetryBuffer,
        recorder: LapRecorder,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="udp-receiver", daemon=True)
        self.host = host
        self.port = port
        self.dashboard_buffer = dashboard_buffer
        self.recorder = recorder
        self.stop_event = stop_event
        self.error: Exception | None = None
        self.packet_count = 0
        self.telemetry_packet_count = 0
        self.malformed_packet_count = 0
        self.last_packet_at: float | None = None
        self.first_packet_at: float | None = None
        self.latest_telemetry: DashboardSample | None = None
        self._stats_lock = threading.Lock()
        self._packet_times: deque[float] = deque(maxlen=20_000)
        self._packet_times_by_id: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=2_000)
        )
        self._last_packet_by_id: dict[int, float] = {}

    @staticmethod
    def _recent_rate(times: deque[float], now: float, window_s: float) -> float:
        if not times:
            return 0.0
        cutoff = now - window_s
        recent = [timestamp for timestamp in times if timestamp >= cutoff]
        if not recent:
            return 0.0
        elapsed = min(window_s, max(0.25, now - recent[0]))
        return len(recent) / elapsed

    def snapshot(self) -> ReceiverSnapshot:
        now = time.monotonic()
        with self._stats_lock:
            last_age = (
                None
                if self.last_packet_at is None
                else max(0.0, now - self.last_packet_at)
            )
            first_age = (
                0.0
                if self.first_packet_at is None
                else max(0.0, now - self.first_packet_at)
            )
            packet_age_by_id = {
                packet_id: max(0.0, now - timestamp)
                for packet_id, timestamp in self._last_packet_by_id.items()
            }
            return ReceiverSnapshot(
                packet_count=self.packet_count,
                telemetry_packet_count=self.telemetry_packet_count,
                malformed_packet_count=self.malformed_packet_count,
                last_packet_age_s=last_age,
                packet_rate_hz=self._recent_rate(
                    self._packet_times, now, window_s=2.0
                ),
                telemetry_rate_hz=self._recent_rate(
                    self._packet_times_by_id[CAR_TELEMETRY_PACKET_ID],
                    now,
                    window_s=2.0,
                ),
                stream_age_s=first_age,
                packet_age_by_id=packet_age_by_id,
                latest_telemetry=self.latest_telemetry,
                error=(None if self.error is None else str(self.error)),
            )

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.5)

        try:
            sock.bind((self.host, self.port))

            while not self.stop_event.is_set():
                try:
                    packet, _sender = sock.recvfrom(4096)
                except socket.timeout:
                    continue

                received_at = time.monotonic()
                with self._stats_lock:
                    self.packet_count += 1
                    self.last_packet_at = received_at
                    if self.first_packet_at is None:
                        self.first_packet_at = received_at
                    self._packet_times.append(received_at)

                try:
                    header = parse_header(packet)
                    if header.packet_format != F1_25_PACKET_FORMAT:
                        continue

                    with self._stats_lock:
                        self._packet_times_by_id[header.packet_id].append(
                            received_at
                        )
                        self._last_packet_by_id[header.packet_id] = received_at

                    if header.packet_id == MOTION_PACKET_ID:
                        self.recorder.put_frame_part(
                            header,
                            "motion",
                            parse_player_motion(packet, header),
                        )

                    elif header.packet_id == LAP_DATA_PACKET_ID:
                        self.recorder.handle_lap_packet(
                            header,
                            parse_player_lap_data(packet, header),
                        )

                    elif header.packet_id == CAR_TELEMETRY_PACKET_ID:
                        telemetry = parse_player_telemetry(packet, header)
                        self.recorder.put_frame_part(
                            header,
                            "telemetry",
                            telemetry,
                        )
                        tyre_rl, tyre_rr, tyre_fl, tyre_fr = (
                            telemetry.tyre_surface_temperatures
                        )
                        dashboard_sample = DashboardSample(
                            timestamp=received_at,
                            speed_kmh=telemetry.speed_kmh,
                            throttle=telemetry.throttle,
                            brake=telemetry.brake,
                            steering=telemetry.steering,
                            gear=telemetry.gear,
                            rpm=telemetry.rpm,
                            drs=telemetry.drs,
                            tyre_rl=tyre_rl,
                            tyre_rr=tyre_rr,
                            tyre_fl=tyre_fl,
                            tyre_fr=tyre_fr,
                        )
                        with self._stats_lock:
                            self.telemetry_packet_count += 1
                            self.latest_telemetry = dashboard_sample
                        self.dashboard_buffer.append(dashboard_sample)

                    elif header.packet_id == CAR_STATUS_PACKET_ID:
                        self.recorder.put_frame_part(
                            header,
                            "status",
                            parse_player_status(packet, header),
                        )

                    elif header.packet_id == CAR_DAMAGE_PACKET_ID:
                        self.recorder.handle_damage_packet(
                            header,
                            parse_player_damage(packet, header),
                        )

                    elif header.packet_id == MOTION_EX_PACKET_ID:
                        self.recorder.put_frame_part(
                            header,
                            "motion_ex",
                            parse_player_motion_ex(packet),
                        )

                except (ValueError, struct.error):
                    with self._stats_lock:
                        self.malformed_packet_count += 1

        except Exception as exc:  # displayed by dashboard instead of dying silently
            self.error = exc
        finally:
            sock.close()
