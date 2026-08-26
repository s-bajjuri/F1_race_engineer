from __future__ import annotations

import json
import math
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from f1_engineer.comparison.manager import LapComparisonManager
from f1_engineer.models import (
    PacketHeader,
    PlayerDamage,
    PlayerLapData,
    PlayerMotion,
    PlayerMotionEx,
    PlayerStatus,
    PlayerTelemetry,
    LapWriteJob,
    RecorderSnapshot,
)
from f1_engineer.recording.writer import LapWriter
from f1_engineer.telemetry.packets import REQUIRED_FRAME_PARTS

class LapRecorder:
    def __init__(
        self,
        track: str,
        output_root: Path,
        writer: LapWriter,
        comparison: LapComparisonManager,
    ) -> None:
        self.track = track
        self.output_root = output_root
        self.raw_lap_dir = output_root / "raw_laps" / track
        self.session_dir = output_root / "sessions"
        self.raw_lap_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.writer = writer
        self.comparison = comparison

        self._lock = threading.RLock()
        self.session_uid: int | None = None
        self.active_lap_num: int | None = None
        self.recording_complete_lap = False
        self.rows: list[dict[str, Any]] = []
        self.pending_frames: dict[int, dict[str, Any]] = {}
        self.latest_damage: PlayerDamage | None = None

        self.previous_frame_identifier: int | None = None
        self.previous_lap_time_ms: int | None = None
        self.previous_lap_distance_m: float | None = None
        self.dropped_incomplete_frames = 0
        self.last_emitted_row: dict[str, Any] | None = None

    def snapshot(self) -> RecorderSnapshot:
        with self._lock:
            latest_row = self.last_emitted_row if self.rows else None
            missing_enrichment: list[str] = []
            implausible: list[str] = []

            if latest_row is not None:
                enrichment_fields = (
                    "fuel_remaining",
                    "ers_deployed",
                    "fl_tyre_wear",
                    "fr_tyre_wear",
                    "rl_tyre_wear",
                    "rr_tyre_wear",
                    "g_force_lat",
                    "g_force_lon",
                    "wheel_slip_fl",
                    "wheel_slip_fr",
                    "wheel_slip_rl",
                    "wheel_slip_rr",
                    "world_position_x",
                    "world_position_y",
                    "world_position_z",
                    "yaw",
                    "wheel_slip_angle_fl",
                    "wheel_slip_angle_fr",
                    "wheel_slip_angle_rl",
                    "wheel_slip_angle_rr",
                    "yaw_rate",
                )
                missing_enrichment = [
                    name
                    for name in enrichment_fields
                    if pd.isna(latest_row.get(name))
                ]

                broad_ranges = {
                    "speed_kmh": (0.0, 400.0),
                    "throttle": (0.0, 1.01),
                    "brake": (0.0, 1.01),
                    "gear": (-1.0, 8.0),
                    "rpm": (0.0, 20_000.0),
                    "fuel_remaining": (-0.1, 120.0),
                    "ers_deployed": (0.0, 5_000_000.0),
                    "g_force_lat": (-15.0, 15.0),
                    "g_force_lon": (-15.0, 15.0),
                    "fl_tyre_wear": (0.0, 1.0),
                    "fr_tyre_wear": (0.0, 1.0),
                    "rl_tyre_wear": (0.0, 1.0),
                    "rr_tyre_wear": (0.0, 1.0),
                    "fl_tyre_temp": (0.0, 200.0),
                    "fr_tyre_temp": (0.0, 200.0),
                    "rl_tyre_temp": (0.0, 200.0),
                    "rr_tyre_temp": (0.0, 200.0),
                    "fl_tyre_pressure": (5.0, 50.0),
                    "fr_tyre_pressure": (5.0, 50.0),
                    "rl_tyre_pressure": (5.0, 50.0),
                    "rr_tyre_pressure": (5.0, 50.0),
                    "wheel_slip_fl": (-5.0, 10.0),
                    "wheel_slip_fr": (-5.0, 10.0),
                    "wheel_slip_rl": (-5.0, 10.0),
                    "wheel_slip_rr": (-5.0, 10.0),
                    # Generous by design, matching the g_force policy: these
                    # catch a mis-parsed struct, not aggressive driving.
                    "world_position_x": (-10_000.0, 10_000.0),
                    "world_position_y": (-1_000.0, 1_000.0),
                    "world_position_z": (-10_000.0, 10_000.0),
                    "g_force_vert": (-15.0, 15.0),
                    "yaw": (-3.2, 3.2),
                    "pitch": (-3.2, 3.2),
                    "roll": (-3.2, 3.2),
                    "wheel_slip_angle_fl": (-3.2, 3.2),
                    "wheel_slip_angle_fr": (-3.2, 3.2),
                    "wheel_slip_angle_rl": (-3.2, 3.2),
                    "wheel_slip_angle_rr": (-3.2, 3.2),
                    "yaw_rate": (-20.0, 20.0),
                    "front_wheels_angle": (-1.5, 1.5),
                }
                for name, (minimum, maximum) in broad_ranges.items():
                    value = latest_row.get(name)
                    if pd.isna(value):
                        continue
                    numeric = float(value)
                    if not minimum <= numeric <= maximum:
                        implausible.append(name)

            return RecorderSnapshot(
                session_uid=self.session_uid,
                active_lap_num=self.active_lap_num,
                recording_complete_lap=self.recording_complete_lap,
                buffered_rows=len(self.rows),
                dropped_incomplete_frames=self.dropped_incomplete_frames,
                latest_damage_available=self.latest_damage is not None,
                missing_enrichment_fields=tuple(missing_enrichment),
                implausible_fields=tuple(implausible),
            )

    def start_session(self, header: PacketHeader) -> None:
        self.session_uid = header.session_uid
        self.active_lap_num = None
        self.recording_complete_lap = False
        self.rows.clear()
        self.pending_frames.clear()
        self.latest_damage = None
        self.previous_frame_identifier = None
        self.previous_lap_time_ms = None
        self.previous_lap_distance_m = None
        self.dropped_incomplete_frames = 0
        self.last_emitted_row = None
        self.comparison.start_lap(None, capture=False)

        metadata = {
            "session_uid": header.session_uid,
            "track": self.track,
            "game_version": (
                f"{header.game_major_version}.{header.game_minor_version}"
            ),
            "udp_format": header.packet_format,
            "udp_port": None,
            "started_at": datetime.now().astimezone().isoformat(),
            "notes": "F1 Race Engineer modular UI, comparison, recording, and data-health session",
        }
        path = self.session_dir / f"session_{header.session_uid}_{self.track}.json"
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(
            "\nNEW SESSION\n"
            f"Session UID: {header.session_uid}\n"
            f"Track label: {self.track}\n"
            f"Session metadata: {path}\n"
        )

    def ensure_session(self, header: PacketHeader) -> None:
        if header.session_uid != self.session_uid:
            self.start_session(header)

    def put_frame_part(self, header: PacketHeader, name: str, value: Any) -> None:
        with self._lock:
            self.ensure_session(header)
            frame = self.pending_frames.setdefault(header.frame_identifier, {})
            frame["header"] = header
            frame[name] = value
            self.try_emit_frame(header.frame_identifier)
            self.drop_stale_frames(header.frame_identifier)

    def handle_lap_packet(self, header: PacketHeader, lap: PlayerLapData) -> None:
        with self._lock:
            self.ensure_session(header)

            if lap.current_lap_num == 0:
                return

            if self.active_lap_num is None:
                self.active_lap_num = lap.current_lap_num
                self.recording_complete_lap = self._is_clean_lap_start(lap)
                self.comparison.start_lap(
                    self.active_lap_num,
                    capture=self.recording_complete_lap,
                )
                state = "recording" if self.recording_complete_lap else "waiting"
                print(
                    f"Observed lap {self.active_lap_num}; "
                    f"{state} for a complete lap."
                )

            flashback = self._is_flashback(header, lap)
            if flashback:
                # Selecting a qualifying flying lap can place the car before the
                # timing line and then reset lap time/distance at the line without
                # incrementing currentLapNum. That looks like a rewind, but it is
                # actually the clean start of the timed lap.
                clean_start_resync = (
                    lap.current_lap_num == self.active_lap_num
                    and self._is_clean_lap_start(lap)
                )

                self.rows.clear()
                self.pending_frames.clear()
                self.active_lap_num = lap.current_lap_num
                self.dropped_incomplete_frames = 0

                if clean_start_resync:
                    self.recording_complete_lap = True
                    self.comparison.start_lap(
                        self.active_lap_num, capture=True
                    )
                    print(
                        "\nLAP START RESYNC DETECTED\n"
                        "Flying-lap/start-line reset accepted; "
                        f"recording lap {self.active_lap_num} from the start.\n"
                    )
                else:
                    self.recording_complete_lap = False
                    self.comparison.start_lap(
                        self.active_lap_num, capture=False
                    )
                    print(
                        "\nFLASHBACK/REWIND DETECTED\n"
                        f"Discarding buffered data for lap {self.active_lap_num}.\n"
                    )

            elif (
                not self.recording_complete_lap
                and lap.current_lap_num == self.active_lap_num
                and self._is_clean_lap_start(lap)
            ):
                # Arm recording when a flying/out lap reaches the timing line,
                # even if the game does not increment the lap number there.
                self.rows.clear()
                self.pending_frames.clear()
                self.recording_complete_lap = True
                self.dropped_incomplete_frames = 0
                self.comparison.start_lap(
                    self.active_lap_num, capture=True
                )
                print(
                    f"Clean start detected; recording complete lap "
                    f"{self.active_lap_num}..."
                )

            elif lap.current_lap_num > self.active_lap_num:
                completed_lap_num = self.active_lap_num

                if self.recording_complete_lap and self.rows:
                    completed_rows = list(self.rows)
                    self.writer.submit(
                        LapWriteJob(
                            rows=completed_rows,
                            lap_num=completed_lap_num,
                            official_lap_time_ms=lap.last_lap_time_ms,
                            dropped_incomplete_frames=(
                                self.dropped_incomplete_frames
                            ),
                        )
                    )

                    completed_lap_is_valid = not any(
                        bool(row["current_lap_invalid"])
                        for row in completed_rows
                    )
                    if completed_lap_is_valid:
                        try:
                            self.comparison.set_reference_from_rows(
                                completed_rows,
                                completed_lap_num,
                                lap.last_lap_time_ms,
                            )
                        except Exception as exc:
                            print(
                                "REFERENCE UPDATE ERROR\n"
                                f"{type(exc).__name__}: {exc}\n"
                            )
                    else:
                        print(
                            f"Lap {completed_lap_num} was invalid; "
                            "keeping the previous comparison reference."
                        )
                else:
                    print(
                        f"Lap {completed_lap_num} was not saved "
                        "(the app did not capture it from the start)."
                    )

                self.rows.clear()
                self.pending_frames.clear()
                self.active_lap_num = lap.current_lap_num
                self.recording_complete_lap = True
                self.dropped_incomplete_frames = 0
                self.comparison.start_lap(
                    self.active_lap_num, capture=True
                )
                print(f"Recording complete lap {self.active_lap_num}...")

            elif lap.current_lap_num < self.active_lap_num:
                print(
                    "\nLAP NUMBER MOVED BACKWARDS\n"
                    "Discarding the current buffer and waiting for a clean boundary.\n"
                )
                self.rows.clear()
                self.pending_frames.clear()
                self.active_lap_num = lap.current_lap_num
                self.recording_complete_lap = False
                self.comparison.start_lap(
                    self.active_lap_num, capture=False
                )

            self.previous_frame_identifier = header.frame_identifier
            self.previous_lap_time_ms = lap.current_lap_time_ms
            self.previous_lap_distance_m = lap.lap_distance_m

            frame = self.pending_frames.setdefault(header.frame_identifier, {})
            frame["header"] = header
            frame["lap"] = lap
            self.try_emit_frame(header.frame_identifier)
            self.drop_stale_frames(header.frame_identifier)

    def handle_damage_packet(
        self,
        header: PacketHeader,
        damage: PlayerDamage,
    ) -> None:
        with self._lock:
            self.ensure_session(header)
            self.latest_damage = damage

    @staticmethod
    def _is_clean_lap_start(lap: PlayerLapData) -> bool:
        # Negative lap distance is common while the game positions the car for
        # a qualifying flying lap. It is not the start of the timed lap yet.
        return bool(
            lap.current_lap_time_ms <= 2_500
            and 0.0 <= lap.lap_distance_m <= 200.0
        )

    def _is_flashback(self, header: PacketHeader, lap: PlayerLapData) -> bool:
        if (
            self.previous_frame_identifier is not None
            and header.frame_identifier < self.previous_frame_identifier
        ):
            return True

        if (
            self.active_lap_num == lap.current_lap_num
            and self.previous_lap_time_ms is not None
            and lap.current_lap_time_ms < self.previous_lap_time_ms - 1_000
        ):
            return True

        return bool(
            self.active_lap_num == lap.current_lap_num
            and self.previous_lap_distance_m is not None
            and lap.lap_distance_m < self.previous_lap_distance_m - 100.0
        )

    def try_emit_frame(self, frame_id: int) -> None:
        frame = self.pending_frames.get(frame_id)
        if not frame or not REQUIRED_FRAME_PARTS.issubset(frame):
            return

        self.pending_frames.pop(frame_id, None)

        if not self.recording_complete_lap:
            return

        header: PacketHeader = frame["header"]
        lap: PlayerLapData = frame["lap"]
        telemetry: PlayerTelemetry = frame["telemetry"]
        motion: PlayerMotion = frame["motion"]
        status: PlayerStatus = frame["status"]
        motion_ex: PlayerMotionEx = frame["motion_ex"]

        if lap.current_lap_num != self.active_lap_num:
            return

        # All F1 wheel arrays are RL, RR, FL, FR.
        brake_rl, brake_rr, brake_fl, brake_fr = telemetry.brake_temperatures
        surface_rl, surface_rr, surface_fl, surface_fr = (
            telemetry.tyre_surface_temperatures
        )
        inner_rl, inner_rr, inner_fl, inner_fr = (
            telemetry.tyre_inner_temperatures
        )
        pressure_rl, pressure_rr, pressure_fl, pressure_fr = (
            telemetry.tyre_pressures
        )
        slip_rl, slip_rr, slip_fl, slip_fr = motion_ex.wheel_slip_ratio
        angle_rl, angle_rr, angle_fl, angle_fr = motion_ex.wheel_slip_angle

        if self.latest_damage is None:
            wear_rl = wear_rr = wear_fl = wear_fr = math.nan
        else:
            wear_rl, wear_rr, wear_fl, wear_fr = self.latest_damage.tyre_wear

        row = {
                "session_uid": header.session_uid,
                "frame_identifier": header.frame_identifier,
                "session_time_s": header.session_time,
                "track": self.track,
                "lap_number": lap.current_lap_num,
                "timestamp_ms": lap.current_lap_time_ms,
                "lap_distance_m": lap.lap_distance_m,
                "lap_distance_pct": math.nan,
                "speed_kmh": float(telemetry.speed_kmh),
                "throttle": telemetry.throttle,
                "brake": telemetry.brake,
                "gear": telemetry.gear,
                "steering": telemetry.steering,
                "rpm": float(telemetry.rpm),
                "drs": telemetry.drs,
                "clutch": telemetry.clutch,
                "ers_deployed": status.ers_deployed_this_lap,
                "fuel_remaining": status.fuel_in_tank,
                "fl_tyre_temp": float(surface_fl),
                "fr_tyre_temp": float(surface_fr),
                "rl_tyre_temp": float(surface_rl),
                "rr_tyre_temp": float(surface_rr),
                "fl_inner_tyre_temp": float(inner_fl),
                "fr_inner_tyre_temp": float(inner_fr),
                "rl_inner_tyre_temp": float(inner_rl),
                "rr_inner_tyre_temp": float(inner_rr),
                "fl_brake_temp": float(brake_fl),
                "fr_brake_temp": float(brake_fr),
                "rl_brake_temp": float(brake_rl),
                "rr_brake_temp": float(brake_rr),
                "fl_tyre_pressure": pressure_fl,
                "fr_tyre_pressure": pressure_fr,
                "rl_tyre_pressure": pressure_rl,
                "rr_tyre_pressure": pressure_rr,
                "engine_temp": float(telemetry.engine_temperature),
                "fl_tyre_wear": wear_fl,
                "fr_tyre_wear": wear_fr,
                "rl_tyre_wear": wear_rl,
                "rr_tyre_wear": wear_rr,
                "g_force_lat": motion.g_force_lateral,
                "g_force_lon": motion.g_force_longitudinal,
                "wheel_slip_fl": slip_fl,
                "wheel_slip_fr": slip_fr,
                "wheel_slip_rl": slip_rl,
                "wheel_slip_rr": slip_rr,
                # --- Track geometry (added after the Spa/Monza/Monaco data) ---
                # Purely additive: laps recorded before this simply lack these
                # columns. Phase 3 validation does not require them, so no
                # historical lap changes eligibility.
                "world_position_x": motion.world_position_x,
                "world_position_y": motion.world_position_y,
                "world_position_z": motion.world_position_z,
                "world_velocity_x": motion.world_velocity_x,
                "world_velocity_y": motion.world_velocity_y,
                "world_velocity_z": motion.world_velocity_z,
                "g_force_vert": motion.g_force_vertical,
                "yaw": motion.yaw,
                "pitch": motion.pitch,
                "roll": motion.roll,
                # --- Lateral grip state (understeer/oversteer evidence) ---
                "wheel_slip_angle_fl": angle_fl,
                "wheel_slip_angle_fr": angle_fr,
                "wheel_slip_angle_rl": angle_rl,
                "wheel_slip_angle_rr": angle_rr,
                "yaw_rate": motion_ex.yaw_rate,
                "front_wheels_angle": motion_ex.front_wheels_angle,
                "local_velocity_x": motion_ex.local_velocity_x,
                "local_velocity_y": motion_ex.local_velocity_y,
                "local_velocity_z": motion_ex.local_velocity_z,
                "car_position": lap.car_position,
                "sector": lap.sector + 1,
                "pit_status": lap.pit_status,
                "current_lap_invalid": lap.current_lap_invalid,
                "lap_time_ms": 0,
            }
        self.rows.append(row)
        self.last_emitted_row = row
        self.comparison.append_row(row)

    def drop_stale_frames(self, newest_frame_id: int) -> None:
        cutoff = newest_frame_id - 15
        stale_ids = [
            frame_id for frame_id in self.pending_frames if frame_id < cutoff
        ]
        for frame_id in stale_ids:
            frame = self.pending_frames.pop(frame_id, None)
            if frame and self.recording_complete_lap:
                self.dropped_incomplete_frames += 1
