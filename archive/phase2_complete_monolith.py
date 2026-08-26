from __future__ import annotations

import argparse
import json
import math
import queue
import socket
import struct
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation


# -----------------------------------------------------------------------------
# F1 25 UDP protocol constants and packed structures
# -----------------------------------------------------------------------------

F1_25_PACKET_FORMAT = 2025

MOTION_PACKET_ID = 0
LAP_DATA_PACKET_ID = 2
CAR_TELEMETRY_PACKET_ID = 6
CAR_STATUS_PACKET_ID = 7
CAR_DAMAGE_PACKET_ID = 10
MOTION_EX_PACKET_ID = 13

HEADER_STRUCT = struct.Struct("<HBBBBBQfIIBB")  # 29 bytes
LAP_DATA_RECORD_SIZE = 57
LAP_DATA_PREFIX_STRUCT = struct.Struct("<IIHBHBHBHBfffBBBBBB")
CAR_TELEMETRY_STRUCT = struct.Struct("<HfffBbHBBH4H4B4BH4f4B")  # 60 bytes
CAR_MOTION_STRUCT = struct.Struct("<6f6h6f")  # 60 bytes
CAR_STATUS_STRUCT = struct.Struct("<5B3f2H2BH3Bb3fB3fB")  # 55 bytes
CAR_DAMAGE_STRUCT = struct.Struct("<4f30B")  # 46 bytes
MOTION_EX_STRUCT = struct.Struct("<61f")  # Player car only, 244 bytes

REQUIRED_FRAME_PARTS = {
    "lap",
    "telemetry",
    "motion",
    "status",
    "motion_ex",
}

PACKET_NAMES = {
    MOTION_PACKET_ID: "Motion",
    LAP_DATA_PACKET_ID: "Lap Data",
    CAR_TELEMETRY_PACKET_ID: "Car Telemetry",
    CAR_STATUS_PACKET_ID: "Car Status",
    CAR_DAMAGE_PACKET_ID: "Car Damage",
    MOTION_EX_PACKET_ID: "Motion Ex",
}

HIGH_RATE_PACKET_IDS = (
    MOTION_PACKET_ID,
    LAP_DATA_PACKET_ID,
    CAR_TELEMETRY_PACKET_ID,
    CAR_STATUS_PACKET_ID,
    MOTION_EX_PACKET_ID,
)


# -----------------------------------------------------------------------------
# Parsed packet models
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PacketHeader:
    packet_format: int
    game_major_version: int
    game_minor_version: int
    packet_id: int
    session_uid: int
    session_time: float
    frame_identifier: int
    overall_frame_identifier: int
    player_car_index: int


@dataclass(frozen=True, slots=True)
class PlayerLapData:
    last_lap_time_ms: int
    current_lap_time_ms: int
    lap_distance_m: float
    total_distance_m: float
    car_position: int
    current_lap_num: int
    pit_status: int
    sector: int
    current_lap_invalid: bool


@dataclass(frozen=True, slots=True)
class PlayerTelemetry:
    speed_kmh: int
    throttle: float
    steering: float
    brake: float
    clutch: int
    gear: int
    rpm: int
    drs: int
    brake_temperatures: tuple[int, int, int, int]
    tyre_surface_temperatures: tuple[int, int, int, int]
    tyre_inner_temperatures: tuple[int, int, int, int]
    engine_temperature: int
    tyre_pressures: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PlayerMotion:
    g_force_lateral: float
    g_force_longitudinal: float


@dataclass(frozen=True, slots=True)
class PlayerStatus:
    fuel_in_tank: float
    ers_deployed_this_lap: float


@dataclass(frozen=True, slots=True)
class PlayerDamage:
    tyre_wear: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PlayerMotionEx:
    wheel_slip_ratio: tuple[float, float, float, float]


# -----------------------------------------------------------------------------
# Packet parsing
# -----------------------------------------------------------------------------

def parse_header(packet: bytes) -> PacketHeader:
    if len(packet) < HEADER_STRUCT.size:
        raise ValueError("Packet is smaller than the F1 25 header.")

    values = HEADER_STRUCT.unpack_from(packet)
    return PacketHeader(
        packet_format=values[0],
        game_major_version=values[2],
        game_minor_version=values[3],
        packet_id=values[5],
        session_uid=values[6],
        session_time=values[7],
        frame_identifier=values[8],
        overall_frame_identifier=values[9],
        player_car_index=values[10],
    )


def _player_record_offset(header: PacketHeader, record_size: int) -> int:
    if not 0 <= header.player_car_index < 22:
        raise ValueError(f"Invalid player car index: {header.player_car_index}")
    return HEADER_STRUCT.size + header.player_car_index * record_size


def parse_player_lap_data(packet: bytes, header: PacketHeader) -> PlayerLapData:
    offset = _player_record_offset(header, LAP_DATA_RECORD_SIZE)
    if len(packet) < offset + LAP_DATA_RECORD_SIZE:
        raise ValueError("Lap Data packet is too small.")

    values = LAP_DATA_PREFIX_STRUCT.unpack_from(packet, offset)
    return PlayerLapData(
        last_lap_time_ms=values[0],
        current_lap_time_ms=values[1],
        lap_distance_m=values[10],
        total_distance_m=values[11],
        car_position=values[13],
        current_lap_num=values[14],
        pit_status=values[15],
        sector=values[17],
        current_lap_invalid=bool(values[18]),
    )


def parse_player_telemetry(packet: bytes, header: PacketHeader) -> PlayerTelemetry:
    offset = _player_record_offset(header, CAR_TELEMETRY_STRUCT.size)
    if len(packet) < offset + CAR_TELEMETRY_STRUCT.size:
        raise ValueError("Car Telemetry packet is too small.")

    values = CAR_TELEMETRY_STRUCT.unpack_from(packet, offset)
    return PlayerTelemetry(
        speed_kmh=values[0],
        throttle=max(0.0, min(1.0, values[1])),
        steering=max(-1.0, min(1.0, values[2])),
        brake=max(0.0, min(1.0, values[3])),
        clutch=values[4],
        gear=values[5],
        rpm=values[6],
        drs=values[7],
        brake_temperatures=tuple(values[10:14]),
        tyre_surface_temperatures=tuple(values[14:18]),
        tyre_inner_temperatures=tuple(values[18:22]),
        engine_temperature=values[22],
        tyre_pressures=tuple(values[23:27]),
    )


def parse_player_motion(packet: bytes, header: PacketHeader) -> PlayerMotion:
    offset = _player_record_offset(header, CAR_MOTION_STRUCT.size)
    if len(packet) < offset + CAR_MOTION_STRUCT.size:
        raise ValueError("Motion packet is too small.")

    values = CAR_MOTION_STRUCT.unpack_from(packet, offset)
    return PlayerMotion(
        g_force_lateral=values[12],
        g_force_longitudinal=values[13],
    )


def parse_player_status(packet: bytes, header: PacketHeader) -> PlayerStatus:
    offset = _player_record_offset(header, CAR_STATUS_STRUCT.size)
    if len(packet) < offset + CAR_STATUS_STRUCT.size:
        raise ValueError("Car Status packet is too small.")

    values = CAR_STATUS_STRUCT.unpack_from(packet, offset)
    return PlayerStatus(
        fuel_in_tank=values[5],
        ers_deployed_this_lap=values[23],
    )


def parse_player_damage(packet: bytes, header: PacketHeader) -> PlayerDamage:
    offset = _player_record_offset(header, CAR_DAMAGE_STRUCT.size)
    if len(packet) < offset + CAR_DAMAGE_STRUCT.size:
        raise ValueError("Car Damage packet is too small.")

    values = CAR_DAMAGE_STRUCT.unpack_from(packet, offset)
    tyre_wear_fraction = tuple(float(value) / 100.0 for value in values[:4])
    return PlayerDamage(tyre_wear=tyre_wear_fraction)


def parse_player_motion_ex(packet: bytes) -> PlayerMotionEx:
    offset = HEADER_STRUCT.size
    if len(packet) < offset + MOTION_EX_STRUCT.size:
        raise ValueError("Motion Ex packet is too small.")

    values = MOTION_EX_STRUCT.unpack_from(packet, offset)
    return PlayerMotionEx(wheel_slip_ratio=tuple(values[16:20]))


# -----------------------------------------------------------------------------
# Dashboard sample buffer
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class DashboardSample:
    timestamp: float
    speed_kmh: int
    throttle: float
    brake: float
    steering: float
    gear: int
    rpm: int
    drs: int
    tyre_rl: int
    tyre_rr: int
    tyre_fl: int
    tyre_fr: int


class TelemetryBuffer:
    def __init__(self, maxlen: int) -> None:
        self._samples: deque[DashboardSample] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, sample: DashboardSample) -> None:
        with self._lock:
            self._samples.append(sample)

    def snapshot(self) -> list[DashboardSample]:
        with self._lock:
            return list(self._samples)



# -----------------------------------------------------------------------------
# Distance-aligned current-lap versus reference-lap comparison
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ReferenceLap:
    source: str
    lap_num: int | None
    official_lap_time_ms: int
    track_length_m: float
    distance_pct: np.ndarray
    elapsed_ms: np.ndarray
    speed_kmh: np.ndarray
    throttle: np.ndarray
    brake: np.ndarray


@dataclass(frozen=True, slots=True)
class ComparisonSnapshot:
    has_reference: bool
    reference_source: str | None
    reference_lap_num: int | None
    reference_lap_time_ms: int | None
    current_lap_num: int | None
    current_progress_pct: float | None
    latest_delta_s: float | None
    current_distance_pct: np.ndarray
    current_speed_kmh: np.ndarray
    current_throttle: np.ndarray
    current_brake: np.ndarray
    current_delta_s: np.ndarray
    reference_distance_pct: np.ndarray
    reference_speed_kmh: np.ndarray
    reference_throttle: np.ndarray
    reference_brake: np.ndarray


class LapComparisonManager:
    """Owns a valid same-track reference lap and the current live comparison."""

    def __init__(self, raw_lap_dir: Path) -> None:
        self.raw_lap_dir = raw_lap_dir
        self._lock = threading.RLock()
        self._reference: ReferenceLap | None = None
        self._current_lap_num: int | None = None
        self._capture_current = False
        self._current_start_time_ms: int | None = None
        self._distance_pct: list[float] = []
        self._speed_kmh: list[float] = []
        self._throttle: list[float] = []
        self._brake: list[float] = []
        self._delta_s: list[float] = []

    @staticmethod
    def _empty_array() -> np.ndarray:
        return np.empty(0, dtype=float)

    @staticmethod
    def _format_lap_time(milliseconds: int) -> str:
        minutes, remainder = divmod(milliseconds, 60_000)
        return f"{minutes}:{remainder / 1000:06.3f}"

    def load_reference(self, explicit_path: Path | None = None) -> None:
        if explicit_path is not None:
            candidates = [explicit_path]
        else:
            candidates = sorted(
                self.raw_lap_dir.glob("*_valid.parquet"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )

        if not candidates:
            print(
                "No valid same-track reference lap found. "
                "The first clean lap will become the reference."
            )
            return

        errors: list[str] = []
        for path in candidates:
            try:
                dataframe = pd.read_parquet(path, engine="pyarrow")
                reference = self._build_reference(
                    dataframe,
                    source=str(path),
                )
            except Exception as exc:
                errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
                continue

            with self._lock:
                self._reference = reference

            print(
                "REFERENCE LAP LOADED\n"
                f"Source: {path}\n"
                f"Lap: {reference.lap_num}\n"
                f"Time: {self._format_lap_time(reference.official_lap_time_ms)}\n"
                f"Track length: {reference.track_length_m:.1f} m\n"
            )
            return

        print("Reference lap loading failed:")
        for error in errors[:5]:
            print(f"  {error}")

    def set_reference_from_rows(
        self,
        rows: list[dict[str, Any]],
        lap_num: int,
        official_lap_time_ms: int,
    ) -> None:
        if not rows:
            return

        dataframe = pd.DataFrame(rows)
        dataframe["lap_time_ms"] = int(official_lap_time_ms)
        reference = self._build_reference(
            dataframe,
            source=f"completed lap {lap_num} (live memory)",
        )

        with self._lock:
            self._reference = reference

        print(
            "REFERENCE LAP UPDATED\n"
            f"Lap {lap_num} is now the comparison lap "
            f"({self._format_lap_time(official_lap_time_ms)}).\n"
        )

    def _build_reference(
        self,
        dataframe: pd.DataFrame,
        source: str,
    ) -> ReferenceLap:
        required = {"timestamp_ms", "speed_kmh", "throttle", "brake"}
        missing = sorted(required.difference(dataframe.columns))
        if missing:
            raise ValueError(f"Missing reference columns: {', '.join(missing)}")

        work = dataframe.copy()
        if "lap_distance_m" in work.columns:
            lap_distance_m = pd.to_numeric(
                work["lap_distance_m"], errors="coerce"
            )
            track_length_m = float(lap_distance_m.max())
        else:
            lap_distance_m = pd.Series(np.nan, index=work.index)
            track_length_m = math.nan

        if (
            "lap_distance_pct" in work.columns
            and pd.to_numeric(work["lap_distance_pct"], errors="coerce")
            .notna()
            .sum()
            >= 100
        ):
            distance_pct = pd.to_numeric(
                work["lap_distance_pct"], errors="coerce"
            )
        elif math.isfinite(track_length_m) and track_length_m > 100.0:
            distance_pct = lap_distance_m / track_length_m
        else:
            raise ValueError("Reference has no usable lap-distance information.")

        clean = pd.DataFrame(
            {
                "distance_pct": distance_pct,
                "timestamp_ms": pd.to_numeric(
                    work["timestamp_ms"], errors="coerce"
                ),
                "speed_kmh": pd.to_numeric(
                    work["speed_kmh"], errors="coerce"
                ),
                "throttle": pd.to_numeric(
                    work["throttle"], errors="coerce"
                ),
                "brake": pd.to_numeric(work["brake"], errors="coerce"),
            }
        ).dropna()

        clean = clean[
            (clean["distance_pct"] >= 0.0)
            & (clean["distance_pct"] <= 1.02)
        ]
        clean = clean.sort_values("distance_pct").drop_duplicates(
            "distance_pct", keep="last"
        )

        if len(clean) < 100:
            raise ValueError("Reference contains fewer than 100 usable samples.")
        if float(clean["distance_pct"].iloc[-1]) < 0.95:
            raise ValueError("Reference does not cover a complete lap.")

        first_timestamp = float(clean["timestamp_ms"].iloc[0])
        elapsed_ms = (
            clean["timestamp_ms"].to_numpy(dtype=float) - first_timestamp
        )

        if "lap_time_ms" in work.columns:
            official_series = pd.to_numeric(
                work["lap_time_ms"], errors="coerce"
            ).dropna()
            official_lap_time_ms = (
                int(official_series.max())
                if not official_series.empty
                else int(elapsed_ms[-1])
            )
        else:
            official_lap_time_ms = int(elapsed_ms[-1])

        lap_num: int | None = None
        if "lap_number" in work.columns:
            lap_series = pd.to_numeric(
                work["lap_number"], errors="coerce"
            ).dropna()
            if not lap_series.empty:
                lap_num = int(lap_series.iloc[0])

        if not math.isfinite(track_length_m) or track_length_m <= 100.0:
            track_length_m = 1.0

        return ReferenceLap(
            source=source,
            lap_num=lap_num,
            official_lap_time_ms=official_lap_time_ms,
            track_length_m=track_length_m,
            distance_pct=clean["distance_pct"].to_numpy(dtype=float),
            elapsed_ms=elapsed_ms,
            speed_kmh=clean["speed_kmh"].to_numpy(dtype=float),
            throttle=clean["throttle"].to_numpy(dtype=float),
            brake=clean["brake"].to_numpy(dtype=float),
        )

    def start_lap(self, lap_num: int | None, capture: bool) -> None:
        with self._lock:
            self._current_lap_num = lap_num
            self._capture_current = capture
            self._current_start_time_ms = None
            self._distance_pct.clear()
            self._speed_kmh.clear()
            self._throttle.clear()
            self._brake.clear()
            self._delta_s.clear()

    def append_row(self, row: dict[str, Any]) -> None:
        with self._lock:
            reference = self._reference
            if not self._capture_current or reference is None:
                return

            distance_m = float(row["lap_distance_m"])
            if not math.isfinite(distance_m):
                return

            distance_pct = max(
                0.0,
                min(1.0, distance_m / reference.track_length_m),
            )

            if (
                self._distance_pct
                and distance_pct < self._distance_pct[-1] - 0.002
            ):
                return

            timestamp_ms = int(row["timestamp_ms"])
            if self._current_start_time_ms is None:
                self._current_start_time_ms = timestamp_ms
            current_elapsed_ms = timestamp_ms - self._current_start_time_ms

            reference_elapsed_ms = float(
                np.interp(
                    distance_pct,
                    reference.distance_pct,
                    reference.elapsed_ms,
                )
            )
            delta_s = (current_elapsed_ms - reference_elapsed_ms) / 1000.0

            self._distance_pct.append(distance_pct)
            self._speed_kmh.append(float(row["speed_kmh"]))
            self._throttle.append(float(row["throttle"]))
            self._brake.append(float(row["brake"]))
            self._delta_s.append(delta_s)

    @staticmethod
    def _downsample(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
        return values if not len(indices) else values[indices]

    def snapshot(self, max_current_points: int = 1_200) -> ComparisonSnapshot:
        with self._lock:
            reference = self._reference
            current_distance = np.asarray(self._distance_pct, dtype=float)
            current_speed = np.asarray(self._speed_kmh, dtype=float)
            current_throttle = np.asarray(self._throttle, dtype=float)
            current_brake = np.asarray(self._brake, dtype=float)
            current_delta = np.asarray(self._delta_s, dtype=float)

            if len(current_distance) > max_current_points:
                indices = np.linspace(
                    0,
                    len(current_distance) - 1,
                    max_current_points,
                    dtype=int,
                )
                current_distance = current_distance[indices]
                current_speed = current_speed[indices]
                current_throttle = current_throttle[indices]
                current_brake = current_brake[indices]
                current_delta = current_delta[indices]

            if reference is None:
                empty = self._empty_array()
                return ComparisonSnapshot(
                    has_reference=False,
                    reference_source=None,
                    reference_lap_num=None,
                    reference_lap_time_ms=None,
                    current_lap_num=self._current_lap_num,
                    current_progress_pct=(
                        float(current_distance[-1] * 100.0)
                        if len(current_distance)
                        else None
                    ),
                    latest_delta_s=(
                        float(current_delta[-1]) if len(current_delta) else None
                    ),
                    current_distance_pct=current_distance,
                    current_speed_kmh=current_speed,
                    current_throttle=current_throttle,
                    current_brake=current_brake,
                    current_delta_s=current_delta,
                    reference_distance_pct=empty,
                    reference_speed_kmh=empty,
                    reference_throttle=empty,
                    reference_brake=empty,
                )

            return ComparisonSnapshot(
                has_reference=True,
                reference_source=reference.source,
                reference_lap_num=reference.lap_num,
                reference_lap_time_ms=reference.official_lap_time_ms,
                current_lap_num=self._current_lap_num,
                current_progress_pct=(
                    float(current_distance[-1] * 100.0)
                    if len(current_distance)
                    else None
                ),
                latest_delta_s=(
                    float(current_delta[-1]) if len(current_delta) else None
                ),
                current_distance_pct=current_distance,
                current_speed_kmh=current_speed,
                current_throttle=current_throttle,
                current_brake=current_brake,
                current_delta_s=current_delta,
                reference_distance_pct=reference.distance_pct,
                reference_speed_kmh=reference.speed_kmh,
                reference_throttle=reference.throttle,
                reference_brake=reference.brake,
            )


# -----------------------------------------------------------------------------
# Background Parquet writer
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class LapWriteJob:
    rows: list[dict[str, Any]]
    lap_num: int
    official_lap_time_ms: int
    dropped_incomplete_frames: int


@dataclass(frozen=True, slots=True)
class WriterSnapshot:
    queued_jobs: int
    saved_laps: int
    last_saved_file: str | None
    last_error: str | None


class LapWriter(threading.Thread):
    def __init__(self, raw_lap_dir: Path) -> None:
        super().__init__(name="lap-writer", daemon=True)
        self.raw_lap_dir = raw_lap_dir
        self._jobs: queue.Queue[LapWriteJob | None] = queue.Queue()
        self._lock = threading.Lock()
        self._saved_laps = 0
        self._last_saved_file: str | None = None
        self._last_error: str | None = None

    def submit(self, job: LapWriteJob) -> None:
        self._jobs.put(job)

    def snapshot(self) -> WriterSnapshot:
        with self._lock:
            return WriterSnapshot(
                queued_jobs=self._jobs.qsize(),
                saved_laps=self._saved_laps,
                last_saved_file=self._last_saved_file,
                last_error=self._last_error,
            )

    def finish(self) -> None:
        self._jobs.join()
        self._jobs.put(None)
        self.join(timeout=10.0)

    def run(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                self._write_lap(job)
            except Exception as exc:  # keep the receiver alive if disk writing fails
                error = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._last_error = error
                print(f"\nLAP WRITE ERROR\n{error}\n")
            finally:
                self._jobs.task_done()

    def _write_lap(self, job: LapWriteJob) -> None:
        dataframe = pd.DataFrame(job.rows)
        if dataframe.empty:
            return

        max_distance = dataframe["lap_distance_m"].clip(lower=0).max()
        if pd.notna(max_distance) and max_distance > 0:
            dataframe["lap_distance_pct"] = (
                dataframe["lap_distance_m"].clip(lower=0) / max_distance
            ).clip(0.0, 1.0)

        dataframe["lap_time_ms"] = int(job.official_lap_time_ms)

        invalid = bool(dataframe["current_lap_invalid"].any())
        validity_label = "invalid" if invalid else "valid"
        # Timestamp-first naming keeps a directory sorted chronologically.
        # Milliseconds make accidental collisions practically impossible.
        timestamp = (
            datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        )
        filename = f"lap_{timestamp}_{validity_label}.parquet"
        path = self.raw_lap_dir / filename

        dataframe.to_parquet(
            path,
            engine="pyarrow",
            index=False,
            compression="zstd",
        )

        verified = pd.read_parquet(path, engine="pyarrow")
        if len(verified) != len(dataframe):
            raise RuntimeError(
                f"Verification failed for {path}: wrote {len(dataframe)} rows, "
                f"read {len(verified)}."
            )

        duration_s = max(
            float(dataframe["timestamp_ms"].iloc[-1]) / 1000.0,
            0.001,
        )
        estimated_hz = len(dataframe) / duration_s
        minutes, remainder = divmod(job.official_lap_time_ms, 60_000)
        seconds = remainder / 1_000

        enriched_columns = [
            "fuel_remaining",
            "ers_deployed",
            "g_force_lat",
            "g_force_lon",
            "wheel_slip_fl",
            "wheel_slip_fr",
            "wheel_slip_rl",
            "wheel_slip_rr",
        ]
        enrichment_complete = int(
            dataframe[enriched_columns].notna().all(axis=1).sum()
        )
        wear_columns = [
            "fl_tyre_wear",
            "fr_tyre_wear",
            "rl_tyre_wear",
            "rr_tyre_wear",
        ]
        wear_complete = int(dataframe[wear_columns].notna().all(axis=1).sum())

        with self._lock:
            self._saved_laps += 1
            self._last_saved_file = str(path)
            self._last_error = None

        print(
            "\nLAP SAVED\n"
            f"Lap: {job.lap_num}\n"
            f"Time: {minutes}:{seconds:06.3f}\n"
            f"Validity: {validity_label}\n"
            f"Rows: {len(dataframe)}\n"
            f"Estimated sample rate: {estimated_hz:.1f} Hz\n"
            f"Status/motion/slip enriched rows: "
            f"{enrichment_complete}/{len(dataframe)}\n"
            f"Tyre-wear populated rows: {wear_complete}/{len(dataframe)}\n"
            f"Incomplete frame groups dropped: "
            f"{job.dropped_incomplete_frames}\n"
            f"File: {path}\n"
        )


# -----------------------------------------------------------------------------
# Lap recorder; all packet mutations occur on the UDP receiver thread
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RecorderSnapshot:
    session_uid: int | None
    active_lap_num: int | None
    recording_complete_lap: bool
    buffered_rows: int
    dropped_incomplete_frames: int
    latest_damage_available: bool
    missing_enrichment_fields: tuple[str, ...]
    implausible_fields: tuple[str, ...]


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
            "notes": "F1 25 Phase 2D dashboard, comparison, recording, and data-health session",
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


# -----------------------------------------------------------------------------
# Single UDP receiver: feeds both dashboard and recorder
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ReceiverSnapshot:
    packet_count: int
    telemetry_packet_count: int
    malformed_packet_count: int
    last_packet_age_s: float | None
    packet_rate_hz: float
    telemetry_rate_hz: float
    stream_age_s: float
    packet_age_by_id: dict[int, float]
    latest_telemetry: DashboardSample | None
    error: str | None


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


# -----------------------------------------------------------------------------
# Dashboard health evaluation
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------

def format_gear(gear: int) -> str:
    if gear == -1:
        return "R"
    if gear == 0:
        return "N"
    return str(gear)


def format_lap_time(milliseconds: int | None) -> str:
    if milliseconds is None:
        return "--:--.---"
    minutes, remainder = divmod(milliseconds, 60_000)
    return f"{minutes}:{remainder / 1000:06.3f}"


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
            "\nPhase 2C app stopped. "
            f"Completed lap files saved this run: {writer_state.saved_laps}"
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "F1 25 Phase 2D app: live distance-aligned lap comparison, "
            "telemetry dashboard, enriched Parquet recorder, and data-health panel."
        )
    )
    parser.add_argument(
        "--track",
        required=True,
        help="Track folder label, for example: spa or monza",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20777)
    parser.add_argument(
        "--window",
        type=float,
        default=20.0,
        help="Number of recent seconds shown in the tyre panel.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Output root directory (default: data)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help=(
            "Optional explicit valid Parquet lap to compare against. "
            "By default the newest valid lap for the selected track is used."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window <= 0:
        raise SystemExit("--window must be greater than zero")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.reference is not None and not args.reference.is_file():
        raise SystemExit(f"Reference file does not exist: {args.reference}")

    track = args.track.strip().lower().replace(" ", "_")
    if not track:
        raise SystemExit("--track cannot be empty")

    build_app(
        host=args.host,
        port=args.port,
        track=track,
        output_root=args.output,
        window_seconds=args.window,
        reference_path=args.reference,
    )


if __name__ == "__main__":
    main()
