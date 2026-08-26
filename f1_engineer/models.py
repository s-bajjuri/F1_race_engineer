from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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
    # World position/orientation are the basis for track-geometry context
    # (curvature, elevation, heading change) that transfers across circuits.
    # The Motion packet already carries them; earlier versions parsed the
    # struct but kept only the two G-force fields.
    world_position_x: float
    world_position_y: float
    world_position_z: float
    world_velocity_x: float
    world_velocity_y: float
    world_velocity_z: float
    g_force_lateral: float
    g_force_longitudinal: float
    g_force_vertical: float
    yaw: float
    pitch: float
    roll: float

@dataclass(frozen=True, slots=True)
class PlayerStatus:
    fuel_in_tank: float
    ers_deployed_this_lap: float

@dataclass(frozen=True, slots=True)
class PlayerDamage:
    tyre_wear: tuple[float, float, float, float]

@dataclass(frozen=True, slots=True)
class PlayerMotionEx:
    # Slip ratio is longitudinal (wheelspin/lockup). Slip angle is lateral and
    # is what separates understeer from oversteer; yaw rate and front wheel
    # angle give the reference the slip angles are judged against.
    wheel_slip_ratio: tuple[float, float, float, float]
    wheel_slip_angle: tuple[float, float, float, float]
    local_velocity_x: float
    local_velocity_y: float
    local_velocity_z: float
    yaw_rate: float
    front_wheels_angle: float

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
