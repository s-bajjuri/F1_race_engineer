from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from f1_engineer.analysis.segmentation import BrakingSegment

TYRE_TEMP_COLUMNS: tuple[str, ...] = (
    "fl_tyre_temp",
    "fr_tyre_temp",
    "rl_tyre_temp",
    "rr_tyre_temp",
)
TYRE_WEAR_COLUMNS: tuple[str, ...] = (
    "fl_tyre_wear",
    "fr_tyre_wear",
    "rl_tyre_wear",
    "rr_tyre_wear",
)
FRONT_SLIP_COLUMNS: tuple[str, ...] = ("wheel_slip_fl", "wheel_slip_fr")
REAR_SLIP_COLUMNS: tuple[str, ...] = ("wheel_slip_rl", "wheel_slip_rr")
ALL_SLIP_COLUMNS: tuple[str, ...] = FRONT_SLIP_COLUMNS + REAR_SLIP_COLUMNS


def _numeric(
    dataframe: pd.DataFrame,
    column: str,
    *,
    default: float = np.nan,
) -> np.ndarray:
    if column not in dataframe.columns:
        return np.full(len(dataframe), default, dtype=float)
    return pd.to_numeric(dataframe[column], errors="coerce").to_numpy(dtype=float)


def _matrix(dataframe: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(dataframe), 0), dtype=float)
    return np.column_stack([_numeric(dataframe, column) for column in columns])


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _safe_mean(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.mean(finite)) if finite.size else float("nan")


def _safe_std(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.std(finite)) if finite.size else float("nan")


def _safe_min(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.min(finite)) if finite.size else float("nan")


def _safe_max(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.max(finite)) if finite.size else float("nan")


def _safe_max_abs(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.max(np.abs(finite))) if finite.size else float("nan")


def _safe_first(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(finite[0]) if finite.size else float("nan")


def _safe_last(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(finite[-1]) if finite.size else float("nan")


def _positive_difference(values: np.ndarray) -> float:
    first = _safe_first(values)
    last = _safe_last(values)
    if not np.isfinite(first) or not np.isfinite(last):
        return float("nan")
    return float(max(0.0, last - first))


def _decrease(values: np.ndarray) -> float:
    first = _safe_first(values)
    last = _safe_last(values)
    if not np.isfinite(first) or not np.isfinite(last):
        return float("nan")
    return float(max(0.0, first - last))


def _elapsed_seconds(dataframe: pd.DataFrame) -> np.ndarray:
    return _numeric(dataframe, "timestamp_ms") / 1_000.0


def _valid_slopes(values: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    if len(values) < 2 or len(times_s) < 2:
        return np.empty(0, dtype=float)
    dv = np.diff(values)
    dt = np.diff(times_s)
    valid = np.isfinite(dv) & np.isfinite(dt) & (dt > 1e-6)
    return dv[valid] / dt[valid]


def _mask_duration_s(mask: np.ndarray, times_s: np.ndarray) -> float:
    if len(mask) < 2 or len(times_s) < 2:
        return 0.0
    dt = np.diff(times_s)
    valid = np.isfinite(dt) & (dt > 0)
    active = np.asarray(mask[:-1], dtype=bool) & valid
    return float(np.sum(dt[active]))


def _lap_duration_s(dataframe: pd.DataFrame) -> float:
    times_s = _elapsed_seconds(dataframe)
    finite = _finite(times_s)
    if finite.size < 2:
        return 0.0
    return float(max(0.0, finite[-1] - finite[0]))


def _matrix_row_max_abs(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.full(matrix.shape[0], np.nan, dtype=float)
    result = np.full(matrix.shape[0], np.nan, dtype=float)
    for row_index, row in enumerate(matrix):
        finite = _finite(row)
        if finite.size:
            result[row_index] = float(np.max(np.abs(finite)))
    return result


def _temperature_metrics(dataframe: pd.DataFrame) -> dict[str, float]:
    matrix = _matrix(dataframe, TYRE_TEMP_COLUMNS)
    finite = _finite(matrix)
    if not finite.size:
        return {
            "mean_tyre_temp_c": float("nan"),
            "max_tyre_temp_c": float("nan"),
            "mean_tyre_temp_spread_c": float("nan"),
        }

    row_spreads: list[float] = []
    for row in matrix:
        row_finite = _finite(row)
        if row_finite.size:
            row_spreads.append(float(np.max(row_finite) - np.min(row_finite)))
    return {
        "mean_tyre_temp_c": float(np.mean(finite)),
        "max_tyre_temp_c": float(np.max(finite)),
        "mean_tyre_temp_spread_c": (
            float(np.mean(row_spreads)) if row_spreads else float("nan")
        ),
    }


def _wear_delta(dataframe: pd.DataFrame) -> float:
    matrix = _matrix(dataframe, TYRE_WEAR_COLUMNS)
    if matrix.size == 0:
        return float("nan")
    deltas: list[float] = []
    for wheel_index in range(matrix.shape[1]):
        delta = _positive_difference(matrix[:, wheel_index])
        if np.isfinite(delta):
            deltas.append(delta)
    return float(np.mean(deltas)) if deltas else float("nan")


def _slip_metrics(
    dataframe: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, float]:
    front = _matrix(dataframe, FRONT_SLIP_COLUMNS)
    rear = _matrix(dataframe, REAR_SLIP_COLUMNS)
    all_slip = _matrix(dataframe, ALL_SLIP_COLUMNS)
    row_peak = _matrix_row_max_abs(all_slip)
    finite_peak = _finite(row_peak)

    front_abs_mean = _safe_mean(np.abs(front))
    rear_abs_mean = _safe_mean(np.abs(rear))
    rear_bias = (
        float(rear_abs_mean - front_abs_mean)
        if np.isfinite(front_abs_mean) and np.isfinite(rear_abs_mean)
        else float("nan")
    )
    return {
        "slip_frequency": (
            float(np.mean(finite_peak > threshold))
            if finite_peak.size
            else float("nan")
        ),
        "peak_front_slip_abs": _safe_max_abs(front),
        "peak_rear_slip_abs": _safe_max_abs(rear),
        "mean_front_slip_abs": front_abs_mean,
        "mean_rear_slip_abs": rear_abs_mean,
        "rear_slip_bias": rear_bias,
    }


def extract_segment_features(
    dataframe: pd.DataFrame,
    segment: BrakingSegment,
    *,
    slip_threshold: float = 0.05,
) -> dict[str, float | int | bool]:
    """Extract one descriptive feature row for a detected braking segment."""

    start = max(0, int(segment.brake_start_index))
    end = min(len(dataframe) - 1, int(segment.segment_end_index))
    if end < start:
        raise ValueError("segment end occurs before segment start")

    window = dataframe.iloc[start : end + 1].reset_index(drop=True)
    times_s = _elapsed_seconds(window)
    speed = _numeric(window, "speed_kmh")
    throttle = np.clip(_numeric(window, "throttle"), 0.0, 1.0)
    brake = np.clip(_numeric(window, "brake"), 0.0, 1.0)
    steering = np.clip(_numeric(window, "steering"), -1.0, 1.0)

    local_brake_end_for_mean = min(
        len(window) - 1, max(0, int(segment.brake_end_index) - start)
    )
    braking_phase = brake[: local_brake_end_for_mean + 1]

    local_peak_index = max(0, int(segment.peak_brake_index) - start)
    local_brake_end_index = min(
        len(window) - 1,
        max(local_peak_index, int(segment.brake_end_index) - start),
    )
    release_brake = brake[local_peak_index : local_brake_end_index + 1]
    release_time = times_s[local_peak_index : local_brake_end_index + 1]

    steering_rates = _valid_slopes(steering, times_s)
    release_slopes = _valid_slopes(release_brake, release_time)
    overlap_mask = (throttle > 0.05) & (brake > 0.05)
    slip = _slip_metrics(window, threshold=float(slip_threshold))
    temperatures = _temperature_metrics(window)

    speed_at_start = float(speed[0]) if len(speed) and np.isfinite(speed[0]) else np.nan
    speed_loss = (
        float(max(0.0, speed_at_start - segment.minimum_speed_kmh))
        if np.isfinite(speed_at_start)
        else float("nan")
    )
    full_throttle_distance = (
        float(max(0.0, segment.full_throttle_m - segment.minimum_speed_m))
        if segment.full_throttle_found and segment.full_throttle_m is not None
        else float("nan")
    )
    minimum_speed_time_s = (
        float(_numeric(dataframe, "timestamp_ms")[segment.minimum_speed_index] / 1_000.0)
        if 0 <= segment.minimum_speed_index < len(dataframe)
        else float("nan")
    )
    full_throttle_delay_s = (
        float(max(0.0, segment.full_throttle_time_s - minimum_speed_time_s))
        if (
            segment.full_throttle_found
            and segment.full_throttle_time_s is not None
            and len(times_s)
        )
        else float("nan")
    )

    return {
        "braking_distance_m": float(
            max(0.0, segment.minimum_speed_m - segment.brake_start_m)
        ),
        "brake_zone_distance_m": float(
            max(0.0, segment.brake_end_m - segment.brake_start_m)
        ),
        "mean_brake": _safe_mean(braking_phase),
        "brake_release_smoothness_per_s": _safe_std(release_slopes),
        "speed_at_brake_start_kmh": speed_at_start,
        "speed_loss_kmh": speed_loss,
        "full_throttle_distance_from_min_speed_m": full_throttle_distance,
        "full_throttle_delay_from_min_speed_s": full_throttle_delay_s,
        "steering_input_std": _safe_std(steering),
        "steering_rate_std_per_s": _safe_std(steering_rates),
        "throttle_brake_overlap_ms": _mask_duration_s(overlap_mask, times_s)
        * 1_000.0,
        "mean_abs_lateral_g": _safe_mean(np.abs(_numeric(window, "g_force_lat"))),
        "peak_abs_lateral_g": _safe_max_abs(_numeric(window, "g_force_lat")),
        "mean_abs_longitudinal_g": _safe_mean(
            np.abs(_numeric(window, "g_force_lon"))
        ),
        "ers_deployed_delta_j": _positive_difference(
            _numeric(window, "ers_deployed")
        ),
        **slip,
        **temperatures,
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _metadata_value(mapping: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = mapping.get(key, default)
    try:
        return default if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


def _ers_usage_by_thirds(dataframe: pd.DataFrame) -> dict[str, float]:
    ers = _numeric(dataframe, "ers_deployed")
    distance = _numeric(dataframe, "lap_distance_pct")
    if len(ers) < 2:
        return {
            "ers_total_deployed_j": float("nan"),
            "ers_early_lap_pct": float("nan"),
            "ers_middle_lap_pct": float("nan"),
            "ers_late_lap_pct": float("nan"),
        }

    increments = np.diff(ers, prepend=ers[0])
    increments[~np.isfinite(increments)] = 0.0
    increments = np.clip(increments, 0.0, None)
    total = float(np.sum(increments))
    if total <= 0:
        return {
            "ers_total_deployed_j": 0.0,
            "ers_early_lap_pct": 0.0,
            "ers_middle_lap_pct": 0.0,
            "ers_late_lap_pct": 0.0,
        }

    finite_distance = np.nan_to_num(distance, nan=0.0)
    early = float(np.sum(increments[finite_distance < 1.0 / 3.0])) / total
    middle = float(
        np.sum(
            increments[
                (finite_distance >= 1.0 / 3.0) & (finite_distance < 2.0 / 3.0)
            ]
        )
    ) / total
    late = float(np.sum(increments[finite_distance >= 2.0 / 3.0])) / total
    return {
        "ers_total_deployed_j": total,
        "ers_early_lap_pct": early,
        "ers_middle_lap_pct": middle,
        "ers_late_lap_pct": late,
    }


def extract_lap_style_features(
    dataframe: pd.DataFrame,
    segments: Sequence[BrakingSegment],
    segment_feature_records: Sequence[Mapping[str, Any]],
    *,
    lap_id: str,
    track: str,
    source_file: str,
    catalog_row: Mapping[str, Any],
    slip_threshold: float = 0.05,
) -> dict[str, Any]:
    """Create one compact style row for a complete lap."""

    times_s = _elapsed_seconds(dataframe)
    duration_s = _lap_duration_s(dataframe)
    throttle = np.clip(_numeric(dataframe, "throttle"), 0.0, 1.0)
    brake = np.clip(_numeric(dataframe, "brake"), 0.0, 1.0)
    steering = np.clip(_numeric(dataframe, "steering"), -1.0, 1.0)
    overlap_mask = (throttle > 0.05) & (brake > 0.05)
    coast_mask = (throttle < 0.05) & (brake < 0.05)
    full_throttle_mask = throttle >= 0.95
    hard_brake_mask = brake >= 0.30

    segment_frame = pd.DataFrame.from_records(segment_feature_records)

    def segment_mean(column: str) -> float:
        if column not in segment_frame.columns:
            return float("nan")
        return _safe_mean(
            pd.to_numeric(segment_frame[column], errors="coerce").to_numpy(dtype=float)
        )

    def segment_std(column: str) -> float:
        if column not in segment_frame.columns:
            return float("nan")
        return _safe_std(
            pd.to_numeric(segment_frame[column], errors="coerce").to_numpy(dtype=float)
        )

    peak_brakes = np.asarray([segment.peak_brake for segment in segments], dtype=float)
    slip = _slip_metrics(dataframe, threshold=float(slip_threshold))
    temperatures = _temperature_metrics(dataframe)
    ers = _ers_usage_by_thirds(dataframe)

    lap_time_ms = _metadata_value(catalog_row, "lap_time_ms")
    if lap_time_ms is None:
        lap_time_ms = _safe_last(_numeric(dataframe, "lap_time_ms"))

    training_eligible = _as_bool(
        _metadata_value(catalog_row, "training_eligible", False)
    )
    record: dict[str, Any] = {
        "lap_id": lap_id,
        "track": track,
        "source_file": source_file,
        "session_uid": _metadata_value(catalog_row, "session_uid"),
        "lap_number": _metadata_value(catalog_row, "lap_number"),
        "lap_time_ms": lap_time_ms,
        "sample_rate_hz": _metadata_value(catalog_row, "sample_rate_hz"),
        "training_eligible": training_eligible,
        "feature_complete": training_eligible,
        "braking_segment_count": int(len(segments)),
        "brake_aggressiveness": _safe_mean(peak_brakes),
        "brake_peak_consistency": _safe_std(peak_brakes),
        "mean_brake_force": segment_mean("mean_brake"),
        "mean_braking_distance_m": segment_mean("braking_distance_m"),
        "brake_release_smoothness": segment_mean(
            "brake_release_smoothness_per_s"
        ),
        "throttle_patience_m": segment_mean(
            "full_throttle_distance_from_min_speed_m"
        ),
        "throttle_patience_s": segment_mean(
            "full_throttle_delay_from_min_speed_s"
        ),
        "steering_smoothness": segment_mean("steering_input_std"),
        "steering_rate_smoothness": segment_mean("steering_rate_std_per_s"),
        "mean_segment_minimum_speed_kmh": _safe_mean(
            np.asarray(
                [segment.minimum_speed_kmh for segment in segments], dtype=float
            )
        ),
        "throttle_brake_overlap_ms": _mask_duration_s(overlap_mask, times_s)
        * 1_000.0,
        "throttle_brake_overlap_pct": (
            _mask_duration_s(overlap_mask, times_s) / duration_s
            if duration_s > 0
            else float("nan")
        ),
        "coast_time_pct": (
            _mask_duration_s(coast_mask, times_s) / duration_s
            if duration_s > 0
            else float("nan")
        ),
        "full_throttle_time_pct": (
            _mask_duration_s(full_throttle_mask, times_s) / duration_s
            if duration_s > 0
            else float("nan")
        ),
        "hard_braking_time_pct": (
            _mask_duration_s(hard_brake_mask, times_s) / duration_s
            if duration_s > 0
            else float("nan")
        ),
        "mean_speed_kmh": _safe_mean(_numeric(dataframe, "speed_kmh")),
        "maximum_speed_kmh": _safe_max(_numeric(dataframe, "speed_kmh")),
        "minimum_speed_kmh": _safe_min(_numeric(dataframe, "speed_kmh")),
        "mean_abs_lateral_g": _safe_mean(
            np.abs(_numeric(dataframe, "g_force_lat"))
        ),
        "peak_abs_lateral_g": _safe_max_abs(_numeric(dataframe, "g_force_lat")),
        "mean_abs_longitudinal_g": _safe_mean(
            np.abs(_numeric(dataframe, "g_force_lon"))
        ),
        "peak_abs_longitudinal_g": _safe_max_abs(
            _numeric(dataframe, "g_force_lon")
        ),
        "fuel_used_kg": _decrease(_numeric(dataframe, "fuel_remaining")),
        "tyre_wear_delta": _wear_delta(dataframe),
        "whole_lap_steering_input_std": _safe_std(steering),
        "whole_lap_steering_rate_std_per_s": _safe_std(
            _valid_slopes(steering, times_s)
        ),
        **slip,
        **temperatures,
        **ers,
    }
    return record
