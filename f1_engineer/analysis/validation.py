from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Minimum schema needed to establish that a file is a complete, usable lap.
CORE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp_ms",
    "lap_distance_pct",
    "lap_distance_m",
    "speed_kmh",
    "throttle",
    "brake",
    "gear",
    "steering",
    "rpm",
    "drs",
    "lap_time_ms",
    "current_lap_invalid",
)

# Columns needed for the full Phase 3 style-feature set. A lap can still be
# useful for core input analysis when one of these is unavailable, but it is
# not admitted to the full training set until the fields are populated.
MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "timestamp_ms",
    "lap_distance_pct",
    "lap_distance_m",
    "speed_kmh",
    "throttle",
    "brake",
    "gear",
    "steering",
    "rpm",
    "drs",
    "ers_deployed",
    "fuel_remaining",
    "fl_tyre_temp",
    "fr_tyre_temp",
    "rl_tyre_temp",
    "rr_tyre_temp",
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

PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "lap_distance_pct": (-0.02, 1.02),
    "speed_kmh": (0.0, 400.0),
    "throttle": (0.0, 1.01),
    "brake": (0.0, 1.01),
    "gear": (-1.0, 8.0),
    "steering": (-1.05, 1.05),
    "rpm": (0.0, 20_000.0),
    "ers_deployed": (0.0, 5_000_000.0),
    "fuel_remaining": (-0.1, 120.0),
    "fl_tyre_temp": (0.0, 200.0),
    "fr_tyre_temp": (0.0, 200.0),
    "rl_tyre_temp": (0.0, 200.0),
    "rr_tyre_temp": (0.0, 200.0),
    "fl_tyre_wear": (0.0, 1.0),
    "fr_tyre_wear": (0.0, 1.0),
    "rl_tyre_wear": (0.0, 1.0),
    "rr_tyre_wear": (0.0, 1.0),
    "g_force_lat": (-15.0, 15.0),
    "g_force_lon": (-15.0, 15.0),
    "wheel_slip_fl": (-5.0, 10.0),
    "wheel_slip_fr": (-5.0, 10.0),
    "wheel_slip_rl": (-5.0, 10.0),
    "wheel_slip_rr": (-5.0, 10.0),
}


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    expected_sample_rate_hz: float = 60.0
    sample_rate_tolerance_hz: float = 7.0
    minimum_rows: int = 1_000
    start_distance_pct_max: float = 0.03
    end_distance_pct_min: float = 0.97
    minimum_distance_coverage_pct: float = 0.94
    distance_backtrack_tolerance_pct: float = 0.002
    maximum_model_null_fraction: float = 0.0
    require_enrichment_for_training: bool = True
    # The game occasionally reports a single stale currentLapTime, so a lap can
    # contain one sub-frame backward step while frame_identifier and
    # lap_distance_m stay perfectly ordered. Rejecting the whole lap for that
    # discarded a fifth of the dataset, so a small amount is tolerated and
    # reported as a quality warning instead.
    maximum_timestamp_backsteps: int = 3
    maximum_timestamp_backstep_ms: float = 34.0


def _safe_first(series: pd.Series, default: Any = None) -> Any:
    values = series.dropna()
    if values.empty:
        return default
    value = values.iloc[0]
    return value.item() if hasattr(value, "item") else value


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _sampling_class(rate_hz: float | None) -> str:
    if rate_hz is None or not np.isfinite(rate_hz):
        return "unknown"
    rates = (10.0, 20.0, 30.0, 60.0)
    nearest = min(rates, key=lambda rate: abs(rate_hz - rate))
    if abs(rate_hz - nearest) <= 4.0:
        return f"{int(nearest)}hz"
    return "other"


def _filename_validity(path: Path) -> str:
    name = path.stem.lower()
    if name.endswith("_invalid") or "_invalid_" in name:
        return "invalid"
    if name.endswith("_valid") or "_valid_" in name:
        return "valid"
    return "unknown"


def _reason_join(reasons: list[str]) -> str:
    return ";".join(dict.fromkeys(reason for reason in reasons if reason))


def validate_lap_dataframe(
    dataframe: pd.DataFrame,
    path: Path,
    config: ValidationConfig | None = None,
) -> dict[str, Any]:
    """Validate one in-memory lap without modifying it.

    The function intentionally returns a flat dictionary so the records can be
    written directly to CSV. Outlier detection is applied later at catalog
    level because it requires comparing laps from the same track.
    """

    cfg = config or ValidationConfig()
    frame = dataframe
    rows = int(len(frame))
    columns = set(frame.columns)

    missing_core = sorted(set(CORE_REQUIRED_COLUMNS) - columns)
    missing_model = sorted(set(MODEL_FEATURE_COLUMNS) - columns)
    schema_ok = not missing_core

    track = path.parent.name
    if "track" in columns:
        track_value = _safe_first(frame["track"], track)
        if track_value is not None and str(track_value).strip():
            track = str(track_value).strip().lower()

    session_uid = (
        _safe_int(_safe_first(frame["session_uid"]))
        if "session_uid" in columns
        else None
    )
    lap_number = (
        _safe_int(_safe_first(frame["lap_number"]))
        if "lap_number" in columns
        else None
    )

    timestamp_start_ms: float | None = None
    timestamp_end_ms: float | None = None
    timestamps_monotonic = False
    duplicate_timestamp_count = 0
    timestamp_backward_steps = 0
    timestamp_worst_backward_ms = 0.0
    sample_rate_hz: float | None = None

    if "timestamp_ms" in columns and rows:
        timestamps = pd.to_numeric(frame["timestamp_ms"], errors="coerce")
        finite_timestamps = timestamps.dropna()
        if not finite_timestamps.empty:
            timestamp_start_ms = float(finite_timestamps.iloc[0])
            timestamp_end_ms = float(finite_timestamps.iloc[-1])
            timestamps_monotonic = bool(finite_timestamps.is_monotonic_increasing)
            duplicate_timestamp_count = int(finite_timestamps.duplicated().sum())
            backward = finite_timestamps.diff().dropna()
            backward = backward[backward < 0]
            timestamp_backward_steps = int(len(backward))
            timestamp_worst_backward_ms = (
                float(backward.min()) if timestamp_backward_steps else 0.0
            )
            duration_s = (timestamp_end_ms - timestamp_start_ms) / 1_000.0
            if duration_s > 0 and len(finite_timestamps) > 1:
                sample_rate_hz = float((len(finite_timestamps) - 1) / duration_s)

    distance_start_pct: float | None = None
    distance_end_pct: float | None = None
    distance_coverage_pct: float | None = None
    distance_monotonic = False
    distance_backtrack_events = 0

    if "lap_distance_pct" in columns and rows:
        distance = pd.to_numeric(frame["lap_distance_pct"], errors="coerce").dropna()
        if not distance.empty:
            distance_start_pct = float(distance.iloc[0])
            distance_end_pct = float(distance.iloc[-1])
            distance_coverage_pct = float(distance_end_pct - distance_start_pct)
            diffs = distance.diff().dropna()
            distance_backtrack_events = int(
                (diffs < -cfg.distance_backtrack_tolerance_pct).sum()
            )
            distance_monotonic = distance_backtrack_events == 0

    official_lap_time_ms: int | None = None
    if "lap_time_ms" in columns:
        lap_times = pd.to_numeric(frame["lap_time_ms"], errors="coerce").dropna()
        positive = lap_times[lap_times > 0]
        if not positive.empty:
            official_lap_time_ms = int(round(float(positive.iloc[-1])))
    if official_lap_time_ms is None and timestamp_end_ms is not None:
        official_lap_time_ms = int(round(timestamp_end_ms))

    filename_validity = _filename_validity(path)
    invalid_samples = 0
    if "current_lap_invalid" in columns:
        invalid_values = frame["current_lap_invalid"].fillna(False).astype(bool)
        invalid_samples = int(invalid_values.sum())
    valid = invalid_samples == 0 and filename_validity != "invalid"

    available_model_columns = [
        column for column in MODEL_FEATURE_COLUMNS if column in columns
    ]
    model_null_cells = 0
    model_null_fraction = 1.0 if not available_model_columns else 0.0
    all_null_model_columns: list[str] = []
    if available_model_columns and rows:
        model_data = frame[available_model_columns]
        model_null_cells = int(model_data.isna().sum().sum())
        model_null_fraction = float(
            model_null_cells / (rows * len(MODEL_FEATURE_COLUMNS))
        )
        all_null_model_columns = sorted(
            column
            for column in available_model_columns
            if bool(model_data[column].isna().all())
        )

    implausible_columns: list[str] = []
    for column, (minimum, maximum) in PLAUSIBLE_RANGES.items():
        if column not in columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        if bool(((values < minimum) | (values > maximum)).any()):
            implausible_columns.append(column)

    sample_rate_ok = bool(
        sample_rate_hz is not None
        and abs(sample_rate_hz - cfg.expected_sample_rate_hz)
        <= cfg.sample_rate_tolerance_hz
    )
    starts_near_zero = bool(
        distance_start_pct is not None
        and -0.01 <= distance_start_pct <= cfg.start_distance_pct_max
    )
    reaches_finish = bool(
        distance_end_pct is not None
        and distance_end_pct >= cfg.end_distance_pct_min
    )
    coverage_ok = bool(
        distance_coverage_pct is not None
        and distance_coverage_pct >= cfg.minimum_distance_coverage_pct
    )
    lap_time_ok = bool(official_lap_time_ms and official_lap_time_ms > 0)
    row_count_ok = rows >= cfg.minimum_rows

    # Strict monotonicity is kept on the record, but eligibility uses a
    # tolerance: a couple of sub-frame glitches do not make a lap unusable.
    timestamps_acceptable = bool(
        timestamps_monotonic
        or (
            timestamp_backward_steps <= cfg.maximum_timestamp_backsteps
            and abs(timestamp_worst_backward_ms) <= cfg.maximum_timestamp_backstep_ms
        )
    )

    complete = bool(
        schema_ok
        and row_count_ok
        and timestamps_acceptable
        and starts_near_zero
        and reaches_finish
        and coverage_ok
        and distance_monotonic
        and lap_time_ok
    )

    core_reasons: list[str] = []
    if missing_core:
        core_reasons.append("missing_core_columns")
    if not row_count_ok:
        core_reasons.append("too_few_rows")
    if not timestamps_acceptable:
        core_reasons.append("timestamps_not_monotonic")
    quality_warnings: list[str] = []
    if timestamp_backward_steps and timestamps_acceptable:
        quality_warnings.append("timestamp_jitter")
    if duplicate_timestamp_count:
        # F1 can repeat currentLapTime on adjacent telemetry frames. Preserve
        # the count for diagnostics, but distance resampling can handle it.
        quality_warnings.append("duplicate_timestamps")
    if not starts_near_zero:
        core_reasons.append("lap_start_not_captured")
    if not reaches_finish:
        core_reasons.append("lap_finish_not_captured")
    if not coverage_ok:
        core_reasons.append("insufficient_distance_coverage")
    if not distance_monotonic:
        core_reasons.append("distance_backtrack")
    if not lap_time_ok:
        core_reasons.append("missing_lap_time")
    if not valid:
        core_reasons.append("invalid_lap")
    if not sample_rate_ok:
        core_reasons.append(f"sample_rate_{_sampling_class(sample_rate_hz)}")
    if implausible_columns:
        core_reasons.append("implausible_values")

    core_eligible = bool(
        complete
        and valid
        and sample_rate_ok
        and not implausible_columns
    )

    training_reasons = list(core_reasons)
    if missing_model:
        training_reasons.append("missing_model_columns")
    if all_null_model_columns:
        training_reasons.append("all_null_model_columns")
    if model_null_fraction > cfg.maximum_model_null_fraction:
        training_reasons.append("model_null_values")

    enrichment_ok = bool(
        not missing_model
        and not all_null_model_columns
        and model_null_fraction <= cfg.maximum_model_null_fraction
    )
    training_eligible_pre_outlier = bool(
        core_eligible
        and (enrichment_ok or not cfg.require_enrichment_for_training)
    )

    return {
        "file_path": str(path),
        "file_name": path.name,
        "track": track,
        "session_uid": session_uid,
        "lap_number": lap_number,
        "lap_time_ms": official_lap_time_ms,
        "lap_time_s": (
            official_lap_time_ms / 1_000.0
            if official_lap_time_ms is not None
            else None
        ),
        "rows": rows,
        "readable": True,
        "schema_ok": schema_ok,
        "valid": valid,
        "filename_validity": filename_validity,
        "invalid_sample_count": invalid_samples,
        "complete": complete,
        "timestamp_start_ms": timestamp_start_ms,
        "timestamp_end_ms": timestamp_end_ms,
        "timestamps_monotonic": timestamps_monotonic,
        "timestamps_acceptable": timestamps_acceptable,
        "timestamp_backward_steps": timestamp_backward_steps,
        "timestamp_worst_backward_ms": timestamp_worst_backward_ms,
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "sample_rate_hz": sample_rate_hz,
        "sample_rate_class": _sampling_class(sample_rate_hz),
        "sample_rate_ok": sample_rate_ok,
        "distance_start_pct": distance_start_pct,
        "distance_end_pct": distance_end_pct,
        "distance_coverage_pct": distance_coverage_pct,
        "distance_monotonic": distance_monotonic,
        "distance_backtrack_events": distance_backtrack_events,
        "missing_core_columns": ",".join(missing_core),
        "missing_model_columns": ",".join(missing_model),
        "all_null_model_columns": ",".join(all_null_model_columns),
        "model_null_cells": model_null_cells,
        "model_null_fraction": model_null_fraction,
        "implausible_columns": ",".join(sorted(implausible_columns)),
        "core_eligible": core_eligible,
        "training_eligible_pre_outlier": training_eligible_pre_outlier,
        "training_eligible": training_eligible_pre_outlier,
        "outlier_lap_time": False,
        "track_median_lap_time_ms": None,
        "lap_time_ratio_to_track_median": None,
        "exclusion_reasons": _reason_join(training_reasons),
        "quality_warnings": _reason_join(quality_warnings),
        "read_error": "",
    }


def unreadable_lap_record(path: Path, error: Exception) -> dict[str, Any]:
    return {
        "file_path": str(path),
        "file_name": path.name,
        "track": path.parent.name.lower(),
        "session_uid": None,
        "lap_number": None,
        "lap_time_ms": None,
        "lap_time_s": None,
        "rows": 0,
        "readable": False,
        "schema_ok": False,
        "valid": False,
        "filename_validity": _filename_validity(path),
        "invalid_sample_count": 0,
        "complete": False,
        "timestamp_start_ms": None,
        "timestamp_end_ms": None,
        "timestamps_monotonic": False,
        "timestamps_acceptable": False,
        "timestamp_backward_steps": 0,
        "timestamp_worst_backward_ms": 0.0,
        "duplicate_timestamp_count": 0,
        "sample_rate_hz": None,
        "sample_rate_class": "unknown",
        "sample_rate_ok": False,
        "distance_start_pct": None,
        "distance_end_pct": None,
        "distance_coverage_pct": None,
        "distance_monotonic": False,
        "distance_backtrack_events": 0,
        "missing_core_columns": "",
        "missing_model_columns": "",
        "all_null_model_columns": "",
        "model_null_cells": 0,
        "model_null_fraction": 1.0,
        "implausible_columns": "",
        "core_eligible": False,
        "training_eligible_pre_outlier": False,
        "training_eligible": False,
        "outlier_lap_time": False,
        "track_median_lap_time_ms": None,
        "lap_time_ratio_to_track_median": None,
        "exclusion_reasons": "unreadable_parquet",
        "quality_warnings": "",
        "read_error": f"{type(error).__name__}: {error}",
    }
