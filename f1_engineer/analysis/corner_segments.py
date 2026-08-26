"""Geometry-defined corner maneuvers with behaviour measured on raw telemetry.

This replaces brake-threshold segmentation for modelling purposes. Two problems
with the older approach are addressed here.

**The window no longer depends on the driver.** A segment that starts where
``brake >= 0.30`` moves whenever the driver brakes differently, so a corner has
no existence independent of how it was driven. Here the window comes from the
track map: it is identical on every lap, which is what makes "how did I take
this corner compared with my normal" a well-posed question.

**Behaviour is measured on the raw 60 Hz samples.** The uniform-distance grid
used for alignment decimates by a factor inversely proportional to speed, so
slow corners are sampled far more coarsely in time than straights. Any
derivative computed on that grid — steering rate, brake release rate — is
smoothed by an amount that depends on how fast the driver was going, which
silently confounds "smooth" with "slow". The grid stays useful for alignment
and comparison; it is the wrong place to measure dynamics.

Context and behaviour are kept in separate column groups, matching the Phase 3F
contract. Context here is real geometry rather than the causal telemetry proxy,
so it is comparable across tracks: that is the property few-shot coaching on an
unseen track depends on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Execution-independent descriptors of the corner itself. These are the k-NN
# query features. Turn direction is deliberately excluded: a left and a right
# corner of the same shape are the same driving problem mirrored, so matching
# uses unsigned geometry and keeps direction as metadata.
GEOMETRY_CONTEXT_COLUMNS: tuple[str, ...] = (
    "ctx_abs_heading_change_deg",
    "ctx_corner_length_m",
    "ctx_minimum_radius_m",
    "ctx_mean_radius_m",
    "ctx_entry_radius_m",
    "ctx_exit_radius_m",
    "ctx_radius_trend_per_100m",
    "ctx_elevation_change_m",
    "ctx_mean_gradient_pct",
    "ctx_preceding_straight_m",
    "ctx_following_straight_m",
    "ctx_preceding_corner_minimum_radius_m",
    "ctx_following_corner_minimum_radius_m",
)

# How the driver executed the maneuver. Prediction targets, never query inputs.
CORNER_BEHAVIOR_COLUMNS: tuple[str, ...] = (
    "brake_application_before_entry_m",
    "hard_brake_before_entry_m",
    "brake_build_rate_per_s",
    "peak_brake",
    "mean_brake",
    "brake_duration_s",
    "brake_release_duration_s",
    "brake_release_rate_per_s",
    "brake_release_smoothness_per_s",
    "trail_brake_past_entry_m",
    "entry_speed_kmh",
    "minimum_speed_kmh",
    "speed_loss_kmh",
    "minimum_speed_offset_from_apex_m",
    "throttle_reapplication_offset_from_apex_m",
    "full_throttle_offset_from_exit_m",
    "throttle_brake_overlap_s",
    "coast_distance_m",
    "peak_abs_steering",
    "steering_rate_std_per_s",
    "steering_correction_count",
    "mean_abs_lateral_g",
    "peak_abs_lateral_g",
    "mean_abs_longitudinal_g",
    "peak_rear_slip_abs",
    "peak_front_slip_abs",
    "rear_slip_bias",
    "slip_frequency",
)

# Behaviour available at every corner, including ones taken flat out.
ALWAYS_MEASURABLE_COLUMNS: tuple[str, ...] = (
    "minimum_speed_kmh",
    "minimum_speed_offset_from_apex_m",
    "peak_abs_steering",
    "steering_rate_std_per_s",
    "mean_abs_lateral_g",
)

# Behaviour that only exists when the driver actually braked.
BRAKING_BEHAVIOR_COLUMNS: tuple[str, ...] = (
    "brake_application_before_entry_m",
    "peak_brake",
    "entry_speed_kmh",
    "speed_loss_kmh",
    "brake_release_smoothness_per_s",
)

FRONT_SLIP_COLUMNS: tuple[str, ...] = ("wheel_slip_fl", "wheel_slip_fr")
REAR_SLIP_COLUMNS: tuple[str, ...] = ("wheel_slip_rl", "wheel_slip_rr")


@dataclass(frozen=True, slots=True)
class CornerSegmentConfig:
    """Window sizing and the thresholds used to read behaviour off a lap."""

    maximum_approach_m: float = 300.0
    maximum_exit_m: float = 200.0
    straight_fraction: float = 0.8
    brake_touch_threshold: float = 0.02
    hard_brake_threshold: float = 0.30
    throttle_touch_threshold: float = 0.05
    full_throttle_threshold: float = 0.95
    sustain_seconds: float = 0.08
    full_throttle_hold_s: float = 0.20
    peak_plateau_tolerance: float = 0.02
    slip_threshold: float = 0.05
    steering_correction_deadband: float = 0.05
    minimum_window_samples: int = 20


@dataclass(frozen=True, slots=True)
class CornerWindow:
    """One corner of one track, with the distance span used to observe it."""

    track: str
    corner_id: int
    direction: str
    entry_m: float
    apex_m: float
    exit_m: float
    approach_start_m: float
    window_end_m: float
    context: dict[str, float] = field(default_factory=dict)

    @property
    def corner_key(self) -> str:
        return f"{self.track}:{self.corner_id}"


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _mean(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.mean(finite)) if finite.size else float("nan")


def _std(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.std(finite)) if finite.size else float("nan")


def _max_abs(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.max(np.abs(finite))) if finite.size else float("nan")


def _value_at(grid_distance: np.ndarray, values: np.ndarray, target_m: float) -> float:
    if not len(grid_distance) or not np.isfinite(target_m):
        return float("nan")
    index = int(np.argmin(np.abs(grid_distance - float(target_m))))
    value = float(values[index])
    return value if np.isfinite(value) else float("nan")


def _slopes(values: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.empty(0, dtype=float)
    dv = np.diff(values)
    dt = np.diff(times_s)
    valid = np.isfinite(dv) & np.isfinite(dt) & (dt > 1e-6)
    return dv[valid] / dt[valid]


def _sustained_index(
    condition: np.ndarray,
    times_s: np.ndarray,
    minimum_duration_s: float,
    *,
    start: int = 0,
) -> int | None:
    """First index from ``start`` where a condition holds long enough.

    Raw telemetry is noisy at the thresholds that matter, so a single sample
    crossing a threshold is not treated as the driver having done something.
    """

    total = len(condition)
    if start >= total:
        return None
    index = start
    while index < total:
        if not condition[index]:
            index += 1
            continue
        end = index
        while end + 1 < total and condition[end + 1]:
            end += 1
        held = float(times_s[end] - times_s[index]) if end > index else 0.0
        if held >= minimum_duration_s or end == total - 1:
            return index
        index = end + 1
    return None


def _direction_change_count(values: np.ndarray, deadband: float) -> int:
    """Count meaningful reversals in a signal.

    Small jitter is ignored: a reversal only counts once the signal has moved
    by ``deadband`` since the last turning point. Applied to steering this is a
    correction count, which is one of the coaching cues the project wants.
    """

    finite = _finite(values)
    if finite.size < 3:
        return 0
    reversals = 0
    extreme = float(finite[0])
    direction = 0
    for value in finite[1:]:
        if direction >= 0 and value > extreme:
            extreme = value
            direction = 1
        elif direction <= 0 and value < extreme:
            extreme = value
            direction = -1
        elif abs(value - extreme) >= deadband:
            reversals += 1
            extreme = value
            direction = 1 if value > extreme else -1
    return reversals


def _distance_travelled(distance_m: np.ndarray, mask: np.ndarray) -> float:
    if len(distance_m) < 2:
        return 0.0
    steps = np.diff(distance_m)
    valid = np.isfinite(steps) & (steps > 0)
    active = np.asarray(mask[:-1], dtype=bool) & valid
    return float(np.sum(steps[active]))


def _duration_s(times_s: np.ndarray, mask: np.ndarray) -> float:
    if len(times_s) < 2:
        return 0.0
    steps = np.diff(times_s)
    valid = np.isfinite(steps) & (steps > 0)
    active = np.asarray(mask[:-1], dtype=bool) & valid
    return float(np.sum(steps[active]))


def build_corner_windows(
    corners: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    track: str,
    config: CornerSegmentConfig | None = None,
) -> list[CornerWindow]:
    """Turn a track's corner table into observation windows with context."""

    cfg = config or CornerSegmentConfig()
    if corners.empty:
        return []

    required = {"corner_id", "entry_m", "apex_m", "exit_m"}
    missing = sorted(required.difference(corners.columns))
    if missing:
        raise ValueError(f"corner table is missing required columns: {missing}")

    ordered = corners.sort_values("entry_m", kind="stable").reset_index(drop=True)
    grid_distance = _numeric(grid, "lap_distance_m")
    grid_radius = _numeric(grid, "radius_m")

    radii = pd.to_numeric(ordered["minimum_radius_m"], errors="coerce").to_numpy(
        dtype=float
    ) if "minimum_radius_m" in ordered.columns else np.full(len(ordered), np.nan)

    windows: list[CornerWindow] = []
    for position, row in ordered.iterrows():
        entry_m = float(row["entry_m"])
        apex_m = float(row["apex_m"])
        exit_m = float(row["exit_m"])
        length_m = float(row.get("length_m", max(0.0, exit_m - entry_m)))

        preceding = float(row.get("preceding_straight_m", np.nan))
        following = float(row.get("following_straight_m", np.nan))
        approach_span = min(
            cfg.maximum_approach_m,
            (preceding * cfg.straight_fraction) if np.isfinite(preceding) else cfg.maximum_approach_m,
        )
        exit_span = min(
            cfg.maximum_exit_m,
            (following * cfg.straight_fraction) if np.isfinite(following) else cfg.maximum_exit_m,
        )

        entry_radius = _value_at(grid_distance, grid_radius, entry_m)
        exit_radius = _value_at(grid_distance, grid_radius, exit_m)
        radius_trend = (
            100.0 * (exit_radius - entry_radius) / length_m
            if length_m > 0 and np.isfinite(entry_radius) and np.isfinite(exit_radius)
            else float("nan")
        )

        previous_radius = float(radii[position - 1]) if len(radii) > 1 else float("nan")
        next_radius = (
            float(radii[(position + 1) % len(radii)]) if len(radii) > 1 else float("nan")
        )

        context = {
            "ctx_abs_heading_change_deg": abs(
                float(row.get("heading_change_deg", np.nan))
            ),
            "ctx_corner_length_m": length_m,
            "ctx_minimum_radius_m": float(row.get("minimum_radius_m", np.nan)),
            "ctx_mean_radius_m": float(row.get("mean_radius_m", np.nan)),
            "ctx_entry_radius_m": entry_radius,
            "ctx_exit_radius_m": exit_radius,
            "ctx_radius_trend_per_100m": radius_trend,
            "ctx_elevation_change_m": float(row.get("elevation_change_m", np.nan)),
            "ctx_mean_gradient_pct": float(row.get("mean_gradient_pct", np.nan)),
            "ctx_preceding_straight_m": preceding,
            "ctx_following_straight_m": following,
            "ctx_preceding_corner_minimum_radius_m": previous_radius,
            "ctx_following_corner_minimum_radius_m": next_radius,
        }

        windows.append(
            CornerWindow(
                track=track,
                corner_id=int(row["corner_id"]),
                direction=str(row.get("direction", "unknown")),
                entry_m=entry_m,
                apex_m=apex_m,
                exit_m=exit_m,
                approach_start_m=entry_m - approach_span,
                window_end_m=exit_m + exit_span,
                context=context,
            )
        )
    return windows


def _slip_metrics(window: pd.DataFrame, threshold: float) -> dict[str, float]:
    front = np.column_stack(
        [_numeric(window, column) for column in FRONT_SLIP_COLUMNS]
    )
    rear = np.column_stack([_numeric(window, column) for column in REAR_SLIP_COLUMNS])
    combined = np.column_stack((front, rear))

    row_peak = np.full(combined.shape[0], np.nan, dtype=float)
    for index, row in enumerate(combined):
        finite = _finite(row)
        if finite.size:
            row_peak[index] = float(np.max(np.abs(finite)))
    finite_peak = _finite(row_peak)

    front_mean = _mean(np.abs(front))
    rear_mean = _mean(np.abs(rear))
    return {
        "peak_front_slip_abs": _max_abs(front),
        "peak_rear_slip_abs": _max_abs(rear),
        "rear_slip_bias": (
            float(rear_mean - front_mean)
            if np.isfinite(front_mean) and np.isfinite(rear_mean)
            else float("nan")
        ),
        "slip_frequency": (
            float(np.mean(finite_peak > threshold)) if finite_peak.size else float("nan")
        ),
    }


def _empty_behavior() -> dict[str, Any]:
    record: dict[str, Any] = {column: float("nan") for column in CORNER_BEHAVIOR_COLUMNS}
    record["window_samples"] = 0
    record["window_sample_rate_hz"] = float("nan")
    record["approach_truncated"] = True
    record["brake_detected"] = False
    record["release_truncated"] = False
    record["behavior_complete"] = False
    record["braking_complete"] = False
    return record


def extract_corner_behavior(
    raw_lap: pd.DataFrame,
    window: CornerWindow,
    *,
    config: CornerSegmentConfig | None = None,
) -> dict[str, Any]:
    """Measure how one lap executed one geometric corner.

    Operates on the raw, time-sampled lap. Distances are used only to select
    the window and to express positions relative to the corner.
    """

    cfg = config or CornerSegmentConfig()
    if "lap_distance_m" not in raw_lap.columns:
        raise ValueError("raw lap requires lap_distance_m")

    distance = _numeric(raw_lap, "lap_distance_m")
    selected = np.flatnonzero(
        np.isfinite(distance)
        & (distance >= window.approach_start_m)
        & (distance <= window.window_end_m)
    )
    if selected.size < cfg.minimum_window_samples:
        return _empty_behavior()

    frame = raw_lap.iloc[selected].reset_index(drop=True)
    distance_m = distance[selected]
    times_s = _numeric(frame, "timestamp_ms") / 1_000.0
    speed = _numeric(frame, "speed_kmh")
    brake = np.clip(np.nan_to_num(_numeric(frame, "brake"), nan=0.0), 0.0, 1.0)
    throttle = np.clip(np.nan_to_num(_numeric(frame, "throttle"), nan=0.0), 0.0, 1.0)
    steering = np.clip(_numeric(frame, "steering"), -1.0, 1.0)

    span_s = float(times_s[-1] - times_s[0]) if len(times_s) > 1 else 0.0
    sample_rate = float((len(times_s) - 1) / span_s) if span_s > 0 else float("nan")

    record: dict[str, Any] = {}
    record["window_samples"] = int(len(frame))
    record["window_sample_rate_hz"] = sample_rate
    record["approach_truncated"] = bool(
        distance_m[0] > window.approach_start_m + 1.0
        or distance_m[-1] < window.window_end_m - 1.0
    )

    # Braking. A dual threshold keeps the measurement origin independent of how
    # progressively the driver applies the pedal: detection uses the firm
    # threshold, but the reported start is where pressure actually began.
    hard_index = _sustained_index(
        brake >= cfg.hard_brake_threshold, times_s, cfg.sustain_seconds
    )
    touch_index: int | None = None
    if hard_index is not None:
        touch_index = hard_index
        while touch_index > 0 and brake[touch_index - 1] > cfg.brake_touch_threshold:
            touch_index -= 1
    else:
        touch_index = _sustained_index(
            brake > cfg.brake_touch_threshold, times_s, cfg.sustain_seconds
        )

    record["brake_detected"] = touch_index is not None

    # Apex speed is a property of the corner, so the minimum is searched inside
    # the corner rather than across the whole window. Searching the window would
    # return the approach speed for a corner taken flat, which is not an apex.
    corner_indices = np.flatnonzero(
        (distance_m >= window.entry_m) & (distance_m <= window.exit_m)
    )
    search = corner_indices if corner_indices.size else np.arange(len(speed))
    finite_search = search[np.isfinite(speed[search])]
    if finite_search.size:
        minimum_speed_index = int(finite_search[np.argmin(speed[finite_search])])
    else:
        minimum_speed_index = int(search[0])
    record["minimum_speed_kmh"] = float(speed[minimum_speed_index])
    record["minimum_speed_offset_from_apex_m"] = float(
        distance_m[minimum_speed_index] - window.apex_m
    )

    if touch_index is not None:
        peak_index = touch_index + int(np.argmax(brake[touch_index:]))
        peak_value = float(brake[peak_index])

        # Peak pressure is usually held for a while. argmax returns the start of
        # that plateau, so the release has to start at its end — otherwise the
        # held portion is averaged into the release and every driver looks
        # gentler than they were.
        release_start = peak_index
        while (
            release_start + 1 < len(brake)
            and brake[release_start + 1] >= peak_value - cfg.peak_plateau_tolerance
        ):
            release_start += 1

        # The release runs all the way back to zero pedal, not to the detection
        # threshold. The last third of the release is the trail-braking part and
        # is the most coaching-relevant portion of the whole event.
        release_end = release_start
        while (
            release_end + 1 < len(brake)
            and brake[release_end + 1] > cfg.brake_touch_threshold
        ):
            release_end += 1
        # The pedal is still loaded at the window edge, so the release was cut
        # short rather than completed. Recorded explicitly instead of quietly
        # producing a shortened release.
        record["release_truncated"] = bool(
            release_end >= len(brake) - 1
            and brake[release_end] > cfg.brake_touch_threshold
        )

        braking_mask = brake > cfg.brake_touch_threshold
        release_brake = brake[release_start : release_end + 1]
        release_times = times_s[release_start : release_end + 1]
        release_slopes = _slopes(release_brake, release_times)
        build_slopes = _slopes(
            brake[touch_index : peak_index + 1], times_s[touch_index : peak_index + 1]
        )

        entry_speed = float(speed[touch_index])
        record.update(
            {
                "brake_application_before_entry_m": float(
                    window.entry_m - distance_m[touch_index]
                ),
                "hard_brake_before_entry_m": (
                    float(window.entry_m - distance_m[hard_index])
                    if hard_index is not None
                    else float("nan")
                ),
                "brake_build_rate_per_s": _mean(build_slopes),
                "peak_brake": peak_value,
                "mean_brake": _mean(brake[touch_index : release_end + 1]),
                "brake_duration_s": _duration_s(times_s, braking_mask),
                "brake_release_duration_s": float(
                    max(0.0, times_s[release_end] - times_s[release_start])
                ),
                "brake_release_rate_per_s": _mean(release_slopes),
                "brake_release_smoothness_per_s": _std(release_slopes),
                "trail_brake_past_entry_m": float(
                    max(0.0, distance_m[release_end] - window.entry_m)
                ),
                "entry_speed_kmh": entry_speed,
                "speed_loss_kmh": float(
                    max(0.0, entry_speed - record["minimum_speed_kmh"])
                ),
            }
        )
    else:
        for column in (
            "brake_application_before_entry_m",
            "hard_brake_before_entry_m",
            "brake_build_rate_per_s",
            "peak_brake",
            "mean_brake",
            "brake_duration_s",
            "brake_release_duration_s",
            "brake_release_rate_per_s",
            "brake_release_smoothness_per_s",
            "trail_brake_past_entry_m",
            "entry_speed_kmh",
            "speed_loss_kmh",
        ):
            record[column] = float("nan")
        record["release_truncated"] = False

    # Throttle recovery, expressed against the geometric apex and exit rather
    # than against the driver's own minimum-speed point.
    throttle_index = _sustained_index(
        throttle > cfg.throttle_touch_threshold,
        times_s,
        cfg.sustain_seconds,
        start=minimum_speed_index,
    )
    full_index = _sustained_index(
        throttle >= cfg.full_throttle_threshold,
        times_s,
        cfg.full_throttle_hold_s,
        start=minimum_speed_index,
    )
    record["throttle_reapplication_offset_from_apex_m"] = (
        float(distance_m[throttle_index] - window.apex_m)
        if throttle_index is not None
        else float("nan")
    )
    record["full_throttle_offset_from_exit_m"] = (
        float(distance_m[full_index] - window.exit_m)
        if full_index is not None
        else float("nan")
    )

    overlap_mask = (throttle > cfg.throttle_touch_threshold) & (
        brake > cfg.throttle_touch_threshold
    )
    coast_mask = (throttle <= cfg.throttle_touch_threshold) & (
        brake <= cfg.brake_touch_threshold
    )
    record["throttle_brake_overlap_s"] = _duration_s(times_s, overlap_mask)
    record["coast_distance_m"] = _distance_travelled(distance_m, coast_mask)

    record["peak_abs_steering"] = _max_abs(steering)
    record["steering_rate_std_per_s"] = _std(_slopes(steering, times_s))
    record["steering_correction_count"] = _direction_change_count(
        steering, cfg.steering_correction_deadband
    )

    lateral_g = _numeric(frame, "g_force_lat")
    longitudinal_g = _numeric(frame, "g_force_lon")
    record["mean_abs_lateral_g"] = _mean(np.abs(lateral_g))
    record["peak_abs_lateral_g"] = _max_abs(lateral_g)
    record["mean_abs_longitudinal_g"] = _mean(np.abs(longitudinal_g))
    record.update(_slip_metrics(frame, cfg.slip_threshold))

    # Completeness is split in two. Plenty of corners are taken flat, and a
    # missing brake trace there is the correct answer rather than missing data.
    def all_finite(columns: tuple[str, ...]) -> bool:
        return all(np.isfinite(float(record.get(column, np.nan))) for column in columns)

    record["behavior_complete"] = all_finite(ALWAYS_MEASURABLE_COLUMNS)
    # A release still in progress at the window edge is either genuinely cut
    # short or is already braking for the next corner. Either way the release
    # figures describe a partial event, so the row is not a clean target.
    record["braking_complete"] = (
        bool(record["brake_detected"])
        and not bool(record.get("release_truncated", False))
        and all_finite(BRAKING_BEHAVIOR_COLUMNS)
    )
    return record


def extract_lap_corner_segments(
    raw_lap: pd.DataFrame,
    windows: list[CornerWindow],
    *,
    metadata: dict[str, Any],
    config: CornerSegmentConfig | None = None,
) -> list[dict[str, Any]]:
    """One row per corner for one lap: identity, context, then behaviour."""

    records: list[dict[str, Any]] = []
    for window in windows:
        behavior = extract_corner_behavior(raw_lap, window, config=config)
        record: dict[str, Any] = {
            **metadata,
            "corner_id": window.corner_id,
            "corner_key": window.corner_key,
            "turn_direction": window.direction,
            "corner_entry_m": window.entry_m,
            "corner_apex_m": window.apex_m,
            "corner_exit_m": window.exit_m,
            "window_start_m": window.approach_start_m,
            "window_end_m": window.window_end_m,
            **window.context,
            **behavior,
        }
        records.append(record)
    return records