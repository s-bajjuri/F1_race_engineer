from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class BrakingSegment:
    segment_id: int
    brake_start_index: int
    brake_end_index: int
    peak_brake_index: int
    minimum_speed_index: int
    full_throttle_index: int | None
    segment_end_index: int
    brake_start_time_s: float
    brake_end_time_s: float
    brake_duration_s: float
    brake_start_pct: float
    brake_end_pct: float
    brake_start_m: float
    brake_end_m: float
    peak_brake: float
    peak_brake_pct: float
    peak_brake_m: float
    minimum_speed_kmh: float
    minimum_speed_pct: float
    minimum_speed_m: float
    full_throttle_found: bool
    full_throttle_time_s: float | None
    full_throttle_pct: float | None
    full_throttle_m: float | None
    segment_end_pct: float
    segment_end_m: float
    segment_length_m: float

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def _numeric(dataframe: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in dataframe.columns:
        return np.full(len(dataframe), default, dtype=float)
    values = pd.to_numeric(dataframe[column], errors="coerce").to_numpy(dtype=float)
    return np.nan_to_num(values, nan=default)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end - 1)) for start, end in changes.reshape(-1, 2)]


def _sample_duration_s(times_s: np.ndarray) -> float:
    diffs = np.diff(times_s)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if len(diffs) else 0.0


def _run_duration_s(times_s: np.ndarray, start: int, end: int) -> float:
    return max(0.0, float(times_s[end] - times_s[start])) + _sample_duration_s(
        times_s[start : end + 1]
    )


def _merge_runs(
    runs: list[tuple[int, int]],
    times_s: np.ndarray,
    maximum_gap_s: float,
) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged: list[tuple[int, int]] = [runs[0]]
    for start, end in runs[1:]:
        previous_start, previous_end = merged[-1]
        gap = float(times_s[start] - times_s[previous_end])
        if gap <= maximum_gap_s:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return merged


def _first_sustained_index(
    condition: np.ndarray,
    times_s: np.ndarray,
    start: int,
    end: int,
    minimum_duration_s: float,
) -> int | None:
    if end < start:
        return None
    local_runs = _runs(condition[start : end + 1])
    for local_start, local_end in local_runs:
        absolute_start = start + local_start
        absolute_end = start + local_end
        if _run_duration_s(times_s, absolute_start, absolute_end) >= minimum_duration_s:
            return absolute_start
    return None


def detect_braking_segments(
    dataframe: pd.DataFrame,
    *,
    brake_threshold: float = 0.30,
    minimum_brake_duration_s: float = 0.50,
    merge_gap_s: float = 0.20,
    full_throttle_threshold: float = 0.95,
    full_throttle_hold_s: float = 0.20,
    maximum_corner_window_s: float = 10.0,
) -> list[BrakingSegment]:
    """Detect meaningful braking/corner segments in one lap.

    Braking events are defined by brake input above ``brake_threshold`` for at
    least ``minimum_brake_duration_s``. Short releases are merged. Each event
    is extended through the minimum-speed point and, when available, the first
    sustained full-throttle application before the next braking event.
    """

    required = {"timestamp_ms", "lap_distance_pct", "brake", "speed_kmh"}
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"missing columns for segmentation: {', '.join(missing)}")
    if dataframe.empty:
        return []

    frame = dataframe.sort_values("lap_distance_pct", kind="stable").reset_index(drop=True)
    times_s = _numeric(frame, "timestamp_ms") / 1_000.0
    distance_pct = np.clip(_numeric(frame, "lap_distance_pct"), 0.0, 1.0)
    if "lap_distance_m" in frame.columns:
        distance_m = np.maximum(_numeric(frame, "lap_distance_m"), 0.0)
    else:
        distance_m = distance_pct * 1.0
    brake = np.clip(_numeric(frame, "brake"), 0.0, 1.0)
    throttle = np.clip(_numeric(frame, "throttle"), 0.0, 1.0)
    speed = np.maximum(_numeric(frame, "speed_kmh"), 0.0)

    raw_runs = _runs(brake >= float(brake_threshold))
    merged_runs = _merge_runs(raw_runs, times_s, float(merge_gap_s))
    braking_runs = [
        (start, end)
        for start, end in merged_runs
        if _run_duration_s(times_s, start, end) >= float(minimum_brake_duration_s)
    ]

    segments: list[BrakingSegment] = []
    for position, (start, end) in enumerate(braking_runs):
        next_start = (
            braking_runs[position + 1][0]
            if position + 1 < len(braking_runs)
            else len(frame)
        )
        maximum_end_time = times_s[end] + float(maximum_corner_window_s)
        time_limited_end = int(np.searchsorted(times_s, maximum_end_time, side="right") - 1)
        search_end = min(len(frame) - 1, next_start - 1, max(end, time_limited_end))

        peak_index = start + int(np.argmax(brake[start : end + 1]))
        minimum_speed_index = start + int(np.argmin(speed[start : search_end + 1]))

        full_throttle_index = _first_sustained_index(
            throttle >= float(full_throttle_threshold),
            times_s,
            minimum_speed_index,
            search_end,
            float(full_throttle_hold_s),
        )
        segment_end_index = (
            full_throttle_index if full_throttle_index is not None else search_end
        )

        segment = BrakingSegment(
            segment_id=position + 1,
            brake_start_index=start,
            brake_end_index=end,
            peak_brake_index=peak_index,
            minimum_speed_index=minimum_speed_index,
            full_throttle_index=full_throttle_index,
            segment_end_index=segment_end_index,
            brake_start_time_s=float(times_s[start]),
            brake_end_time_s=float(times_s[end]),
            brake_duration_s=_run_duration_s(times_s, start, end),
            brake_start_pct=float(distance_pct[start]),
            brake_end_pct=float(distance_pct[end]),
            brake_start_m=float(distance_m[start]),
            brake_end_m=float(distance_m[end]),
            peak_brake=float(brake[peak_index]),
            peak_brake_pct=float(distance_pct[peak_index]),
            peak_brake_m=float(distance_m[peak_index]),
            minimum_speed_kmh=float(speed[minimum_speed_index]),
            minimum_speed_pct=float(distance_pct[minimum_speed_index]),
            minimum_speed_m=float(distance_m[minimum_speed_index]),
            full_throttle_found=full_throttle_index is not None,
            full_throttle_time_s=(
                float(times_s[full_throttle_index])
                if full_throttle_index is not None
                else None
            ),
            full_throttle_pct=(
                float(distance_pct[full_throttle_index])
                if full_throttle_index is not None
                else None
            ),
            full_throttle_m=(
                float(distance_m[full_throttle_index])
                if full_throttle_index is not None
                else None
            ),
            segment_end_pct=float(distance_pct[segment_end_index]),
            segment_end_m=float(distance_m[segment_end_index]),
            segment_length_m=max(
                0.0, float(distance_m[segment_end_index] - distance_m[start])
            ),
        )
        segments.append(segment)

    return segments


def plot_braking_segments(
    dataframe: pd.DataFrame,
    segments: list[BrakingSegment],
    output_path: Path,
    *,
    title: str,
) -> None:
    """Save a compact visual inspection plot for one segmented lap."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distance_m = _numeric(dataframe, "lap_distance_m")
    use_metres = bool(
        np.isfinite(distance_m).any() and float(np.nanmax(distance_m)) > 1.0
    )
    if not use_metres:
        distance_m = _numeric(dataframe, "lap_distance_pct") * 100.0
        x_label = "Lap distance (%)"
    else:
        x_label = "Lap distance (m)"

    speed = _numeric(dataframe, "speed_kmh")
    throttle = _numeric(dataframe, "throttle") * 100.0
    brake = _numeric(dataframe, "brake") * 100.0

    figure, (speed_axis, controls_axis) = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
    )
    speed_axis.plot(distance_m, speed, label="Speed")
    speed_axis.set_ylabel("Speed (km/h)")
    speed_axis.grid(True, alpha=0.25)

    controls_axis.plot(distance_m, throttle, label="Throttle")
    controls_axis.plot(distance_m, brake, label="Brake")
    controls_axis.set_ylabel("Input (%)")
    controls_axis.set_xlabel(x_label)
    controls_axis.set_ylim(-2.0, 102.0)
    controls_axis.grid(True, alpha=0.25)
    controls_axis.legend(loc="upper right")

    for segment in segments:
        start_x = (
            segment.brake_start_m
            if use_metres
            else segment.brake_start_pct * 100.0
        )
        end_x = (
            segment.segment_end_m
            if use_metres
            else segment.segment_end_pct * 100.0
        )
        for axis in (speed_axis, controls_axis):
            axis.axvspan(start_x, end_x, alpha=0.10)
            axis.axvline(start_x, linestyle="--", alpha=0.55)
        speed_axis.text(
            start_x,
            max(10.0, float(np.nanmax(speed)) * 0.97),
            f"S{segment.segment_id}",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
        )

    speed_axis.set_title(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
