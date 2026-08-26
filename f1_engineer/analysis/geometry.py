"""Track geometry maps derived from recorded Motion telemetry.

Braking segmentation describes what the driver did. This module describes what
the track is: a driver-independent map of heading, curvature and elevation
against ``lap_distance_m``, plus a corner table derived from it.

Two properties make the map reusable:

* ``lap_distance_m`` is measured along the track spline rather than along the
  driven line, so a map built from a handful of laps joins onto every other lap
  recorded at the same track, including laps captured before Motion fields were
  stored.
* The corner descriptors are geometric, so corners at different tracks can be
  compared with each other. That comparison is what makes few-shot coaching on
  an unseen track possible.

Geometry laps do not need to be fast or valid. An invalid or slow lap describes
the shape of the track just as well as a quick one.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_engineer.analysis.resampling import wrap_to_pi

# World position is required. Yaw is recorded alongside it and is kept as a
# cross-check on the path-derived heading.
REQUIRED_GEOMETRY_COLUMNS: tuple[str, ...] = (
    "lap_distance_m",
    "world_position_x",
    "world_position_y",
    "world_position_z",
)

OPTIONAL_GEOMETRY_COLUMNS: tuple[str, ...] = (
    "yaw",
    "steering",
    "speed_kmh",
)


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    """Thresholds for building a track map and detecting its corners."""

    grid_spacing_m: float = 2.0
    path_smoothing_m: float = 12.0
    heading_smoothing_m: float = 20.0
    corner_minimum_radius_m: float = 400.0
    minimum_corner_length_m: float = 25.0
    corner_merge_gap_m: float = 20.0
    corner_split_ratio: float = 0.6
    corner_split_balance_ratio: float = 0.5
    minimum_coverage_fraction: float = 0.95
    maximum_radius_m: float = 10_000.0

    @property
    def corner_curvature_threshold(self) -> float:
        return 1.0 / max(1.0, float(self.corner_minimum_radius_m))


@dataclass(frozen=True, slots=True)
class TrackGeometry:
    """A built track map: per-distance grid, corner table, and provenance."""

    track: str
    grid: pd.DataFrame
    corners: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)


def has_geometry(frame: pd.DataFrame) -> bool:
    """Report whether a lap carries the Motion fields the map needs."""

    return all(column in frame.columns for column in REQUIRED_GEOMETRY_COLUMNS)


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _monotonic_pairs(
    distance: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(distance) & np.isfinite(values)
    x = distance[valid]
    y = values[valid]
    if not len(x):
        return x, y
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    keep = np.ones(len(x), dtype=bool)
    keep[:-1] = x[:-1] != x[1:]
    return x[keep], y[keep]


def _interpolate_to_grid(
    distance: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
    *,
    angular: bool = False,
) -> np.ndarray:
    x, y = _monotonic_pairs(distance, values)
    if len(x) < 2:
        return np.full(len(grid), np.nan, dtype=float)
    if angular:
        return wrap_to_pi(np.interp(grid, x, np.unwrap(y)))
    return np.interp(grid, x, y)


def _window_samples(window_m: float, spacing_m: float) -> int:
    samples = int(round(float(window_m) / max(1e-6, float(spacing_m))))
    if samples < 3:
        return 0
    return samples if samples % 2 else samples + 1


def _circular_smooth(values: np.ndarray, samples: int) -> np.ndarray:
    """Moving average that treats the lap as the closed loop it is."""

    if samples < 3 or len(values) <= samples:
        return values
    kernel = np.ones(samples, dtype=float) / float(samples)
    padded = np.concatenate((values[-samples:], values, values[:samples]))
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[samples:-samples]


def _circular_smooth_angle(values: np.ndarray, samples: int) -> np.ndarray:
    if samples < 3:
        return values
    sin_part = _circular_smooth(np.sin(values), samples)
    cos_part = _circular_smooth(np.cos(values), samples)
    return np.arctan2(sin_part, cos_part)


def _circular_gradient(values: np.ndarray, spacing_m: float) -> np.ndarray:
    return (np.roll(values, -1) - np.roll(values, 1)) / (2.0 * float(spacing_m))


def _heading_gradient(heading: np.ndarray, spacing_m: float) -> np.ndarray:
    """Differentiate a wrapped heading without a discontinuity at the seam.

    Working from wrapped differences means the start/finish line is not a
    special place on the map.
    """

    forward = wrap_to_pi(np.roll(heading, -1) - heading)
    backward = wrap_to_pi(heading - np.roll(heading, 1))
    return (forward + backward) / (2.0 * float(spacing_m))


def _circular_indices(start: int, end: int, total: int) -> np.ndarray:
    return np.arange(start, end + 1) % total


def _circular_state_runs(state: np.ndarray) -> list[tuple[int, int, int]]:
    """Runs of equal non-zero state, joining a run that spans the seam.

    Turn direction is part of the state, so a chicane is never returned as one
    run: the sign change between the left and the right half always splits it.
    """

    total = len(state)
    if total == 0 or not np.any(state != 0):
        return []
    if np.all(state == state[0]):
        return [(0, total - 1, int(state[0]))]

    boundaries = np.flatnonzero(state != np.roll(state, 1))
    offset = int(boundaries[0])
    rotated = np.roll(state, -offset)

    runs: list[tuple[int, int, int]] = []
    begin = 0
    for index in range(1, total + 1):
        if index == total or rotated[index] != rotated[begin]:
            if rotated[begin] != 0:
                runs.append((begin + offset, index - 1 + offset, int(rotated[begin])))
            begin = index
    return runs


def _merge_signed_runs(
    runs: list[tuple[int, int, int]],
    total: int,
    gap_samples: int,
) -> list[tuple[int, int, int]]:
    """Close short sub-threshold dips inside a single corner.

    Only runs that turn the same way are merged, so this cannot fuse the two
    halves of a chicane.
    """

    if len(runs) < 2:
        return runs
    merged = [runs[0]]
    for start, end, sign in runs[1:]:
        previous_start, previous_end, previous_sign = merged[-1]
        if sign == previous_sign and (start - previous_end - 1) <= gap_samples:
            merged[-1] = (previous_start, end, sign)
        else:
            merged.append((start, end, sign))

    if len(merged) > 1:
        first_start, first_end, first_sign = merged[0]
        last_start, last_end, last_sign = merged[-1]
        gap = (first_start + total) - last_end - 1
        if first_sign == last_sign and gap <= gap_samples:
            merged = merged[1:-1]
            merged.insert(0, (last_start, first_end + total, first_sign))
    return merged


def _split_by_prominence(
    magnitude: np.ndarray,
    minimum_samples: int,
    split_ratio: float,
    balance_ratio: float,
) -> list[tuple[int, int]]:
    """Split one same-direction run into separate apexes.

    A long stretch of same-sign curvature can hold several distinct corners
    joined by a straighter link. Splitting where curvature falls well below
    both neighbouring peaks separates them; a stretch with one apex is
    returned untouched.

    Two conditions must both hold, because the map follows the driven line and
    a single long corner taken with a late apex has genuinely uneven curvature:

    * the valley must be deep relative to the peaks on either side, and
    * the two peaks must be comparable in size.

    The second condition is what stops a decaying corner exit being split off
    as a corner of its own. Parabolica is the standard example.
    """

    length = len(magnitude)
    if length < 2 * minimum_samples:
        return [(0, length - 1)]

    best_index: int | None = None
    best_depth = float("inf")
    for index in range(minimum_samples, length - minimum_samples):
        value = float(magnitude[index])
        if value > magnitude[index - 1] or value > magnitude[index + 1]:
            continue
        left_peak = float(np.max(magnitude[:index]))
        right_peak = float(np.max(magnitude[index + 1 :]))
        if value > split_ratio * min(left_peak, right_peak):
            continue
        weaker, stronger = sorted((left_peak, right_peak))
        if stronger <= 0.0 or weaker < balance_ratio * stronger:
            continue
        if value < best_depth:
            best_depth = value
            best_index = index

    if best_index is None:
        return [(0, length - 1)]

    left = _split_by_prominence(
        magnitude[:best_index], minimum_samples, split_ratio, balance_ratio
    )
    right = _split_by_prominence(
        magnitude[best_index:], minimum_samples, split_ratio, balance_ratio
    )
    return left + [(start + best_index, end + best_index) for start, end in right]


def _resolve_track_length_m(laps: Sequence[pd.DataFrame]) -> float:
    lengths: list[float] = []
    for lap in laps:
        distance = _numeric(lap, "lap_distance_m")
        finite = distance[np.isfinite(distance)]
        if finite.size:
            lengths.append(float(np.max(finite)))
    if not lengths:
        raise ValueError("no lap provides a usable lap_distance_m range")
    return float(np.median(lengths))


def _lap_coverage_fraction(lap: pd.DataFrame, track_length_m: float) -> float:
    distance = _numeric(lap, "lap_distance_m")
    finite = distance[np.isfinite(distance)]
    if finite.size < 2 or track_length_m <= 0:
        return 0.0
    return float((np.max(finite) - np.min(finite)) / track_length_m)


def _stack_laps_on_grid(
    laps: Sequence[pd.DataFrame],
    grid: np.ndarray,
    column: str,
    *,
    angular: bool = False,
) -> np.ndarray:
    rows = [
        _interpolate_to_grid(
            _numeric(lap, "lap_distance_m"),
            _numeric(lap, column),
            grid,
            angular=angular,
        )
        for lap in laps
    ]
    return np.vstack(rows) if rows else np.empty((0, len(grid)), dtype=float)


def _median_ignoring_nan(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0 or not np.isfinite(matrix).any():
        return np.full(matrix.shape[1] if matrix.ndim == 2 else 0, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmedian(matrix, axis=0)


def _circular_mean_ignoring_nan(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.full(matrix.shape[1] if matrix.ndim == 2 else 0, np.nan)
    with np.errstate(all="ignore"):
        sin_part = np.nanmean(np.sin(matrix), axis=0)
        cos_part = np.nanmean(np.cos(matrix), axis=0)
    return np.arctan2(sin_part, cos_part)


def _curvature_sign_is_right(
    curvature: np.ndarray,
    steering: np.ndarray,
    threshold: float,
) -> bool | None:
    """Decide whether positive curvature means a right-hand corner.

    The F1 UDP steering axis runs -1 full left to +1 full right, so the sign
    the driver used is the ground truth for labelling the map. Returning None
    means steering was unavailable and the caller should fall back.
    """

    mask = (
        np.isfinite(curvature)
        & np.isfinite(steering)
        & (np.abs(curvature) > threshold)
    )
    if not mask.any():
        return None
    agreement = float(np.sum(np.sign(curvature[mask]) * np.sign(steering[mask])))
    if agreement == 0.0:
        return None
    return agreement > 0.0


def build_track_geometry(
    laps: Sequence[pd.DataFrame],
    *,
    track: str,
    lap_ids: Sequence[str] | None = None,
    config: GeometryConfig | None = None,
) -> TrackGeometry:
    """Build a distance-indexed track map and corner table from Motion laps."""

    cfg = config or GeometryConfig()
    usable = [lap for lap in laps if has_geometry(lap) and len(lap) > 1]
    if not usable:
        raise ValueError(
            "no lap carries the Motion fields required for a geometry map: "
            + ", ".join(REQUIRED_GEOMETRY_COLUMNS)
        )
    names = list(lap_ids) if lap_ids is not None else [f"lap_{i}" for i in range(len(laps))]
    usable_names = [
        name for lap, name in zip(laps, names) if has_geometry(lap) and len(lap) > 1
    ]

    track_length_m = _resolve_track_length_m(usable)
    coverage = [_lap_coverage_fraction(lap, track_length_m) for lap in usable]
    keep = [value >= cfg.minimum_coverage_fraction for value in coverage]
    partial_laps = [name for name, ok in zip(usable_names, keep) if not ok]
    selected = [lap for lap, ok in zip(usable, keep) if ok]
    selected_names = [name for name, ok in zip(usable_names, keep) if ok]
    if not selected:
        raise ValueError(
            "every geometry lap falls below the minimum distance coverage of "
            f"{cfg.minimum_coverage_fraction:.0%}"
        )

    points = max(16, int(round(track_length_m / cfg.grid_spacing_m)))
    spacing_m = track_length_m / float(points)
    grid = np.arange(points, dtype=float) * spacing_m

    position_x = _median_ignoring_nan(
        _stack_laps_on_grid(selected, grid, "world_position_x")
    )
    position_y = _median_ignoring_nan(
        _stack_laps_on_grid(selected, grid, "world_position_y")
    )
    position_z = _median_ignoring_nan(
        _stack_laps_on_grid(selected, grid, "world_position_z")
    )
    recorded_yaw = _circular_mean_ignoring_nan(
        _stack_laps_on_grid(selected, grid, "yaw", angular=True)
    )
    steering = _median_ignoring_nan(_stack_laps_on_grid(selected, grid, "steering"))
    speed = _median_ignoring_nan(_stack_laps_on_grid(selected, grid, "speed_kmh"))

    path_samples = _window_samples(cfg.path_smoothing_m, spacing_m)
    heading_samples = _window_samples(cfg.heading_smoothing_m, spacing_m)

    smooth_x = _circular_smooth(position_x, path_samples)
    smooth_y = _circular_smooth(position_y, path_samples)
    smooth_z = _circular_smooth(position_z, path_samples)

    # Heading comes from the path tangent rather than from yaw, so it is not
    # offset by the car's slip angle. Recorded yaw is retained for comparison.
    heading = np.arctan2(
        _circular_gradient(smooth_x, spacing_m),
        _circular_gradient(smooth_z, spacing_m),
    )
    heading = _circular_smooth_angle(heading, heading_samples)
    curvature = _heading_gradient(heading, spacing_m)
    curvature = _circular_smooth(curvature, heading_samples)

    elevation_gradient = _circular_gradient(smooth_y, spacing_m)

    with np.errstate(divide="ignore", invalid="ignore"):
        radius = np.where(
            np.abs(curvature) > 1.0 / cfg.maximum_radius_m,
            1.0 / np.abs(curvature),
            cfg.maximum_radius_m,
        )

    positive_is_right = _curvature_sign_is_right(
        curvature, steering, cfg.corner_curvature_threshold
    )
    sign_source = "steering" if positive_is_right is not None else "assumed"
    if positive_is_right is None:
        positive_is_right = True

    corner_id = np.zeros(points, dtype=int)
    corner_records: list[dict[str, Any]] = []

    threshold = cfg.corner_curvature_threshold
    state = np.where(curvature >= threshold, 1, np.where(curvature <= -threshold, -1, 0))
    signed_runs = _merge_signed_runs(
        _circular_state_runs(state),
        points,
        max(1, int(round(cfg.corner_merge_gap_m / spacing_m))),
    )
    minimum_samples = max(1, int(round(cfg.minimum_corner_length_m / spacing_m)))

    runs: list[tuple[int, int]] = []
    for start, end, _sign in signed_runs:
        indices = _circular_indices(start, end, points)
        magnitude = np.abs(curvature[indices])
        for local_start, local_end in _split_by_prominence(
            magnitude,
            minimum_samples,
            float(cfg.corner_split_ratio),
            float(cfg.corner_split_balance_ratio),
        ):
            if (local_end - local_start + 1) < minimum_samples:
                continue
            runs.append((start + local_start, start + local_end))
    runs.sort(key=lambda run: run[0] % points)

    for position, (start, end) in enumerate(runs):
        indices = _circular_indices(start, end, points)
        corner_id[indices] = position + 1
        local_curvature = curvature[indices]
        apex_offset = int(np.argmax(np.abs(local_curvature)))
        apex_index = int(indices[apex_offset])
        mean_signed_curvature = float(np.mean(local_curvature))
        turns_right = (mean_signed_curvature > 0.0) == bool(positive_is_right)
        length_m = float(len(indices) * spacing_m)
        elevations = smooth_y[indices]

        corner_records.append(
            {
                "track": track,
                "corner_id": position + 1,
                "direction": "right" if turns_right else "left",
                "entry_m": float(grid[int(indices[0])]),
                "apex_m": float(grid[apex_index]),
                "exit_m": float(grid[int(indices[-1])]),
                "length_m": length_m,
                "entry_pct": float(grid[int(indices[0])] / track_length_m),
                "apex_pct": float(grid[apex_index] / track_length_m),
                "exit_pct": float(grid[int(indices[-1])] / track_length_m),
                "mean_abs_curvature_1pm": float(np.mean(np.abs(local_curvature))),
                "peak_abs_curvature_1pm": float(np.max(np.abs(local_curvature))),
                "minimum_radius_m": float(np.min(radius[indices])),
                "mean_radius_m": float(np.mean(radius[indices])),
                "heading_change_deg": float(
                    np.degrees(np.sum(local_curvature) * spacing_m)
                ),
                "entry_elevation_m": float(elevations[0]),
                "apex_elevation_m": float(smooth_y[apex_index]),
                "exit_elevation_m": float(elevations[-1]),
                "elevation_change_m": float(elevations[-1] - elevations[0]),
                "mean_gradient_pct": float(
                    100.0 * np.mean(elevation_gradient[indices])
                ),
                "apex_world_x": float(smooth_x[apex_index]),
                "apex_world_z": float(smooth_z[apex_index]),
                "reference_minimum_speed_kmh": (
                    float(np.nanmin(speed[indices]))
                    if np.isfinite(speed[indices]).any()
                    else float("nan")
                ),
            }
        )

    for position, record in enumerate(corner_records):
        previous = corner_records[position - 1]
        following = corner_records[(position + 1) % len(corner_records)]
        preceding_gap = record["entry_m"] - previous["exit_m"]
        following_gap = following["entry_m"] - record["exit_m"]
        record["preceding_straight_m"] = float(
            preceding_gap if preceding_gap >= 0 else preceding_gap + track_length_m
        )
        record["following_straight_m"] = float(
            following_gap if following_gap >= 0 else following_gap + track_length_m
        )

    grid_frame = pd.DataFrame(
        {
            "track": track,
            "grid_index": np.arange(points, dtype=np.int32),
            "lap_distance_m": grid,
            "lap_distance_pct": grid / track_length_m,
            "world_position_x": smooth_x,
            "world_position_y": smooth_y,
            "world_position_z": smooth_z,
            "elevation_m": smooth_y,
            "elevation_gradient_pct": 100.0 * elevation_gradient,
            "heading_rad": heading,
            "recorded_yaw_rad": recorded_yaw,
            "curvature_1pm": curvature,
            "radius_m": radius,
            "corner_id": corner_id,
            "reference_speed_kmh": speed,
            "reference_steering": steering,
        }
    )
    corners_frame = pd.DataFrame.from_records(corner_records)

    heading_agreement = float(
        np.degrees(
            np.nanmean(np.abs(wrap_to_pi(heading - recorded_yaw)))
        )
    ) if np.isfinite(recorded_yaw).any() else float("nan")

    summary: dict[str, Any] = {
        "track": track,
        "track_length_m": track_length_m,
        "grid_points": points,
        "grid_spacing_m": spacing_m,
        "laps_available": len(usable),
        "laps_used": len(selected),
        "laps_used_ids": selected_names,
        "laps_rejected_low_coverage": partial_laps,
        "corner_count": len(corner_records),
        "left_corners": sum(
            1 for record in corner_records if record["direction"] == "left"
        ),
        "right_corners": sum(
            1 for record in corner_records if record["direction"] == "right"
        ),
        "curvature_sign_convention": (
            "positive_is_right" if positive_is_right else "positive_is_left"
        ),
        "curvature_sign_source": sign_source,
        "mean_heading_vs_yaw_deg": heading_agreement,
        "config": {
            "grid_spacing_m": cfg.grid_spacing_m,
            "path_smoothing_m": cfg.path_smoothing_m,
            "heading_smoothing_m": cfg.heading_smoothing_m,
            "corner_minimum_radius_m": cfg.corner_minimum_radius_m,
            "minimum_corner_length_m": cfg.minimum_corner_length_m,
            "corner_merge_gap_m": cfg.corner_merge_gap_m,
        },
    }
    return TrackGeometry(
        track=track, grid=grid_frame, corners=corners_frame, summary=summary
    )


def attach_track_geometry(
    lap: pd.DataFrame,
    geometry: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Join a track map onto any lap recorded at that track.

    The join key is ``lap_distance_m``, which the game measures along the track
    spline. Laps recorded before Motion fields were stored therefore gain full
    geometry without being re-driven.
    """

    if "lap_distance_m" not in lap.columns:
        raise ValueError("lap requires lap_distance_m to attach geometry")
    if "lap_distance_m" not in geometry.columns:
        raise ValueError("geometry map requires lap_distance_m")

    default_columns = (
        "elevation_m",
        "elevation_gradient_pct",
        "heading_rad",
        "curvature_1pm",
        "radius_m",
        "corner_id",
    )
    wanted = [
        column
        for column in (columns if columns is not None else default_columns)
        if column in geometry.columns
    ]

    source_distance = pd.to_numeric(
        geometry["lap_distance_m"], errors="coerce"
    ).to_numpy(dtype=float)
    target_distance = pd.to_numeric(
        lap["lap_distance_m"], errors="coerce"
    ).to_numpy(dtype=float)

    track_length_m = float(np.nanmax(source_distance)) if len(source_distance) else 0.0
    indices = np.searchsorted(source_distance, target_distance, side="left")
    indices = np.clip(indices, 0, len(source_distance) - 1)
    left = np.clip(indices - 1, 0, len(source_distance) - 1)
    use_right = np.abs(source_distance[indices] - target_distance) < np.abs(
        target_distance - source_distance[left]
    )
    nearest = np.where(use_right, indices, left)
    nearest = np.where(np.isfinite(target_distance), nearest, 0)

    enriched = lap.copy()
    for column in wanted:
        values = geometry[column].to_numpy()
        assigned = values[nearest]
        if np.issubdtype(np.asarray(assigned).dtype, np.floating):
            assigned = np.where(np.isfinite(target_distance), assigned, np.nan)
        enriched[column] = assigned
    if track_length_m > 0:
        enriched["track_length_m"] = track_length_m
    return enriched


def track_output_dir(output_root: Path, track: str) -> Path:
    """Per-track subfolder for geometry outputs.

    Each track produces several files, so they are kept together rather than
    flattened into one directory.
    """

    return Path(output_root) / str(track).lower()


def write_track_geometry(output_root: Path, geometry: TrackGeometry) -> dict[str, str]:
    """Persist a built map. Raw laps are never modified."""

    track_root = track_output_dir(output_root, geometry.track)
    track_root.mkdir(parents=True, exist_ok=True)

    grid_path = track_root / f"{geometry.track}_geometry.parquet"
    grid_csv = track_root / f"{geometry.track}_geometry.csv"
    corners_path = track_root / f"{geometry.track}_corners.csv"
    summary_path = track_root / f"{geometry.track}_summary.json"

    geometry.grid.to_parquet(
        grid_path, engine="pyarrow", index=False, compression="zstd"
    )
    geometry.grid.to_csv(grid_csv, index=False)
    geometry.corners.to_csv(corners_path, index=False)
    summary_path.write_text(json.dumps(geometry.summary, indent=2), encoding="utf-8")

    return {
        "geometry_parquet": str(grid_path),
        "geometry_csv": str(grid_csv),
        "corners_csv": str(corners_path),
        "summary_json": str(summary_path),
    }


def plot_track_geometry(geometry: TrackGeometry, output_path: Path) -> None:
    """Save a map and curvature trace for visual confirmation."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = geometry.grid
    figure = plt.figure(figsize=(15, 7))
    map_axis = figure.add_subplot(1, 2, 1)
    curvature_axis = figure.add_subplot(2, 2, 2)
    elevation_axis = figure.add_subplot(2, 2, 4, sharex=curvature_axis)

    x = grid["world_position_x"].to_numpy(dtype=float)
    z = grid["world_position_z"].to_numpy(dtype=float)
    corner_ids = grid["corner_id"].to_numpy(dtype=int)

    map_axis.plot(x, z, color="0.75", linewidth=3.0, solid_capstyle="round")
    in_corner = corner_ids > 0
    map_axis.scatter(x[in_corner], z[in_corner], s=6.0, c=corner_ids[in_corner], cmap="turbo")
    for _, corner in geometry.corners.iterrows():
        map_axis.annotate(
            f"{int(corner['corner_id'])}{corner['direction'][0].upper()}",
            (corner["apex_world_x"], corner["apex_world_z"]),
            fontsize=8,
            ha="center",
            va="center",
        )
    map_axis.set_aspect("equal", adjustable="datalim")
    map_axis.set_title(f"{geometry.track.upper()} — {len(geometry.corners)} corners")
    map_axis.set_xlabel("world X (m)")
    map_axis.set_ylabel("world Z (m)")
    map_axis.grid(True, alpha=0.2)

    distance = grid["lap_distance_m"].to_numpy(dtype=float)
    curvature_axis.plot(distance, grid["curvature_1pm"].to_numpy(dtype=float))
    curvature_axis.axhline(0.0, color="0.6", linewidth=0.8)
    curvature_axis.set_ylabel("curvature (1/m)")
    curvature_axis.grid(True, alpha=0.25)
    for _, corner in geometry.corners.iterrows():
        if corner["entry_m"] <= corner["exit_m"]:
            curvature_axis.axvspan(corner["entry_m"], corner["exit_m"], alpha=0.12)

    elevation_axis.plot(distance, grid["elevation_m"].to_numpy(dtype=float))
    elevation_axis.set_ylabel("elevation (m)")
    elevation_axis.set_xlabel("lap distance (m)")
    elevation_axis.grid(True, alpha=0.25)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def audit_geometry_coverage(data_root: Path) -> pd.DataFrame:
    """Report, per track, how many raw laps carry Motion geometry fields."""

    import pyarrow.parquet as pq

    data_root = Path(data_root)
    records: list[dict[str, Any]] = []
    for lap_path in sorted(data_root.glob("*/*.parquet")):
        try:
            columns = set(pq.ParquetFile(lap_path).schema.names)
        except Exception:
            continue
        records.append(
            {
                "track": lap_path.parent.name.lower(),
                "lap_id": lap_path.stem,
                "has_geometry": all(
                    column in columns for column in REQUIRED_GEOMETRY_COLUMNS
                ),
            }
        )
    if not records:
        return pd.DataFrame(columns=["track", "laps", "with_geometry", "without_geometry"])

    frame = pd.DataFrame.from_records(records)
    summary = (
        frame.groupby("track")
        .agg(laps=("lap_id", "size"), with_geometry=("has_geometry", "sum"))
        .reset_index()
    )
    summary["without_geometry"] = summary["laps"] - summary["with_geometry"]
    return summary.sort_values("track").reset_index(drop=True)