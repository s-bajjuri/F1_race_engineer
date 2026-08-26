from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

# These fields represent states or identifiers. Nearest-neighbour selection is
# preferable to linear interpolation for them.
DEFAULT_DISCRETE_COLUMNS: frozenset[str] = frozenset(
    {
        "session_uid",
        "frame_identifier",
        "lap_number",
        "gear",
        "drs",
        "clutch",
        "car_position",
        "sector",
        "pit_status",
        "current_lap_invalid",
        "lap_time_ms",
    }
)

# Radian angles that wrap at +/-pi. Interpolating them linearly turns a small
# step across the wrap into a sweep through the entire range, which would make
# any derived heading or curvature meaningless. They are unwrapped before
# interpolation and wrapped back afterwards.
DEFAULT_ANGULAR_COLUMNS: frozenset[str] = frozenset({"yaw", "pitch", "roll"})


def wrap_to_pi(values: np.ndarray) -> np.ndarray:
    """Wrap radian angles into the half-open interval (-pi, pi]."""

    return (np.asarray(values, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _nearest_values(
    source_x: np.ndarray,
    source_values: np.ndarray,
    target_x: np.ndarray,
) -> np.ndarray:
    """Return nearest-neighbour values on ``target_x``.

    ``source_x`` must already be finite and strictly increasing.
    """

    if len(source_x) == 0:
        return np.full(len(target_x), np.nan)
    if len(source_x) == 1:
        return np.repeat(source_values[0], len(target_x))

    right = np.searchsorted(source_x, target_x, side="left")
    right = np.clip(right, 0, len(source_x) - 1)
    left = np.clip(right - 1, 0, len(source_x) - 1)

    use_right = np.abs(source_x[right] - target_x) < np.abs(
        target_x - source_x[left]
    )
    indices = np.where(use_right, right, left)
    return source_values[indices]


def _finite_numeric_pairs(
    distance: pd.Series,
    values: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(distance, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if not len(x):
        return x, y

    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    # Distance can repeat while the car is stationary or because packets share
    # the same rounded position. Keep the final state at each repeated distance.
    keep = np.ones(len(x), dtype=bool)
    keep[:-1] = x[:-1] != x[1:]
    return x[keep], y[keep]


def _first_non_null(series: pd.Series) -> object:
    values = series.dropna()
    return values.iloc[0] if not values.empty else None


def resample_lap_by_distance(
    dataframe: pd.DataFrame,
    *,
    points: int = 2_001,
    discrete_columns: Iterable[str] = DEFAULT_DISCRETE_COLUMNS,
    angular_columns: Iterable[str] = DEFAULT_ANGULAR_COLUMNS,
) -> pd.DataFrame:
    """Resample a complete lap onto a fixed 0-to-1 distance grid.

    Raw telemetry files are left untouched. Numeric telemetry is linearly
    interpolated, radian angles are interpolated on their unwrapped values,
    and identifiers and state fields use nearest-neighbour selection. String
    metadata is carried forward as a constant value.
    """

    if points < 2:
        raise ValueError("points must be at least 2")
    if "lap_distance_pct" not in dataframe.columns:
        raise ValueError("lap_distance_pct is required for distance resampling")
    if dataframe.empty:
        raise ValueError("cannot resample an empty lap")

    distance = pd.to_numeric(
        dataframe["lap_distance_pct"], errors="coerce"
    ).clip(0.0, 1.0)
    finite_distance = distance[np.isfinite(distance)]
    if len(finite_distance) < 2:
        raise ValueError("lap requires at least two finite distance samples")
    if float(finite_distance.max() - finite_distance.min()) < 0.90:
        raise ValueError("lap distance coverage is too small for resampling")

    target = np.linspace(0.0, 1.0, int(points), dtype=float)
    result: dict[str, object] = {
        "resample_grid_index": np.arange(points, dtype=np.int32),
        "lap_distance_pct": target,
    }

    discrete = set(discrete_columns)
    angular = set(angular_columns)

    for column in dataframe.columns:
        if column == "lap_distance_pct":
            continue

        series = dataframe[column]
        if pd.api.types.is_bool_dtype(series.dtype):
            x, y = _finite_numeric_pairs(distance, series.astype(float))
            sampled = _nearest_values(x, y, target)
            result[column] = sampled >= 0.5
            continue

        if pd.api.types.is_numeric_dtype(series.dtype) or column in discrete:
            x, y = _finite_numeric_pairs(distance, series)
            if not len(x):
                result[column] = np.full(points, np.nan)
            elif column in discrete:
                sampled = _nearest_values(x, y, target)
                if pd.api.types.is_integer_dtype(series.dtype):
                    sampled = np.rint(sampled)
                result[column] = sampled
            elif len(x) == 1:
                result[column] = np.repeat(y[0], points)
            elif column in angular:
                result[column] = wrap_to_pi(np.interp(target, x, np.unwrap(y)))
            else:
                result[column] = np.interp(target, x, y)
            continue

        # Text/categorical metadata such as track remains constant per lap.
        result[column] = np.repeat(_first_non_null(series), points)

    resampled = pd.DataFrame(result)

    # Keep a convenient and stable ordering near the front of the file.
    preferred = [
        "resample_grid_index",
        "session_uid",
        "track",
        "lap_number",
        "timestamp_ms",
        "lap_distance_m",
        "lap_distance_pct",
        "speed_kmh",
        "throttle",
        "brake",
        "gear",
        "steering",
    ]
    ordered = [column for column in preferred if column in resampled.columns]
    ordered.extend(column for column in resampled.columns if column not in ordered)
    return resampled.loc[:, ordered]
