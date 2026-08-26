from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_engineer.models import ComparisonSnapshot, ReferenceLap

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
