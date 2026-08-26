from __future__ import annotations

import queue
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd

from f1_engineer.models import LapWriteJob, WriterSnapshot

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

        # Geometry is optional so that laps recorded before it was captured
        # still save. Reported per lap so a silent regression is visible.
        geometry_columns = [
            "world_position_x",
            "world_position_y",
            "world_position_z",
            "yaw",
            "wheel_slip_angle_fl",
            "wheel_slip_angle_rl",
        ]
        if all(column in dataframe.columns for column in geometry_columns):
            geometry_rows = int(
                dataframe[geometry_columns].notna().all(axis=1).sum()
            )
            geometry_status = f"{geometry_rows}/{len(dataframe)}"
        else:
            geometry_status = "not recorded"

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
            f"Geometry/slip-angle rows: {geometry_status}\n"
            f"Incomplete frame groups dropped: "
            f"{job.dropped_incomplete_frames}\n"
            f"File: {path}\n"
        )
