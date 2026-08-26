from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from f1_engineer.analysis.validation import (
    MODEL_FEATURE_COLUMNS,
    ValidationConfig,
    validate_lap_dataframe,
)


def make_lap(
    *,
    rate_hz: float = 60.0,
    lap_seconds: float = 100.0,
    valid: bool = True,
    include_enrichment: bool = True,
) -> pd.DataFrame:
    rows = int(rate_hz * lap_seconds) + 1
    timestamp_ms = np.linspace(0.0, lap_seconds * 1_000.0, rows)
    distance_pct = np.linspace(0.0, 1.0, rows)
    frame = pd.DataFrame(
        {
            "session_uid": 123,
            "track": "spa",
            "lap_number": 2,
            "timestamp_ms": timestamp_ms,
            "lap_distance_pct": distance_pct,
            "lap_distance_m": distance_pct * 7_004.0,
            "speed_kmh": 200.0,
            "throttle": 0.8,
            "brake": 0.0,
            "gear": 6,
            "steering": 0.0,
            "rpm": 11_000.0,
            "drs": 0,
            "lap_time_ms": int(lap_seconds * 1_000),
            "current_lap_invalid": not valid,
        }
    )

    if include_enrichment:
        defaults = {
            "ers_deployed": 500_000.0,
            "fuel_remaining": 10.0,
            "fl_tyre_temp": 95.0,
            "fr_tyre_temp": 95.0,
            "rl_tyre_temp": 92.0,
            "rr_tyre_temp": 92.0,
            "fl_tyre_wear": 0.10,
            "fr_tyre_wear": 0.10,
            "rl_tyre_wear": 0.10,
            "rr_tyre_wear": 0.10,
            "g_force_lat": 0.0,
            "g_force_lon": 0.0,
            "wheel_slip_fl": 0.0,
            "wheel_slip_fr": 0.0,
            "wheel_slip_rl": 0.0,
            "wheel_slip_rr": 0.0,
        }
        for column, value in defaults.items():
            frame[column] = value

    return frame


class LapValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ValidationConfig(minimum_rows=100)

    def test_clean_enriched_60hz_lap_is_training_eligible(self) -> None:
        result = validate_lap_dataframe(
            make_lap(), Path("data/raw_laps/spa/lap_valid.parquet"), self.config
        )
        self.assertTrue(result["readable"])
        self.assertTrue(result["complete"])
        self.assertTrue(result["core_eligible"])
        self.assertTrue(result["training_eligible"])
        self.assertAlmostEqual(result["sample_rate_hz"], 60.0, places=1)

    def test_20hz_lap_is_kept_but_not_training_eligible(self) -> None:
        result = validate_lap_dataframe(
            make_lap(rate_hz=20.0),
            Path("data/raw_laps/spa/lap_20hz_valid.parquet"),
            self.config,
        )
        self.assertTrue(result["complete"])
        self.assertFalse(result["training_eligible"])
        self.assertIn("sample_rate_20hz", result["exclusion_reasons"])

    def test_invalid_lap_is_not_training_eligible(self) -> None:
        result = validate_lap_dataframe(
            make_lap(valid=False),
            Path("data/raw_laps/spa/lap_invalid.parquet"),
            self.config,
        )
        self.assertFalse(result["valid"])
        self.assertFalse(result["training_eligible"])
        self.assertIn("invalid_lap", result["exclusion_reasons"])

    def test_core_only_lap_remains_core_eligible(self) -> None:
        result = validate_lap_dataframe(
            make_lap(include_enrichment=False),
            Path("data/raw_laps/spa/lap_core_valid.parquet"),
            self.config,
        )
        self.assertTrue(result["core_eligible"])
        self.assertFalse(result["training_eligible"])
        self.assertIn("missing_model_columns", result["exclusion_reasons"])

    def test_required_model_columns_are_declared(self) -> None:
        self.assertIn("wheel_slip_rr", MODEL_FEATURE_COLUMNS)
        self.assertIn("ers_deployed", MODEL_FEATURE_COLUMNS)


class TimestampJitterTests(unittest.TestCase):
    """The game occasionally reports one stale currentLapTime. Rejecting the
    whole lap for a single sub-frame step discarded a fifth of the dataset."""

    def setUp(self) -> None:
        self.config = ValidationConfig(minimum_rows=100)

    def _lap_with_backsteps(self, offsets_ms: dict[int, float]) -> pd.DataFrame:
        frame = make_lap()
        timestamps = frame["timestamp_ms"].to_numpy(dtype=float).copy()
        for index, offset in offsets_ms.items():
            timestamps[index] += offset
        frame["timestamp_ms"] = timestamps
        return frame

    def test_single_sub_frame_backstep_stays_eligible(self) -> None:
        # One frame reported ~1.5 frames late, then corrected: a 25 ms step back.
        lap = self._lap_with_backsteps({2_000: 25.0})
        result = validate_lap_dataframe(
            lap, Path("data/raw_laps/spa/lap_jitter_valid.parquet"), self.config
        )

        self.assertFalse(result["timestamps_monotonic"])
        self.assertTrue(result["timestamps_acceptable"])
        self.assertEqual(result["timestamp_backward_steps"], 1)
        self.assertTrue(result["core_eligible"])
        self.assertTrue(result["training_eligible"])
        self.assertNotIn("timestamps_not_monotonic", result["exclusion_reasons"])
        self.assertIn("timestamp_jitter", result["quality_warnings"])

    def test_a_large_jump_back_still_disqualifies(self) -> None:
        # Seconds of rewind is a flashback or a lap-boundary error, not jitter.
        lap = self._lap_with_backsteps({2_000: 5_000.0})
        result = validate_lap_dataframe(
            lap, Path("data/raw_laps/spa/lap_rewind_valid.parquet"), self.config
        )

        self.assertFalse(result["timestamps_acceptable"])
        self.assertFalse(result["core_eligible"])
        self.assertIn("timestamps_not_monotonic", result["exclusion_reasons"])

    def test_many_small_backsteps_still_disqualify(self) -> None:
        lap = self._lap_with_backsteps(
            {index: 25.0 for index in (500, 1_000, 1_500, 2_000, 2_500)}
        )
        result = validate_lap_dataframe(
            lap, Path("data/raw_laps/spa/lap_noisy_valid.parquet"), self.config
        )

        self.assertGreater(
            result["timestamp_backward_steps"], self.config.maximum_timestamp_backsteps
        )
        self.assertFalse(result["timestamps_acceptable"])
        self.assertFalse(result["core_eligible"])


if __name__ == "__main__":
    unittest.main()
