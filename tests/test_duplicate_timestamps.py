from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from f1_engineer.analysis.validation import ValidationConfig, validate_lap_dataframe


def make_enriched_lap() -> pd.DataFrame:
    rows = 6_001
    timestamp_ms = np.linspace(0.0, 100_000.0, rows)
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
            "lap_time_ms": 100_000,
            "current_lap_invalid": False,
        }
    )
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


class DuplicateTimestampTests(unittest.TestCase):
    def test_repeated_f1_lap_time_is_warning_not_exclusion(self) -> None:
        frame = make_enriched_lap()
        frame.loc[100, "timestamp_ms"] = frame.loc[99, "timestamp_ms"]
        result = validate_lap_dataframe(
            frame,
            Path("data/raw_laps/spa/lap_duplicate_time_valid.parquet"),
            ValidationConfig(minimum_rows=100),
        )
        self.assertTrue(result["core_eligible"])
        self.assertIn("duplicate_timestamps", result["quality_warnings"])
        self.assertNotIn("duplicate_timestamps", result["exclusion_reasons"])


if __name__ == "__main__":
    unittest.main()
