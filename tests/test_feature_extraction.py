from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from f1_engineer.analysis.features import (
    extract_lap_style_features,
    extract_segment_features,
)
from f1_engineer.analysis.segmentation import detect_braking_segments


class FeatureExtractionTests(unittest.TestCase):
    def make_lap(self) -> pd.DataFrame:
        points = 2001
        distance = np.linspace(0.0, 1.0, points)
        time_s = np.linspace(0.0, 100.0, points)
        brake = np.zeros(points)
        throttle = np.ones(points)
        speed = np.full(points, 250.0)
        steering = np.zeros(points)

        event = (distance >= 0.20) & (distance <= 0.24)
        corner = (distance >= 0.20) & (distance <= 0.30)
        brake[event] = 0.8
        throttle[corner] = 0.0
        speed[corner] = np.interp(
            distance[corner],
            [0.20, 0.25, 0.30],
            [250.0, 90.0, 220.0],
        )
        steering[corner] = np.sin(np.linspace(0.0, np.pi, int(corner.sum()))) * 0.5
        throttle[(distance >= 0.28) & (distance < 0.30)] = 0.5
        throttle[distance >= 0.30] = 1.0

        return pd.DataFrame(
            {
                "timestamp_ms": time_s * 1000.0,
                "lap_distance_pct": distance,
                "lap_distance_m": distance * 7000.0,
                "speed_kmh": speed,
                "throttle": throttle,
                "brake": brake,
                "gear": 6,
                "steering": steering,
                "rpm": 11000.0,
                "drs": 0,
                "ers_deployed": distance * 2_000_000.0,
                "fuel_remaining": 20.0 - distance * 2.0,
                "fl_tyre_temp": 90.0 + distance,
                "fr_tyre_temp": 91.0 + distance,
                "rl_tyre_temp": 92.0 + distance,
                "rr_tyre_temp": 93.0 + distance,
                "fl_tyre_wear": 0.10 + distance * 0.01,
                "fr_tyre_wear": 0.10 + distance * 0.01,
                "rl_tyre_wear": 0.11 + distance * 0.01,
                "rr_tyre_wear": 0.11 + distance * 0.01,
                "g_force_lat": steering * 5.0,
                "g_force_lon": -brake * 4.0 + throttle * 1.0,
                "wheel_slip_fl": -brake * 0.02,
                "wheel_slip_fr": -brake * 0.03,
                "wheel_slip_rl": throttle * 0.01,
                "wheel_slip_rr": throttle * 0.02,
                "lap_time_ms": 100_000,
                "current_lap_invalid": False,
            }
        )

    def test_segment_features_are_finite(self) -> None:
        lap = self.make_lap()
        segments = detect_braking_segments(lap)
        self.assertEqual(len(segments), 1)
        features = extract_segment_features(lap, segments[0])
        self.assertGreater(features["braking_distance_m"], 0.0)
        self.assertGreater(features["mean_brake"], 0.0)
        self.assertTrue(np.isfinite(features["steering_input_std"]))
        self.assertTrue(np.isfinite(features["mean_tyre_temp_c"]))

    def test_lap_style_features_create_one_row(self) -> None:
        lap = self.make_lap()
        segments = detect_braking_segments(lap)
        segment_features = [
            {**segment.to_record(), **extract_segment_features(lap, segment)}
            for segment in segments
        ]
        result = extract_lap_style_features(
            lap,
            segments,
            segment_features,
            lap_id="synthetic",
            track="spa",
            source_file="synthetic.parquet",
            catalog_row={
                "lap_time_ms": 100_000,
                "sample_rate_hz": 20.0,
                "training_eligible": True,
            },
        )
        self.assertEqual(result["lap_id"], "synthetic")
        self.assertEqual(result["braking_segment_count"], 1)
        self.assertAlmostEqual(result["fuel_used_kg"], 2.0, places=4)
        self.assertAlmostEqual(result["tyre_wear_delta"], 0.01, places=4)
        self.assertAlmostEqual(result["ers_total_deployed_j"], 2_000_000.0, places=0)
        self.assertTrue(result["feature_complete"])


if __name__ == "__main__":
    unittest.main()
