from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from f1_engineer.analysis.segmentation import detect_braking_segments


class BrakingSegmentationTests(unittest.TestCase):
    def make_lap(self) -> pd.DataFrame:
        rate_hz = 60.0
        duration_s = 20.0
        rows = int(rate_hz * duration_s) + 1
        time_s = np.linspace(0.0, duration_s, rows)
        distance_pct = time_s / duration_s

        brake = np.zeros(rows)
        throttle = np.ones(rows)
        speed = np.full(rows, 260.0)

        first_brake = (time_s >= 2.0) & (time_s <= 3.0)
        brake[first_brake] = 0.8
        throttle[(time_s >= 2.0) & (time_s < 5.0)] = 0.0
        speed[(time_s >= 2.0) & (time_s <= 4.0)] = np.linspace(
            260.0, 90.0, int(((time_s >= 2.0) & (time_s <= 4.0)).sum())
        )
        speed[(time_s > 4.0) & (time_s <= 5.0)] = np.linspace(
            90.0, 180.0, int(((time_s > 4.0) & (time_s <= 5.0)).sum())
        )

        second_brake = (time_s >= 10.0) & (time_s <= 10.8)
        brake[second_brake] = 0.7
        throttle[(time_s >= 10.0) & (time_s < 12.0)] = 0.0
        speed[(time_s >= 10.0) & (time_s <= 11.0)] = np.linspace(
            260.0, 120.0, int(((time_s >= 10.0) & (time_s <= 11.0)).sum())
        )

        # Too short to qualify as a braking segment.
        brake[(time_s >= 16.0) & (time_s <= 16.2)] = 0.9

        return pd.DataFrame(
            {
                "timestamp_ms": time_s * 1_000.0,
                "lap_distance_pct": distance_pct,
                "lap_distance_m": distance_pct * 7_000.0,
                "speed_kmh": speed,
                "throttle": throttle,
                "brake": brake,
            }
        )

    def test_detects_only_sustained_braking_events(self) -> None:
        segments = detect_braking_segments(self.make_lap())
        self.assertEqual(len(segments), 2)
        self.assertGreaterEqual(segments[0].brake_duration_s, 0.5)
        self.assertAlmostEqual(segments[0].brake_start_time_s, 2.0, places=1)
        self.assertLess(segments[0].minimum_speed_kmh, 100.0)
        self.assertTrue(segments[0].full_throttle_found)
        self.assertGreater(segments[0].segment_end_m, segments[0].brake_start_m)


if __name__ == "__main__":
    unittest.main()
