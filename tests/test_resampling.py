from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from f1_engineer.analysis.resampling import resample_lap_by_distance


class DistanceResamplingTests(unittest.TestCase):
    def test_resamples_to_fixed_grid_and_handles_repeated_distance(self) -> None:
        frame = pd.DataFrame(
            {
                "track": ["spa"] * 5,
                "lap_distance_pct": [0.0, 0.2, 0.2, 0.6, 1.0],
                "lap_distance_m": [0.0, 1_400.0, 1_400.0, 4_200.0, 7_000.0],
                "timestamp_ms": [0.0, 20_000.0, 20_050.0, 60_000.0, 100_000.0],
                "speed_kmh": [100.0, 200.0, 210.0, 250.0, 300.0],
                "throttle": [0.0, 0.5, 0.6, 0.8, 1.0],
                "brake": [1.0, 0.5, 0.4, 0.0, 0.0],
                "gear": [2, 4, 4, 6, 8],
                "steering": [0.2, 0.1, 0.1, 0.0, 0.0],
            }
        )

        result = resample_lap_by_distance(frame, points=11)

        self.assertEqual(len(result), 11)
        self.assertAlmostEqual(float(result["lap_distance_pct"].iloc[0]), 0.0)
        self.assertAlmostEqual(float(result["lap_distance_pct"].iloc[-1]), 1.0)
        self.assertTrue(result["lap_distance_pct"].is_monotonic_increasing)
        self.assertEqual(set(result["track"]), {"spa"})
        self.assertTrue(np.isfinite(result["speed_kmh"]).all())
        self.assertTrue(set(np.rint(result["gear"]).astype(int)).issubset({2, 4, 6, 8}))

    def test_rejects_insufficient_distance_coverage(self) -> None:
        frame = pd.DataFrame(
            {
                "lap_distance_pct": [0.2, 0.3, 0.4],
                "speed_kmh": [100.0, 120.0, 140.0],
            }
        )
        with self.assertRaises(ValueError):
            resample_lap_by_distance(frame, points=101)


if __name__ == "__main__":
    unittest.main()
