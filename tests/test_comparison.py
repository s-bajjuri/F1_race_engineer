from __future__ import annotations

import unittest
from pathlib import Path

from f1_engineer.comparison.manager import LapComparisonManager


class ComparisonTests(unittest.TestCase):
    def test_faster_current_lap_has_negative_delta(self) -> None:
        rows = [
            {
                "timestamp_ms": i * 1_000,
                "lap_distance_m": i * 70.0,
                "lap_distance_pct": i / 100,
                "speed_kmh": 100.0 + i,
                "throttle": i / 100,
                "brake": 1.0 - i / 100,
                "lap_number": 7,
            }
            for i in range(101)
        ]

        manager = LapComparisonManager(Path("."))
        manager.set_reference_from_rows(
            rows,
            lap_num=7,
            official_lap_time_ms=100_000,
        )
        manager.start_lap(8, capture=True)

        for i in range(101):
            manager.append_row(
                {
                    "timestamp_ms": i * 990,
                    "lap_distance_m": i * 70.0,
                    "speed_kmh": 100.0 + i,
                    "throttle": i / 100,
                    "brake": 1.0 - i / 100,
                }
            )

        snapshot = manager.snapshot()
        self.assertTrue(snapshot.has_reference)
        self.assertEqual(snapshot.reference_lap_num, 7)
        self.assertEqual(snapshot.current_lap_num, 8)
        self.assertAlmostEqual(snapshot.latest_delta_s or 0.0, -1.0, places=3)


if __name__ == "__main__":
    unittest.main()
