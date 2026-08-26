from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from f1_engineer.analysis.corner_segments import (
    CornerSegmentConfig,
    build_corner_windows,
    extract_corner_behavior,
)


def build_braking_lap() -> tuple[pd.DataFrame, dict[str, float]]:
    """A 60 Hz lap with one braking event and a long progressive release.

    Brake ramps up from 6.0 s, peaks at 6.5 s, is held to 8.0 s, then bleeds off
    to zero by 9.8 s. It falls below 0.30 at roughly 9.1 s, so a release
    measured only down to the firm threshold would miss the final second.
    """

    dt = 1.0 / 60.0
    times = np.arange(0.0, 18.0, dt)

    speed_kmh = np.full_like(times, 250.0)
    speed_kmh = np.where((times >= 6.0) & (times < 9.0), 250.0 - (times - 6.0) * 56.67, speed_kmh)
    speed_kmh = np.where((times >= 9.0) & (times < 10.5), 80.0, speed_kmh)
    speed_kmh = np.where(times >= 10.5, np.minimum(250.0, 80.0 + (times - 10.5) * 37.8), speed_kmh)

    distance = np.cumsum(speed_kmh / 3.6 * dt)

    brake = np.zeros_like(times)
    brake = np.where((times >= 6.0) & (times < 6.5), (times - 6.0) / 0.5, brake)
    brake = np.where((times >= 6.5) & (times < 8.0), 1.0, brake)
    brake = np.where((times >= 8.0) & (times < 9.8), 1.0 - (times - 8.0) / 1.8, brake)
    brake = np.clip(brake, 0.0, 1.0)

    throttle = np.where(times >= 10.5, np.minimum(1.0, (times - 10.5) / 0.8), 0.0)
    throttle = np.where(times < 6.0, 1.0, throttle)

    steering = np.where(
        (times >= 7.0) & (times <= 11.5),
        0.6 * np.sin(np.pi * (times - 7.0) / 4.5),
        0.0,
    )

    lap = pd.DataFrame(
        {
            "timestamp_ms": times * 1_000.0,
            "lap_distance_m": distance,
            "speed_kmh": speed_kmh,
            "brake": brake,
            "throttle": throttle,
            "steering": steering,
            "g_force_lat": steering * 3.0,
            "g_force_lon": np.gradient(speed_kmh / 3.6, dt) / 9.80665,
        }
    )

    def distance_at(second: float) -> float:
        return float(np.interp(second, times, distance))

    marks = {
        "brake_touch": distance_at(6.0),
        "hard_brake": distance_at(6.16),
        "entry": distance_at(7.5),
        "apex": distance_at(9.4),
        "exit": distance_at(11.0),
    }
    return lap, marks


class CornerWindowTests(unittest.TestCase):
    def test_context_is_geometry_only(self) -> None:
        corners = pd.DataFrame(
            [
                {
                    "corner_id": 1,
                    "direction": "right",
                    "entry_m": 500.0,
                    "apex_m": 560.0,
                    "exit_m": 640.0,
                    "length_m": 140.0,
                    "minimum_radius_m": 80.0,
                    "mean_radius_m": 120.0,
                    "heading_change_deg": -95.0,
                    "elevation_change_m": -3.0,
                    "mean_gradient_pct": -2.0,
                    "preceding_straight_m": 900.0,
                    "following_straight_m": 400.0,
                }
            ]
        )
        grid = pd.DataFrame(
            {
                "lap_distance_m": np.arange(0.0, 1_200.0, 2.0),
                "radius_m": np.full(600, 100.0),
            }
        )
        windows = build_corner_windows(corners, grid, track="test")

        self.assertEqual(len(windows), 1)
        window = windows[0]
        self.assertEqual(window.corner_key, "test:1")
        # Heading change is unsigned so a left and a right corner of the same
        # shape match each other; direction stays as metadata.
        self.assertAlmostEqual(window.context["ctx_abs_heading_change_deg"], 95.0)
        self.assertEqual(window.direction, "right")

        # The approach is capped by the straight in front of the corner.
        self.assertAlmostEqual(window.approach_start_m, 500.0 - 300.0)
        self.assertAlmostEqual(window.window_end_m, 640.0 + 200.0)

    def test_short_straight_shrinks_the_window(self) -> None:
        corners = pd.DataFrame(
            [
                {
                    "corner_id": 2,
                    "direction": "left",
                    "entry_m": 300.0,
                    "apex_m": 320.0,
                    "exit_m": 350.0,
                    "length_m": 50.0,
                    "minimum_radius_m": 30.0,
                    "mean_radius_m": 40.0,
                    "heading_change_deg": 70.0,
                    "elevation_change_m": 0.0,
                    "mean_gradient_pct": 0.0,
                    "preceding_straight_m": 20.0,
                    "following_straight_m": 10.0,
                }
            ]
        )
        grid = pd.DataFrame(
            {"lap_distance_m": np.arange(0.0, 600.0, 2.0), "radius_m": np.full(300, 40.0)}
        )
        window = build_corner_windows(corners, grid, track="test")[0]
        self.assertAlmostEqual(window.approach_start_m, 300.0 - 16.0)
        self.assertAlmostEqual(window.window_end_m, 350.0 + 8.0)


class CornerBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lap, self.marks = build_braking_lap()
        corners = pd.DataFrame(
            [
                {
                    "corner_id": 1,
                    "direction": "right",
                    "entry_m": self.marks["entry"],
                    "apex_m": self.marks["apex"],
                    "exit_m": self.marks["exit"],
                    "length_m": self.marks["exit"] - self.marks["entry"],
                    "minimum_radius_m": 90.0,
                    "mean_radius_m": 130.0,
                    "heading_change_deg": -90.0,
                    "elevation_change_m": 0.0,
                    "mean_gradient_pct": 0.0,
                    "preceding_straight_m": 800.0,
                    "following_straight_m": 800.0,
                }
            ]
        )
        grid = pd.DataFrame(
            {
                "lap_distance_m": np.arange(0.0, 2_000.0, 2.0),
                "radius_m": np.full(1_000, 120.0),
            }
        )
        self.window = build_corner_windows(corners, grid, track="test")[0]

    def test_braking_point_uses_first_pressure_not_the_firm_threshold(self) -> None:
        result = extract_corner_behavior(self.lap, self.window)

        self.assertTrue(result["brake_detected"])
        expected = self.window.entry_m - self.marks["brake_touch"]
        self.assertAlmostEqual(
            result["brake_application_before_entry_m"], expected, delta=6.0
        )
        # The firm threshold is reached later, so it reports a shorter distance.
        # Measuring from it would make a progressive application look late.
        self.assertLess(
            result["hard_brake_before_entry_m"],
            result["brake_application_before_entry_m"],
        )

    def test_release_is_measured_all_the_way_to_zero_pedal(self) -> None:
        result = extract_corner_behavior(self.lap, self.window)

        # Peak is at 8.0 s and the pedal reaches zero at 9.8 s. Stopping at the
        # 0.30 threshold near 9.1 s would report roughly 1.1 s instead.
        self.assertGreater(result["brake_release_duration_s"], 1.5)
        self.assertFalse(result["release_truncated"])
        self.assertTrue(np.isfinite(result["brake_release_smoothness_per_s"]))
        # A perfectly linear release has a near-constant slope.
        self.assertLess(result["brake_release_smoothness_per_s"], 0.5)
        self.assertAlmostEqual(result["brake_release_rate_per_s"], -1.0 / 1.8, delta=0.15)

    def test_apex_speed_is_searched_inside_the_corner(self) -> None:
        result = extract_corner_behavior(self.lap, self.window)

        self.assertAlmostEqual(result["minimum_speed_kmh"], 80.0, delta=2.0)
        self.assertAlmostEqual(result["peak_brake"], 1.0, delta=0.01)
        self.assertTrue(result["behavior_complete"])
        self.assertTrue(result["braking_complete"])

    def test_behaviour_is_measured_at_the_raw_sample_rate(self) -> None:
        result = extract_corner_behavior(self.lap, self.window)
        self.assertAlmostEqual(result["window_sample_rate_hz"], 60.0, delta=1.0)

    def test_a_corner_taken_flat_is_complete_without_braking(self) -> None:
        flat = self.lap.copy()
        flat["brake"] = 0.0
        flat["throttle"] = 1.0
        result = extract_corner_behavior(flat, self.window)

        self.assertFalse(result["brake_detected"])
        self.assertFalse(result["braking_complete"])
        # Missing brake data at a flat-out corner is the correct answer, not a
        # gap, so the row still counts as behaviourally complete.
        self.assertTrue(result["behavior_complete"])

    def test_steering_corrections_are_counted(self) -> None:
        result = extract_corner_behavior(self.lap, self.window)
        self.assertEqual(result["steering_correction_count"], 1)

        wobbly = self.lap.copy()
        times = wobbly["timestamp_ms"].to_numpy(dtype=float) / 1_000.0
        wobbly["steering"] = wobbly["steering"].to_numpy(dtype=float) + np.where(
            (times >= 8.0) & (times <= 10.0), 0.2 * np.sin(times * 12.0), 0.0
        )
        noisy = extract_corner_behavior(wobbly, self.window)
        self.assertGreater(
            noisy["steering_correction_count"], result["steering_correction_count"]
        )


if __name__ == "__main__":
    unittest.main()