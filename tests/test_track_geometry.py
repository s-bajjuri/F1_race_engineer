from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from f1_engineer.analysis.geometry import (
    GeometryConfig,
    _split_by_prominence,
    build_track_geometry,
)
from f1_engineer.analysis.resampling import resample_lap_by_distance


def build_oval_lap(
    *,
    straight_m: float = 500.0,
    radius_m: float = 150.0,
    spacing_m: float = 1.0,
) -> pd.DataFrame:
    """A synthetic oval: two straights joined by two 180 degree bends."""

    bend_m = math.pi * radius_m
    total_m = 2.0 * straight_m + 2.0 * bend_m
    distance = np.arange(0.0, total_m, spacing_m)

    x = np.zeros_like(distance)
    z = np.zeros_like(distance)
    steering = np.zeros_like(distance)
    speed = np.full_like(distance, 200.0)

    for index, s in enumerate(distance):
        if s < straight_m:
            x[index] = s
            z[index] = -radius_m
        elif s < straight_m + bend_m:
            angle = (s - straight_m) / radius_m
            x[index] = straight_m + radius_m * math.sin(angle)
            z[index] = -radius_m * math.cos(angle)
            steering[index] = 0.5
            speed[index] = 120.0
        elif s < 2.0 * straight_m + bend_m:
            x[index] = straight_m - (s - straight_m - bend_m)
            z[index] = radius_m
        else:
            angle = (s - 2.0 * straight_m - bend_m) / radius_m
            x[index] = -radius_m * math.sin(angle)
            z[index] = radius_m * math.cos(angle)
            steering[index] = 0.5
            speed[index] = 120.0

    return pd.DataFrame(
        {
            "lap_distance_m": distance,
            "world_position_x": x,
            "world_position_y": np.zeros_like(distance),
            "world_position_z": z,
            "yaw": np.zeros_like(distance),
            "steering": steering,
            "speed_kmh": speed,
        }
    )


class TrackGeometryTests(unittest.TestCase):
    def test_oval_yields_two_bends_with_the_expected_radius(self) -> None:
        lap = build_oval_lap()
        geometry = build_track_geometry(
            [lap], track="oval", config=GeometryConfig(grid_spacing_m=2.0)
        )

        self.assertEqual(len(geometry.corners), 2)
        expected_length = 2.0 * 500.0 + 2.0 * math.pi * 150.0
        self.assertAlmostEqual(
            geometry.summary["track_length_m"], expected_length, delta=5.0
        )

        for _, corner in geometry.corners.iterrows():
            self.assertAlmostEqual(corner["minimum_radius_m"], 150.0, delta=15.0)
            self.assertAlmostEqual(abs(corner["heading_change_deg"]), 180.0, delta=12.0)

        directions = set(geometry.corners["direction"])
        self.assertEqual(len(directions), 1, "both bends turn the same way")
        self.assertEqual(geometry.summary["curvature_sign_source"], "steering")

    def test_straights_are_not_reported_as_corners(self) -> None:
        lap = build_oval_lap()
        geometry = build_track_geometry(
            [lap], track="oval", config=GeometryConfig(grid_spacing_m=2.0)
        )
        in_corner = geometry.grid["corner_id"].to_numpy() > 0
        bend_fraction = 2.0 * math.pi * 150.0 / (2.0 * 500.0 + 2.0 * math.pi * 150.0)
        self.assertAlmostEqual(float(np.mean(in_corner)), bend_fraction, delta=0.08)


class CornerSplittingTests(unittest.TestCase):
    """The map follows the driven line, so a single corner has uneven
    curvature. Splitting must separate genuine apexes without shearing a
    decaying exit off the corner it belongs to."""

    def test_a_weak_decaying_tail_is_not_split_off(self) -> None:
        magnitude = np.concatenate(
            [
                np.linspace(0.5, 1.0, 10),
                np.linspace(1.0, 0.2, 10),
                np.linspace(0.2, 0.35, 7),
                np.linspace(0.35, 0.1, 10),
            ]
        )
        pieces = _split_by_prominence(magnitude, 5, 0.6, 0.5)
        self.assertEqual(len(pieces), 1)

    def test_two_comparable_apexes_are_split(self) -> None:
        magnitude = np.concatenate(
            [
                np.linspace(0.5, 1.0, 10),
                np.linspace(1.0, 0.2, 10),
                np.linspace(0.2, 0.9, 10),
                np.linspace(0.9, 0.3, 10),
            ]
        )
        pieces = _split_by_prominence(magnitude, 5, 0.6, 0.5)
        self.assertEqual(len(pieces), 2)

    def test_disabling_the_balance_rule_reproduces_the_over_split(self) -> None:
        magnitude = np.concatenate(
            [
                np.linspace(0.5, 1.0, 10),
                np.linspace(1.0, 0.2, 10),
                np.linspace(0.2, 0.35, 7),
                np.linspace(0.35, 0.1, 10),
            ]
        )
        pieces = _split_by_prominence(magnitude, 5, 0.6, 0.0)
        self.assertEqual(len(pieces), 2)


class YawResamplingTests(unittest.TestCase):
    def test_yaw_is_interpolated_across_the_pi_wrap(self) -> None:
        # Heading sweeping past pi wraps to -pi. Interpolating the wrapped
        # values would sweep back through zero and destroy any curvature.
        points = 400
        distance = np.linspace(0.0, 1.0, points)
        true_heading = np.linspace(math.pi - 0.4, math.pi + 0.4, points)
        wrapped = (true_heading + math.pi) % (2.0 * math.pi) - math.pi

        frame = pd.DataFrame(
            {
                "lap_distance_pct": distance,
                "timestamp_ms": np.linspace(0.0, 60_000.0, points),
                "speed_kmh": np.full(points, 150.0),
                "yaw": wrapped,
            }
        )
        result = resample_lap_by_distance(frame, points=201)
        yaw = result["yaw"].to_numpy(dtype=float)

        self.assertTrue(np.all(yaw >= -math.pi - 1e-9))
        self.assertTrue(np.all(yaw <= math.pi + 1e-9))

        steps = np.abs(np.diff(np.unwrap(yaw)))
        self.assertLess(
            float(np.max(steps)),
            0.05,
            "unwrapped heading must stay continuous across the wrap",
        )


if __name__ == "__main__":
    unittest.main()