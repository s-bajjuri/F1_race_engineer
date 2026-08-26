from __future__ import annotations

import unittest

from f1_engineer.analysis.causal_segments import (
    LiveTelemetrySample,
    StreamingBrakeSegmenter,
)


def sample(t_ms: int, *, brake: float, throttle: float, speed: float) -> LiveTelemetrySample:
    return LiveTelemetrySample(
        timestamp_ms=t_ms,
        lap_distance_pct=t_ms / 10_000.0,
        lap_distance_m=t_ms / 10.0,
        speed_kmh=speed,
        throttle=throttle,
        brake=brake,
        steering=0.1,
        wheel_slip_fl=0.01,
        wheel_slip_fr=0.02,
        wheel_slip_rl=0.03,
        wheel_slip_rr=0.04,
    )


class CausalSegmentTests(unittest.TestCase):
    def test_confirms_without_lookahead_and_finishes_on_full_throttle(self) -> None:
        detector = StreamingBrakeSegmenter(
            minimum_brake_duration_s=0.50,
            merge_gap_s=0.20,
            full_throttle_hold_s=0.20,
        )
        completed = []
        # 60-ish Hz samples, with braking starting at 200 ms.
        for t in range(0, 1501, 20):
            if 200 <= t <= 800:
                brake = 0.8
                throttle = 0.0
                speed = 220.0 - (t - 200) * 0.18
            elif t >= 1060:
                brake = 0.0
                throttle = 1.0
                speed = 115.0 + (t - 1060) * 0.08
            else:
                brake = 0.0
                throttle = 0.35
                speed = 115.0
            completed.extend(detector.feed(sample(t, brake=brake, throttle=throttle, speed=speed)))

        self.assertEqual(len(completed), 1)
        result = completed[0]
        self.assertEqual(result.start_timestamp_ms, 200)
        self.assertGreaterEqual(result.brake_end_timestamp_ms, 780)
        self.assertTrue(result.full_throttle_recovered)
        self.assertAlmostEqual(result.peak_brake, 0.8, places=6)
        self.assertLess(result.minimum_speed_kmh, 130.0)

    def test_short_brake_pulse_never_becomes_segment(self) -> None:
        detector = StreamingBrakeSegmenter(minimum_brake_duration_s=0.50)
        completed = []
        for t in range(0, 1001, 20):
            brake = 0.9 if 200 <= t <= 400 else 0.0
            completed.extend(detector.feed(sample(t, brake=brake, throttle=0.2, speed=180.0)))
        self.assertEqual(completed, [])
        self.assertFalse(detector.active)


if __name__ == "__main__":
    unittest.main()
