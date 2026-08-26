from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from f1_engineer.modeling.style_envelope import (
    StyleEnvelope,
    StyleEnvelopeConfig,
    corner_dependence,
    describe_deviations,
    evaluate_envelope_transfer,
    summarise_transfer,
)

TARGET = "brake_release_smoothness_per_s"


def make_style_rows(
    *,
    track: str,
    corners: int = 8,
    laps: int = 8,
    tight_centre: float = 0.9,
    wide_centre: float = 1.4,
    spread: float = 0.3,
    seed: int = 0,
) -> pd.DataFrame:
    """Rows whose style descriptor depends on the corner band, not the track."""

    generator = np.random.default_rng(seed)
    records = []
    for corner in range(1, corners + 1):
        radius = 30.0 if corner <= corners // 2 else 200.0
        centre = tight_centre if radius < 60.0 else wide_centre
        for lap in range(laps):
            records.append(
                {
                    "track": track,
                    "lap_id": f"{track}_lap{lap:02d}",
                    "corner_key": f"{track}:{corner}",
                    "turn_direction": "right",
                    "ctx_minimum_radius_m": radius,
                    "behavior_complete": True,
                    "braking_complete": True,
                    TARGET: float(generator.normal(centre, spread)),
                    "steering_correction_count": float(
                        generator.normal(2.0, 0.8)
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


class BandTests(unittest.TestCase):
    def test_radius_maps_to_the_expected_band(self) -> None:
        config = StyleEnvelopeConfig()
        self.assertEqual(config.band_for(30.0), "tight")
        self.assertEqual(config.band_for(100.0), "medium")
        self.assertEqual(config.band_for(400.0), "fast")
        self.assertEqual(config.band_for(float("nan")), "unknown")

    def test_a_thin_band_falls_back_to_all_corners(self) -> None:
        # The synthetic corners are all 30 m or 200 m, so nothing lands in the
        # medium band and it has no evidence to stand on.
        frame = make_style_rows(track="a", corners=8, laps=8)
        envelope = StyleEnvelope(targets=(TARGET,)).fit(frame)

        _, _, medium_source = envelope.reference(TARGET, "medium")
        self.assertEqual(medium_source, "all")

        _, _, tight_source = envelope.reference(TARGET, "tight")
        self.assertEqual(tight_source, "tight")


class EnvelopeTests(unittest.TestCase):
    def test_each_band_gets_its_own_centre(self) -> None:
        frame = make_style_rows(track="a", tight_centre=0.9, wide_centre=1.4)
        envelope = StyleEnvelope(targets=(TARGET,)).fit(frame)

        tight_centre, _, tight_source = envelope.reference(TARGET, "tight")
        fast_centre, _, fast_source = envelope.reference(TARGET, "fast")
        self.assertEqual(tight_source, "tight")
        self.assertEqual(fast_source, "fast")
        self.assertAlmostEqual(tight_centre, 0.9, delta=0.15)
        self.assertAlmostEqual(fast_centre, 1.4, delta=0.15)

    def test_a_typical_lap_scores_near_zero(self) -> None:
        frame = make_style_rows(track="a", laps=12)
        envelope = StyleEnvelope(targets=(TARGET,)).fit(frame)
        scores = envelope.score(frame)
        z = scores[f"z_{TARGET}"].to_numpy(dtype=float)
        self.assertAlmostEqual(float(np.median(z)), 0.0, delta=0.2)
        self.assertGreater(float(np.mean(np.abs(z) <= 2.0)), 0.9)

    def test_an_abrupt_release_is_flagged(self) -> None:
        frame = make_style_rows(track="a", laps=12)
        envelope = StyleEnvelope(targets=(TARGET,)).fit(frame)

        odd = frame.iloc[[0]].copy()
        odd[TARGET] = 0.9 + 5.0 * 0.3
        scores = envelope.score(odd)
        self.assertGreater(float(scores[f"z_{TARGET}"].iloc[0]), 3.0)

        lines = describe_deviations(odd, scores, envelope)
        self.assertEqual(len(lines), 1)
        self.assertIn("more abrupt", lines[0])

    def test_the_band_is_used_rather_than_one_global_normal(self) -> None:
        """A wide corner value must be judged against wide-corner behaviour."""

        frame = make_style_rows(track="a", tight_centre=0.9, wide_centre=1.4, laps=12)
        envelope = StyleEnvelope(targets=(TARGET,)).fit(frame)

        typical_wide = frame[frame["ctx_minimum_radius_m"] > 60.0].iloc[[0]].copy()
        typical_wide[TARGET] = 1.4
        z = envelope.score(typical_wide)[f"z_{TARGET}"].iloc[0]
        # Against one pooled normal this would look high; against its own band
        # it is exactly typical.
        self.assertLess(abs(float(z)), 0.5)

    def test_saving_produces_loadable_json(self) -> None:
        frame = make_style_rows(track="a")
        envelope = StyleEnvelope(targets=(TARGET,)).fit(frame)
        with tempfile.TemporaryDirectory() as directory:
            path = envelope.save(Path(directory) / "envelope.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("bands", payload)
        self.assertIn("global", payload)
        self.assertTrue(any(entry["target"] == TARGET for entry in payload["global"]))


class TransferTests(unittest.TestCase):
    def test_an_envelope_transfers_when_style_is_track_independent(self) -> None:
        frame = pd.concat(
            [
                make_style_rows(track=name, seed=index)
                for index, name in enumerate(("a", "b", "c"))
            ],
            ignore_index=True,
        )
        results = evaluate_envelope_transfer(frame, targets=(TARGET,))
        summary = summarise_transfer(results)

        row = summary[summary["target"] == TARGET].iloc[0]
        self.assertTrue(bool(row["transfers"]))
        self.assertAlmostEqual(row["centre_z"], 0.0, delta=0.3)
        self.assertAlmostEqual(row["spread_z"], 1.0, delta=0.3)
        self.assertGreater(row["within_2sd"], 0.9)

    def test_a_track_specific_style_fails_the_calibration(self) -> None:
        """A descriptor that shifts per track must not be reported as transferring."""

        frame = pd.concat(
            [
                make_style_rows(track="a", tight_centre=0.9, wide_centre=1.4, seed=0),
                make_style_rows(track="b", tight_centre=0.9, wide_centre=1.4, seed=1),
                # Track c is wildly different from the others.
                make_style_rows(track="c", tight_centre=4.0, wide_centre=5.0, seed=2),
            ],
            ignore_index=True,
        )
        results = evaluate_envelope_transfer(frame, targets=(TARGET,))
        summary = summarise_transfer(results)

        row = summary[summary["target"] == TARGET].iloc[0]
        self.assertFalse(bool(row["transfers"]))
        self.assertLess(int(row["tracks_passing"]), int(row["tracks"]))

    def test_transfer_needs_more_than_one_track(self) -> None:
        frame = make_style_rows(track="a")
        results = evaluate_envelope_transfer(frame, targets=(TARGET,))
        self.assertIn("note", results.columns)


class CornerDependenceTests(unittest.TestCase):
    def test_a_corner_driven_the_same_way_every_lap_scores_high(self) -> None:
        frame = make_style_rows(track="a", spread=0.01, laps=10)
        dependence = corner_dependence(frame, targets=(TARGET,))
        self.assertGreater(float(dependence.iloc[0]["corner_share"]), 0.8)

    def test_a_purely_random_descriptor_scores_low(self) -> None:
        frame = make_style_rows(track="a", tight_centre=1.0, wide_centre=1.0, laps=10)
        dependence = corner_dependence(frame, targets=(TARGET,))
        self.assertLess(float(dependence.iloc[0]["corner_share"]), 0.3)


if __name__ == "__main__":
    unittest.main()