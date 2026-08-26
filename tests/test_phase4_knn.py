from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from f1_engineer.analysis.corner_segments import GEOMETRY_CONTEXT_COLUMNS
from f1_engineer.modeling.evaluation import (
    EvaluationConfig,
    _blend,
    _select_adaptation_laps,
    evaluate_few_shot,
    evaluate_track_coverage,
)
from f1_engineer.modeling.knn import (
    ContextScaler,
    CornerBehaviorKNN,
    KNNConfig,
    TrackNormaliser,
    robust_scale,
    score_residuals,
)


def make_corner_rows(
    *,
    track: str,
    corners: int,
    laps: int,
    sessions: int = 1,
    radius_base: float = 50.0,
    behavior_from_radius=lambda radius: radius * 0.8,
    noise: float = 0.0,
) -> pd.DataFrame:
    """Corner rows whose behaviour is a clean function of geometry."""

    generator = np.random.default_rng(abs(hash(track)) % (2**31))
    records = []
    for corner in range(1, corners + 1):
        radius = radius_base * corner
        for lap in range(laps):
            record = {
                "track": track,
                "lap_id": f"{track}_lap{lap:02d}",
                "session_key": f"{track}:s{lap % max(1, sessions)}",
                "corner_key": f"{track}:{corner}",
                "behavior_complete": True,
                "braking_complete": True,
            }
            for column in GEOMETRY_CONTEXT_COLUMNS:
                record[column] = radius
            record["ctx_minimum_radius_m"] = radius
            value = behavior_from_radius(radius)
            if noise:
                value += float(generator.normal(0.0, noise))
            for target in ("minimum_speed_kmh", "peak_brake"):
                record[target] = value
            records.append(record)
    return pd.DataFrame.from_records(records)


class ContextScalerTests(unittest.TestCase):
    def test_constant_feature_does_not_divide_by_zero(self) -> None:
        frame = make_corner_rows(track="a", corners=3, laps=2)
        matrix = ContextScaler().fit_transform(frame)
        self.assertTrue(np.isfinite(matrix).all())

    def test_missing_context_is_filled_from_the_training_median(self) -> None:
        train = make_corner_rows(track="a", corners=4, laps=2)
        scaler = ContextScaler().fit(train)

        query = train.iloc[:1].copy()
        query["ctx_minimum_radius_m"] = np.nan
        transformed = scaler.transform(query)
        self.assertTrue(np.isfinite(transformed).all())


class KNNTests(unittest.TestCase):
    def test_geometry_predicts_behaviour_when_the_relationship_is_clean(self) -> None:
        train = make_corner_rows(track="a", corners=8, laps=3)
        test = make_corner_rows(track="b", corners=8, laps=1)

        model = CornerBehaviorKNN(
            targets=("minimum_speed_kmh",), config=KNNConfig(neighbours=3)
        ).fit(train)
        predictions = model.predict(test)

        actual = test["minimum_speed_kmh"].to_numpy(dtype=float)
        predicted = predictions["predicted_minimum_speed_kmh"].to_numpy(dtype=float)
        error = float(np.mean(np.abs(actual - predicted)))
        spread = float(np.mean(np.abs(actual - actual.mean())))
        self.assertLess(error, 0.35 * spread)

    def test_neighbours_with_a_missing_target_are_skipped(self) -> None:
        train = make_corner_rows(track="a", corners=6, laps=2)
        train.loc[train.index[:6], "peak_brake"] = np.nan
        test = make_corner_rows(track="b", corners=6, laps=1)

        model = CornerBehaviorKNN(
            targets=("minimum_speed_kmh", "peak_brake"), config=KNNConfig(neighbours=4)
        ).fit(train)
        predictions = model.predict(test)
        self.assertTrue(
            np.isfinite(predictions["predicted_peak_brake"].to_numpy(dtype=float)).any()
        )

    def test_explain_names_the_nearest_corners(self) -> None:
        train = make_corner_rows(track="a", corners=5, laps=2)
        test = make_corner_rows(track="b", corners=5, laps=1)
        model = CornerBehaviorKNN(targets=("minimum_speed_kmh",)).fit(train)

        nearest = model.explain(test, row=0, count=3)
        self.assertEqual(len(nearest), 3)
        self.assertIn("corner_key", nearest.columns)
        self.assertTrue(nearest["distance"].is_monotonic_increasing)


class ResidualTests(unittest.TestCase):
    def test_robust_scale_resists_a_single_wild_value(self) -> None:
        clean = np.array([10.0, 10.5, 9.5, 10.2, 9.8])
        contaminated = np.append(clean, 500.0)

        # Standard deviation is destroyed by the outlier; the MAD-based scale
        # stays the same order of magnitude, which is the point of using it.
        self.assertGreater(float(np.std(contaminated)), 100.0)
        self.assertLess(robust_scale(contaminated), 3.0 * robust_scale(clean))

    def test_residuals_are_expressed_in_deviations(self) -> None:
        actual = pd.Series([10.0, 12.0, 14.0, 10.0, 11.0])
        predicted = pd.Series([11.0] * 5)
        residuals = score_residuals(actual, predicted)
        self.assertEqual(len(residuals), 5)
        self.assertTrue(np.isfinite(residuals).all())
        self.assertGreater(residuals.iloc[2], residuals.iloc[0])


class TrackNormaliserTests(unittest.TestCase):
    def test_each_track_is_centred_on_its_own_level(self) -> None:
        frame = pd.concat(
            [
                make_corner_rows(track="a", corners=6, laps=3),
                make_corner_rows(
                    track="b",
                    corners=6,
                    laps=3,
                    behavior_from_radius=lambda radius: radius * 0.8 + 100.0,
                ),
            ],
            ignore_index=True,
        )
        normalised = TrackNormaliser(targets=("minimum_speed_kmh",)).fit(
            frame
        ).transform(frame)

        for track in ("a", "b"):
            values = normalised.loc[
                normalised["track"] == track, "minimum_speed_kmh"
            ].to_numpy(dtype=float)
            self.assertAlmostEqual(float(np.median(values)), 0.0, places=6)

    def test_an_unseen_track_falls_back_to_pooled_statistics(self) -> None:
        train = make_corner_rows(track="a", corners=6, laps=3)
        normaliser = TrackNormaliser(targets=("minimum_speed_kmh",)).fit(train)

        unseen = make_corner_rows(track="zzz", corners=6, laps=1)
        normalised = normaliser.transform(unseen)
        values = normalised["minimum_speed_kmh"].to_numpy(dtype=float)
        self.assertTrue(np.isfinite(values).all())

        location, scale = normaliser.pooled("minimum_speed_kmh")
        self.assertTrue(np.isfinite(location) and np.isfinite(scale))


class TrackCoverageTests(unittest.TestCase):
    def test_more_tracks_help_when_shapes_are_the_only_signal(self) -> None:
        frame = pd.concat(
            [
                make_corner_rows(
                    track=name, corners=5, laps=3, radius_base=base, noise=1.0
                )
                for name, base in (
                    ("a", 20.0),
                    ("b", 45.0),
                    ("c", 70.0),
                    ("d", 95.0),
                )
            ],
            ignore_index=True,
        )
        coverage = evaluate_track_coverage(
            frame,
            ("minimum_speed_kmh",),
            config=EvaluationConfig(minimum_reference_rows=5),
        )
        trend = coverage.groupby("training_tracks")["skill"].mean().sort_index()
        self.assertGreater(trend.iloc[-1], trend.iloc[0])

    def test_a_curve_needs_at_least_three_tracks(self) -> None:
        frame = pd.concat(
            [
                make_corner_rows(track="a", corners=4, laps=2),
                make_corner_rows(track="b", corners=4, laps=2),
            ],
            ignore_index=True,
        )
        coverage = evaluate_track_coverage(
            frame, ("minimum_speed_kmh",), config=EvaluationConfig()
        )
        self.assertIn("note", coverage.columns)


class BlendTests(unittest.TestCase):
    def test_no_observations_leaves_the_prior_untouched(self) -> None:
        blended = _blend(
            np.array([5.0, 7.0]), np.array([np.nan, np.nan]), np.array([0.0, 0.0]), 3.0
        )
        np.testing.assert_allclose(blended, [5.0, 7.0])

    def test_evidence_pulls_the_prediction_toward_the_driver(self) -> None:
        prior = np.array([10.0])
        own = np.array([20.0])
        one_lap = _blend(prior, own, np.array([1.0]), 3.0)[0]
        five_laps = _blend(prior, own, np.array([5.0]), 3.0)[0]

        self.assertLess(one_lap, five_laps)
        self.assertAlmostEqual(one_lap, (1 * 20 + 3 * 10) / 4)
        self.assertAlmostEqual(five_laps, (5 * 20 + 3 * 10) / 8)

    def test_zero_strength_uses_the_driver_where_they_have_data(self) -> None:
        blended = _blend(
            np.array([10.0, 10.0]),
            np.array([20.0, np.nan]),
            np.array([1.0, 0.0]),
            0.0,
        )
        np.testing.assert_allclose(blended, [20.0, 10.0])


class AdaptationSplitTests(unittest.TestCase):
    def test_practice_and_test_laps_never_overlap(self) -> None:
        frame = make_corner_rows(track="a", corners=4, laps=8, sessions=2)
        adaptation, test, _ = _select_adaptation_laps(frame, shots=3, repeat=0)
        self.assertEqual(set(adaptation) & set(test), set())
        self.assertEqual(len(adaptation), 3)

    def test_multi_session_tracks_avoid_sharing_a_session(self) -> None:
        frame = make_corner_rows(track="a", corners=4, laps=8, sessions=2)
        adaptation, test, leakage = _select_adaptation_laps(frame, shots=2, repeat=0)
        self.assertFalse(leakage)

        sessions = frame.drop_duplicates("lap_id").set_index("lap_id")["session_key"]
        self.assertEqual(
            set(sessions.loc[adaptation]) & set(sessions.loc[test]), set()
        )

    def test_single_session_tracks_report_the_leakage(self) -> None:
        frame = make_corner_rows(track="a", corners=4, laps=6, sessions=1)
        _, _, leakage = _select_adaptation_laps(frame, shots=2, repeat=0)
        self.assertTrue(leakage)


class FewShotTests(unittest.TestCase):
    def test_practice_laps_are_kept_out_of_the_neighbour_pool(self) -> None:
        """The geometry score must not improve as practice laps are added.

        If it does, the practice laps have leaked into the prior and the model
        is being credited for geometry when it is really recognising the same
        corner.
        """

        frame = pd.concat(
            [
                make_corner_rows(track="a", corners=6, laps=6, sessions=2, noise=1.0),
                make_corner_rows(track="b", corners=6, laps=6, sessions=2, noise=1.0),
            ],
            ignore_index=True,
        )
        results = evaluate_few_shot(
            frame,
            ("minimum_speed_kmh",),
            config=EvaluationConfig(
                shots=(0, 1, 3), repeats=1, minimum_reference_rows=5
            ),
        )
        for track, subset in results.groupby("unseen_track"):
            sizes = subset["prior_reference_rows"].unique()
            self.assertEqual(
                len(sizes),
                1,
                f"{track}: prior pool changed size with shots, so practice laps leaked in",
            )

    def test_more_practice_laps_improve_the_blended_prediction(self) -> None:
        frame = pd.concat(
            [
                make_corner_rows(track="a", corners=6, laps=8, sessions=2, noise=1.0),
                make_corner_rows(
                    track="b",
                    corners=6,
                    laps=8,
                    sessions=2,
                    noise=1.0,
                    behavior_from_radius=lambda radius: radius * 0.8 + 40.0,
                ),
            ],
            ignore_index=True,
        )
        results = evaluate_few_shot(
            frame,
            ("minimum_speed_kmh",),
            config=EvaluationConfig(
                shots=(0, 1, 5),
                repeats=2,
                minimum_reference_rows=5,
                prior_strengths=(1.0,),
            ),
        )
        by_shots = results.groupby("shots")["blended_mae"].mean()
        # Track b is offset from track a, so geometry alone cannot know it and
        # the practice laps have to carry the correction.
        self.assertLess(by_shots.loc[5], by_shots.loc[1])
        self.assertLess(by_shots.loc[1], by_shots.loc[0])


if __name__ == "__main__":
    unittest.main()