from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from f1_engineer.analysis.quality import (
    FeatureQualityConfig,
    write_feature_quality_outputs,
)


class FeatureQualityTests(unittest.TestCase):
    def _style_rows(self) -> pd.DataFrame:
        base = {
            "track": "spa",
            "source_file": "lap.parquet",
            "session_uid": 1,
            "lap_number": 1,
            "sample_rate_hz": 60.0,
            "training_eligible": True,
            "feature_complete": True,
            "braking_segment_count": 2,
            "brake_aggressiveness": 0.95,
            "brake_peak_consistency": 0.02,
            "mean_brake_force": 0.80,
            "mean_braking_distance_m": 120.0,
            "brake_release_smoothness": 0.70,
            "throttle_patience_m": 80.0,
            "throttle_patience_s": 2.0,
            "steering_smoothness": 0.20,
            "steering_rate_smoothness": 0.30,
            "mean_segment_minimum_speed_kmh": 110.0,
            "throttle_brake_overlap_pct": 0.02,
            "coast_time_pct": 0.01,
            "full_throttle_time_pct": 0.60,
            "hard_braking_time_pct": 0.08,
            "mean_speed_kmh": 250.0,
            "maximum_speed_kmh": 335.0,
            "minimum_speed_kmh": 65.0,
            "whole_lap_steering_input_std": 0.15,
            "whole_lap_steering_rate_std_per_s": 0.22,
            "mean_abs_lateral_g": 1.3,
            "peak_abs_lateral_g": 5.0,
            "mean_abs_longitudinal_g": 0.6,
            "peak_abs_longitudinal_g": 4.2,
            "slip_frequency": 0.10,
            "peak_front_slip_abs": 0.10,
            "peak_rear_slip_abs": 0.20,
            "mean_front_slip_abs": 0.003,
            "mean_rear_slip_abs": 0.022,
            "rear_slip_bias": 0.019,
            "fuel_used_kg": 0.0,
            "tyre_wear_delta": 0.0,
            "mean_tyre_temp_c": 93.0,
            "max_tyre_temp_c": 93.0,
            "mean_tyre_temp_spread_c": 0.0,
            "ers_total_deployed_j": 0.0,
            "ers_early_lap_pct": 0.0,
            "ers_middle_lap_pct": 0.0,
            "ers_late_lap_pct": 0.0,
        }
        static = {**base, "lap_id": "lap_static", "lap_time_ms": 108000}
        dynamic = {
            **base,
            "lap_id": "lap_dynamic",
            "lap_time_ms": 107000,
            "fuel_used_kg": 2.0,
            "tyre_wear_delta": 0.02,
            "mean_tyre_temp_c": 70.0,
            "max_tyre_temp_c": 120.0,
            "mean_tyre_temp_spread_c": 15.0,
            "ers_total_deployed_j": 1_500_000.0,
            "ers_early_lap_pct": 0.35,
            "ers_middle_lap_pct": 0.30,
            "ers_late_lap_pct": 0.35,
        }
        return pd.DataFrame([static, dynamic])

    def _segments(self) -> pd.DataFrame:
        rows = []
        for lap_id in ("lap_static", "lap_dynamic"):
            for segment_id, start in ((1, 100.0), (2, 500.0)):
                rows.append(
                    {
                        "lap_id": lap_id,
                        "track": "spa",
                        "segment_id": segment_id,
                        "brake_start_m": start + (2.0 if lap_id == "lap_dynamic" else 0.0),
                        "brake_end_m": start + 50.0,
                        "peak_brake": 1.0,
                        "minimum_speed_kmh": 100.0,
                        "minimum_speed_m": start + 80.0,
                        "segment_end_m": start + 150.0,
                        "braking_distance_m": 80.0,
                        "mean_brake": 0.8,
                        "brake_release_smoothness_per_s": 0.7,
                        "steering_input_std": 0.2,
                        "steering_rate_std_per_s": 0.3,
                        "full_throttle_found": True,
                    }
                )
        rows.append(
            {
                "lap_id": "lap_dynamic",
                "track": "spa",
                "segment_id": 3,
                "brake_start_m": 900.0,
                "brake_end_m": 930.0,
                "peak_brake": 0.6,
                "minimum_speed_kmh": 180.0,
                "minimum_speed_m": 940.0,
                "segment_end_m": 980.0,
                "braking_distance_m": 40.0,
                "mean_brake": 0.5,
                "brake_release_smoothness_per_s": 0.5,
                "steering_input_std": 0.1,
                "steering_rate_std_per_s": 0.2,
                "full_throttle_found": True,
            }
        )
        return pd.DataFrame(rows)

    def test_quality_outputs_find_context_and_rare_zone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = write_feature_quality_outputs(
                Path(directory),
                self._style_rows(),
                self._segments(),
                config=FeatureQualityConfig(
                    zone_gap_m=100.0,
                    canonical_zone_min_support=0.75,
                ),
            )
            self.assertEqual(summary["canonical_zone_count"], 2)
            self.assertEqual(summary["rare_zone_count"], 1)
            self.assertEqual(summary["rare_segment_rows"], 1)
            self.assertEqual(summary["style_model_ready_rows"], 2)
            self.assertEqual(summary["clean_baseline_candidate_rows"], 1)
            self.assertEqual(summary["full_context_model_ready_rows"], 1)
            self.assertEqual(
                summary["context_profiles"]["static_strategy_context"], 1
            )
            self.assertEqual(
                summary["context_profiles"]["dynamic_strategy_context"], 1
            )
            self.assertTrue((Path(directory) / "feature_quality_report.md").exists())
            self.assertTrue((Path(directory) / "style_features_validated.csv").exists())


if __name__ == "__main__":
    unittest.main()
