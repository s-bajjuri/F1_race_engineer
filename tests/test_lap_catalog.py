from __future__ import annotations

import unittest

import pandas as pd

from f1_engineer.analysis.catalog import apply_track_outliers


class CatalogOutlierTests(unittest.TestCase):
    def test_slower_than_120_percent_of_median_is_flagged(self) -> None:
        catalog = pd.DataFrame(
            {
                "track": ["spa", "spa", "spa"],
                "readable": [True, True, True],
                "complete": [True, True, True],
                "valid": [True, True, True],
                "lap_time_ms": [100_000, 101_000, 140_000],
                "training_eligible": [True, True, True],
                "exclusion_reasons": ["", "", ""],
                "outlier_lap_time": [False, False, False],
                "track_median_lap_time_ms": [None, None, None],
                "lap_time_ratio_to_track_median": [None, None, None],
            }
        )
        result = apply_track_outliers(catalog, outlier_factor=1.20)
        self.assertFalse(bool(result.loc[2, "training_eligible"]))
        self.assertTrue(bool(result.loc[2, "outlier_lap_time"]))
        self.assertIn("lap_time_outlier", result.loc[2, "exclusion_reasons"])
        self.assertTrue(bool(result.loc[0, "training_eligible"]))


if __name__ == "__main__":
    unittest.main()
