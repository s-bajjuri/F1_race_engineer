from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from f1_engineer.analysis.modeling import (
    CONTEXT_FEATURE_COLUMNS,
    build_modeling_segment_table,
    extract_causal_pre_entry_context,
)


class Phase3FModelingTests(unittest.TestCase):
    def test_context_uses_only_pre_entry_window(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp_ms": np.arange(0, 2000, 100),
                "speed_kmh": np.linspace(200, 100, 20),
                "throttle": np.linspace(1.0, 0.0, 20),
                "steering": np.linspace(0.0, 0.5, 20),
                "g_force_lat": np.linspace(0.0, 2.0, 20),
                "g_force_lon": np.linspace(0.0, -2.0, 20),
            }
        )
        baseline = extract_causal_pre_entry_context(frame, 10, lookback_seconds=0.5)
        # Mutating data AFTER the start index must not change causal context.
        changed = frame.copy()
        changed.loc[11:, ["speed_kmh", "throttle", "steering", "g_force_lat", "g_force_lon"]] = 999.0
        mutated = extract_causal_pre_entry_context(changed, 10, lookback_seconds=0.5)
        for column in CONTEXT_FEATURE_COLUMNS:
            self.assertAlmostEqual(baseline[column], mutated[column], places=10)

    def test_track_and_zone_are_not_context_features(self) -> None:
        self.assertNotIn("track", CONTEXT_FEATURE_COLUMNS)
        self.assertNotIn("zone_id", CONTEXT_FEATURE_COLUMNS)
        self.assertNotIn("peak_brake", CONTEXT_FEATURE_COLUMNS)
        self.assertNotIn("minimum_speed_kmh", CONTEXT_FEATURE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
