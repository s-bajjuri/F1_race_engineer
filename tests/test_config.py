from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from f1_engineer.config import load_config


class ConfigTests(unittest.TestCase):
    def test_yaml_values_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                """
app:
  track: monza
udp:
  host: 127.0.0.1
  port: 20778
recording:
  data_directory: telemetry_data
ui:
  dashboard_window_seconds: 18
  start_view: analysis
  drive_trace_seconds: 2.5
  tyre_warning_c: 104
  tyre_critical_c: 116
overlay:
  enabled: true
  visible_on_start: false
  width: 600
  height: 280
  x: 44
  y: 55
  opacity: 0.85
  click_through: false
  trace_seconds: 6.5
  refresh_hz: 25
""".strip(),
                encoding="utf-8",
            )
            config = load_config(path)

        self.assertEqual(config.track, "monza")
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 20778)
        self.assertEqual(config.output_root, Path("telemetry_data"))
        self.assertEqual(config.window_seconds, 18.0)
        self.assertEqual(config.start_view, "analysis")
        self.assertEqual(config.drive_trace_seconds, 2.5)
        self.assertEqual(config.tyre_warning_c, 104.0)
        self.assertEqual(config.tyre_critical_c, 116.0)
        self.assertTrue(config.overlay_enabled)
        self.assertFalse(config.overlay_visible_on_start)
        self.assertEqual(config.overlay_width, 600)
        self.assertEqual(config.overlay_height, 280)
        self.assertEqual(config.overlay_x, 44)
        self.assertEqual(config.overlay_y, 55)
        self.assertEqual(config.overlay_opacity, 0.85)
        self.assertFalse(config.overlay_click_through)
        self.assertEqual(config.overlay_trace_seconds, 6.5)
        self.assertEqual(config.overlay_refresh_hz, 25.0)


if __name__ == "__main__":
    unittest.main()
