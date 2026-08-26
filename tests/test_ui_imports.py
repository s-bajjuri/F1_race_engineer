from __future__ import annotations

import unittest


class UIImportTests(unittest.TestCase):
    def test_ui_modules_import(self) -> None:
        from f1_engineer.ui.analysis_view import AnalysisView
        from f1_engineer.ui.dashboard import DashboardController
        from f1_engineer.ui.diagnostics_view import DiagnosticsView
        from f1_engineer.ui.drive_overlay import DriveOverlay
        from f1_engineer.ui.drive_view import DriveView

        self.assertIsNotNone(AnalysisView)
        self.assertIsNotNone(DashboardController)
        self.assertIsNotNone(DiagnosticsView)
        self.assertIsNotNone(DriveOverlay)
        self.assertIsNotNone(DriveView)


if __name__ == "__main__":
    unittest.main()
