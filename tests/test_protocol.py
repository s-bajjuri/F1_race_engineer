from __future__ import annotations

import unittest

from f1_engineer.telemetry.packets import (
    CAR_DAMAGE_STRUCT,
    CAR_MOTION_STRUCT,
    CAR_STATUS_STRUCT,
    CAR_TELEMETRY_STRUCT,
    HEADER_STRUCT,
    MOTION_EX_STRUCT,
)


class ProtocolStructureTests(unittest.TestCase):
    def test_known_structure_sizes(self) -> None:
        self.assertEqual(HEADER_STRUCT.size, 29)
        self.assertEqual(CAR_TELEMETRY_STRUCT.size, 60)
        self.assertEqual(CAR_MOTION_STRUCT.size, 60)
        self.assertEqual(CAR_STATUS_STRUCT.size, 55)
        self.assertEqual(CAR_DAMAGE_STRUCT.size, 46)
        self.assertEqual(MOTION_EX_STRUCT.size, 244)


if __name__ == "__main__":
    unittest.main()
