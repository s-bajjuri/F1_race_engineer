from __future__ import annotations

from dataclasses import dataclass

from f1_engineer.health import HealthItem
from f1_engineer.models import (
    ComparisonSnapshot,
    DashboardSample,
    ReceiverSnapshot,
    RecorderSnapshot,
    WriterSnapshot,
)


@dataclass(frozen=True, slots=True)
class DashboardState:
    receiver: ReceiverSnapshot
    recorder: RecorderSnapshot
    writer: WriterSnapshot
    comparison: ComparisonSnapshot
    samples: list[DashboardSample]
    health_overall: str
    health_items: list[HealthItem]
