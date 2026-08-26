from __future__ import annotations

import threading
from collections import deque

from f1_engineer.models import DashboardSample

class TelemetryBuffer:
    def __init__(self, maxlen: int) -> None:
        self._samples: deque[DashboardSample] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, sample: DashboardSample) -> None:
        with self._lock:
            self._samples.append(sample)

    def snapshot(self) -> list[DashboardSample]:
        with self._lock:
            return list(self._samples)
