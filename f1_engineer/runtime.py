from __future__ import annotations

import threading
from pathlib import Path

from f1_engineer.comparison.manager import LapComparisonManager
from f1_engineer.core.telemetry_buffer import TelemetryBuffer
from f1_engineer.health import evaluate_health
from f1_engineer.recording.recorder import LapRecorder
from f1_engineer.recording.writer import LapWriter
from f1_engineer.telemetry.receiver import UDPReceiver
from f1_engineer.ui.view_state import DashboardState


class AppRuntime:
    """Own the live services independently from the dashboard presentation."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        track: str,
        output_root: Path,
        buffer_seconds: float,
        reference_path: Path | None,
    ) -> None:
        self.host = host
        self.port = port
        self.track = track
        self.output_root = output_root
        self.raw_lap_dir = output_root / "raw_laps" / track
        self.raw_lap_dir.mkdir(parents=True, exist_ok=True)

        self.comparison = LapComparisonManager(raw_lap_dir=self.raw_lap_dir)
        self.comparison.load_reference(reference_path)

        self.writer = LapWriter(raw_lap_dir=self.raw_lap_dir)
        self.recorder = LapRecorder(
            track=track,
            output_root=output_root,
            writer=self.writer,
            comparison=self.comparison,
        )
        self.dashboard_buffer = TelemetryBuffer(
            maxlen=max(600, int(buffer_seconds * 75))
        )
        self.stop_event = threading.Event()
        self.receiver = UDPReceiver(
            host=host,
            port=port,
            dashboard_buffer=self.dashboard_buffer,
            recorder=self.recorder,
            stop_event=self.stop_event,
        )
        self._started = False
        self._stopped = False

    def start(self) -> None:
        if self._started:
            return
        self.writer.start()
        self.receiver.start()
        self._started = True

    def request_stop(self) -> None:
        self.stop_event.set()

    def snapshot(self) -> DashboardState:
        receiver = self.receiver.snapshot()
        recorder = self.recorder.snapshot()
        writer = self.writer.snapshot()
        comparison = self.comparison.snapshot()
        overall, items = evaluate_health(
            receiver=receiver,
            recorder=recorder,
            writer=writer,
            comparison=comparison,
            receiver_alive=self.receiver.is_alive(),
            writer_alive=self.writer.is_alive(),
        )
        return DashboardState(
            receiver=receiver,
            recorder=recorder,
            writer=writer,
            comparison=comparison,
            samples=self.dashboard_buffer.snapshot(),
            health_overall=overall,
            health_items=items,
        )

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self.stop_event.set()
        if self._started:
            self.receiver.join(timeout=2.0)
            self.writer.finish()
