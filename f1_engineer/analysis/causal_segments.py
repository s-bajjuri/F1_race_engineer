from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import pstdev
from typing import Literal


@dataclass(frozen=True, slots=True)
class LiveTelemetrySample:
    timestamp_ms: int
    lap_distance_pct: float
    lap_distance_m: float
    speed_kmh: float
    throttle: float
    brake: float
    steering: float = 0.0
    g_force_lat: float = 0.0
    g_force_lon: float = 0.0
    wheel_slip_fl: float = 0.0
    wheel_slip_fr: float = 0.0
    wheel_slip_rl: float = 0.0
    wheel_slip_rr: float = 0.0


@dataclass(frozen=True, slots=True)
class LiveSegmentSnapshot:
    state: str
    confirmed: bool
    start_timestamp_ms: int | None
    elapsed_s: float
    sample_count: int
    peak_brake: float
    minimum_speed_kmh: float
    mean_brake: float
    peak_abs_steering: float
    peak_front_slip_abs: float
    peak_rear_slip_abs: float


@dataclass(frozen=True, slots=True)
class CompletedLiveSegment:
    start_timestamp_ms: int
    brake_end_timestamp_ms: int
    end_timestamp_ms: int
    start_distance_pct: float
    start_distance_m: float
    end_distance_pct: float
    end_distance_m: float
    duration_s: float
    peak_brake: float
    mean_brake: float
    minimum_speed_kmh: float
    steering_std: float
    peak_abs_steering: float
    peak_front_slip_abs: float
    peak_rear_slip_abs: float
    full_throttle_recovered: bool


class StreamingBrakeSegmenter:
    """Causal companion to the offline braking-segment detector.

    It never looks ahead. A brake event is confirmed only after the configured
    minimum duration, with the logical start backdated to the first buffered
    threshold crossing. Running features are available through ``snapshot``.
    The segment finishes after sustained full throttle, the next distinct brake
    event, or a maximum recovery window.

    This state machine is intentionally braking-based only. Phase 3F does not
    introduce a hand-tuned maneuver taxonomy.
    """

    def __init__(
        self,
        *,
        brake_threshold: float = 0.30,
        minimum_brake_duration_s: float = 0.50,
        merge_gap_s: float = 0.20,
        full_throttle_threshold: float = 0.95,
        full_throttle_hold_s: float = 0.20,
        maximum_corner_window_s: float = 10.0,
    ) -> None:
        self.brake_threshold = float(brake_threshold)
        self.minimum_brake_duration_ms = int(round(float(minimum_brake_duration_s) * 1000))
        self.merge_gap_ms = int(round(float(merge_gap_s) * 1000))
        self.full_throttle_threshold = float(full_throttle_threshold)
        self.full_throttle_hold_ms = int(round(float(full_throttle_hold_s) * 1000))
        self.maximum_corner_window_ms = int(round(float(maximum_corner_window_s) * 1000))
        self.reset()

    def reset(self) -> None:
        self.state: Literal["idle", "candidate", "braking", "recovery"] = "idle"
        self._candidate: list[LiveTelemetrySample] = []
        self._active: list[LiveTelemetrySample] = []
        self._last_above_brake: LiveTelemetrySample | None = None
        self._full_throttle_start: LiveTelemetrySample | None = None
        self._brake_end: LiveTelemetrySample | None = None

    @property
    def active(self) -> bool:
        return self.state in {"braking", "recovery"}

    def _elapsed_ms(self, first: LiveTelemetrySample, last: LiveTelemetrySample) -> int:
        return max(0, int(last.timestamp_ms) - int(first.timestamp_ms))

    def snapshot(self) -> LiveSegmentSnapshot:
        samples = self._active if self._active else self._candidate
        if not samples:
            return LiveSegmentSnapshot(
                state=self.state,
                confirmed=False,
                start_timestamp_ms=None,
                elapsed_s=0.0,
                sample_count=0,
                peak_brake=0.0,
                minimum_speed_kmh=math.nan,
                mean_brake=0.0,
                peak_abs_steering=0.0,
                peak_front_slip_abs=0.0,
                peak_rear_slip_abs=0.0,
            )
        start = samples[0]
        end = samples[-1]
        brakes = [max(0.0, min(1.0, sample.brake)) for sample in samples]
        front_slip = [
            max(abs(sample.wheel_slip_fl), abs(sample.wheel_slip_fr))
            for sample in samples
        ]
        rear_slip = [
            max(abs(sample.wheel_slip_rl), abs(sample.wheel_slip_rr))
            for sample in samples
        ]
        return LiveSegmentSnapshot(
            state=self.state,
            confirmed=self.active,
            start_timestamp_ms=int(start.timestamp_ms),
            elapsed_s=self._elapsed_ms(start, end) / 1000.0,
            sample_count=len(samples),
            peak_brake=max(brakes, default=0.0),
            minimum_speed_kmh=min((sample.speed_kmh for sample in samples), default=math.nan),
            mean_brake=(sum(brakes) / len(brakes)) if brakes else 0.0,
            peak_abs_steering=max((abs(sample.steering) for sample in samples), default=0.0),
            peak_front_slip_abs=max(front_slip, default=0.0),
            peak_rear_slip_abs=max(rear_slip, default=0.0),
        )

    def _complete(self, *, end_sample: LiveTelemetrySample, full_throttle: bool) -> CompletedLiveSegment:
        samples = list(self._active)
        start = samples[0]
        brake_end = self._brake_end or self._last_above_brake or end_sample
        brakes = [max(0.0, min(1.0, sample.brake)) for sample in samples]
        steering = [sample.steering for sample in samples]
        result = CompletedLiveSegment(
            start_timestamp_ms=int(start.timestamp_ms),
            brake_end_timestamp_ms=int(brake_end.timestamp_ms),
            end_timestamp_ms=int(end_sample.timestamp_ms),
            start_distance_pct=float(start.lap_distance_pct),
            start_distance_m=float(start.lap_distance_m),
            end_distance_pct=float(end_sample.lap_distance_pct),
            end_distance_m=float(end_sample.lap_distance_m),
            duration_s=self._elapsed_ms(start, end_sample) / 1000.0,
            peak_brake=max(brakes, default=0.0),
            mean_brake=(sum(brakes) / len(brakes)) if brakes else 0.0,
            minimum_speed_kmh=min((sample.speed_kmh for sample in samples), default=math.nan),
            steering_std=pstdev(steering) if len(steering) > 1 else 0.0,
            peak_abs_steering=max((abs(value) for value in steering), default=0.0),
            peak_front_slip_abs=max(
                (max(abs(s.wheel_slip_fl), abs(s.wheel_slip_fr)) for s in samples),
                default=0.0,
            ),
            peak_rear_slip_abs=max(
                (max(abs(s.wheel_slip_rl), abs(s.wheel_slip_rr)) for s in samples),
                default=0.0,
            ),
            full_throttle_recovered=bool(full_throttle),
        )
        self.reset()
        return result

    def feed(self, sample: LiveTelemetrySample) -> list[CompletedLiveSegment]:
        completed: list[CompletedLiveSegment] = []
        above_brake = float(sample.brake) >= self.brake_threshold

        if self.state == "idle":
            if above_brake:
                self.state = "candidate"
                self._candidate = [sample]
            return completed

        if self.state == "candidate":
            if not above_brake:
                self.reset()
                return completed
            self._candidate.append(sample)
            if self._elapsed_ms(self._candidate[0], sample) >= self.minimum_brake_duration_ms:
                self.state = "braking"
                self._active = list(self._candidate)
                self._candidate = []
                self._last_above_brake = sample
            return completed

        if self.state == "braking":
            self._active.append(sample)
            if above_brake:
                self._last_above_brake = sample
                return completed

            if self._last_above_brake is not None:
                if self._elapsed_ms(self._last_above_brake, sample) >= self.merge_gap_ms:
                    self.state = "recovery"
                    self._brake_end = self._last_above_brake
                    if sample.throttle >= self.full_throttle_threshold:
                        self._full_throttle_start = sample
            return completed

        # recovery
        if above_brake:
            # The gap has already exceeded merge_gap_s, therefore this is a new
            # braking event. Finish the previous maneuver causally at the sample
            # just before this new threshold crossing, then seed a new candidate.
            previous_end = self._active[-1]
            completed.append(self._complete(end_sample=previous_end, full_throttle=False))
            self.state = "candidate"
            self._candidate = [sample]
            return completed

        self._active.append(sample)
        if sample.throttle >= self.full_throttle_threshold:
            if self._full_throttle_start is None:
                self._full_throttle_start = sample
            elif self._elapsed_ms(self._full_throttle_start, sample) >= self.full_throttle_hold_ms:
                completed.append(
                    self._complete(end_sample=self._full_throttle_start, full_throttle=True)
                )
                return completed
        else:
            self._full_throttle_start = None

        if self._brake_end is not None:
            if self._elapsed_ms(self._brake_end, sample) >= self.maximum_corner_window_ms:
                completed.append(self._complete(end_sample=sample, full_throttle=False))
        return completed

    def flush(self) -> CompletedLiveSegment | None:
        """Finalize a confirmed active segment at session/lap shutdown."""
        if not self._active:
            self.reset()
            return None
        return self._complete(end_sample=self._active[-1], full_throttle=False)
