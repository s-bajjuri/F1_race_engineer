"""k-NN from corner geometry to this driver's behaviour.

The query is what a corner *is* — radius, length, elevation, the straights
either side. The answer is what this driver *did* there. Because the query
carries no track identity, a corner on a track never driven before can be
matched against geometrically similar corners the driver already knows, which
is the mechanism few-shot coaching depends on.

Deliberately interpretable: neighbours can be named, and a prediction can be
explained as "corners like this one, which you have driven, you take like so".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from f1_engineer.analysis.corner_segments import GEOMETRY_CONTEXT_COLUMNS

# Behaviour every corner has, including ones taken flat.
PRIMARY_TARGETS: tuple[str, ...] = (
    "minimum_speed_kmh",
    "minimum_speed_offset_from_apex_m",
    "peak_abs_steering",
    "steering_rate_std_per_s",
    "mean_abs_lateral_g",
)

# Behaviour that only exists when the driver braked.
BRAKING_TARGETS: tuple[str, ...] = (
    "brake_application_before_entry_m",
    "peak_brake",
    "entry_speed_kmh",
    "speed_loss_kmh",
    "brake_release_smoothness_per_s",
    "trail_brake_past_entry_m",
)

# Lengths and radii span two orders of magnitude, from a 12 m hairpin to a
# 900 m straight. Compressing them first stops the widest features dominating
# the distance and makes similarity proportional rather than absolute: 30 m
# versus 60 m is a bigger difference in kind than 800 m versus 830 m.
LOG_SCALE_CONTEXT: frozenset[str] = frozenset(
    {
        "ctx_corner_length_m",
        "ctx_minimum_radius_m",
        "ctx_mean_radius_m",
        "ctx_entry_radius_m",
        "ctx_exit_radius_m",
        "ctx_preceding_straight_m",
        "ctx_following_straight_m",
        "ctx_preceding_corner_minimum_radius_m",
        "ctx_following_corner_minimum_radius_m",
    }
)


@dataclass
class ContextScaler:
    """Standardise geometry context, fitted on the training pool only."""

    columns: tuple[str, ...] = GEOMETRY_CONTEXT_COLUMNS
    log_columns: frozenset[str] = LOG_SCALE_CONTEXT
    centre_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    fill_: np.ndarray | None = None

    def _raw_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        columns = []
        for column in self.columns:
            values = (
                pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
                if column in frame.columns
                else np.full(len(frame), np.nan)
            )
            if column in self.log_columns:
                values = np.log1p(np.clip(values, 0.0, None))
            columns.append(values)
        return np.column_stack(columns) if columns else np.empty((len(frame), 0))

    def fit(self, frame: pd.DataFrame) -> "ContextScaler":
        matrix = self._raw_matrix(frame)
        with np.errstate(all="ignore"):
            self.fill_ = np.nanmedian(matrix, axis=0)
            self.centre_ = np.nanmean(matrix, axis=0)
            scale = np.nanstd(matrix, axis=0)
        self.fill_ = np.nan_to_num(self.fill_, nan=0.0)
        self.centre_ = np.nan_to_num(self.centre_, nan=0.0)
        # A constant feature carries no information; leaving its scale at zero
        # would divide by zero rather than simply contributing nothing.
        self.scale_ = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.centre_ is None or self.scale_ is None or self.fill_ is None:
            raise ValueError("scaler must be fitted before transforming")
        matrix = self._raw_matrix(frame)
        filled = np.where(np.isfinite(matrix), matrix, self.fill_)
        return (filled - self.centre_) / self.scale_

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


@dataclass(frozen=True, slots=True)
class KNNConfig:
    neighbours: int = 10
    distance_epsilon: float = 1e-3
    minimum_neighbours: int = 2


@dataclass
class CornerBehaviorKNN:
    """Predict behaviour targets from geometry by distance-weighted averaging."""

    targets: tuple[str, ...]
    config: KNNConfig = field(default_factory=KNNConfig)
    scaler: ContextScaler = field(default_factory=ContextScaler)
    reference_context_: np.ndarray | None = None
    reference_targets_: np.ndarray | None = None
    reference_labels_: list[str] = field(default_factory=list)
    fallback_: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "CornerBehaviorKNN":
        self.reference_context_ = self.scaler.fit_transform(frame)
        self.reference_targets_ = np.column_stack(
            [
                pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
                if column in frame.columns
                else np.full(len(frame), np.nan)
                for column in self.targets
            ]
        )
        labels = (
            frame["corner_key"].astype(str).tolist()
            if "corner_key" in frame.columns
            else [str(index) for index in range(len(frame))]
        )
        self.reference_labels_ = labels
        with np.errstate(all="ignore"):
            fallback = np.nanmean(self.reference_targets_, axis=0)
        self.fallback_ = np.nan_to_num(fallback, nan=0.0)
        return self

    @property
    def reference_size(self) -> int:
        return 0 if self.reference_context_ is None else len(self.reference_context_)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict every target, ignoring neighbours whose value is missing.

        A neighbour that was taken flat has no braking figures. Dropping it per
        target rather than per row keeps the usable part of every neighbour.
        """

        if self.reference_context_ is None or self.reference_targets_ is None:
            raise ValueError("model must be fitted before predicting")
        if self.reference_size == 0:
            raise ValueError("model was fitted on an empty reference pool")

        query = self.scaler.transform(frame)
        # (queries, references) squared distances via the expansion of |a-b|^2.
        distances = np.sqrt(
            np.maximum(
                0.0,
                np.sum(query**2, axis=1)[:, None]
                + np.sum(self.reference_context_**2, axis=1)[None, :]
                - 2.0 * query @ self.reference_context_.T,
            )
        )

        neighbours = max(1, min(self.config.neighbours, self.reference_size))
        predictions = np.full((len(frame), len(self.targets)), np.nan)
        used = np.zeros((len(frame), len(self.targets)), dtype=int)
        mean_distance = np.full(len(frame), np.nan)

        order = np.argsort(distances, axis=1, kind="stable")
        for target_index in range(len(self.targets)):
            available = np.isfinite(self.reference_targets_[:, target_index])
            if not available.any():
                continue
            for row in range(len(frame)):
                candidates = order[row][available[order[row]]][:neighbours]
                if candidates.size < self.config.minimum_neighbours:
                    continue
                weights = 1.0 / (distances[row, candidates] + self.config.distance_epsilon)
                values = self.reference_targets_[candidates, target_index]
                predictions[row, target_index] = float(
                    np.sum(weights * values) / np.sum(weights)
                )
                used[row, target_index] = int(candidates.size)
                if target_index == 0:
                    mean_distance[row] = float(np.mean(distances[row, candidates]))

        result = pd.DataFrame(
            predictions,
            columns=[f"predicted_{name}" for name in self.targets],
            index=frame.index,
        )
        for position, name in enumerate(self.targets):
            result[f"neighbours_{name}"] = used[:, position]
        result["mean_neighbour_distance"] = mean_distance
        return result

    def explain(self, frame: pd.DataFrame, row: int = 0, count: int = 5) -> pd.DataFrame:
        """Name the nearest reference corners for one query, for inspection."""

        if self.reference_context_ is None:
            raise ValueError("model must be fitted before explaining")
        query = self.scaler.transform(frame.iloc[[row]])
        distances = np.linalg.norm(self.reference_context_ - query, axis=1)
        nearest = np.argsort(distances, kind="stable")[:count]
        return pd.DataFrame(
            {
                "corner_key": [self.reference_labels_[index] for index in nearest],
                "distance": distances[nearest],
                **{
                    name: self.reference_targets_[nearest, position]
                    for position, name in enumerate(self.targets)
                },
            }
        )


@dataclass
class TrackNormaliser:
    """Split behaviour into a track level and a within-track shape.

    A 90 m radius corner is not the same driving problem at Monza and at
    Monaco: downforce, gearing and grip differ, so the same shape demands
    different absolute speeds. Asking geometry to predict raw behaviour forces
    it to average across those regimes.

    Normalising per track asks it something it can plausibly know instead —
    *which corners on this track do you take faster or slower than your typical
    corner here* — and leaves the track's own level and spread to be supplied
    by whatever laps are available on the day.
    """

    targets: tuple[str, ...]
    location_: dict[tuple[str, str], float] = field(default_factory=dict)
    scale_: dict[tuple[str, str], float] = field(default_factory=dict)
    pooled_location_: dict[str, float] = field(default_factory=dict)
    pooled_scale_: dict[str, float] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "TrackNormaliser":
        for target in self.targets:
            values = pd.to_numeric(frame.get(target), errors="coerce").to_numpy(
                dtype=float
            )
            location, scale = robust_location_scale(values)
            self.pooled_location_[target] = location
            self.pooled_scale_[target] = scale
        for track, track_frame in frame.groupby(frame["track"].astype(str)):
            for target in self.targets:
                values = pd.to_numeric(
                    track_frame.get(target), errors="coerce"
                ).to_numpy(dtype=float)
                location, scale = robust_location_scale(values)
                self.location_[(track, target)] = location
                self.scale_[(track, target)] = scale
        return self

    def _pair(self, track: str, target: str) -> tuple[float, float]:
        location = self.location_.get((track, target), np.nan)
        scale = self.scale_.get((track, target), np.nan)
        if not np.isfinite(location):
            location = self.pooled_location_.get(target, np.nan)
        if not np.isfinite(scale) or scale <= 0:
            scale = self.pooled_scale_.get(target, np.nan)
        return location, scale

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Replace each target with its within-track deviation."""

        result = frame.copy()
        tracks = frame["track"].astype(str).to_numpy()
        for target in self.targets:
            values = pd.to_numeric(frame.get(target), errors="coerce").to_numpy(
                dtype=float
            )
            locations = np.array(
                [self._pair(track, target)[0] for track in tracks], dtype=float
            )
            scales = np.array(
                [self._pair(track, target)[1] for track in tracks], dtype=float
            )
            usable = np.isfinite(locations) & np.isfinite(scales) & (scales > 0)
            normalised = np.full(len(frame), np.nan)
            normalised[usable] = (values[usable] - locations[usable]) / scales[usable]
            result[target] = normalised
        return result

    def pooled(self, target: str) -> tuple[float, float]:
        return (
            self.pooled_location_.get(target, np.nan),
            self.pooled_scale_.get(target, np.nan),
        )


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    """Median and a MAD-based spread, both resistant to one wild lap."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 3:
        return float("nan"), float("nan")
    location = float(np.median(finite))
    deviation = float(np.median(np.abs(finite - location)))
    if deviation > 1e-9:
        return location, 1.4826 * deviation
    spread = float(np.std(finite))
    return location, spread if spread > 1e-9 else float("nan")


def robust_scale(values: np.ndarray) -> float:
    """Spread estimate that a single wild lap cannot inflate."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan")
    median = float(np.median(finite))
    deviation = float(np.median(np.abs(finite - median)))
    if deviation > 1e-9:
        return 1.4826 * deviation
    spread = float(np.std(finite))
    return spread if spread > 1e-9 else float("nan")


__all__ = [
    "BRAKING_TARGETS",
    "PRIMARY_TARGETS",
    "ContextScaler",
    "CornerBehaviorKNN",
    "KNNConfig",
    "TrackNormaliser",
    "robust_location_scale",
    "robust_scale",
    "score_residuals",
]


def score_residuals(
    actual: pd.Series,
    predicted: pd.Series,
    reference: pd.Series | None = None,
) -> pd.Series:
    """Express the gap between done and expected in robust standard deviations.

    This is the number coaching would speak from: "you braked two and a half
    deviations later than usual here" rather than a raw metre count that means
    nothing without knowing the corner.
    """

    actual_values = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    predicted_values = pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=float)
    basis = actual_values if reference is None else pd.to_numeric(
        reference, errors="coerce"
    ).to_numpy(dtype=float)
    scale = robust_scale(basis)
    if not np.isfinite(scale):
        return pd.Series(np.full(len(actual_values), np.nan), index=actual.index)
    return pd.Series((actual_values - predicted_values) / scale, index=actual.index)