"""A personal style envelope: what this driver's execution normally looks like.

The k-NN in `knn.py` answers "what should this corner demand", which the corner
largely dictates and which therefore needs the track to be known. This module
answers a different question — "was that a normal way for *you* to do it" — and
the variance decomposition says that question barely depends on the corner:

```text
behaviour                        corner explains   varies lap to lap
brake_release_smoothness_per_s              21%                 69%
steering_rate_std_per_s                     60%                 34%
minimum_speed_kmh                           86%                  2%
```

Release smoothness is almost entirely the driver, not the corner. Knowing which
lap a row came from explains a further 17% of its variance beyond the corner,
so a lap has a consistent character — you were rougher *that lap*, everywhere.
That is a real driver state rather than sensor noise, and it is coachable.

Because the envelope is built from every corner the driver has ever driven and
keyed only on a coarse corner band, it applies to a circuit that has never been
driven. That is the part an in-game ghost cannot do: a ghost needs a reference
lap on the track and reports a time delta, where this reports what you did.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from f1_engineer.modeling.knn import robust_location_scale

# Execution descriptors. Absolute corner demands such as minimum speed and
# entry speed are deliberately excluded: the corner dictates those, so they
# belong to the geometry model, not to a personal envelope.
STYLE_TARGETS: tuple[str, ...] = (
    "brake_release_smoothness_per_s",
    "brake_build_rate_per_s",
    "steering_rate_std_per_s",
    "steering_correction_count",
    "trail_brake_past_entry_m",
    "throttle_brake_overlap_s",
    "coast_distance_m",
    "minimum_speed_offset_from_apex_m",
    "peak_rear_slip_abs",
    "rear_slip_bias",
)

# Targets read off the brake trace need a complete, untruncated release.
BRAKE_DERIVED_TARGETS: frozenset[str] = frozenset(
    {
        "brake_release_smoothness_per_s",
        "brake_build_rate_per_s",
        "trail_brake_past_entry_m",
        "throttle_brake_overlap_s",
    }
)

# What a high or low reading means, for deterministic phrasing. No live model
# is involved in speaking: the wording is selected, not generated.
DEVIATION_PHRASES: dict[str, tuple[str, str]] = {
    "brake_release_smoothness_per_s": (
        "brake release more abrupt than your normal",
        "brake release smoother than your normal",
    ),
    "brake_build_rate_per_s": (
        "brake applied harder and faster than your normal",
        "brake applied more gradually than your normal",
    ),
    "steering_rate_std_per_s": (
        "busier hands than your normal",
        "calmer steering than your normal",
    ),
    "steering_correction_count": (
        "more steering corrections than your normal",
        "fewer steering corrections than your normal",
    ),
    "trail_brake_past_entry_m": (
        "carrying the brake deeper than your normal",
        "off the brake earlier than your normal",
    ),
    "throttle_brake_overlap_s": (
        "more throttle and brake overlap than your normal",
        "less throttle and brake overlap than your normal",
    ),
    "coast_distance_m": (
        "coasting more than your normal",
        "coasting less than your normal",
    ),
    "minimum_speed_offset_from_apex_m": (
        "slowest point later than your normal",
        "slowest point earlier than your normal",
    ),
    "peak_rear_slip_abs": (
        "more rear slip than your normal",
        "less rear slip than your normal",
    ),
    "rear_slip_bias": (
        "more rear-biased than your normal",
        "more front-biased than your normal",
    ),
}


@dataclass(frozen=True, slots=True)
class StyleEnvelopeConfig:
    """Corner bands, and how much evidence a band needs to stand alone."""

    band_edges: tuple[float, ...] = (60.0, 150.0)
    band_names: tuple[str, ...] = ("tight", "medium", "fast")
    minimum_band_rows: int = 20
    deviation_threshold: float = 2.0

    def band_for(self, radius: float) -> str:
        if not np.isfinite(radius):
            return "unknown"
        for edge, name in zip(self.band_edges, self.band_names):
            if radius < edge:
                return name
        return self.band_names[-1]


def assign_bands(frame: pd.DataFrame, config: StyleEnvelopeConfig) -> pd.Series:
    radius = pd.to_numeric(
        frame.get("ctx_minimum_radius_m"), errors="coerce"
    ).to_numpy(dtype=float)
    return pd.Series(
        [config.band_for(value) for value in radius], index=frame.index, name="band"
    )


def usable_rows(frame: pd.DataFrame, target: str) -> pd.Series:
    """Rows where this target means something."""

    values = pd.to_numeric(frame.get(target), errors="coerce")
    ok = values.notna() & np.isfinite(values)
    if target in BRAKE_DERIVED_TARGETS and "braking_complete" in frame.columns:
        ok &= frame["braking_complete"].fillna(False).astype(bool)
    elif "behavior_complete" in frame.columns:
        ok &= frame["behavior_complete"].fillna(False).astype(bool)
    return ok


@dataclass
class StyleEnvelope:
    """Per-band robust centre and spread for each style descriptor."""

    targets: tuple[str, ...] = STYLE_TARGETS
    config: StyleEnvelopeConfig = field(default_factory=StyleEnvelopeConfig)
    bands_: dict[tuple[str, str], tuple[float, float, int]] = field(
        default_factory=dict
    )
    global_: dict[str, tuple[float, float, int]] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "StyleEnvelope":
        bands = assign_bands(frame, self.config)
        for target in self.targets:
            usable = usable_rows(frame, target)
            values = pd.to_numeric(frame.loc[usable, target], errors="coerce")
            location, scale = robust_location_scale(values.to_numpy(dtype=float))
            self.global_[target] = (location, scale, int(usable.sum()))
            for band in set(bands):
                selected = usable & (bands == band)
                count = int(selected.sum())
                if count < self.config.minimum_band_rows:
                    continue
                band_values = pd.to_numeric(
                    frame.loc[selected, target], errors="coerce"
                ).to_numpy(dtype=float)
                band_location, band_scale = robust_location_scale(band_values)
                if np.isfinite(band_location) and np.isfinite(band_scale):
                    self.bands_[(target, band)] = (band_location, band_scale, count)
        return self

    def reference(self, target: str, band: str) -> tuple[float, float, str]:
        """Centre and spread for a corner band, falling back to all corners."""

        entry = self.bands_.get((target, band))
        if entry is not None:
            return entry[0], entry[1], band
        location, scale, _ = self.global_.get(
            target, (float("nan"), float("nan"), 0)
        )
        return location, scale, "all"

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Deviation of each row from this driver's normal, in robust sigmas."""

        bands = assign_bands(frame, self.config)
        result = pd.DataFrame(index=frame.index)
        result["band"] = bands
        for target in self.targets:
            values = pd.to_numeric(frame.get(target), errors="coerce").to_numpy(
                dtype=float
            )
            usable = usable_rows(frame, target).to_numpy()
            centres = np.full(len(frame), np.nan)
            spreads = np.full(len(frame), np.nan)
            for band in set(bands):
                location, scale, _ = self.reference(target, band)
                mask = (bands == band).to_numpy()
                centres[mask] = location
                spreads[mask] = scale
            with np.errstate(all="ignore"):
                z = (values - centres) / spreads
            z[~usable] = np.nan
            z[~np.isfinite(spreads) | (spreads <= 0)] = np.nan
            result[f"z_{target}"] = z
        return result

    def lap_offsets(self, frame: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
        """How far a whole lap sits from normal, per target.

        A lap can be rough everywhere. Reporting that once is more useful than
        firing the same alert at every corner, so the offset is subtracted to
        leave the corners that stand out *within* the lap.
        """

        working = scores.copy()
        working["lap_id"] = frame["lap_id"].to_numpy()
        columns = [f"z_{target}" for target in self.targets if f"z_{target}" in working]
        return working.groupby("lap_id")[columns].median()

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": list(self.targets),
            "band_edges": list(self.config.band_edges),
            "band_names": list(self.config.band_names),
            "deviation_threshold": self.config.deviation_threshold,
            "bands": [
                {
                    "target": target,
                    "band": band,
                    "centre": centre,
                    "spread": spread,
                    "rows": rows,
                }
                for (target, band), (centre, spread, rows) in sorted(
                    self.bands_.items()
                )
            ],
            "global": [
                {"target": target, "centre": centre, "spread": spread, "rows": rows}
                for target, (centre, spread, rows) in sorted(self.global_.items())
            ],
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def describe_deviations(
    frame: pd.DataFrame,
    scores: pd.DataFrame,
    envelope: StyleEnvelope,
    *,
    limit: int = 10,
) -> list[str]:
    """Turn the largest deviations into the sentences coaching would speak."""

    threshold = envelope.config.deviation_threshold
    entries: list[tuple[float, str]] = []
    for target in envelope.targets:
        column = f"z_{target}"
        if column not in scores.columns:
            continue
        z = scores[column]
        flagged = z.abs() >= threshold
        for index in scores.index[flagged.fillna(False)]:
            value = float(z.loc[index])
            high, low = DEVIATION_PHRASES.get(target, (f"{target} high", f"{target} low"))
            corner = frame.loc[index].get("corner_key", "?")
            direction = frame.loc[index].get("turn_direction", "")
            band = scores.loc[index, "band"]
            phrase = high if value > 0 else low
            entries.append(
                (
                    abs(value),
                    f"{corner} ({direction}, {band}): {phrase} "
                    f"[{value:+.1f} sigma]",
                )
            )
    entries.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in entries[:limit]]


def evaluate_envelope_transfer(
    frame: pd.DataFrame,
    *,
    config: StyleEnvelopeConfig | None = None,
    targets: Sequence[str] = STYLE_TARGETS,
) -> pd.DataFrame:
    """Does an envelope built elsewhere describe a track it has never seen?

    This is a calibration test, not an accuracy test. If the envelope transfers,
    a held-out track's deviations should sit near zero with a spread near one,
    which puts roughly 68 per cent inside one sigma and 95 inside two. A centre
    far from zero means the level does not carry across tracks; a spread far
    from one means the width does not.
    """

    cfg = config or StyleEnvelopeConfig()
    tracks = sorted(frame["track"].astype(str).unique())
    if len(tracks) < 2:
        return pd.DataFrame([{"note": "need at least two tracks"}])

    records: list[dict[str, Any]] = []
    for held in tracks:
        is_held = frame["track"].astype(str) == held
        reference = frame[~is_held]
        test = frame[is_held]
        if reference.empty or test.empty:
            continue

        envelope = StyleEnvelope(targets=tuple(targets), config=cfg).fit(reference)
        scores = envelope.score(test)
        # Upper bound: an envelope fitted on the held-out track itself. The gap
        # between the two is what transfer actually costs.
        oracle = StyleEnvelope(targets=tuple(targets), config=cfg).fit(test)
        oracle_scores = oracle.score(test)

        lap_offsets = envelope.lap_offsets(test, scores)
        lap_ids = test["lap_id"].to_numpy()

        for target in targets:
            column = f"z_{target}"
            z = scores[column].to_numpy(dtype=float)
            z = z[np.isfinite(z)]
            if z.size < 20:
                continue
            centre, spread = robust_location_scale(z)
            oracle_z = oracle_scores[column].to_numpy(dtype=float)
            oracle_z = oracle_z[np.isfinite(oracle_z)]

            offsets = lap_offsets[column].reindex(lap_ids).to_numpy(dtype=float)
            adjusted = scores[column].to_numpy(dtype=float) - offsets
            adjusted = adjusted[np.isfinite(adjusted)]

            records.append(
                {
                    "unseen_track": held,
                    "target": target,
                    "rows": int(z.size),
                    "centre_z": centre,
                    "spread_z": spread,
                    "within_1sd": float(np.mean(np.abs(z) <= 1.0)),
                    "within_2sd": float(np.mean(np.abs(z) <= 2.0)),
                    # How chatty a live overlay would be, and how far the worst
                    # reading strays. A robust spread assumes tails no heavier
                    # than a normal distribution; several of these have much
                    # heavier tails, which shows up here rather than silently.
                    "flag_rate": float(
                        np.mean(np.abs(z) >= cfg.deviation_threshold)
                    ),
                    "worst_abs_z": float(np.max(np.abs(z))),
                    "oracle_within_2sd": (
                        float(np.mean(np.abs(oracle_z) <= 2.0))
                        if oracle_z.size
                        else float("nan")
                    ),
                    "within_2sd_after_lap_offset": (
                        float(np.mean(np.abs(adjusted) <= 2.0))
                        if adjusted.size
                        else float("nan")
                    ),
                    "transfers": bool(
                        np.isfinite(centre)
                        and np.isfinite(spread)
                        and abs(centre) < 0.5
                        and 0.7 < spread < 1.4
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def summarise_transfer(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "target" not in results.columns:
        return results
    grouped = (
        results.groupby("target")
        .agg(
            tracks=("unseen_track", "nunique"),
            tracks_passing=("transfers", "sum"),
            rows=("rows", "sum"),
            centre_z=("centre_z", "mean"),
            spread_z=("spread_z", "mean"),
            worst_centre_z=("centre_z", lambda values: values.abs().max()),
            within_1sd=("within_1sd", "mean"),
            within_2sd=("within_2sd", "mean"),
            flag_rate=("flag_rate", "mean"),
            worst_abs_z=("worst_abs_z", "max"),
            oracle_within_2sd=("oracle_within_2sd", "mean"),
            within_2sd_after_lap_offset=("within_2sd_after_lap_offset", "mean"),
            transfers=("transfers", "all"),
        )
        .reset_index()
    )
    grouped["tracks_passing"] = grouped["tracks_passing"].astype(int)
    return grouped.sort_values(
        ["tracks_passing", "within_2sd"], ascending=False
    ).reset_index(drop=True)


def corner_dependence(
    frame: pd.DataFrame,
    targets: Iterable[str] = STYLE_TARGETS,
) -> pd.DataFrame:
    """Share of each target's variance explained by which corner it was."""

    records: list[dict[str, Any]] = []
    for target in targets:
        usable = usable_rows(frame, target)
        subset = frame[usable]
        if len(subset) < 30:
            continue
        values = pd.to_numeric(subset[target], errors="coerce").to_numpy(dtype=float)
        grand = float(np.mean(values))
        total = float(np.sum((values - grand) ** 2))
        if total <= 0:
            continue
        corner_means = subset.groupby("corner_key")[target].transform("mean")
        residual = float(
            np.sum((values - pd.to_numeric(corner_means, errors="coerce").to_numpy()) ** 2)
        )
        records.append(
            {
                "target": target,
                "rows": int(len(subset)),
                "corner_share": 1.0 - residual / total,
            }
        )
    return pd.DataFrame.from_records(records)