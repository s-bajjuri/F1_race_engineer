"""Evaluation protocols for the Phase 4 corner model.

Three questions, in increasing order of difficulty:

1. **Known track** — on a track in the training data, can the model predict how
   this driver takes a corner it has seen? Held out by session, never by lap,
   because two laps from one session share tyre state, fuel and conditions and
   would leak into each other.
2. **Leave one track out** — with a whole track removed, can geometry alone
   carry the driver's behaviour onto it? This is the zero-shot case and the
   real test of whether corner shape transfers.
3. **Few shot** — after n practice laps on the unseen track, how much better
   does it get? This is the curve the project is actually aiming at.

Every number is reported against a baseline that ignores geometry, because a
model that cannot beat "predict the average" has not demonstrated anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from f1_engineer.modeling.knn import (
    BRAKING_TARGETS,
    PRIMARY_TARGETS,
    CornerBehaviorKNN,
    KNNConfig,
    TrackNormaliser,
    robust_location_scale,
)

TARGET_GROUPS: dict[str, tuple[tuple[str, ...], str]] = {
    "primary": (PRIMARY_TARGETS, "behavior_complete"),
    "braking": (BRAKING_TARGETS, "braking_complete"),
}


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    shots: tuple[int, ...] = (0, 1, 3, 5)
    repeats: int = 4
    minimum_test_laps: int = 2
    minimum_reference_rows: int = 30
    # How many observations of a corner the geometric prior is worth. Swept
    # rather than assumed: if the best value is zero, the prior earns no weight
    # and geometry is not helping, which is a result worth knowing rather than
    # hiding behind a hand-picked constant. Zero means "use your own laps where
    # you have them, the prior only where you do not".
    #
    # The practice laps are deliberately kept out of the neighbour pool. Put
    # them in and the identical corner sits at distance zero, swamping every
    # other neighbour, and the model silently collapses into "average your own
    # laps here" while still looking like a geometry model.
    prior_strengths: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0)
    coverage_subsets: int = 3
    knn: KNNConfig = KNNConfig()


def _as_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(True, index=frame.index)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, int]:
    usable = np.isfinite(actual) & np.isfinite(predicted)
    if not usable.any():
        return float("nan"), 0
    return float(np.mean(np.abs(actual[usable] - predicted[usable]))), int(usable.sum())


def _skill(model_mae: float, baseline_mae: float) -> float:
    """Fraction of the baseline's error removed. Zero means no better."""

    if not np.isfinite(model_mae) or not np.isfinite(baseline_mae) or baseline_mae <= 0:
        return float("nan")
    return float(1.0 - model_mae / baseline_mae)


def _score_split(
    reference: pd.DataFrame,
    test: pd.DataFrame,
    targets: Sequence[str],
    *,
    config: KNNConfig,
    extra_baseline: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    model = CornerBehaviorKNN(targets=tuple(targets), config=config).fit(reference)
    predictions = model.predict(test)

    records: list[dict[str, Any]] = []
    for target in targets:
        actual = pd.to_numeric(test[target], errors="coerce").to_numpy(dtype=float)
        predicted = predictions[f"predicted_{target}"].to_numpy(dtype=float)
        model_mae, count = _mean_absolute_error(actual, predicted)

        # Baseline: the pool average, which uses no geometry at all.
        pool_values = pd.to_numeric(reference[target], errors="coerce").to_numpy(
            dtype=float
        )
        pool_finite = pool_values[np.isfinite(pool_values)]
        pool_mean = float(np.mean(pool_finite)) if pool_finite.size else float("nan")
        baseline_mae, _ = _mean_absolute_error(
            actual, np.full_like(actual, pool_mean)
        )

        record: dict[str, Any] = {
            "target": target,
            "rows_scored": count,
            "model_mae": model_mae,
            "pool_mean_mae": baseline_mae,
            "skill_vs_pool_mean": _skill(model_mae, baseline_mae),
            "reference_rows": int(len(reference)),
        }

        # Second baseline for the few-shot case: what the driver did at this
        # exact corner during the practice laps. Strong, and the honest thing
        # to compare against once the corner has actually been driven.
        if extra_baseline is not None and not extra_baseline.empty:
            corner_means = (
                extra_baseline.groupby("corner_key")[target]
                .apply(lambda values: pd.to_numeric(values, errors="coerce").mean())
                .to_dict()
            )
            shot_overall = pd.to_numeric(
                extra_baseline[target], errors="coerce"
            ).mean()
            shot_predicted = np.array(
                [
                    corner_means.get(key, shot_overall)
                    for key in test["corner_key"].astype(str)
                ],
                dtype=float,
            )
            shot_mae, _ = _mean_absolute_error(actual, shot_predicted)
            record["same_corner_mean_mae"] = shot_mae
            record["skill_vs_same_corner_mean"] = _skill(model_mae, shot_mae)

        records.append(record)
    return records


def evaluate_known_track(
    frame: pd.DataFrame,
    targets: Sequence[str],
    *,
    config: EvaluationConfig | None = None,
) -> pd.DataFrame:
    """Hold out one session at a time, within each track."""

    cfg = config or EvaluationConfig()
    records: list[dict[str, Any]] = []
    for track, track_frame in frame.groupby("track", sort=True):
        sessions = sorted(track_frame["session_key"].astype(str).unique())
        if len(sessions) < 2:
            records.append(
                {
                    "track": track,
                    "target": "-",
                    "note": "only one session; cannot hold out without leakage",
                    "rows_scored": 0,
                }
            )
            continue
        for session in sessions:
            held = track_frame["session_key"].astype(str) == session
            reference = track_frame[~held]
            test = track_frame[held]
            if len(reference) < cfg.minimum_reference_rows or test.empty:
                continue
            for record in _score_split(
                reference, test, targets, config=cfg.knn
            ):
                records.append({"track": track, "held_out_session": session, **record})
    return pd.DataFrame.from_records(records)


def _select_adaptation_laps(
    track_frame: pd.DataFrame,
    shots: int,
    repeat: int,
) -> tuple[list[str], list[str], bool]:
    """Pick practice laps, preferring a session the test laps do not share.

    Returns the adaptation laps, the test laps, and whether the two ended up
    sharing a session, which is recorded rather than hidden.
    """

    laps = sorted(track_frame["lap_id"].astype(str).unique())
    if shots <= 0:
        return [], laps, False

    sessions = sorted(track_frame["session_key"].astype(str).unique())
    if len(sessions) > 1:
        chosen_session = sessions[repeat % len(sessions)]
        in_session = track_frame["session_key"].astype(str) == chosen_session
        session_laps = sorted(track_frame.loc[in_session, "lap_id"].astype(str).unique())
        other_laps = [lap for lap in laps if lap not in set(session_laps)]
        if len(session_laps) >= shots and other_laps:
            offset = (repeat // len(sessions)) * shots
            rotated = [
                session_laps[(offset + index) % len(session_laps)]
                for index in range(shots)
            ]
            return sorted(set(rotated)), other_laps, False

    if len(laps) <= shots:
        return [], laps, False
    offset = repeat * shots
    adaptation = sorted({laps[(offset + index) % len(laps)] for index in range(shots)})
    test = [lap for lap in laps if lap not in set(adaptation)]
    return adaptation, test, True


def _own_corner_estimate(
    adaptation: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-corner mean and observation count from the practice laps."""

    keys = test["corner_key"].astype(str).to_numpy()
    if adaptation.empty:
        return np.full(len(test), np.nan), np.zeros(len(test))

    working = adaptation.loc[:, ["corner_key", target]].copy()
    working[target] = pd.to_numeric(working[target], errors="coerce")
    grouped = working.groupby(working["corner_key"].astype(str))[target].agg(
        ["mean", "count"]
    )
    means = grouped["mean"].to_dict()
    counts = grouped["count"].to_dict()
    return (
        np.array([means.get(key, np.nan) for key in keys], dtype=float),
        np.array([counts.get(key, 0) for key in keys], dtype=float),
    )


def _blend(
    prior: np.ndarray,
    own_mean: np.ndarray,
    own_count: np.ndarray,
    prior_strength: float,
) -> np.ndarray:
    """Shrink the driver's own record toward the geometric prior.

    With no observations this is the prior. With many it is the driver's own
    average. In between it is what a race engineer actually does: start from
    what corners of this shape usually demand, and update as laps come in.
    """

    blended = np.full(len(prior), np.nan)
    has_prior = np.isfinite(prior)
    has_own = np.isfinite(own_mean) & (own_count > 0)

    both = has_prior & has_own
    weight = own_count[both] + prior_strength
    blended[both] = (
        own_count[both] * own_mean[both] + prior_strength * prior[both]
    ) / weight
    blended[has_prior & ~has_own] = prior[has_prior & ~has_own]
    blended[~has_prior & has_own] = own_mean[~has_prior & has_own]
    return blended


def evaluate_few_shot(
    frame: pd.DataFrame,
    targets: Sequence[str],
    *,
    config: EvaluationConfig | None = None,
) -> pd.DataFrame:
    """Leave one track out, then feed back n practice laps from that track.

    Three predictors are scored side by side so the contribution of geometry
    can be separated from the contribution of simply having driven the corner:

    * ``geometry`` — the k-NN prior, fitted on other tracks only. Identical at
      every shot count, because it never sees the practice laps.
    * ``own`` — the driver's mean at that corner across the practice laps,
      falling back to their overall mean where the corner has no usable row.
    * ``blended`` — the prior updated by the practice laps.
    """

    cfg = config or EvaluationConfig()
    records: list[dict[str, Any]] = []
    tracks = sorted(frame["track"].astype(str).unique())
    if len(tracks) < 2:
        return pd.DataFrame(
            [{"note": "need at least two tracks for leave-one-track-out"}]
        )

    for track in tracks:
        held = frame["track"].astype(str) == track
        other_tracks = frame[~held]
        track_frame = frame[held]
        if len(other_tracks) < cfg.minimum_reference_rows or track_frame.empty:
            continue

        for shots in cfg.shots:
            repeats = 1 if shots == 0 else cfg.repeats
            for repeat in range(repeats):
                adaptation_laps, test_laps, leakage = _select_adaptation_laps(
                    track_frame, shots, repeat
                )
                if len(test_laps) < cfg.minimum_test_laps:
                    continue
                lap_ids = track_frame["lap_id"].astype(str)
                adaptation = track_frame[lap_ids.isin(adaptation_laps)]
                test = track_frame[lap_ids.isin(test_laps)]
                if test.empty:
                    continue

                # The prior never sees the practice laps, so its score is a
                # clean measure of what corner shape alone carries across.
                model = CornerBehaviorKNN(
                    targets=tuple(targets), config=cfg.knn
                ).fit(other_tracks)
                priors = model.predict(test)

                # Phase 4B: the same prior trained on within-track deviations,
                # so geometry only ranks corners against their own track. The
                # level and spread are restored from the practice laps.
                normaliser = TrackNormaliser(targets=tuple(targets)).fit(other_tracks)
                normalised_model = CornerBehaviorKNN(
                    targets=tuple(targets), config=cfg.knn
                ).fit(normaliser.transform(other_tracks))
                normalised_priors = normalised_model.predict(test)

                for target in targets:
                    actual = pd.to_numeric(test[target], errors="coerce").to_numpy(
                        dtype=float
                    )
                    prior = priors[f"predicted_{target}"].to_numpy(dtype=float)
                    own_mean, own_count = _own_corner_estimate(
                        adaptation, test, target
                    )

                    # Where a practice lap exists but that corner has no usable
                    # row, fall back to the driver's overall practice average.
                    if not adaptation.empty:
                        overall = pd.to_numeric(
                            adaptation[target], errors="coerce"
                        ).mean()
                        own_only = np.where(
                            np.isfinite(own_mean) & (own_count > 0), own_mean, overall
                        )
                    else:
                        own_only = np.full(len(test), np.nan)

                    pool_values = pd.to_numeric(
                        other_tracks[target], errors="coerce"
                    ).to_numpy(dtype=float)
                    pool_finite = pool_values[np.isfinite(pool_values)]
                    pool_mean = (
                        float(np.mean(pool_finite)) if pool_finite.size else float("nan")
                    )

                    # Restore the track level and spread. With practice laps
                    # available they come from the driver's own laps on this
                    # track; with none, from the pooled statistics, which is
                    # the best guess anyone could make.
                    deviation = normalised_priors[f"predicted_{target}"].to_numpy(
                        dtype=float
                    )
                    if not adaptation.empty:
                        location, scale = robust_location_scale(
                            pd.to_numeric(adaptation[target], errors="coerce").to_numpy(
                                dtype=float
                            )
                        )
                    else:
                        location, scale = float("nan"), float("nan")
                    pooled_location, pooled_scale = normaliser.pooled(target)
                    if not np.isfinite(location):
                        location = pooled_location
                    if not np.isfinite(scale) or scale <= 0:
                        scale = pooled_scale
                    normalised_prior = (
                        location + scale * deviation
                        if np.isfinite(location) and np.isfinite(scale)
                        else np.full(len(test), np.nan)
                    )

                    geometry_mae, scored = _mean_absolute_error(actual, prior)
                    normalised_mae, _ = _mean_absolute_error(actual, normalised_prior)
                    own_mae, _ = _mean_absolute_error(actual, own_only)
                    baseline_mae, _ = _mean_absolute_error(
                        actual, np.full_like(actual, pool_mean)
                    )

                    common = {
                        "unseen_track": track,
                        "shots": shots,
                        "repeat": repeat,
                        "target": target,
                        "adaptation_laps": len(adaptation_laps),
                        "test_laps": len(test_laps),
                        "session_shared_with_test": leakage,
                        "rows_scored": scored,
                        "prior_reference_rows": int(len(other_tracks)),
                        "corners_seen_in_practice": int(np.sum(own_count > 0)),
                        "geometry_mae": geometry_mae,
                        "normalised_mae": normalised_mae,
                        "own_mae": own_mae,
                        "pool_mean_mae": baseline_mae,
                        "skill_geometry": _skill(geometry_mae, baseline_mae),
                        "skill_normalised": _skill(normalised_mae, baseline_mae),
                        "skill_own": _skill(own_mae, baseline_mae),
                        "normalised_gain_over_raw": _skill(
                            normalised_mae, geometry_mae
                        ),
                    }
                    strengths = cfg.prior_strengths if shots > 0 else (0.0,)
                    for strength in strengths:
                        blended = _blend(prior, own_mean, own_count, strength)
                        blended_mae, _ = _mean_absolute_error(actual, blended)
                        blended_normalised = _blend(
                            normalised_prior, own_mean, own_count, strength
                        )
                        blended_normalised_mae, _ = _mean_absolute_error(
                            actual, blended_normalised
                        )
                        records.append(
                            {
                                **common,
                                "prior_strength": float(strength),
                                "blended_mae": blended_mae,
                                "blended_normalised_mae": blended_normalised_mae,
                                "skill_blended": _skill(blended_mae, baseline_mae),
                                "skill_blended_normalised": _skill(
                                    blended_normalised_mae, baseline_mae
                                ),
                                "blended_gain_over_own": _skill(blended_mae, own_mae),
                                "blended_normalised_gain_over_own": _skill(
                                    blended_normalised_mae, own_mae
                                ),
                            }
                        )
    return pd.DataFrame.from_records(records)


def choose_prior_strength(results: pd.DataFrame) -> pd.DataFrame:
    """Pick the prior weight that minimises error, per target group.

    Reported rather than assumed. A chosen weight of zero says plainly that the
    geometric prior is not earning its place once the driver has laps of their
    own at that corner.
    """

    if results.empty or "prior_strength" not in results.columns:
        return pd.DataFrame()
    scored = results[results["shots"] > 0]
    if scored.empty:
        return pd.DataFrame()
    grouped = (
        scored.groupby(["target", "prior_strength"], sort=True)
        .agg(
            blended_mae=("blended_mae", "mean"),
            blended_normalised_mae=("blended_normalised_mae", "mean"),
            own_mae=("own_mae", "mean"),
            skill_blended=("skill_blended", "mean"),
            skill_blended_normalised=("skill_blended_normalised", "mean"),
        )
        .reset_index()
    )
    # The chosen weight is the one that minimises the better of the two priors,
    # so normalisation is judged on its own best configuration rather than one
    # tuned for the raw prior.
    best = grouped.loc[
        grouped.groupby("target")["blended_normalised_mae"].idxmin()
    ].copy()
    best["gain_over_own"] = 1.0 - best["blended_mae"] / best["own_mae"]
    best["normalised_gain_over_own"] = (
        1.0 - best["blended_normalised_mae"] / best["own_mae"]
    )
    return best.reset_index(drop=True)


def summarise_few_shot(results: pd.DataFrame) -> pd.DataFrame:
    """Average the repeats away, keeping track, shots and target."""

    if results.empty or "shots" not in results.columns:
        return results
    if "prior_strength" in results.columns:
        # Summarise the best weight per target so the curve is one line.
        best = choose_prior_strength(results)
        if not best.empty:
            chosen = dict(zip(best["target"], best["prior_strength"]))
            keep = results.apply(
                lambda row: row["shots"] == 0
                or row["prior_strength"] == chosen.get(row["target"], 0.0),
                axis=1,
            )
            results = results[keep]
    columns = {
        "geometry_mae": "mean",
        "normalised_mae": "mean",
        "own_mae": "mean",
        "blended_mae": "mean",
        "blended_normalised_mae": "mean",
        "pool_mean_mae": "mean",
        "skill_geometry": "mean",
        "skill_normalised": "mean",
        "skill_own": "mean",
        "skill_blended": "mean",
        "skill_blended_normalised": "mean",
        "blended_gain_over_own": "mean",
        "blended_normalised_gain_over_own": "mean",
        "normalised_gain_over_raw": "mean",
        "corners_seen_in_practice": "mean",
        "rows_scored": "mean",
    }
    available = {key: value for key, value in columns.items() if key in results.columns}
    grouped = (
        results.groupby(["unseen_track", "target", "shots"], sort=True)
        .agg(available)
        .reset_index()
    )
    if "session_shared_with_test" in results.columns:
        leakage = (
            results.groupby(["unseen_track", "target", "shots"], sort=True)[
                "session_shared_with_test"
            ]
            .any()
            .reset_index()
        )
        grouped = grouped.merge(leakage, on=["unseen_track", "target", "shots"])
    return grouped


def evaluate_track_coverage(
    frame: pd.DataFrame,
    targets: Sequence[str],
    *,
    config: EvaluationConfig | None = None,
) -> pd.DataFrame:
    """Zero-shot skill against the number of tracks the prior was trained on.

    The question this answers is whether weak transfer is a shortage of corner
    shapes or a shortage of method. A rising curve says collect more tracks; a
    flat one says more of the same data will not help and the model has to
    change. Worth knowing before spending evenings driving.
    """

    from itertools import combinations

    cfg = config or EvaluationConfig()
    tracks = sorted(frame["track"].astype(str).unique())
    if len(tracks) < 3:
        return pd.DataFrame([{"note": "need at least three tracks for a curve"}])

    records: list[dict[str, Any]] = []
    for held in tracks:
        others = [track for track in tracks if track != held]
        test = frame[frame["track"].astype(str) == held]
        if test.empty:
            continue
        for size in range(1, len(others) + 1):
            subsets = list(combinations(others, size))[: cfg.coverage_subsets]
            for subset in subsets:
                reference = frame[frame["track"].astype(str).isin(subset)]
                if len(reference) < cfg.minimum_reference_rows:
                    continue
                model = CornerBehaviorKNN(
                    targets=tuple(targets), config=cfg.knn
                ).fit(reference)
                predictions = model.predict(test)
                for target in targets:
                    actual = pd.to_numeric(test[target], errors="coerce").to_numpy(
                        dtype=float
                    )
                    predicted = predictions[f"predicted_{target}"].to_numpy(dtype=float)
                    model_mae, scored = _mean_absolute_error(actual, predicted)
                    pool = pd.to_numeric(reference[target], errors="coerce").to_numpy(
                        dtype=float
                    )
                    pool = pool[np.isfinite(pool)]
                    pool_mean = float(np.mean(pool)) if pool.size else float("nan")
                    baseline_mae, _ = _mean_absolute_error(
                        actual, np.full_like(actual, pool_mean)
                    )
                    records.append(
                        {
                            "unseen_track": held,
                            "training_tracks": size,
                            "training_set": "+".join(subset),
                            "target": target,
                            "rows_scored": scored,
                            "model_mae": model_mae,
                            "pool_mean_mae": baseline_mae,
                            "skill": _skill(model_mae, baseline_mae),
                        }
                    )
    return pd.DataFrame.from_records(records)


def evaluate_all(
    frame: pd.DataFrame,
    *,
    config: EvaluationConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Run every protocol over both target groups."""

    cfg = config or EvaluationConfig()
    output: dict[str, pd.DataFrame] = {}
    for group, (targets, mask_column) in TARGET_GROUPS.items():
        usable = frame[_as_bool(frame, mask_column)].reset_index(drop=True)
        if usable.empty:
            continue
        output[f"known_track_{group}"] = evaluate_known_track(
            usable, targets, config=cfg
        )
        few_shot = evaluate_few_shot(usable, targets, config=cfg)
        output[f"few_shot_{group}"] = few_shot
        output[f"few_shot_summary_{group}"] = summarise_few_shot(few_shot)
        chosen = choose_prior_strength(few_shot)
        if not chosen.empty:
            output[f"prior_strength_{group}"] = chosen
        output[f"track_coverage_{group}"] = evaluate_track_coverage(
            usable, targets, config=cfg
        )
    return output