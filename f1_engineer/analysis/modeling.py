from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Phase 3F deliberately separates features observable at segment entry from
# features that describe how the driver subsequently executed the maneuver.
# Track/zone identifiers are metadata for evaluation only and are NOT k-NN
# context features.
CONTEXT_FEATURE_COLUMNS: tuple[str, ...] = (
    "approach_speed_kmh",
    "approach_speed_delta_kmh",
    "approach_throttle_mean",
    "approach_abs_steering_mean",
    "approach_abs_steering_max",
    "approach_mean_abs_lateral_g",
    "approach_mean_abs_longitudinal_g",
    "approach_curvature_proxy_signed_mean_1pm",
    "approach_curvature_proxy_abs_mean_1pm",
)

CORE_BEHAVIOR_TARGET_COLUMNS: tuple[str, ...] = (
    "peak_brake",
    "mean_brake",
    "brake_release_smoothness_per_s",
    "braking_distance_m",
    "speed_loss_kmh",
    "minimum_speed_kmh",
    "full_throttle_distance_from_min_speed_m",
    "full_throttle_delay_from_min_speed_s",
    "steering_input_std",
    "steering_rate_std_per_s",
    "throttle_brake_overlap_ms",
)

MOTION_BEHAVIOR_TARGET_COLUMNS: tuple[str, ...] = (
    "slip_frequency",
    "peak_front_slip_abs",
    "peak_rear_slip_abs",
    "mean_front_slip_abs",
    "mean_rear_slip_abs",
    "rear_slip_bias",
    "mean_abs_lateral_g",
    "peak_abs_lateral_g",
    "mean_abs_longitudinal_g",
)

IDENTITY_COLUMNS: tuple[str, ...] = (
    "lap_id",
    "track",
    "source_file",
    "session_uid",
    "lap_number",
    "segment_id",
    "zone_id",
)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _numeric(frame: pd.DataFrame, column: str, default: float = math.nan) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _finite_row(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool, index=frame.index)
    if any(column not in frame.columns for column in columns):
        return pd.Series(False, index=frame.index, dtype=bool)
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return numeric.notna().all(axis=1) & np.isfinite(numeric).all(axis=1)


def _resolve_existing_path(value: object, *, processed_root: Path) -> Path:
    raw = str(value)
    candidates = [Path(raw)]
    if "\\" in raw:
        candidates.append(Path(raw.replace("\\", os.sep)))
    if "/" in raw and os.sep != "/":
        candidates.append(Path(raw.replace("/", os.sep)))

    anchors = [Path.cwd(), processed_root, processed_root.parent, processed_root.parent.parent]
    for candidate in candidates:
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if candidate.exists():
            return candidate
        for anchor in anchors:
            anchored = anchor / candidate
            if anchored.exists():
                return anchored
    return candidates[0]


def extract_causal_pre_entry_context(
    resampled_lap: pd.DataFrame,
    brake_start_index: int,
    *,
    lookback_seconds: float = 0.75,
) -> dict[str, float]:
    """Extract only telemetry available at or before braking confirmation.

    These descriptors are intentionally causal proxies, not claimed to be
    pure track geometry. They let Phase 4 test context->behavior matching on
    the existing dataset without leaking minimum speed, peak brake, throttle
    recovery, or other future segment outcomes into the neighbor query.
    """

    if resampled_lap.empty:
        raise ValueError("resampled lap is empty")
    if "timestamp_ms" not in resampled_lap.columns:
        raise ValueError("resampled lap is missing timestamp_ms")

    start = int(brake_start_index)
    if not 0 <= start < len(resampled_lap):
        raise IndexError(f"brake_start_index {start} is outside lap length {len(resampled_lap)}")

    times_ms = _numeric(resampled_lap, "timestamp_ms")
    start_time_ms = float(times_ms[start])
    lower_ms = start_time_ms - max(0.05, float(lookback_seconds)) * 1_000.0
    indices = np.flatnonzero(
        np.isfinite(times_ms)
        & (times_ms >= lower_ms)
        & (times_ms <= start_time_ms)
    )
    indices = indices[indices <= start]
    if len(indices) == 0:
        indices = np.array([start], dtype=int)

    speed = _numeric(resampled_lap, "speed_kmh")
    throttle = _numeric(resampled_lap, "throttle", 0.0)
    steering = _numeric(resampled_lap, "steering", 0.0)
    lat_g = _numeric(resampled_lap, "g_force_lat")
    lon_g = _numeric(resampled_lap, "g_force_lon")

    window_speed = speed[indices]
    finite_speed = window_speed[np.isfinite(window_speed)]
    approach_speed = float(speed[start]) if np.isfinite(speed[start]) else math.nan
    speed_delta = (
        approach_speed - float(finite_speed[0])
        if np.isfinite(approach_speed) and len(finite_speed)
        else math.nan
    )

    def finite_mean(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if len(finite) else math.nan

    def finite_max(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.max(finite)) if len(finite) else math.nan

    speed_mps = window_speed / 3.6
    window_lat_g = lat_g[indices]
    curvature = np.full(len(indices), np.nan, dtype=float)
    valid_curve = (
        np.isfinite(speed_mps)
        & np.isfinite(window_lat_g)
        & (np.abs(speed_mps) >= 5.0)
    )
    # a_lat = v^2 / r, therefore signed curvature ~= a_lat / v^2.
    curvature[valid_curve] = (
        window_lat_g[valid_curve] * 9.80665
    ) / np.square(speed_mps[valid_curve])

    window_times = times_ms[indices]
    duration_s = (
        max(0.0, float(window_times[-1] - window_times[0]) / 1_000.0)
        if len(window_times) > 1
        else 0.0
    )

    return {
        "context_lookback_s": duration_s,
        "approach_speed_kmh": approach_speed,
        "approach_speed_delta_kmh": speed_delta,
        "approach_throttle_mean": finite_mean(throttle[indices]),
        "approach_abs_steering_mean": finite_mean(np.abs(steering[indices])),
        "approach_abs_steering_max": finite_max(np.abs(steering[indices])),
        "approach_mean_abs_lateral_g": finite_mean(np.abs(window_lat_g)),
        "approach_mean_abs_longitudinal_g": finite_mean(np.abs(lon_g[indices])),
        "approach_curvature_proxy_signed_mean_1pm": finite_mean(curvature),
        "approach_curvature_proxy_abs_mean_1pm": finite_mean(np.abs(curvature)),
    }


def build_modeling_segment_table(
    segment_features: pd.DataFrame,
    style_features: pd.DataFrame,
    *,
    processed_root: Path,
    lookback_seconds: float = 0.75,
) -> pd.DataFrame:
    required = {
        "lap_id",
        "track",
        "resampled_file",
        "brake_start_index",
        "segment_id",
    }
    missing = sorted(required.difference(segment_features.columns))
    if missing:
        raise ValueError(f"validated segment features missing required columns: {missing}")

    style_flags = style_features.copy()
    keep_style = [
        column
        for column in (
            "lap_id",
            "clean_baseline_candidate",
            "style_model_ready",
            "context_profile",
        )
        if column in style_flags.columns
    ]
    if "lap_id" not in keep_style:
        raise ValueError("validated style features are missing lap_id")
    style_flags = style_flags.loc[:, keep_style].drop_duplicates("lap_id")

    working = segment_features.copy().merge(
        style_flags,
        on="lap_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_lap"),
    )

    context_records: list[dict[str, float]] = []
    cache: dict[str, pd.DataFrame] = {}
    for _, row in working.iterrows():
        resolved = _resolve_existing_path(row["resampled_file"], processed_root=processed_root)
        cache_key = str(resolved)
        try:
            if cache_key not in cache:
                cache[cache_key] = pd.read_parquet(resolved, engine="pyarrow")
            context = extract_causal_pre_entry_context(
                cache[cache_key],
                int(row["brake_start_index"]),
                lookback_seconds=float(lookback_seconds),
            )
            context["context_error"] = ""
        except Exception as exc:
            context = {column: math.nan for column in CONTEXT_FEATURE_COLUMNS}
            context["context_lookback_s"] = math.nan
            context["context_error"] = f"{type(exc).__name__}: {exc}"
        context_records.append(context)

    contexts = pd.DataFrame.from_records(context_records, index=working.index)
    working = pd.concat([working, contexts], axis=1)

    session_uid = working.get("session_uid", pd.Series(index=working.index, dtype=object))
    session_uid_text = session_uid.fillna("").astype(str).str.strip()
    working["session_key"] = (
        working["track"].astype(str).str.lower() + ":" + session_uid_text
    )
    working["session_uid_available"] = session_uid_text.ne("") & session_uid_text.ne("nan")
    working["context_complete"] = _finite_row(working, CONTEXT_FEATURE_COLUMNS)
    working["core_behavior_complete"] = _finite_row(
        working, CORE_BEHAVIOR_TARGET_COLUMNS
    )
    working["motion_behavior_complete"] = _finite_row(
        working, MOTION_BEHAVIOR_TARGET_COLUMNS
    )

    segment_ok = (
        _as_bool(working["segment_quality_eligible"])
        if "segment_quality_eligible" in working.columns
        else pd.Series(True, index=working.index)
    )
    lap_clean = (
        _as_bool(working["clean_baseline_candidate"])
        if "clean_baseline_candidate" in working.columns
        else pd.Series(False, index=working.index)
    )
    working["baseline_segment_candidate"] = segment_ok & lap_clean
    working["knn_v1_candidate"] = (
        working["baseline_segment_candidate"].astype(bool)
        & working["session_uid_available"].astype(bool)
        & working["context_complete"].astype(bool)
        & working["core_behavior_complete"].astype(bool)
    )

    # Important contract: these IDs stay available for grouped evaluation, but
    # are not part of CONTEXT_FEATURE_COLUMNS used by cross-track k-NN.
    working["track_identity_for_evaluation_only"] = working["track"].astype(str)
    working["context_version"] = "causal_proxy_v1"
    return working


def build_session_table(modeling_segments: pd.DataFrame) -> pd.DataFrame:
    if modeling_segments.empty:
        return pd.DataFrame(
            columns=[
                "session_key",
                "track",
                "session_uid",
                "lap_count",
                "segment_count",
                "baseline_segment_count",
                "knn_v1_candidate_count",
            ]
        )

    grouped = (
        modeling_segments.groupby(
            ["session_key", "track", "session_uid"],
            dropna=False,
            sort=True,
        )
        .agg(
            lap_count=("lap_id", "nunique"),
            segment_count=("segment_id", "size"),
            baseline_segment_count=("baseline_segment_candidate", "sum"),
            knn_v1_candidate_count=("knn_v1_candidate", "sum"),
        )
        .reset_index()
    )
    for column in (
        "lap_count",
        "segment_count",
        "baseline_segment_count",
        "knn_v1_candidate_count",
    ):
        grouped[column] = grouped[column].astype(int)
    return grouped


def _markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 3F Modeling Contract",
            "",
            "Phase 3F freezes the interface between offline feature extraction and Phase 4 modeling.",
            "No raw lap or Phase 3A-3E output is modified.",
            "",
            "## Output",
            "",
            f"- Modeling segment rows: {summary['modeling_segment_rows']}",
            f"- k-NN v1 candidate rows: {summary['knn_v1_candidate_rows']}",
            f"- Sessions represented: {summary['session_count']}",
            f"- Tracks represented: {summary['track_count']}",
            f"- Rows with usable session IDs: {summary['session_uid_rows']}",
            "",
            "## Locked modeling rules",
            "",
            "- Query context and driver-behavior targets are separate column groups.",
            "- Neighbor lookup must not use peak brake, minimum speed, throttle recovery, or other future outcomes.",
            "- `track`, `zone_id`, and lap identity are evaluation metadata, not cross-track k-NN context features.",
            "- Dataset splits must be grouped by `session_key`; do not randomly split laps from the same session across train/test.",
            "- The current context is a causal telemetry proxy derived only from the pre-entry window. It is not claimed to be pure track geometry.",
            "- Explicit maneuver category labels are deferred; descriptors stay continuous.",
            "",
            "## Phase 4 entry point",
            "",
            "Use `modeling_segments.csv`, filter `knn_v1_candidate == True`, normalize only the CONTEXT feature group on the training split, retrieve neighbors by context similarity, and predict BEHAVIOR targets from neighbor behavior.",
            "",
            "A geometry-rich context can be added later without changing the behavior-target contract.",
            "",
        ]
    )


def write_phase3f_outputs(
    processed_root: Path,
    *,
    lookback_seconds: float = 0.75,
) -> dict[str, Any]:
    processed_root = Path(processed_root)
    segment_path = processed_root / "segment_features_validated.csv"
    style_path = processed_root / "style_features_validated.csv"
    if not segment_path.exists():
        raise FileNotFoundError(f"missing Phase 3E output: {segment_path}")
    if not style_path.exists():
        raise FileNotFoundError(f"missing Phase 3E output: {style_path}")

    segments = pd.read_csv(segment_path)
    styles = pd.read_csv(style_path)
    modeling = build_modeling_segment_table(
        segments,
        styles,
        processed_root=processed_root,
        lookback_seconds=float(lookback_seconds),
    )
    sessions = build_session_table(modeling)

    modeling_path = processed_root / "modeling_segments.csv"
    sessions_path = processed_root / "modeling_sessions.csv"
    modeling.to_csv(modeling_path, index=False)
    sessions.to_csv(sessions_path, index=False)

    try:
        modeling.to_parquet(
            processed_root / "modeling_segments.parquet",
            engine="pyarrow",
            index=False,
            compression="zstd",
        )
    except (ImportError, ModuleNotFoundError):
        pass

    session_ok = int(_as_bool(modeling["session_uid_available"]).sum()) if not modeling.empty else 0
    context_ok = int(_as_bool(modeling["context_complete"]).sum()) if not modeling.empty else 0
    knn_ok = int(_as_bool(modeling["knn_v1_candidate"]).sum()) if not modeling.empty else 0
    summary: dict[str, Any] = {
        "phase": "3F",
        "context_version": "causal_proxy_v1",
        "context_lookback_seconds": float(lookback_seconds),
        "modeling_segment_rows": int(len(modeling)),
        "context_complete_rows": context_ok,
        "knn_v1_candidate_rows": knn_ok,
        "session_uid_rows": session_ok,
        "session_count": int(modeling.loc[modeling["session_uid_available"], "session_key"].nunique()) if not modeling.empty else 0,
        "track_count": int(modeling["track"].nunique()) if not modeling.empty else 0,
        "context_features": list(CONTEXT_FEATURE_COLUMNS),
        "core_behavior_targets": list(CORE_BEHAVIOR_TARGET_COLUMNS),
        "motion_behavior_targets": list(MOTION_BEHAVIOR_TARGET_COLUMNS),
        "cross_track_geometry_status": "not_required_for_phase3f; causal_proxy_v1 only",
        "session_split_rule": "group by session_key; never random lap-level leakage",
        "maneuver_taxonomy": "deferred; continuous descriptors only",
        "modeling_segments_csv": str(modeling_path),
        "modeling_sessions_csv": str(sessions_path),
    }

    json_path = processed_root / "phase3f_contract.json"
    markdown_path = processed_root / "phase3f_contract.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(summary), encoding="utf-8")
    summary["contract_json"] = str(json_path)
    summary["contract_markdown"] = str(markdown_path)
    return summary


def format_phase3f_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "PHASE 3F MODELING CONTRACT + CAUSAL PREP",
            "------------------------------------------",
            f"Modeling segment rows:       {summary['modeling_segment_rows']}",
            f"Context-complete rows:       {summary['context_complete_rows']}",
            f"k-NN v1 candidate rows:      {summary['knn_v1_candidate_rows']}",
            f"Sessions represented:        {summary['session_count']}",
            f"Tracks represented:          {summary['track_count']}",
            f"Session-ID rows:             {summary['session_uid_rows']}",
            "Context contract:            causal_proxy_v1",
            "Taxonomy:                    deferred (continuous descriptors)",
            "Split rule:                  session-grouped",
        ]
    )
