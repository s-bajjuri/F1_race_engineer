from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class FeatureQualityConfig:
    """Thresholds for Phase 3E feature-quality checks.

    The checks are deliberately descriptive. They do not delete or overwrite
    raw telemetry or the original Phase 3D feature tables.
    """

    zone_gap_m: float = 150.0
    canonical_zone_min_support: float = 0.50
    fuel_dynamic_min_kg: float = 0.05
    ers_dynamic_min_j: float = 10_000.0
    tyre_wear_dynamic_min: float = 0.001
    tyre_temp_dynamic_min_c: float = 0.25
    stationary_speed_threshold_kmh: float = 20.0
    maximum_feature_missing_fraction: float = 0.25
    minimum_correlation_rows: int = 5


STYLE_METADATA_COLUMNS: frozenset[str] = frozenset(
    {
        "lap_id",
        "track",
        "source_file",
        "session_uid",
        "lap_number",
        "sample_rate_hz",
        "training_eligible",
        "feature_complete",
    }
)

CORE_STYLE_COLUMNS: tuple[str, ...] = (
    "brake_aggressiveness",
    "brake_peak_consistency",
    "mean_brake_force",
    "mean_braking_distance_m",
    "brake_release_smoothness",
    "throttle_patience_m",
    "throttle_patience_s",
    "steering_smoothness",
    "steering_rate_smoothness",
    "mean_segment_minimum_speed_kmh",
    "throttle_brake_overlap_pct",
    "coast_time_pct",
    "full_throttle_time_pct",
    "hard_braking_time_pct",
    "mean_speed_kmh",
    "maximum_speed_kmh",
    "whole_lap_steering_input_std",
    "whole_lap_steering_rate_std_per_s",
)

MOTION_STYLE_COLUMNS: tuple[str, ...] = (
    "mean_abs_lateral_g",
    "peak_abs_lateral_g",
    "mean_abs_longitudinal_g",
    "peak_abs_longitudinal_g",
    "slip_frequency",
    "peak_front_slip_abs",
    "peak_rear_slip_abs",
    "mean_front_slip_abs",
    "mean_rear_slip_abs",
    "rear_slip_bias",
)

STRATEGY_STYLE_COLUMNS: tuple[str, ...] = (
    "fuel_used_kg",
    "tyre_wear_delta",
    "mean_tyre_temp_c",
    "max_tyre_temp_c",
    "mean_tyre_temp_spread_c",
    "ers_total_deployed_j",
    "ers_early_lap_pct",
    "ers_middle_lap_pct",
    "ers_late_lap_pct",
)

CONTEXT_SENSITIVE_FEATURES: frozenset[str] = frozenset(STRATEGY_STYLE_COLUMNS)

SEGMENT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "brake_start_m",
    "brake_end_m",
    "peak_brake",
    "minimum_speed_kmh",
    "minimum_speed_m",
    "segment_end_m",
    "braking_distance_m",
    "mean_brake",
    "brake_release_smoothness_per_s",
    "steering_input_std",
    "steering_rate_std_per_s",
)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _all_finite(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    columns = tuple(columns)
    if frame.empty:
        return pd.Series(dtype=bool, index=frame.index)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        return pd.Series(False, index=frame.index, dtype=bool)
    matrix = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return matrix.notna().all(axis=1) & np.isfinite(matrix).all(axis=1)


def _cluster_segment_zones(
    segment_features: pd.DataFrame,
    *,
    config: FeatureQualityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign track-local braking-zone IDs using start-distance proximity."""

    required = {"lap_id", "track", "brake_start_m"}
    missing = sorted(required.difference(segment_features.columns))
    if missing:
        raise ValueError(f"segment feature table is missing required columns: {missing}")

    validated_parts: list[pd.DataFrame] = []
    zone_records: list[dict[str, Any]] = []

    for track, track_frame in segment_features.groupby("track", dropna=False, sort=True):
        working = track_frame.copy()
        working["_original_index"] = working.index
        working["brake_start_m"] = _numeric(working["brake_start_m"])
        working = working.sort_values("brake_start_m", kind="mergesort")

        zone_numbers: list[int] = []
        current_zone = 0
        previous_start: float | None = None
        for start in working["brake_start_m"].to_numpy(dtype=float):
            if not np.isfinite(start):
                current_zone += 1
            elif previous_start is None or start - previous_start > config.zone_gap_m:
                current_zone += 1
            zone_numbers.append(current_zone)
            if np.isfinite(start):
                previous_start = float(start)

        working["zone_number"] = zone_numbers
        total_laps = max(1, int(working["lap_id"].nunique()))

        zone_stats = (
            working.groupby("zone_number", sort=True)
            .agg(
                zone_center_m=("brake_start_m", "mean"),
                zone_start_std_m=("brake_start_m", "std"),
                zone_segment_count=("lap_id", "size"),
                zone_support_laps=("lap_id", "nunique"),
            )
            .reset_index()
        )
        zone_stats["zone_support_fraction"] = (
            zone_stats["zone_support_laps"] / float(total_laps)
        )
        zone_stats["canonical_zone"] = (
            zone_stats["zone_support_fraction"]
            >= float(config.canonical_zone_min_support)
        )
        track_label = str(track)
        zone_stats["track"] = track_label
        zone_stats["zone_id"] = zone_stats["zone_number"].map(
            lambda value: f"{track_label}_z{int(value):02d}"
        )

        working = working.merge(
            zone_stats[
                [
                    "zone_number",
                    "zone_id",
                    "zone_center_m",
                    "zone_start_std_m",
                    "zone_segment_count",
                    "zone_support_laps",
                    "zone_support_fraction",
                    "canonical_zone",
                ]
            ],
            on="zone_number",
            how="left",
            validate="many_to_one",
        )

        available_required = [
            column for column in SEGMENT_REQUIRED_COLUMNS if column in working.columns
        ]
        if len(available_required) != len(SEGMENT_REQUIRED_COLUMNS):
            working["segment_required_complete"] = False
        else:
            working["segment_required_complete"] = _all_finite(
                working, SEGMENT_REQUIRED_COLUMNS
            )
        working["rare_zone"] = ~_as_bool(working["canonical_zone"])
        working["segment_quality_eligible"] = (
            _as_bool(working["canonical_zone"])
            & _as_bool(working["segment_required_complete"])
        )

        def segment_flags(row: pd.Series) -> str:
            flags: list[str] = []
            if bool(row.get("rare_zone", False)):
                flags.append("rare_braking_zone")
            if not bool(row.get("segment_required_complete", False)):
                flags.append("missing_required_segment_feature")
            if not bool(row.get("full_throttle_found", True)):
                flags.append("no_sustained_full_throttle_before_segment_end")
            return ";".join(flags)

        working["segment_quality_flags"] = working.apply(segment_flags, axis=1)
        validated_parts.append(working.sort_values("_original_index"))
        zone_records.extend(zone_stats.to_dict(orient="records"))

    if not validated_parts:
        return segment_features.copy(), pd.DataFrame.from_records(zone_records)

    validated = pd.concat(validated_parts, ignore_index=True)
    validated = validated.drop(columns=["_original_index"], errors="ignore")
    zones = pd.DataFrame.from_records(zone_records)
    if not zones.empty:
        zones = zones[
            [
                "track",
                "zone_id",
                "zone_number",
                "zone_center_m",
                "zone_start_std_m",
                "zone_segment_count",
                "zone_support_laps",
                "zone_support_fraction",
                "canonical_zone",
            ]
        ].sort_values(["track", "zone_number"], ignore_index=True)
    return validated, zones


def _context_profile(
    row: pd.Series,
    *,
    config: FeatureQualityConfig,
) -> tuple[str, int]:
    if not bool(row.get("motion_features_complete", False)):
        return "missing_enrichment", 0

    dynamic_signals = 0

    fuel = pd.to_numeric(pd.Series([row.get("fuel_used_kg")]), errors="coerce").iloc[0]
    ers = pd.to_numeric(
        pd.Series([row.get("ers_total_deployed_j")]), errors="coerce"
    ).iloc[0]
    wear = pd.to_numeric(
        pd.Series([row.get("tyre_wear_delta")]), errors="coerce"
    ).iloc[0]
    tyre_spread = pd.to_numeric(
        pd.Series([row.get("mean_tyre_temp_spread_c")]), errors="coerce"
    ).iloc[0]
    tyre_mean = pd.to_numeric(
        pd.Series([row.get("mean_tyre_temp_c")]), errors="coerce"
    ).iloc[0]
    tyre_max = pd.to_numeric(
        pd.Series([row.get("max_tyre_temp_c")]), errors="coerce"
    ).iloc[0]

    if np.isfinite(fuel) and fuel > config.fuel_dynamic_min_kg:
        dynamic_signals += 1
    if np.isfinite(ers) and ers > config.ers_dynamic_min_j:
        dynamic_signals += 1
    if np.isfinite(wear) and wear > config.tyre_wear_dynamic_min:
        dynamic_signals += 1
    tyre_dynamic = False
    if np.isfinite(tyre_spread) and tyre_spread > config.tyre_temp_dynamic_min_c:
        tyre_dynamic = True
    if (
        np.isfinite(tyre_mean)
        and np.isfinite(tyre_max)
        and abs(float(tyre_max) - float(tyre_mean)) > config.tyre_temp_dynamic_min_c
    ):
        tyre_dynamic = True
    if tyre_dynamic:
        dynamic_signals += 1

    if dynamic_signals == 0:
        return "static_strategy_context", dynamic_signals
    if dynamic_signals >= 2:
        return "dynamic_strategy_context", dynamic_signals
    return "mixed_strategy_context", dynamic_signals


def validate_style_features(
    style_features: pd.DataFrame,
    segment_quality: pd.DataFrame,
    zones: pd.DataFrame,
    *,
    config: FeatureQualityConfig,
) -> pd.DataFrame:
    """Add model-readiness and context flags without changing source features."""

    required = {"lap_id", "track", "lap_time_ms"}
    missing = sorted(required.difference(style_features.columns))
    if missing:
        raise ValueError(f"style feature table is missing required columns: {missing}")

    validated = style_features.copy()
    validated["core_style_complete"] = _all_finite(validated, CORE_STYLE_COLUMNS)
    validated["motion_features_complete"] = _all_finite(
        validated, MOTION_STYLE_COLUMNS
    )
    validated["strategy_features_complete"] = _all_finite(
        validated, STRATEGY_STYLE_COLUMNS
    )

    profile_values = validated.apply(
        lambda row: _context_profile(row, config=config), axis=1
    )
    validated["context_profile"] = [value[0] for value in profile_values]
    validated["strategy_dynamic_signal_count"] = [value[1] for value in profile_values]

    minimum_speed = (
        _numeric(validated["minimum_speed_kmh"])
        if "minimum_speed_kmh" in validated.columns
        else pd.Series(np.nan, index=validated.index)
    )
    validated["standing_start_or_stationary_context"] = (
        minimum_speed < float(config.stationary_speed_threshold_kmh)
    ).fillna(False)

    if segment_quality.empty:
        per_lap = pd.DataFrame(index=validated["lap_id"])
    else:
        per_lap = (
            segment_quality.groupby("lap_id", sort=False)
            .agg(
                detected_segment_count=("lap_id", "size"),
                canonical_segment_count=("canonical_zone", lambda value: int(_as_bool(value).sum())),
                rare_segment_count=("rare_zone", lambda value: int(_as_bool(value).sum())),
                required_segment_failures=(
                    "segment_required_complete",
                    lambda value: int((~_as_bool(value)).sum()),
                ),
            )
        )

    expected_by_track: dict[str, int] = {}
    if not zones.empty:
        for track, group in zones.groupby("track", sort=False):
            expected_by_track[str(track)] = int(_as_bool(group["canonical_zone"]).sum())

    validated = validated.merge(
        per_lap.reset_index(),
        on="lap_id",
        how="left",
        validate="one_to_one",
    )
    for column in (
        "detected_segment_count",
        "canonical_segment_count",
        "rare_segment_count",
        "required_segment_failures",
    ):
        validated[column] = (
            pd.to_numeric(validated[column], errors="coerce").fillna(0).astype(int)
        )
    validated["expected_canonical_zone_count"] = validated["track"].map(
        lambda track: expected_by_track.get(str(track), 0)
    )
    validated["canonical_zone_coverage_ok"] = (
        validated["canonical_segment_count"]
        >= validated["expected_canonical_zone_count"]
    )

    validated["style_model_ready"] = (
        _as_bool(validated["core_style_complete"])
        & _as_bool(validated["motion_features_complete"])
        & ~_as_bool(validated["standing_start_or_stationary_context"])
        & _as_bool(validated["canonical_zone_coverage_ok"])
    )
    validated["clean_baseline_candidate"] = (
        _as_bool(validated["style_model_ready"])
        & (validated["rare_segment_count"] == 0)
        & (validated["required_segment_failures"] == 0)
    )
    validated["full_context_model_ready"] = (
        _as_bool(validated["style_model_ready"])
        & _as_bool(validated["strategy_features_complete"])
        & (validated["context_profile"] == "dynamic_strategy_context")
    )

    def row_flags(row: pd.Series) -> str:
        flags: list[str] = []
        if not bool(row.get("core_style_complete", False)):
            flags.append("missing_core_style_feature")
        if not bool(row.get("motion_features_complete", False)):
            flags.append("missing_motion_enrichment")
        if row.get("context_profile") == "static_strategy_context":
            flags.append("static_strategy_telemetry")
        elif row.get("context_profile") == "mixed_strategy_context":
            flags.append("mixed_strategy_telemetry")
        if bool(row.get("standing_start_or_stationary_context", False)):
            flags.append("standing_start_or_stationary_context")
        if int(row.get("rare_segment_count", 0)) > 0:
            flags.append("rare_braking_event")
        if not bool(row.get("canonical_zone_coverage_ok", False)):
            flags.append("missing_canonical_braking_zone")
        if int(row.get("required_segment_failures", 0)) > 0:
            flags.append("incomplete_segment_feature")
        return ";".join(flags)

    validated["quality_flags"] = validated.apply(row_flags, axis=1)
    return validated


def build_feature_quality_table(
    style_features: pd.DataFrame,
    *,
    config: FeatureQualityConfig,
) -> pd.DataFrame:
    """Create one descriptive quality row per numeric style feature."""

    records: list[dict[str, Any]] = []
    lap_time = (
        _numeric(style_features["lap_time_ms"])
        if "lap_time_ms" in style_features.columns
        else pd.Series(dtype=float)
    )

    for column in style_features.columns:
        if column in STYLE_METADATA_COLUMNS:
            continue
        numeric = pd.to_numeric(style_features[column], errors="coerce")
        finite_mask = numeric.notna() & np.isfinite(numeric)
        finite = numeric[finite_mask]
        if finite.empty:
            continue

        unique_count = int(finite.nunique(dropna=True))
        mean = float(finite.mean())
        std = float(finite.std(ddof=0))
        minimum = float(finite.min())
        maximum = float(finite.max())
        coefficient_of_variation = (
            float(abs(std / mean)) if abs(mean) > 1e-12 else float("nan")
        )
        exact_constant = unique_count <= 1
        low_variation = (
            not exact_constant
            and std <= max(abs(mean) * 0.001, 1e-9)
        )

        correlation = float("nan")
        correlation_rows = 0
        if column != "lap_time_ms" and not lap_time.empty:
            pair = pd.DataFrame({"feature": numeric, "lap_time_ms": lap_time}).dropna()
            pair = pair[np.isfinite(pair["feature"]) & np.isfinite(pair["lap_time_ms"])]
            correlation_rows = int(len(pair))
            if correlation_rows >= int(config.minimum_correlation_rows):
                correlation = float(pair["feature"].corr(pair["lap_time_ms"]))

        missing_fraction = float(1.0 - finite_mask.mean())
        context_sensitive = column in CONTEXT_SENSITIVE_FEATURES
        if column == "lap_time_ms":
            recommendation = "target_or_metadata"
        elif exact_constant:
            recommendation = "exclude_constant_until_more_data"
        elif missing_fraction > config.maximum_feature_missing_fraction:
            recommendation = "needs_more_complete_laps"
        elif context_sensitive:
            recommendation = "use_only_with_matching_session_context"
        else:
            recommendation = "candidate_style_feature"

        records.append(
            {
                "feature": column,
                "non_null_rows": int(finite_mask.sum()),
                "missing_rows": int((~finite_mask).sum()),
                "missing_fraction": missing_fraction,
                "unique_count": unique_count,
                "minimum": minimum,
                "maximum": maximum,
                "mean": mean,
                "std": std,
                "coefficient_of_variation": coefficient_of_variation,
                "exact_constant": exact_constant,
                "low_variation": low_variation,
                "context_sensitive": context_sensitive,
                "correlation_with_lap_time": correlation,
                "correlation_rows": correlation_rows,
                "recommended_use": recommendation,
            }
        )

    return pd.DataFrame.from_records(records)


def _top_correlations(feature_quality: pd.DataFrame, limit: int = 6) -> list[dict[str, Any]]:
    if feature_quality.empty:
        return []
    candidates = feature_quality.copy()
    candidates = candidates[
        candidates["correlation_with_lap_time"].notna()
        & ~candidates["context_sensitive"].astype(bool)
        & (candidates["feature"] != "lap_time_ms")
    ]
    if candidates.empty:
        return []
    candidates["absolute_correlation"] = candidates[
        "correlation_with_lap_time"
    ].abs()
    return (
        candidates.sort_values("absolute_correlation", ascending=False)
        .head(limit)[
            [
                "feature",
                "correlation_with_lap_time",
                "correlation_rows",
            ]
        ]
        .to_dict(orient="records")
    )


def _markdown_report(summary: dict[str, Any], zones: pd.DataFrame) -> str:
    lines = [
        "# Phase 3E Feature Quality Report",
        "",
        "This report validates derived feature tables without modifying raw laps or the original Phase 3D outputs.",
        "",
        "## Dataset",
        "",
        f"- Style rows: {summary['style_rows']}",
        f"- Segment rows: {summary['segment_rows']}",
        f"- Technically complete style rows: {summary['technically_complete_style_rows']}",
        f"- Style-model-ready rows: {summary['style_model_ready_rows']}",
        f"- Clean baseline candidates: {summary['clean_baseline_candidate_rows']}",
        f"- Full dynamic-context rows: {summary['full_context_model_ready_rows']}",
        "",
        "## Session-context profiles",
        "",
    ]
    for profile, count in summary.get("context_profiles", {}).items():
        lines.append(f"- {profile}: {count}")

    lines.extend(["", "## Braking zones", ""])
    if zones.empty:
        lines.append("No braking zones were available.")
    else:
        for track, group in zones.groupby("track", sort=True):
            lines.append(f"### {str(track).upper()}")
            lines.append("")
            for _, row in group.iterrows():
                label = "canonical" if bool(row["canonical_zone"]) else "rare"
                lines.append(
                    f"- {row['zone_id']}: {float(row['zone_center_m']):.1f} m, "
                    f"support {int(row['zone_support_laps'])} laps "
                    f"({float(row['zone_support_fraction']):.0%}), {label}"
                )
            lines.append("")

    lines.extend(
        [
            "## Interpretation rules",
            "",
            "- `feature_complete` means the values are present, not that every session mode makes them informative.",
            "- Static fuel, ERS, wear, or tyre-temperature values are retained but marked as context-sensitive.",
            "- Rare braking zones are preserved as possible anomalies and excluded from canonical-zone summaries.",
            "- Correlations are descriptive only because the dataset is still small.",
            "",
            "## Strongest descriptive lap-time relationships",
            "",
        ]
    )
    top = summary.get("top_descriptive_correlations", [])
    if top:
        for item in top:
            lines.append(
                f"- {item['feature']}: r={float(item['correlation_with_lap_time']):+.3f} "
                f"across {int(item['correlation_rows'])} laps"
            )
    else:
        lines.append("Not enough usable rows for descriptive correlations.")
    lines.append("")
    return "\n".join(lines)


def write_feature_quality_outputs(
    output_root: Path,
    style_features: pd.DataFrame,
    segment_features: pd.DataFrame,
    *,
    config: FeatureQualityConfig | None = None,
) -> dict[str, Any]:
    """Run Phase 3E checks and write non-destructive validated outputs."""

    config = config or FeatureQualityConfig()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    segment_quality, zones = _cluster_segment_zones(
        segment_features, config=config
    )
    style_validated = validate_style_features(
        style_features,
        segment_quality,
        zones,
        config=config,
    )
    feature_quality = build_feature_quality_table(
        style_features, config=config
    )

    segment_quality_path = output_root / "segment_features_validated.csv"
    style_validated_path = output_root / "style_features_validated.csv"
    feature_quality_path = output_root / "feature_quality.csv"
    zone_path = output_root / "braking_zone_quality.csv"

    segment_quality.to_csv(segment_quality_path, index=False)
    style_validated.to_csv(style_validated_path, index=False)
    feature_quality.to_csv(feature_quality_path, index=False)
    zones.to_csv(zone_path, index=False)

    # Parquet mirrors are convenient for later PyTorch/pandas work. If pyarrow is
    # unexpectedly unavailable, the CSV outputs and report still remain usable.
    parquet_paths: dict[str, str | None] = {
        "validated_segment_features_parquet": None,
        "validated_style_features_parquet": None,
    }
    try:
        segment_parquet = output_root / "segment_features_validated.parquet"
        style_parquet = output_root / "style_features_validated.parquet"
        segment_quality.to_parquet(
            segment_parquet, engine="pyarrow", index=False, compression="zstd"
        )
        style_validated.to_parquet(
            style_parquet, engine="pyarrow", index=False, compression="zstd"
        )
        parquet_paths = {
            "validated_segment_features_parquet": str(segment_parquet),
            "validated_style_features_parquet": str(style_parquet),
        }
    except (ImportError, ModuleNotFoundError):
        pass

    context_profiles = (
        style_validated["context_profile"].value_counts(dropna=False).to_dict()
        if not style_validated.empty
        else {}
    )
    technically_complete = 0
    if "feature_complete" in style_validated.columns:
        technically_complete = int(_as_bool(style_validated["feature_complete"]).sum())

    summary: dict[str, Any] = {
        "style_rows": int(len(style_validated)),
        "segment_rows": int(len(segment_quality)),
        "technically_complete_style_rows": technically_complete,
        "style_model_ready_rows": int(_as_bool(style_validated["style_model_ready"]).sum()),
        "clean_baseline_candidate_rows": int(
            _as_bool(style_validated["clean_baseline_candidate"]).sum()
        ),
        "full_context_model_ready_rows": int(
            _as_bool(style_validated["full_context_model_ready"]).sum()
        ),
        "standing_start_or_stationary_rows": int(
            _as_bool(style_validated["standing_start_or_stationary_context"]).sum()
        ),
        "rare_segment_rows": int(_as_bool(segment_quality["rare_zone"]).sum()),
        "canonical_zone_count": int(_as_bool(zones["canonical_zone"]).sum())
        if not zones.empty
        else 0,
        "rare_zone_count": int((~_as_bool(zones["canonical_zone"])).sum())
        if not zones.empty
        else 0,
        "context_profiles": {str(key): int(value) for key, value in context_profiles.items()},
        "top_descriptive_correlations": _top_correlations(feature_quality),
        "parameters": {
            "zone_gap_m": float(config.zone_gap_m),
            "canonical_zone_min_support": float(
                config.canonical_zone_min_support
            ),
            "stationary_speed_threshold_kmh": float(
                config.stationary_speed_threshold_kmh
            ),
        },
        "validated_segment_features_csv": str(segment_quality_path),
        "validated_style_features_csv": str(style_validated_path),
        "feature_quality_csv": str(feature_quality_path),
        "braking_zone_quality_csv": str(zone_path),
        **parquet_paths,
    }

    report_json = output_root / "feature_quality_report.json"
    report_markdown = output_root / "feature_quality_report.md"
    report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_markdown.write_text(_markdown_report(summary, zones), encoding="utf-8")
    summary["feature_quality_report_json"] = str(report_json)
    summary["feature_quality_report_markdown"] = str(report_markdown)
    return summary


def format_quality_report(summary: dict[str, Any]) -> str:
    profiles = summary.get("context_profiles", {})
    profile_text = ", ".join(
        f"{name}={count}" for name, count in sorted(profiles.items())
    ) or "none"
    return "\n".join(
        [
            "PHASE 3E FEATURE QUALITY VALIDATION",
            "------------------------------------",
            f"Style-model-ready rows:       {summary['style_model_ready_rows']}",
            f"Clean baseline candidates:    {summary['clean_baseline_candidate_rows']}",
            f"Full dynamic-context rows:    {summary['full_context_model_ready_rows']}",
            f"Canonical braking zones:      {summary['canonical_zone_count']}",
            f"Rare braking zones:           {summary['rare_zone_count']}",
            f"Rare segment rows:            {summary['rare_segment_rows']}",
            f"Standing/stationary rows:     {summary['standing_start_or_stationary_rows']}",
            f"Context profiles:             {profile_text}",
        ]
    )
