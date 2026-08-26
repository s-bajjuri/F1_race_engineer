from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from f1_engineer.analysis.features import (
    extract_lap_style_features,
    extract_segment_features,
)
from f1_engineer.analysis.quality import (
    FeatureQualityConfig,
    format_quality_report,
    write_feature_quality_outputs,
)
from f1_engineer.analysis.resampling import resample_lap_by_distance
from f1_engineer.analysis.segmentation import (
    detect_braking_segments,
    plot_braking_segments,
)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _resolve_lap_path(value: object, catalog_path: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path

    if path.exists():
        return path
    for parent in [catalog_path.parent, *catalog_path.parents]:
        candidate = parent / path
        if candidate.exists():
            return candidate
    return path


def _write_feature_outputs(
    output_root: Path,
    segment_features: pd.DataFrame,
    style_features: pd.DataFrame,
) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)

    segment_csv = output_root / "segment_features.csv"
    segment_parquet = output_root / "segment_features.parquet"
    style_csv = output_root / "style_features.csv"
    style_parquet = output_root / "style_features.parquet"

    segment_features.to_csv(segment_csv, index=False)
    style_features.to_csv(style_csv, index=False)
    segment_features.to_parquet(
        segment_parquet,
        engine="pyarrow",
        index=False,
        compression="zstd",
    )
    style_features.to_parquet(
        style_parquet,
        engine="pyarrow",
        index=False,
        compression="zstd",
    )
    return {
        "segment_features_csv": str(segment_csv),
        "segment_features_parquet": str(segment_parquet),
        "style_features_csv": str(style_csv),
        "style_features_parquet": str(style_parquet),
    }


def process_catalog_laps(
    catalog_path: Path,
    output_root: Path,
    *,
    training_only: bool = False,
    include_outliers: bool = False,
    grid_points: int = 2_001,
    brake_threshold: float = 0.30,
    minimum_brake_duration_s: float = 0.50,
    merge_gap_s: float = 0.20,
    full_throttle_threshold: float = 0.95,
    slip_threshold: float = 0.05,
    quality_zone_gap_m: float = 150.0,
    quality_canonical_zone_min_support: float = 0.50,
    make_plots: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Resample eligible laps, detect segments, and extract style features.

    Returns a per-lap processing table, a boundary table, and a JSON-ready
    summary. Individual file failures are recorded without stopping the batch.
    """

    catalog_path = Path(catalog_path)
    output_root = Path(output_root)
    catalog = pd.read_csv(catalog_path)

    eligibility_column = "training_eligible" if training_only else "core_eligible"
    if eligibility_column not in catalog.columns:
        raise ValueError(f"catalog does not contain {eligibility_column}")

    selected = catalog[_as_bool(catalog[eligibility_column])].copy()
    if not include_outliers and "outlier_lap_time" in selected.columns:
        selected = selected[~_as_bool(selected["outlier_lap_time"])]

    resampled_root = output_root / "resampled"
    plot_root = output_root / "segment_plots"
    processing_records: list[dict[str, object]] = []
    segment_records: list[dict[str, object]] = []
    segment_feature_records: list[dict[str, object]] = []
    style_feature_records: list[dict[str, object]] = []

    for _, catalog_row in selected.iterrows():
        source_path = _resolve_lap_path(catalog_row["file_path"], catalog_path)
        track = str(catalog_row.get("track", source_path.parent.name)).lower()
        lap_id = source_path.stem
        record: dict[str, object] = {
            "lap_id": lap_id,
            "track": track,
            "source_file": str(source_path),
            "status": "error",
            "resampled_file": "",
            "plot_file": "",
            "grid_points": int(grid_points),
            "segment_count": 0,
            "segment_features": 0,
            "style_features": 0,
            "error": "",
        }

        try:
            raw = pd.read_parquet(source_path, engine="pyarrow")
            resampled = resample_lap_by_distance(raw, points=int(grid_points))

            resampled_path = (
                resampled_root / track / f"{source_path.stem}_resampled.parquet"
            )
            resampled_path.parent.mkdir(parents=True, exist_ok=True)
            resampled.to_parquet(
                resampled_path,
                engine="pyarrow",
                index=False,
                compression="zstd",
            )

            segments = detect_braking_segments(
                resampled,
                brake_threshold=float(brake_threshold),
                minimum_brake_duration_s=float(minimum_brake_duration_s),
                merge_gap_s=float(merge_gap_s),
                full_throttle_threshold=float(full_throttle_threshold),
            )

            lap_segment_features: list[dict[str, object]] = []
            for segment in segments:
                common = {
                    "lap_id": lap_id,
                    "track": track,
                    "source_file": str(source_path),
                    "resampled_file": str(resampled_path),
                    "lap_time_ms": catalog_row.get("lap_time_ms"),
                    "session_uid": catalog_row.get("session_uid"),
                    "lap_number": catalog_row.get("lap_number"),
                }
                boundary_record = {**common, **segment.to_record()}
                segment_records.append(boundary_record)

                feature_values = extract_segment_features(
                    resampled,
                    segment,
                    slip_threshold=float(slip_threshold),
                )
                feature_record = {**boundary_record, **feature_values}
                segment_feature_records.append(feature_record)
                lap_segment_features.append(feature_record)

            style_record = extract_lap_style_features(
                resampled,
                segments,
                lap_segment_features,
                lap_id=lap_id,
                track=track,
                source_file=str(source_path),
                catalog_row=catalog_row.to_dict(),
                slip_threshold=float(slip_threshold),
            )
            style_feature_records.append(style_record)

            plot_path = plot_root / track / f"{source_path.stem}_segments.png"
            if make_plots:
                lap_time = catalog_row.get("lap_time_s")
                time_label = (
                    f"{float(lap_time):.3f} s"
                    if pd.notna(lap_time)
                    else "unknown time"
                )
                plot_braking_segments(
                    resampled,
                    segments,
                    plot_path,
                    title=f"{track.upper()} — {source_path.stem} — {time_label}",
                )

            record.update(
                {
                    "status": "ok",
                    "resampled_file": str(resampled_path),
                    "plot_file": str(plot_path) if make_plots else "",
                    "segment_count": len(segments),
                    "segment_features": len(lap_segment_features),
                    "style_features": 1,
                }
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"

        processing_records.append(record)

    processing = pd.DataFrame.from_records(processing_records)
    segments_frame = pd.DataFrame.from_records(segment_records)
    segment_features_frame = pd.DataFrame.from_records(segment_feature_records)
    style_features_frame = pd.DataFrame.from_records(style_feature_records)

    output_root.mkdir(parents=True, exist_ok=True)
    processing_path = output_root / "segment_processing.csv"
    segments_path = output_root / "braking_segments.csv"
    processing.to_csv(processing_path, index=False)
    segments_frame.to_csv(segments_path, index=False)
    feature_paths = _write_feature_outputs(
        output_root,
        segment_features_frame,
        style_features_frame,
    )
    quality_summary = write_feature_quality_outputs(
        output_root,
        style_features_frame,
        segment_features_frame,
        config=FeatureQualityConfig(
            zone_gap_m=float(quality_zone_gap_m),
            canonical_zone_min_support=float(quality_canonical_zone_min_support),
        ),
    )

    successful = (
        int((processing["status"] == "ok").sum()) if not processing.empty else 0
    )
    failed = int(len(processing) - successful)
    complete_style_rows = 0
    if not style_features_frame.empty and "feature_complete" in style_features_frame:
        complete_style_rows = int(_as_bool(style_features_frame["feature_complete"]).sum())

    summary: dict[str, Any] = {
        "catalog_path": str(catalog_path),
        "output_root": str(output_root),
        "eligibility": eligibility_column,
        "include_outliers": bool(include_outliers),
        "selected_laps": int(len(selected)),
        "processed_laps": successful,
        "failed_laps": failed,
        "detected_segments": int(len(segments_frame)),
        "segment_feature_rows": int(len(segment_features_frame)),
        "style_feature_rows": int(len(style_features_frame)),
        "complete_style_feature_rows": complete_style_rows,
        "grid_points": int(grid_points),
        "parameters": {
            "brake_threshold": float(brake_threshold),
            "minimum_brake_duration_s": float(minimum_brake_duration_s),
            "merge_gap_s": float(merge_gap_s),
            "full_throttle_threshold": float(full_throttle_threshold),
            "slip_threshold": float(slip_threshold),
            "quality_zone_gap_m": float(quality_zone_gap_m),
            "quality_canonical_zone_min_support": float(
                quality_canonical_zone_min_support
            ),
        },
        "processing_catalog": str(processing_path),
        "segments_catalog": str(segments_path),
        "resampled_directory": str(resampled_root),
        "plot_directory": str(plot_root) if make_plots else None,
        **feature_paths,
        "quality": quality_summary,
    }
    (output_root / "segment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_root / "feature_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return processing, segments_frame, summary


def format_processing_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "PHASE 3B–3D DISTANCE, SEGMENT + FEATURE PIPELINE",
            "-------------------------------------------------",
            f"Eligible laps selected:    {summary['selected_laps']}",
            f"Processed successfully:    {summary['processed_laps']}",
            f"Failed:                    {summary['failed_laps']}",
            f"Braking segments:          {summary['detected_segments']}",
            f"Segment feature rows:      {summary['segment_feature_rows']}",
            f"Style feature rows:        {summary['style_feature_rows']}",
            f"Complete style rows:       {summary['complete_style_feature_rows']}",
            f"Distance grid points:      {summary['grid_points']}",
            "",
            format_quality_report(summary["quality"]),
        ]
    )
