"""Offline lap validation, feature preparation, and causal modeling utilities."""

from f1_engineer.analysis.catalog import build_lap_catalog, scan_lap_files
from f1_engineer.analysis.causal_segments import (
    CompletedLiveSegment,
    LiveSegmentSnapshot,
    LiveTelemetrySample,
    StreamingBrakeSegmenter,
)
from f1_engineer.analysis.corner_segments import (
    CORNER_BEHAVIOR_COLUMNS,
    GEOMETRY_CONTEXT_COLUMNS,
    CornerSegmentConfig,
    CornerWindow,
    build_corner_windows,
    extract_corner_behavior,
    extract_lap_corner_segments,
)
from f1_engineer.analysis.features import (
    extract_lap_style_features,
    extract_segment_features,
)
from f1_engineer.analysis.geometry import (
    GeometryConfig,
    TrackGeometry,
    attach_track_geometry,
    audit_geometry_coverage,
    build_track_geometry,
    has_geometry,
    track_output_dir,
    write_track_geometry,
)
from f1_engineer.analysis.modeling import (
    CONTEXT_FEATURE_COLUMNS,
    CORE_BEHAVIOR_TARGET_COLUMNS,
    MOTION_BEHAVIOR_TARGET_COLUMNS,
    build_modeling_segment_table,
    extract_causal_pre_entry_context,
    write_phase3f_outputs,
)
from f1_engineer.analysis.processing import process_catalog_laps
from f1_engineer.analysis.quality import (
    FeatureQualityConfig,
    build_feature_quality_table,
    validate_style_features,
    write_feature_quality_outputs,
)
from f1_engineer.analysis.resampling import resample_lap_by_distance
from f1_engineer.analysis.segmentation import (
    BrakingSegment,
    detect_braking_segments,
)
from f1_engineer.analysis.validation import (
    ValidationConfig,
    validate_lap_dataframe,
)

__all__ = [
    "BrakingSegment",
    "CompletedLiveSegment",
    "CONTEXT_FEATURE_COLUMNS",
    "CORE_BEHAVIOR_TARGET_COLUMNS",
    "CORNER_BEHAVIOR_COLUMNS",
    "CornerSegmentConfig",
    "CornerWindow",
    "FeatureQualityConfig",
    "GEOMETRY_CONTEXT_COLUMNS",
    "GeometryConfig",
    "LiveSegmentSnapshot",
    "LiveTelemetrySample",
    "MOTION_BEHAVIOR_TARGET_COLUMNS",
    "StreamingBrakeSegmenter",
    "TrackGeometry",
    "ValidationConfig",
    "attach_track_geometry",
    "audit_geometry_coverage",
    "build_corner_windows",
    "build_feature_quality_table",
    "build_lap_catalog",
    "build_modeling_segment_table",
    "build_track_geometry",
    "detect_braking_segments",
    "extract_corner_behavior",
    "extract_lap_corner_segments",
    "has_geometry",
    "extract_causal_pre_entry_context",
    "extract_lap_style_features",
    "extract_segment_features",
    "process_catalog_laps",
    "resample_lap_by_distance",
    "scan_lap_files",
    "track_output_dir",
    "validate_lap_dataframe",
    "validate_style_features",
    "write_feature_quality_outputs",
    "write_phase3f_outputs",
    "write_track_geometry",
]
