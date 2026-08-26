from __future__ import annotations

import argparse
from pathlib import Path

from f1_engineer.analysis.catalog import build_lap_catalog, format_catalog_report
from f1_engineer.analysis.processing import (
    format_processing_report,
    process_catalog_laps,
)
from f1_engineer.analysis.validation import ValidationConfig


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate F1 lap Parquet files, resample eligible laps by distance, "
            "and detect braking/corner segments."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw_laps"),
        help="Root directory containing per-track Parquet lap folders.",
    )
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="Optional track folder to scan, for example spa.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/lap_catalog.csv"),
        help="CSV catalog destination. Other processed outputs use its folder.",
    )
    parser.add_argument("--expected-hz", type=float, default=60.0)
    parser.add_argument("--hz-tolerance", type=float, default=7.0)
    parser.add_argument("--minimum-rows", type=int, default=1_000)
    parser.add_argument("--outlier-factor", type=float, default=1.20)
    parser.add_argument(
        "--allow-missing-enrichment",
        action="store_true",
        help=(
            "Allow otherwise healthy core-only laps into training eligibility. "
            "By default the complete Phase 3 feature schema is required."
        ),
    )
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="Stop after Phase 3A validation/catalog generation.",
    )
    parser.add_argument(
        "--training-only",
        action="store_true",
        help=(
            "Process only fully enriched training-eligible laps. By default all "
            "core-eligible laps are resampled and segmented."
        ),
    )
    parser.add_argument(
        "--include-outliers",
        action="store_true",
        help="Also process laps flagged above the same-track outlier threshold.",
    )
    parser.add_argument("--grid-points", type=int, default=2_001)
    parser.add_argument("--brake-threshold", type=float, default=0.30)
    parser.add_argument("--min-brake-seconds", type=float, default=0.50)
    parser.add_argument("--merge-gap-seconds", type=float, default=0.20)
    parser.add_argument("--full-throttle-threshold", type=float, default=0.95)
    parser.add_argument(
        "--slip-threshold",
        type=float,
        default=0.05,
        help="Absolute wheel-slip threshold used for slip-frequency features.",
    )
    parser.add_argument(
        "--quality-zone-gap-m",
        type=float,
        default=150.0,
        help="Maximum brake-start gap used to group the same braking zone.",
    )
    parser.add_argument(
        "--quality-zone-support",
        type=float,
        default=0.50,
        help="Minimum fraction of laps required for a canonical braking zone.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip per-lap PNG plots of detected braking segments.",
    )
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    config = ValidationConfig(
        expected_sample_rate_hz=float(args.expected_hz),
        sample_rate_tolerance_hz=float(args.hz_tolerance),
        minimum_rows=int(args.minimum_rows),
        require_enrichment_for_training=not bool(args.allow_missing_enrichment),
    )
    catalog = build_lap_catalog(
        data_root=args.data,
        output_path=args.output,
        track=(args.track.lower() if args.track else None),
        config=config,
        outlier_factor=float(args.outlier_factor),
    )
    print(format_catalog_report(catalog))
    print(f"\nCatalog: {args.output}")
    print(f"Summary: {args.output.with_suffix('.summary.json')}")

    if args.catalog_only:
        return 0

    _, _, summary = process_catalog_laps(
        catalog_path=args.output,
        output_root=args.output.parent,
        training_only=bool(args.training_only),
        include_outliers=bool(args.include_outliers),
        grid_points=int(args.grid_points),
        brake_threshold=float(args.brake_threshold),
        minimum_brake_duration_s=float(args.min_brake_seconds),
        merge_gap_s=float(args.merge_gap_seconds),
        full_throttle_threshold=float(args.full_throttle_threshold),
        slip_threshold=float(args.slip_threshold),
        quality_zone_gap_m=float(args.quality_zone_gap_m),
        quality_canonical_zone_min_support=float(args.quality_zone_support),
        make_plots=not bool(args.no_plots),
    )
    print("\n" + format_processing_report(summary))
    print(f"\nSegments: {args.output.parent / 'braking_segments.csv'}")
    print(f"Processing: {args.output.parent / 'segment_processing.csv'}")
    print(f"Summary: {args.output.parent / 'segment_summary.json'}")
    print(f"Segment features: {args.output.parent / 'segment_features.csv'}")
    print(f"Style features:   {args.output.parent / 'style_features.csv'}")
    print(f"Feature summary:  {args.output.parent / 'feature_summary.json'}")
    print(f"Quality report:   {args.output.parent / 'feature_quality_report.md'}")
    print(f"Validated styles: {args.output.parent / 'style_features_validated.csv'}")
    print(f"Validated zones:  {args.output.parent / 'braking_zone_quality.csv'}")
    if not args.no_plots:
        print(f"Plots: {args.output.parent / 'segment_plots'}")
    return 0 if int(summary["failed_laps"]) == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    return run_from_args(args)
