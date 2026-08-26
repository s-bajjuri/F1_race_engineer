from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from f1_engineer.analysis.geometry import (
    GeometryConfig,
    REQUIRED_GEOMETRY_COLUMNS,
    audit_geometry_coverage,
    build_track_geometry,
    plot_track_geometry,
    track_output_dir,
    write_track_geometry,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a distance-indexed track geometry map and corner table from "
            "raw laps that carry Motion telemetry."
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
        help="Optional track folder to build, for example spa. Default is every track.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/track_geometry"),
        help="Destination directory for geometry maps.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Report per-track Motion geometry coverage and exit without building.",
    )
    parser.add_argument("--grid-spacing-m", type=float, default=2.0)
    parser.add_argument("--path-smoothing-m", type=float, default=12.0)
    parser.add_argument("--heading-smoothing-m", type=float, default=20.0)
    parser.add_argument(
        "--corner-minimum-radius-m",
        type=float,
        default=400.0,
        help="Curvature tighter than this radius counts as a corner.",
    )
    parser.add_argument("--minimum-corner-length-m", type=float, default=25.0)
    parser.add_argument("--corner-merge-gap-m", type=float, default=20.0)
    parser.add_argument(
        "--corner-split-ratio",
        type=float,
        default=0.6,
        help=(
            "Curvature must fall to this fraction of both neighbouring peaks "
            "before one turning stretch is split into separate corners."
        ),
    )
    parser.add_argument(
        "--corner-split-balance-ratio",
        type=float,
        default=0.5,
        help=(
            "Two apexes must be within this ratio of each other to be split "
            "apart. Stops a decaying corner exit becoming its own corner."
        ),
    )
    parser.add_argument(
        "--minimum-coverage",
        type=float,
        default=0.95,
        help="Minimum fraction of the lap a geometry lap must cover to be used.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip the track map PNG used for visual confirmation.",
    )
    return parser


def _geometry_lap_paths(data_root: Path, track: str | None) -> dict[str, list[Path]]:
    import pyarrow.parquet as pq

    by_track: dict[str, list[Path]] = {}
    pattern = f"{track.lower()}/*.parquet" if track else "*/*.parquet"
    for lap_path in sorted(Path(data_root).glob(pattern)):
        try:
            columns = set(pq.ParquetFile(lap_path).schema.names)
        except Exception:
            continue
        if not all(column in columns for column in REQUIRED_GEOMETRY_COLUMNS):
            continue
        by_track.setdefault(lap_path.parent.name.lower(), []).append(lap_path)
    return by_track


def format_audit_report(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "No raw laps found."
    lines = [
        "MOTION GEOMETRY COVERAGE",
        "------------------------",
        f"{'track':12}{'laps':>7}{'with geometry':>16}{'missing':>10}",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{str(row['track']):12}{int(row['laps']):>7}"
            f"{int(row['with_geometry']):>16}{int(row['without_geometry']):>10}"
        )
    missing_tracks = summary[summary["with_geometry"] == 0]["track"].tolist()
    if missing_tracks:
        lines.append("")
        lines.append(
            "No geometry map can be built for: " + ", ".join(missing_tracks)
        )
        lines.append(
            "Record a few laps at those tracks with the current recorder. "
            "They do not need to be quick or valid."
        )
    return "\n".join(lines)


def run_from_args(args: argparse.Namespace) -> int:
    if args.audit:
        print(format_audit_report(audit_geometry_coverage(args.data)))
        return 0

    config = GeometryConfig(
        grid_spacing_m=float(args.grid_spacing_m),
        path_smoothing_m=float(args.path_smoothing_m),
        heading_smoothing_m=float(args.heading_smoothing_m),
        corner_minimum_radius_m=float(args.corner_minimum_radius_m),
        minimum_corner_length_m=float(args.minimum_corner_length_m),
        corner_merge_gap_m=float(args.corner_merge_gap_m),
        corner_split_ratio=float(args.corner_split_ratio),
        corner_split_balance_ratio=float(args.corner_split_balance_ratio),
        minimum_coverage_fraction=float(args.minimum_coverage),
    )

    by_track = _geometry_lap_paths(args.data, args.track)
    if not by_track:
        print(format_audit_report(audit_geometry_coverage(args.data)))
        print("\nNothing to build: no lap carries Motion geometry fields.")
        return 1

    failures = 0
    for track, lap_paths in sorted(by_track.items()):
        frames = []
        lap_ids = []
        for lap_path in lap_paths:
            try:
                frames.append(pd.read_parquet(lap_path, engine="pyarrow"))
                lap_ids.append(lap_path.stem)
            except Exception as exc:
                print(f"{track}: skipped {lap_path.name} ({type(exc).__name__}: {exc})")

        try:
            geometry = build_track_geometry(
                frames, track=track, lap_ids=lap_ids, config=config
            )
        except ValueError as exc:
            print(f"{track}: {exc}")
            failures += 1
            continue

        paths = write_track_geometry(args.output, geometry)
        if not args.no_plots:
            plot_path = track_output_dir(args.output, track) / f"{track}_map.png"
            plot_track_geometry(geometry, plot_path)
            paths["map_png"] = str(plot_path)
        print(format_geometry_report(geometry.summary, paths))
        print()

    return 0 if failures == 0 else 1


def format_geometry_report(
    summary: dict[str, Any],
    paths: dict[str, str],
) -> str:
    lines = [
        f"TRACK GEOMETRY — {str(summary['track']).upper()}",
        "-" * (17 + len(str(summary["track"]))),
        f"Track length:        {summary['track_length_m']:.1f} m",
        f"Grid:                {summary['grid_points']} points "
        f"at {summary['grid_spacing_m']:.2f} m",
        f"Laps used:           {summary['laps_used']} of {summary['laps_available']}",
        f"Corners detected:    {summary['corner_count']} "
        f"({summary['left_corners']} left, {summary['right_corners']} right)",
        f"Curvature sign:      {summary['curvature_sign_convention']} "
        f"(from {summary['curvature_sign_source']})",
        f"Heading vs yaw:      {summary['mean_heading_vs_yaw_deg']:.2f} deg mean difference",
    ]
    if summary["laps_rejected_low_coverage"]:
        lines.append(
            "Rejected (coverage): "
            + ", ".join(summary["laps_rejected_low_coverage"])
        )
    if summary["laps_used"] < 3:
        lines.append(
            "NOTE: fewer than 3 laps. The map is usable but a few more laps "
            "will average out line variation."
        )
    lines.append("")
    for label, path in paths.items():
        lines.append(f"{label:20} {path}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    return run_from_args(args)