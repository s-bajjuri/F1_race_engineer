from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from f1_engineer.analysis.catalog import build_lap_catalog
from f1_engineer.analysis.corner_segments import (
    CORNER_BEHAVIOR_COLUMNS,
    GEOMETRY_CONTEXT_COLUMNS,
    CornerSegmentConfig,
    build_corner_windows,
    extract_lap_corner_segments,
)
from f1_engineer.analysis.geometry import track_output_dir
from f1_engineer.analysis.validation import ValidationConfig


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract geometry-defined corner maneuvers from raw laps. Corner "
            "windows come from the track map, and behaviour is measured on the "
            "raw 60 Hz samples rather than the distance grid."
        )
    )
    parser.add_argument("--data", type=Path, default=Path("data/raw_laps"))
    parser.add_argument("--geometry", type=Path, default=Path("data/track_geometry"))
    parser.add_argument("--output", type=Path, default=Path("data/corner_segments"))
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="Optional single track. Default is every track with a geometry map.",
    )
    parser.add_argument(
        "--eligibility",
        action="store_true",
        help=(
            "Print per-lap eligibility with the reason each excluded lap was "
            "dropped, then exit without extracting anything."
        ),
    )
    parser.add_argument(
        "--include-outliers",
        action="store_true",
        help="Also process laps flagged above the same-track outlier threshold.",
    )
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Process only fully enriched training-eligible laps.",
    )
    parser.add_argument("--maximum-approach-m", type=float, default=300.0)
    parser.add_argument("--maximum-exit-m", type=float, default=200.0)
    parser.add_argument("--hard-brake-threshold", type=float, default=0.30)
    parser.add_argument(
        "--brake-touch-threshold",
        type=float,
        default=0.02,
        help=(
            "Pressure that counts as the driver having begun braking. The "
            "reported braking point uses this, not the firm threshold, so a "
            "progressive application is not measured as a late one."
        ),
    )
    parser.add_argument("--expected-hz", type=float, default=60.0)
    parser.add_argument("--hz-tolerance", type=float, default=7.0)
    parser.add_argument("--minimum-rows", type=int, default=1_000)
    parser.add_argument("--outlier-factor", type=float, default=1.20)
    return parser


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _combine_track_outputs(output_root: Path) -> pd.DataFrame:
    """Rebuild the all-tracks table from every per-track output on disk.

    Reading from disk rather than from the frames produced by this run means a
    single-track run refreshes that track without dropping the others out of
    the combined table.
    """

    frames: list[pd.DataFrame] = []
    for path in sorted(Path(output_root).glob("*/corner_segments.parquet")):
        try:
            frame = pd.read_parquet(path, engine="pyarrow")
        except Exception:
            continue
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _tracks_with_geometry(geometry_root: Path, track: str | None) -> list[str]:
    names: list[str] = []
    for corners_path in sorted(Path(geometry_root).glob("*/*_corners.csv")):
        name = corners_path.parent.name.lower()
        if track and name != track.lower():
            continue
        names.append(name)
    return names


def _load_track_geometry(geometry_root: Path, track: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = track_output_dir(geometry_root, track)
    corners_path = root / f"{track}_corners.csv"
    grid_path = root / f"{track}_geometry.parquet"
    if not corners_path.exists():
        raise FileNotFoundError(f"no corner table for {track}: {corners_path}")
    corners = pd.read_csv(corners_path)
    if grid_path.exists():
        grid = pd.read_parquet(grid_path, engine="pyarrow")
    else:
        grid = pd.read_csv(root / f"{track}_geometry.csv")
    return corners, grid


def process_track(
    track: str,
    *,
    data_root: Path,
    geometry_root: Path,
    output_root: Path,
    validation: ValidationConfig,
    segment_config: CornerSegmentConfig,
    training_only: bool,
    include_outliers: bool,
    outlier_factor: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    corners, grid = _load_track_geometry(geometry_root, track)
    windows = build_corner_windows(corners, grid, track=track, config=segment_config)

    track_root = Path(output_root) / track
    track_root.mkdir(parents=True, exist_ok=True)

    catalog = build_lap_catalog(
        data_root=data_root,
        output_path=track_root / "lap_catalog.csv",
        track=track,
        config=validation,
        outlier_factor=outlier_factor,
    )

    eligibility = "training_eligible" if training_only else "core_eligible"
    selected = catalog[_as_bool(catalog[eligibility])].copy()
    if not include_outliers and "outlier_lap_time" in selected.columns:
        selected = selected[~_as_bool(selected["outlier_lap_time"])]

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for _, row in selected.iterrows():
        source = Path(str(row["file_path"]))
        try:
            raw = pd.read_parquet(source, engine="pyarrow")
        except Exception as exc:
            failures.append({"lap": source.stem, "error": f"{type(exc).__name__}: {exc}"})
            continue

        metadata = {
            "lap_id": source.stem,
            "track": track,
            "source_file": str(source),
            "session_uid": row.get("session_uid"),
            "lap_number": row.get("lap_number"),
            "lap_time_ms": row.get("lap_time_ms"),
            "training_eligible": bool(_as_bool(pd.Series([row.get("training_eligible")]))[0]),
        }
        records.extend(
            extract_lap_corner_segments(
                raw, windows, metadata=metadata, config=segment_config
            )
        )

    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        frame["session_key"] = (
            frame["track"].astype(str) + ":" + frame["session_uid"].astype(str)
        )
        frame["context_version"] = "track_geometry_v1"
        frame["behavior_source"] = "raw_time_series"

    frame.to_csv(track_root / "corner_segments.csv", index=False)
    if not frame.empty:
        frame.to_parquet(
            track_root / "corner_segments.parquet",
            engine="pyarrow",
            index=False,
            compression="zstd",
        )

    complete = (
        int(_as_bool(frame["behavior_complete"]).sum()) if not frame.empty else 0
    )
    braking_complete = (
        int(_as_bool(frame["braking_complete"]).sum()) if not frame.empty else 0
    )
    braked = int(_as_bool(frame["brake_detected"]).sum()) if not frame.empty else 0
    summary: dict[str, Any] = {
        "track": track,
        "corners": len(windows),
        "laps_selected": int(len(selected)),
        "laps_failed": len(failures),
        "corner_rows": int(len(frame)),
        "behavior_complete_rows": complete,
        "braking_complete_rows": braking_complete,
        "rows_with_braking": braked,
        "failures": failures,
    }
    (track_root / "corner_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return frame, summary


def format_eligibility_report(catalog: pd.DataFrame, track: str) -> str:
    if catalog.empty:
        return f"{track}: no laps found."

    working = catalog.copy()
    working["lap"] = working["file_path"].astype(str).map(lambda value: Path(value).stem)
    core = _as_bool(working["core_eligible"])
    outlier = (
        _as_bool(working["outlier_lap_time"])
        if "outlier_lap_time" in working.columns
        else pd.Series(False, index=working.index)
    )

    lines = [
        f"{track.upper()} — {len(working)} laps, {int(core.sum())} eligible",
        "-" * 72,
        f"{'lap':40}{'time':>9}{'use':>6}  reason",
    ]
    for index, row in working.iterrows():
        usable = bool(core[index]) and not bool(outlier[index])
        reason = str(row.get("exclusion_reasons") or "")
        if reason in {"nan", "None"}:
            reason = ""
        if usable and bool(outlier[index]):
            reason = "lap_time_outlier"
        warnings = str(row.get("quality_warnings") or "")
        if warnings in {"nan", "None"}:
            warnings = ""
        if usable and warnings:
            reason = f"ok ({warnings})"
        seconds = pd.to_numeric(pd.Series([row.get("lap_time_s")]), errors="coerce")[0]
        time_text = f"{seconds:8.3f}" if pd.notna(seconds) else "       -"
        lines.append(
            f"{str(row['lap'])[:39]:40}{time_text}{'yes' if usable else 'NO':>6}  {reason}"
        )

    excluded = working[~core]
    if not excluded.empty:
        counts = (
            excluded["exclusion_reasons"].fillna("unknown").astype(str).value_counts()
        )
        lines.append("")
        lines.append("Excluded because:")
        for reason, count in counts.items():
            lines.append(f"  {reason:40} {count}")
    return "\n".join(lines)


def run_eligibility(args: argparse.Namespace, validation: ValidationConfig) -> int:
    tracks = _tracks_with_geometry(args.geometry, args.track)
    if not tracks and args.track:
        tracks = [args.track.lower()]
    if not tracks:
        tracks = sorted(
            path.name.lower() for path in Path(args.data).iterdir() if path.is_dir()
        )

    for track in tracks:
        catalog = build_lap_catalog(
            data_root=args.data,
            output_path=Path(args.output) / track / "lap_catalog.csv",
            track=track,
            config=validation,
            outlier_factor=float(args.outlier_factor),
        )
        print(format_eligibility_report(catalog, track))
        print()
    return 0


def run_from_args(args: argparse.Namespace) -> int:
    validation = ValidationConfig(
        expected_sample_rate_hz=float(args.expected_hz),
        sample_rate_tolerance_hz=float(args.hz_tolerance),
        minimum_rows=int(args.minimum_rows),
    )
    if args.eligibility:
        return run_eligibility(args, validation)

    segment_config = CornerSegmentConfig(
        maximum_approach_m=float(args.maximum_approach_m),
        maximum_exit_m=float(args.maximum_exit_m),
        hard_brake_threshold=float(args.hard_brake_threshold),
        brake_touch_threshold=float(args.brake_touch_threshold),
    )

    tracks = _tracks_with_geometry(args.geometry, args.track)
    if not tracks:
        print(
            "No track geometry maps found. Build them first:\n"
            "  python tools/build_track_geometry.py --audit"
        )
        return 1

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for track in tracks:
        _, summary = process_track(
            track,
            data_root=args.data,
            geometry_root=args.geometry,
            output_root=output_root,
            validation=validation,
            segment_config=segment_config,
            training_only=bool(args.training_only),
            include_outliers=bool(args.include_outliers),
            outlier_factor=float(args.outlier_factor),
        )
        summaries.append(summary)

    # Every track lands in one table. Phase 4 needs more than one track
    # represented, and per-track runs previously overwrote each other.
    combined = _combine_track_outputs(output_root)
    combined_csv = output_root / "corner_segments_all.csv"
    combined.to_csv(combined_csv, index=False)
    if not combined.empty:
        combined.to_parquet(
            output_root / "corner_segments_all.parquet",
            engine="pyarrow",
            index=False,
            compression="zstd",
        )

    overall: dict[str, Any] = {
        "context_version": "track_geometry_v1",
        "behavior_source": "raw_time_series",
        "context_features": list(GEOMETRY_CONTEXT_COLUMNS),
        "behavior_targets": list(CORNER_BEHAVIOR_COLUMNS),
        "tracks_processed_this_run": [summary["track"] for summary in summaries],
        "tracks": summaries,
        "total_corner_rows": int(len(combined)),
        "tracks_represented": int(combined["track"].nunique()) if not combined.empty else 0,
        "sessions_represented": (
            int(combined["session_key"].nunique()) if not combined.empty else 0
        ),
        "combined_csv": str(combined_csv),
    }
    (output_root / "corner_segments_summary.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    print(format_corner_report(overall))
    return 0


def format_corner_report(summary: dict[str, Any]) -> str:
    lines = [
        "GEOMETRY CORNER SEGMENTS",
        "------------------------",
        f"{'track':10}{'corners':>9}{'laps':>7}{'rows':>8}{'complete':>10}"
        f"{'braked':>8}{'brk cmpl':>10}",
    ]
    for track in summary["tracks"]:
        lines.append(
            f"{track['track']:10}{track['corners']:>9}{track['laps_selected']:>7}"
            f"{track['corner_rows']:>8}{track['behavior_complete_rows']:>10}"
            f"{track['rows_with_braking']:>8}{track['braking_complete_rows']:>10}"
        )
    lines.extend(
        [
            "",
            "Combined table (every track on disk, not only this run):",
            f"Total corner rows:      {summary['total_corner_rows']}",
            f"Tracks represented:     {summary['tracks_represented']}",
            f"Sessions represented:   {summary['sessions_represented']}",
            f"Context features:       {len(summary['context_features'])} "
            "(track geometry, execution-independent)",
            f"Behaviour targets:      {len(summary['behavior_targets'])} "
            "(measured on raw 60 Hz samples)",
            "",
            f"Combined table:         {summary['combined_csv']}",
        ]
    )
    failures = [
        failure for track in summary["tracks"] for failure in track["failures"]
    ]
    if failures:
        lines.append("")
        lines.append(f"Lap read failures: {len(failures)}")
        for failure in failures[:5]:
            lines.append(f"  {failure['lap']}: {failure['error']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    return run_from_args(args)