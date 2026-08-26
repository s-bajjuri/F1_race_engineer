from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from f1_engineer.analysis.validation import (
    ValidationConfig,
    unreadable_lap_record,
    validate_lap_dataframe,
)


def scan_lap_files(data_root: Path, track: str | None = None) -> list[Path]:
    """Return Parquet laps in deterministic path order."""

    root = Path(data_root)
    if track:
        root = root / track.lower()
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.parquet") if path.is_file())


def _append_reason(existing: object, reason: str) -> str:
    parts = [part for part in str(existing or "").split(";") if part]
    if reason not in parts:
        parts.append(reason)
    return ";".join(parts)


def apply_track_outliers(
    catalog: pd.DataFrame,
    outlier_factor: float = 1.20,
) -> pd.DataFrame:
    """Flag laps slower than a same-track median multiplier.

    The median is calculated only from readable, complete, valid laps with a
    positive official time. The raw files are never changed or deleted.
    """

    if catalog.empty:
        return catalog.copy()

    result = catalog.copy()
    for track_name, indices in result.groupby("track", dropna=False).groups.items():
        subset = result.loc[list(indices)]
        candidates = subset[
            subset["readable"].fillna(False)
            & subset["complete"].fillna(False)
            & subset["valid"].fillna(False)
            & pd.to_numeric(subset["lap_time_ms"], errors="coerce").gt(0)
        ]
        if candidates.empty:
            continue

        median_ms = float(
            pd.to_numeric(candidates["lap_time_ms"], errors="coerce").median()
        )
        if not np.isfinite(median_ms) or median_ms <= 0:
            continue

        track_indices = list(indices)
        result.loc[track_indices, "track_median_lap_time_ms"] = median_ms
        times = pd.to_numeric(result.loc[track_indices, "lap_time_ms"], errors="coerce")
        ratios = times / median_ms
        result.loc[track_indices, "lap_time_ratio_to_track_median"] = ratios

        outlier_mask = ratios > float(outlier_factor)
        outlier_indices = list(ratios.index[outlier_mask.fillna(False)])
        if not outlier_indices:
            continue

        result.loc[outlier_indices, "outlier_lap_time"] = True
        result.loc[outlier_indices, "training_eligible"] = False
        for index in outlier_indices:
            result.at[index, "exclusion_reasons"] = _append_reason(
                result.at[index, "exclusion_reasons"],
                "lap_time_outlier",
            )

    return result


def build_lap_catalog(
    data_root: Path,
    output_path: Path,
    *,
    track: str | None = None,
    config: ValidationConfig | None = None,
    outlier_factor: float = 1.20,
) -> pd.DataFrame:
    """Scan raw laps, validate them, and save a CSV catalog plus JSON summary."""

    cfg = config or ValidationConfig()
    files = scan_lap_files(data_root, track=track)
    records: list[dict[str, object]] = []

    for path in files:
        try:
            dataframe = pd.read_parquet(path, engine="pyarrow")
            record = validate_lap_dataframe(dataframe, path, cfg)
        except Exception as exc:  # catalog must survive individual bad files
            record = unreadable_lap_record(path, exc)
        records.append(record)

    catalog = apply_track_outliers(pd.DataFrame.from_records(records), outlier_factor)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_path, index=False)

    summary_path = output_path.with_suffix(".summary.json")
    summary = build_catalog_summary(
        catalog,
        data_root=Path(data_root),
        output_path=output_path,
        track=track,
        config=cfg,
        outlier_factor=outlier_factor,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return catalog


def build_catalog_summary(
    catalog: pd.DataFrame,
    *,
    data_root: Path,
    output_path: Path,
    track: str | None,
    config: ValidationConfig,
    outlier_factor: float,
) -> dict[str, object]:
    def count_true(column: str) -> int:
        if column not in catalog:
            return 0
        return int(catalog[column].fillna(False).astype(bool).sum())

    by_track: dict[str, dict[str, int]] = {}
    if not catalog.empty and "track" in catalog:
        for track_name, frame in catalog.groupby("track", dropna=False):
            name = str(track_name)
            by_track[name] = {
                "files": int(len(frame)),
                "readable": int(frame["readable"].fillna(False).sum()),
                "valid": int(frame["valid"].fillna(False).sum()),
                "complete": int(frame["complete"].fillna(False).sum()),
                "core_eligible": int(frame["core_eligible"].fillna(False).sum()),
                "training_eligible": int(
                    frame["training_eligible"].fillna(False).sum()
                ),
            }

    return {
        "data_root": str(data_root),
        "catalog_path": str(output_path),
        "track_filter": track,
        "files_found": int(len(catalog)),
        "readable_files": count_true("readable"),
        "valid_laps": count_true("valid"),
        "complete_laps": count_true("complete"),
        "core_eligible_laps": count_true("core_eligible"),
        "training_eligible_laps": count_true("training_eligible"),
        "outlier_laps": count_true("outlier_lap_time"),
        "configuration": {
            **asdict(config),
            "outlier_factor": float(outlier_factor),
        },
        "by_track": by_track,
    }


def format_catalog_report(catalog: pd.DataFrame) -> str:
    if catalog.empty:
        return "No Parquet lap files were found."

    def count(column: str) -> int:
        return int(catalog[column].fillna(False).astype(bool).sum())

    lines = [
        "PHASE 3A DATASET CATALOG",
        "------------------------",
        f"Files found:          {len(catalog)}",
        f"Readable:             {count('readable')}",
        f"Valid:                {count('valid')}",
        f"Complete:             {count('complete')}",
        f"Core eligible:        {count('core_eligible')}",
        f"Training eligible:    {count('training_eligible')}",
        f"Lap-time outliers:    {count('outlier_lap_time')}",
    ]

    excluded = catalog[~catalog["training_eligible"].fillna(False)]
    if not excluded.empty:
        reason_counts: dict[str, int] = {}
        for value in excluded["exclusion_reasons"].fillna(""):
            for reason in str(value).split(";"):
                if reason:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if reason_counts:
            lines.append("")
            lines.append("Exclusion reasons:")
            for reason, amount in sorted(
                reason_counts.items(), key=lambda item: (-item[1], item[0])
            ):
                lines.append(f"  {reason:<32} {amount}")

    return "\n".join(lines)
