from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from f1_engineer.modeling.style_envelope import (
    STYLE_TARGETS,
    StyleEnvelope,
    StyleEnvelopeConfig,
    corner_dependence,
    describe_deviations,
    evaluate_envelope_transfer,
    summarise_transfer,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a personal style envelope and test whether it describes a "
            "track it has never seen."
        )
    )
    parser.add_argument(
        "--segments",
        type=Path,
        default=Path("data/corner_segments/corner_segments_all.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/style_envelope"))
    parser.add_argument(
        "--band-edges",
        type=float,
        nargs="+",
        default=[60.0, 150.0],
        help="Corner radius boundaries, in metres, for the coarse bands.",
    )
    parser.add_argument("--minimum-band-rows", type=int, default=20)
    parser.add_argument("--deviation-threshold", type=float, default=2.0)
    parser.add_argument(
        "--example-lap",
        type=str,
        default=None,
        help="Lap id to print worked coaching lines for. Default: the last lap.",
    )
    return parser


def _load(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path, engine="pyarrow")
        return pd.read_csv(path)
    alternative = path.with_suffix(".csv")
    if alternative.exists():
        return pd.read_csv(alternative)
    raise FileNotFoundError(
        f"no corner segment table at {path}. Build it with:\n"
        "  python tools/analyze_corners.py"
    )


def format_report(
    transfer: pd.DataFrame,
    dependence: pd.DataFrame,
    overview: dict[str, Any],
    coaching: list[str],
) -> str:
    lines = [
        "PERSONAL STYLE ENVELOPE",
        "=======================",
        f"Corner rows:        {overview['rows']}",
        f"Tracks:             {overview['tracks']}",
        f"Bands:              {overview['bands']}",
        "",
        "Question: was that a normal way for YOU to do it — not what the corner",
        "demands. Built from every track, keyed only on a coarse corner band, so",
        "it applies to a circuit that has never been driven.",
        "",
        "TRANSFER TO AN UNSEEN TRACK (leave one track out)",
        "-" * 78,
        "A calibration test. If the envelope carries, a held-out track sits near",
        "centre 0.0 and spread 1.0, putting about 68% inside 1 sigma and 95%",
        "inside 2. oracle = the same test with the envelope fitted on that very",
        "track, which is the best any envelope could do.",
        "",
    ]

    merged = transfer.merge(dependence, on="target", how="left")
    lines.append(
        f"{'target':34}{'corner%':>9}{'centre':>8}{'spread':>8}"
        f"{'<2sd':>7}{'oracle':>8}{'flag%':>7}{'worst':>8}{'pass':>7}"
    )
    for _, row in merged.iterrows():
        share = row.get("corner_share")
        share_text = f"{share:>8.0%}" if pd.notna(share) else f"{'-':>8}"
        passed = f"{int(row['tracks_passing'])}/{int(row['tracks'])}"
        lines.append(
            f"{str(row['target'])[:33]:34}{share_text} {row['centre_z']:>7.2f}"
            f"{row['spread_z']:>8.2f}{row['within_2sd']:>7.0%}"
            f"{row['oracle_within_2sd']:>8.0%}{row['flag_rate']:>7.0%}"
            f"{row['worst_abs_z']:>8.1f}{passed:>7}"
        )

    passing = merged[merged["transfers"]]
    lines.append("")
    lines.append(
        "pass = how many held-out tracks landed within centre +-0.5 and spread "
        "0.7-1.4."
    )
    lines.append(
        "flag% = share of corners a 2-sigma alert would fire on; worst = the "
        "largest deviation seen."
    )
    lines.append(
        "A worst far above 3 means the tail is heavier than a normal "
        "distribution, so sigma"
    )
    lines.append("overstates how unusual the extremes really are.")
    lines.append("")
    lines.append(
        f"{len(passing)} of {len(merged)} descriptors transfer on every held-out track."
    )
    if not passing.empty:
        lines.append("Usable on lap 1 of an unseen circuit:")
        for _, row in passing.iterrows():
            lines.append(
                f"  - {row['target']:34} fires on {row['flag_rate']:.0%} of corners"
            )
    near = merged[(~merged["transfers"]) & (merged["tracks_passing"] >= 2)]
    if not near.empty:
        lines.append("")
        lines.append("Transfer on most but not all tracks; usable with more data:")
        for _, row in near.iterrows():
            lines.append(
                f"  - {row['target']:34} "
                f"{int(row['tracks_passing'])}/{int(row['tracks'])} tracks"
            )

    if coaching:
        lines.append("")
        lines.append("WORKED EXAMPLE — deviations on one lap")
        lines.append("-" * 78)
        for line in coaching:
            lines.append(f"  {line}")

    lines.append("")
    lines.append(f"Envelope: {overview['envelope_path']}")
    lines.append(f"Outputs:  {overview['output_root']}")
    return "\n".join(lines)


def run_from_args(args: argparse.Namespace) -> int:
    frame = _load(args.segments)
    config = StyleEnvelopeConfig(
        band_edges=tuple(float(edge) for edge in args.band_edges),
        band_names=tuple(
            ["tight", "medium", "fast"][: len(args.band_edges) + 1]
        ),
        minimum_band_rows=int(args.minimum_band_rows),
        deviation_threshold=float(args.deviation_threshold),
    )

    transfer = evaluate_envelope_transfer(frame, config=config)
    summary = summarise_transfer(transfer)
    dependence = corner_dependence(frame)

    # Coaching speaks only through descriptors that survived the calibration
    # test. Anything that fails on a held-out track would be guessing on a
    # circuit the driver has never seen.
    trusted = tuple(
        summary.loc[summary["transfers"], "target"].tolist()
    ) or STYLE_TARGETS
    envelope = StyleEnvelope(config=config).fit(frame)
    speaking = StyleEnvelope(targets=trusted, config=config).fit(frame)

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    transfer.to_csv(output_root / "envelope_transfer_by_track.csv", index=False)
    summary.to_csv(output_root / "envelope_transfer.csv", index=False)
    dependence.to_csv(output_root / "corner_dependence.csv", index=False)
    envelope_path = envelope.save(output_root / "style_envelope.json")

    # Worked example: score one real lap and speak the largest deviations.
    lap_id = args.example_lap
    if lap_id is None and "lap_id" in frame.columns:
        lap_id = str(sorted(frame["lap_id"].astype(str).unique())[-1])
    coaching: list[str] = []
    if lap_id:
        lap = frame[frame["lap_id"].astype(str) == lap_id]
        if not lap.empty:
            scores = speaking.score(lap)
            coaching = describe_deviations(lap, scores, speaking, limit=8)
            header = f"{lap_id} — using only the {len(trusted)} descriptors that transfer:"
            coaching = (
                [header, *coaching]
                if coaching
                else [header, "nothing outside your normal envelope this lap."]
            )

    overview: dict[str, Any] = {
        "rows": int(len(frame)),
        "tracks": int(frame["track"].nunique()),
        "bands": list(config.band_names),
        "band_edges": list(config.band_edges),
        "deviation_threshold": config.deviation_threshold,
        "targets": list(STYLE_TARGETS),
        "transferring_targets": summary.loc[
            summary["transfers"], "target"
        ].tolist()
        if not summary.empty
        else [],
        "envelope_path": str(envelope_path),
        "output_root": str(output_root),
    }
    (output_root / "style_summary.json").write_text(
        json.dumps(overview, indent=2), encoding="utf-8"
    )

    report = format_report(summary, dependence, overview, coaching)
    (output_root / "style_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    return run_from_args(args)