from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from f1_engineer.app import run
from f1_engineer.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "F1 Race Engineer: minimal Drive view, detailed Analysis view, "
            "Diagnostics, enriched recording, and live lap comparison."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="YAML configuration file (default: config.yaml)",
    )
    parser.add_argument("--track", help="Track folder label, such as spa or monza")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--window", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--view",
        choices=("drive", "analysis", "diagnostics"),
        help="View shown at startup",
    )
    parser.add_argument(
        "--drive-trace",
        type=float,
        help="Seconds of throttle/brake history in Drive view",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable the floating driver overlay for this run",
    )
    parser.add_argument(
        "--overlay-trace",
        type=float,
        help="Seconds of throttle/brake history in the floating overlay",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
        config = replace(
            config,
            host=args.host if args.host is not None else config.host,
            port=args.port if args.port is not None else config.port,
            track=args.track if args.track is not None else config.track,
            output_root=(
                args.output if args.output is not None else config.output_root
            ),
            window_seconds=(
                args.window if args.window is not None else config.window_seconds
            ),
            reference_path=(
                args.reference
                if args.reference is not None
                else config.reference_path
            ),
            start_view=args.view if args.view is not None else config.start_view,
            drive_trace_seconds=(
                args.drive_trace
                if args.drive_trace is not None
                else config.drive_trace_seconds
            ),
            overlay_enabled=(
                False if args.no_overlay else config.overlay_enabled
            ),
            overlay_trace_seconds=(
                args.overlay_trace
                if args.overlay_trace is not None
                else config.overlay_trace_seconds
            ),
        )
        run(config)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
