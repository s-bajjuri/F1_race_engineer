from __future__ import annotations

import argparse
from pathlib import Path
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from f1_engineer.analysis.modeling import (
    format_phase3f_report,
    write_phase3f_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3F: freeze causal context/behavior modeling contract."
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing Phase 3E validated feature tables.",
    )
    parser.add_argument(
        "--lookback-seconds",
        type=float,
        default=0.75,
        help="Causal pre-entry window used for context descriptors.",
    )
    args = parser.parse_args()

    summary = write_phase3f_outputs(
        args.processed,
        lookback_seconds=float(args.lookback_seconds),
    )
    print(format_phase3f_report(summary))
    print(f"\nModeling segments: {summary['modeling_segments_csv']}")
    print(f"Sessions:          {summary['modeling_sessions_csv']}")
    print(f"Contract:          {summary['contract_markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
