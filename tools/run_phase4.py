from __future__ import annotations

import sys
from pathlib import Path

# Make the project package importable when this file is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from f1_engineer.modeling.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())