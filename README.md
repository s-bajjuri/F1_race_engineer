# F1 Race Engineer — Modular UI Build

This build keeps the verified F1 25 UDP receiver, enriched Parquet recorder, live distance-aligned reference comparison, and data-health checks while separating the interface into three focused views.

## Views

- **Drive:** large delta, 3-second throttle/brake trace, tyre status, and minimal session state.
- **Analysis:** detailed speed, inputs, delta, and tyre graphs.
- **Diagnostics:** packet, parser, recorder, enrichment, writer, and reference health.

The dark interface is intentionally functional rather than final. Recording and comparison continue regardless of which view is selected.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py --track spa
```

Useful overrides:

```powershell
python main.py --track spa --view analysis --drive-trace 3
```

Keyboard shortcuts:

```text
1 or D  Drive
2 or A  Analysis
3 or G  Diagnostics
```

Only run one UDP-listening F1 tool at a time on port 20777.

## Module map

- `runtime.py`: owns receiver, recorder, writer, comparison, and shared snapshots
- `ui/dashboard.py`: window, navigation, update loop, and shutdown
- `ui/drive_view.py`: minimal driver-facing view
- `ui/analysis_view.py`: detailed lap-comparison workspace
- `ui/diagnostics_view.py`: developer health information
- `ui/theme.py`: dark visual system
- `telemetry/`: packet parsing and UDP reception
- `recording/`: lap detection and Parquet saving
- `comparison/`: reference selection and distance-aligned delta

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Phase 3A — Dataset catalog

Scan and validate accumulated raw laps before feature extraction:

```powershell
python tools/analyze_laps.py --data data/raw_laps
```

See `PHASE3A.md` for validation rules and outputs.
