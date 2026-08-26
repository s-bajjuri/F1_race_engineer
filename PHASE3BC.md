# Phase 3B/3C — Distance Resampling and Braking Segmentation

This patch extends the Phase 3A catalog tool. Raw Parquet laps are never changed.

## Install

Extract the patch into the project folder containing `main.py` and allow Windows
to merge/replace the included files. Do not remove the existing `data/` folder.

## Test

```powershell
python -m unittest discover -s tests -v
```

Expected: 14 tests pass.

## Run Spa processing

```powershell
python tools/analyze_laps.py --data data/raw_laps --track spa
```

The command now performs all of the following:

1. Rebuilds `data/processed/lap_catalog.csv`.
2. Selects complete, valid, approximately 60 Hz core-eligible laps.
3. Resamples each selected lap onto 2,001 points from 0.0 to 1.0 lap distance.
4. Detects braking events where brake input is above 30% for at least 0.5 seconds.
5. Merges brief releases separated by no more than 0.2 seconds.
6. Extends each event through minimum speed and sustained full-throttle application.
7. Saves visual plots for manual inspection.

## Outputs

```text
data/processed/
├── lap_catalog.csv
├── lap_catalog.summary.json
├── resampled/
│   └── spa/
│       └── <lap>_resampled.parquet
├── braking_segments.csv
├── segment_processing.csv
├── segment_summary.json
└── segment_plots/
    └── spa/
        └── <lap>_segments.png
```

`braking_segments.csv` contains one row per detected segment, including braking
start/end, peak brake, minimum speed, first sustained full throttle, and segment
end in both lap percentage and metres.

## Useful options

Process only fully enriched training laps:

```powershell
python tools/analyze_laps.py --data data/raw_laps --track spa --training-only
```

Skip PNG plots:

```powershell
python tools/analyze_laps.py --data data/raw_laps --track spa --no-plots
```

Adjust the detector for testing:

```powershell
python tools/analyze_laps.py --data data/raw_laps --track spa `
  --brake-threshold 0.30 `
  --min-brake-seconds 0.50 `
  --merge-gap-seconds 0.20
```

Run only the Phase 3A catalog:

```powershell
python tools/analyze_laps.py --data data/raw_laps --track spa --catalog-only
```

## Interpretation

The detected items are called **segments**, not official corner numbers. A
braking-based detector may combine linked corners or omit flat-out corners. The
PNG plots are the deliberate validation step before extracting style features.
