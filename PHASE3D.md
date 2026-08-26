# Phase 3D — Feature Extraction

This patch adds per-braking-segment and per-lap style feature extraction to the existing Phase 3 pipeline.

## Install

Extract the ZIP into the project directory containing `main.py` and allow files to merge/replace. Do not delete `data/`.

## Test

```powershell
python -m unittest discover -s tests -v
```

Expected result after the previous Phase 3 patches: 16 tests pass.

## Run

```powershell
python tools/analyze_laps.py --data data/raw_laps --track spa
```

## New outputs

```text
data/processed/segment_features.csv
data/processed/segment_features.parquet
data/processed/style_features.csv
data/processed/style_features.parquet
data/processed/feature_summary.json
```

`segment_features` contains one row per detected braking/corner segment. `style_features` contains one compact row per processed lap. Core-only laps are retained; `feature_complete` identifies rows with the full enriched schema.
