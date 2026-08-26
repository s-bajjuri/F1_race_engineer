# Phase 3A — Dataset Scanner and Validator

This step catalogs every raw lap without modifying or deleting the original Parquet files.

## Run all tracks

```powershell
python tools/analyze_laps.py --data data/raw_laps
```

## Run Spa only

```powershell
python tools/analyze_laps.py --data data/raw_laps --track spa
```

Outputs:

```text
data/processed/lap_catalog.csv
data/processed/lap_catalog.summary.json
```

The catalog reports schema health, completeness, sampling rate, validity, enrichment coverage, plausible-value checks, and training eligibility. It also flags laps slower than 120% of the valid same-track median. Flagged files remain untouched and can be retained as anomaly examples.

A lap can be `core_eligible` but not `training_eligible`. This is intentional: older laps with speed, throttle, brake, gear, steering, and timing are still useful, while the full Phase 3 feature set additionally requires tyre, ERS, fuel, G-force, and wheel-slip columns.

## Useful overrides

```powershell
python tools/analyze_laps.py --expected-hz 60 --hz-tolerance 7
python tools/analyze_laps.py --outlier-factor 1.20
python tools/analyze_laps.py --allow-missing-enrichment
```

## Tests

```powershell
python -m unittest discover -s tests -v
```
