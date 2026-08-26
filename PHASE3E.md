# Phase 3E — Feature Quality Validation

This patch adds a non-destructive quality pass after Phase 3D.

It creates:

- `data/processed/style_features_validated.csv`
- `data/processed/segment_features_validated.csv`
- `data/processed/braking_zone_quality.csv`
- `data/processed/feature_quality.csv`
- `data/processed/feature_quality_report.json`
- `data/processed/feature_quality_report.md`

The original raw laps, `style_features.csv`, and `segment_features.csv` are not modified.

Run:

```powershell
python -m unittest discover -s tests -v
python tools/analyze_laps.py --data data/raw_laps --track spa
```

Important distinction: `feature_complete` means values are present. The new
`context_profile` and model-readiness fields indicate whether context-sensitive
fuel, ERS, tyre-wear, and tyre-temperature features are actually informative.
