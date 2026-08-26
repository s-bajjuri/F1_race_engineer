# Phase 3E Feature Quality Report

This report validates derived feature tables without modifying raw laps or the original Phase 3D outputs.

## Dataset

- Style rows: 11
- Segment rows: 74
- Technically complete style rows: 11
- Style-model-ready rows: 9
- Clean baseline candidates: 8
- Full dynamic-context rows: 0

## Session-context profiles

- static_strategy_context: 11

## Braking zones

### MONACO

- monaco_z01: 115.5 m, support 11 laps (100%), canonical
- monaco_z02: 674.8 m, support 7 laps (64%), canonical
- monaco_z03: 1127.8 m, support 11 laps (100%), canonical
- monaco_z04: 1996.7 m, support 11 laps (100%), canonical
- monaco_z05: 2344.9 m, support 1 laps (9%), rare
- monaco_z06: 2639.8 m, support 10 laps (91%), canonical
- monaco_z07: 2865.7 m, support 10 laps (91%), canonical

## Interpretation rules

- `feature_complete` means the values are present, not that every session mode makes them informative.
- Static fuel, ERS, wear, or tyre-temperature values are retained but marked as context-sensitive.
- Rare braking zones are preserved as possible anomalies and excluded from canonical-zone summaries.
- Correlations are descriptive only because the dataset is still small.

## Strongest descriptive lap-time relationships

- mean_speed_kmh: r=-0.904 across 11 laps
- coast_time_pct: r=+0.794 across 11 laps
- peak_rear_slip_abs: r=+0.770 across 11 laps
- minimum_speed_kmh: r=-0.583 across 11 laps
- mean_abs_longitudinal_g: r=+0.580 across 11 laps
- mean_abs_lateral_g: r=-0.570 across 11 laps
