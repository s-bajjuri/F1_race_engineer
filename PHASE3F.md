# Phase 3F — Modeling Contract & Causal Readiness

This patch closes Phase 3 without changing or deleting any raw telemetry.

It adds two things needed before Phase 4:

1. A strict **context -> behavior** modeling contract for few-shot k-NN experiments.
2. A streaming braking-segment state machine whose running statistics are causal.

## Why existing laps remain useful

All existing laps remain valid inputs to the Phase 3A-3E pipeline. Phase 3F reads the already-created resampled laps and validated segment features. It derives a pre-entry context window from telemetry that existed **before** each braking segment and keeps the post-entry behavior as a separate prediction/assessment target.

No maneuver category taxonomy is introduced.

## Run

From the folder containing `main.py`:

```powershell
python -m unittest discover -s tests -v
python tools/prepare_modeling.py --processed data/processed
```

Outputs:

```text
data/processed/modeling_segments.csv
data/processed/modeling_segments.parquet
data/processed/modeling_sessions.csv
data/processed/phase3f_contract.json
data/processed/phase3f_contract.md
```

## Important contract

`CONTEXT_FEATURE_COLUMNS` are the only Phase 4 v1 values permitted for neighbor lookup. They use a causal 0.75-second pre-entry window.

`CORE_BEHAVIOR_TARGET_COLUMNS` and `MOTION_BEHAVIOR_TARGET_COLUMNS` describe how the maneuver was executed and therefore must never be used to choose neighbors when those same values are being scored.

`track`, `zone_id`, `lap_id`, and `session_uid` are metadata for grouping/evaluation, not cross-track similarity inputs.

All model evaluation must split by `session_key`, not random individual laps.

The current context is deliberately named `causal_proxy_v1`: it is useful for the first few-shot experiment but is not claimed to be pure track geometry. Geometry-rich context can replace/extend it later without changing the behavior-target interface.
