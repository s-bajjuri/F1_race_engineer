# Phase 3F Modeling Contract

Phase 3F freezes the interface between offline feature extraction and Phase 4 modeling.
No raw lap or Phase 3A-3E output is modified.

## Output

- Modeling segment rows: 74
- k-NN v1 candidate rows: 34
- Sessions represented: 2
- Tracks represented: 1
- Rows with usable session IDs: 74

## Locked modeling rules

- Query context and driver-behavior targets are separate column groups.
- Neighbor lookup must not use peak brake, minimum speed, throttle recovery, or other future outcomes.
- `track`, `zone_id`, and lap identity are evaluation metadata, not cross-track k-NN context features.
- Dataset splits must be grouped by `session_key`; do not randomly split laps from the same session across train/test.
- The current context is a causal telemetry proxy derived only from the pre-entry window. It is not claimed to be pure track geometry.
- Explicit maneuver category labels are deferred; descriptors stay continuous.

## Phase 4 entry point

Use `modeling_segments.csv`, filter `knn_v1_candidate == True`, normalize only the CONTEXT feature group on the training split, retrieve neighbors by context similarity, and predict BEHAVIOR targets from neighbor behavior.

A geometry-rich context can be added later without changing the behavior-target contract.
