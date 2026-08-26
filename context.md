# F1 Race Engineer AI — Current Project Context

**Purpose of this file:** Paste or upload this file at the start of a new AI chat so the assistant has the current project state, architecture, dataset status, decisions, known issues, and next steps.

**Current project state:** Phase 1 complete, Phase 2 complete, Phase 3 code/architecture complete through Phase 3F, plus Phase 3G track geometry (new — see section 27A). Dataset collection is still in progress before Phase 4 modeling begins, and the outstanding item is now **track geometry laps at Spa and Monza**, not more behaviour laps.

**Important:** The original `PHASE PLAN.pdf` and old `Context.txt` were written before most of the implementation happened. They are useful historical references, but the current decisions in this file supersede their old phase-status tables and the original assumption that Phase 4 must begin with a whole-lap LSTM autoencoder.

---

# 1. What this project is

This is a personal/student project to build a **standalone Python F1 race engineer assistant for Codemasters F1 25**.

The application should:

1. Read F1 25 native UDP telemetry in real time.
2. Save essentially all useful driving data, lap by lap.
3. Learn the driver's personal driving behavior/style.
4. Detect when the driver executes a maneuver differently from their learned normal behavior.
5. Convert those deviations into useful coaching.
6. Deliver concise coaching through a lightweight in-game overlay and eventually optional voice.
7. Work locally/free without depending on a live LLM API during driving.

The project is not meant to be a generic telemetry dashboard only. The end goal is a personalized engineer that can say things conceptually like:

- braking later than your normal behavior here
- unusually aggressive brake release
- throttle reapplied earlier/later than expected
- excessive rear slip compared with your typical behavior
- unusual steering correction
- tyre/ERS/fuel behavior worth noting
- corner execution differs meaningfully from your normal pattern

The desired long-term capability is **few-shot coaching on unfamiliar tracks**, not just memorizing previously driven tracks.

A realistic first version of "unknown-track" coaching is:

- First lap on an unseen track: collect local geometry + initial behavior.
- Laps 2–3: use general driving-style knowledge plus newly observed local track structure.
- Lap 4+: increasingly personalized, track-aware coaching.

True zero-shot, useful corner-specific coaching on the very first lap is a harder later extension and is not required for the first working system.

---

# 2. Constraints and philosophy

## Constraints

- Solo developer / student project.
- Python desktop application.
- $0/free tooling where practical.
- F1 game telemetry only.
- No dependency on proprietary real-F1 telemetry.
- No live LLM required during driving.
- Real-time latency and reliability are more important than fancy generated wording.
- Prefer interpretable and shippable modeling before more complex neural models.

## Core philosophy

**Collect everything early.**

Do not throw away raw laps merely because they are invalid, messy, slow, contain a crash, or are unsuitable for baseline training.

Raw data should remain immutable.

Quality logic decides what is suitable for:

- clean baseline training
- general style modeling
- strategy/race-context modeling
- anomaly examples
- debugging only

This preserves potentially valuable anomaly examples and prevents subjective manual cleanup from silently changing the dataset.

---

# 3. Technology stack

Current/expected stack:

- Python 3.x
- Native UDP socket on port `20777`
- Custom `struct.unpack` implementation for F1 25 native packet format
- pandas / NumPy
- Parquet storage
- PyTorch available for later neural experiments
- Matplotlib / Plotly as appropriate
- Tkinter desktop UI / floating overlay
- `pyttsx3` planned for offline TTS
- Git for version control

No live LLM is required in-session. Coaching phrases can be pre-generated/stored and selected deterministically from telemetry events.

---

# 4. Current application architecture

The project was deliberately refactored away from a giant monolithic `main.py`.

The active project root is conceptually:

```text
main.py
config.yaml
requirements.txt
README.md
.gitignore

f1_engineer/
    ...telemetry modules...
    ...recording modules...
    ...comparison/reference modules...
    ...analysis modules...
    ...UI modules...
    ui/
        drive_overlay.py

tools/
    analyze_laps.py
    prepare_modeling.py
    ...

tests/
docs/

data/
    raw_laps/
        spa/
        monza/
        monaco/
        suzuka/
        ...
    processed/
    sessions/
```

`main.py` is intended to remain a small launcher rather than absorbing application logic.

The app has **one UDP listener**. The UI, recorder, comparison logic, diagnostics, etc. consume shared parsed state rather than opening multiple competing listeners on the same telemetry port.

Background work is used so disk writes/UI work do not block telemetry ingestion.

---

# 5. Phase 1 — Telemetry foundation — COMPLETE

## UDP/native packet parsing

F1 25 telemetry is successfully received on:

```text
UDP port: 20777
```

The native header format that was confirmed working:

```python
<HBBBBBQfIIBB
```

Header size:

```text
29 bytes
```

Car telemetry record size:

```text
60 bytes
```

A live test confirmed:

- UDP format: 2025
- game version at the time of testing: 1.24
- player car index: 0
- native F1 25 packet flow was received correctly

## Packet IDs observed

The parser observed these packet IDs:

```text
0  Motion
1  Session
2  Lap Data
3  Event
4  Participants
5  Car Setups
6  Car Telemetry
7  Car Status
8  Final Classification
10 Car Damage
11 Session History
12 Tyre Sets
13 Motion Ex
14 Time Trial
15 Lap Positions
```

Not all packet types are necessarily used by every downstream feature yet, but the telemetry foundation recognizes the native F1 25 stream rather than relying on a backward-compatible third-party legacy parser.

## Core telemetry confirmed live

Confirmed fields include:

- speed
- throttle
- brake
- steering
- gear
- RPM
- DRS
- tyre temperatures
- lap timing / lap distance

Later enrichment added:

- fuel remaining
- ERS deployed
- FL/FR/RL/RR tyre wear
- lateral G
- longitudinal G
- FL/FR/RL/RR wheel slip

Healthy example ranges seen during testing included approximately:

```text
fuel:          ~11.29–13.46 kg
ERS:           0–1.45e6 J
tyre wear:     ~0.07–0.11
lateral G:     roughly -5.35 to +5.05 in normal driving
longitudinal G roughly -4.20 to +1.44 in normal driving
wheel slip:    meaningfully variable
```

Impact laps can exceed normal G ranges dramatically and are quality-filtered later.

---

# 6. Lap recording behavior — COMPLETE

The recorder:

- detects lap boundaries from Lap Data
- automatically saves completed laps
- handles normal start/finish crossings
- handles qualifying/flying-lap starts
- handles flashback/rewind behavior
- discards the current in-progress buffer when a rewind/flashback invalidates the temporal sequence
- saves invalid completed laps rather than silently deleting them
- writes Parquet in the background
- records approximately 60 Hz telemetry

## Important flying-lap bug that was fixed

An early version incorrectly interpreted the start of a flying lap as a flashback because timing/distance reset around the start line.

The fixed logic recognizes transitional/negative pre-line distance and accepts the legitimate start-line timer/distance reset.

Do not reintroduce the old "any time/distance reset = flashback" behavior.

## Storage format

Every lap is stored as one `.parquet` file.

Current files have roughly **50 columns** and typically thousands of rows per lap at ~60 Hz.

A known clean Spa qualifying lap was:

```text
6592 rows × 50 columns
~60 Hz
no nulls
complete
```

Raw laps are the source of truth.

## Filename decision

An earlier naming scheme used lap number first:

```text
lap_{lap_number}_{timestamp}_{validity}.parquet
```

That caused problems because lap numbers repeat across sessions and lexicographical ordering could select the wrong "newest" reference.

Current intended naming is timestamp-first:

```text
lap_YYYYMMDD_HHMMSS_mmm_validity.parquet
```

Reference loading should use file modification time / metadata rather than trusting lexical filename order.

---

# 7. Data-retention rules

Do **not** delete laps merely because they are:

- invalid
- slow
- impact laps
- lap-time outliers
- missing some enrichment
- unsuitable for the current model
- old test laps that still contain genuine telemetry

Delete only things that are clearly useless storage artifacts such as:

- zero-byte files
- exact duplicates
- corrupted test files with no recoverable telemetry

Invalid/impact laps may later become valuable anomaly examples.

Some older data was recorded at ~20 Hz. Keep it, but exclude it from the first native-60-Hz model unless explicitly resampled/handled.

---

# 8. Phase 2 — Dashboard / application UI — COMPLETE

The recorder and dashboard were combined around a single UDP listener.

Implemented:

- live telemetry display
- background telemetry receiver
- background Parquet writer
- current-vs-reference lap comparison
- reference alignment by normalized lap distance
- live delta
- data-health diagnostics
- automatic reference update using newly completed valid laps
- invalid laps are saved but do not replace the valid reference

## Distance alignment

A major design rule:

```text
Compare laps by lap_distance_pct, NOT timestamp.
```

This is important because the same corner occurs at different times on laps with different pace.

## UI views

The modular UI includes:

```text
Drive
Analysis
Diagnostics
```

The Drive view was later turned into a floating always-on-top overlay while keeping the same backend and same UDP listener.

Relevant file:

```text
f1_engineer/ui/drive_overlay.py
```

The main analysis window can be minimized while the lightweight driver overlay stays visible.

## Driver overlay

Current/implemented overlay concepts include:

- large live delta
- rolling throttle/brake trace
- compact tyre information/status
- connection/recording information
- reserved space for later coaching messages

The trace duration was adjusted from roughly 3 seconds to about 6–7 seconds through config.

Config contains or has contained options conceptually like:

```yaml
overlay:
  enabled: true
  visible: true
  width: ...
  height: ...
  x: ...
  y: ...
  opacity: ...
  click_through: ...
  trace_seconds: ...
  refresh_hz: ...
```

The overlay is intended for borderless/windowed F1 use.

The driver-facing screen should remain minimal. Full engineering plots belong in Analysis, packet/debug information in Diagnostics.

---

# 9. Phase 3 — analysis / feature pipeline — COMPLETE IN CODE

Phase 3 has been implemented as multiple stages rather than one giant script.

Current status:

```text
3A Dataset validation/catalog           COMPLETE
3B Distance normalization               COMPLETE
3C Braking-event segmentation           COMPLETE
3D Segment + lap style features         COMPLETE
3E Feature-quality validation           COMPLETE
3F Causal modeling contract/prep        COMPLETE
```

Dataset collection is still ongoing, but Phase 3 development itself is considered complete unless a concrete Phase 4 failure reveals a necessary fix.

Do not keep expanding Phase 3 merely because more sophisticated features are imaginable.

---

# 10. Phase 3A — Dataset scanner / validator

Implemented modules/tools included:

```text
f1_engineer/analysis/validation.py
f1_engineer/analysis/catalog.py
f1_engineer/analysis/pipeline.py
tools/analyze_laps.py
tests/...
```

Main outputs:

```text
data/processed/lap_catalog.csv
data/processed/lap_catalog.summary.json
```

The catalog checks things including:

- file readability
- required columns
- timestamp behavior
- lap distance plausibility / 0–1 normalization
- approximate sample rate
- nulls
- completion status
- game validity
- track/lap time metadata
- implausible sensor values
- model-column nulls
- lap-time outliers

## Lap-time outlier rule

The original plan recommended excluding laps slower than:

```text
120% of track median lap time
```

This is used as a quality rule, not as permission to delete the raw lap.

## Duplicate timestamp issue that was fixed

The validator initially required duplicate timestamp count == 0 for core eligibility.

Real F1 telemetry can contain adjacent frames sharing `timestamp_ms`, and this caused legitimate laps to be rejected.

The duplicate-timestamp exclusion was removed.

Important distinction:

- duplicate timestamps alone can be benign
- `timestamps_not_monotonic` is still an exclusion and should not be relaxed automatically without inspection

Several later laps have been preserved but excluded for non-monotonic timestamps.

---

# 11. Phase 3B — Distance normalization

Eligible laps are interpolated to a common normalized-distance grid:

```text
2001 points
0.0000 → 1.0000 lap distance
```

This representation is primarily for:

- lap alignment
- consistent feature extraction
- segment comparison
- plots
- offline analysis

Raw ~60 Hz data remains untouched.

Important current modeling decision:

**Do not assume the final neural model should ingest an entire 2001-point lap.**

For later sequence modeling, segment-level windows around 64/128 samples are now considered more appropriate.

---

# 12. Phase 3C — Braking-event segmentation

Because there was no hard-coded corner map, the first segment detector uses braking events.

The original starting rule was:

```text
brake > 0.3 for approximately 0.5 s+
```

The pipeline also merges sufficiently brief brake releases and derives event landmarks such as:

- braking start
- peak braking
- minimum speed
- throttle reapplication
- full-throttle recovery

Outputs include:

```text
braking_segments.csv
segment_processing.csv
segment_summary.json
```

and segment/resampled diagnostic outputs/plots.

Across repeated laps, recurring braking locations become canonical braking zones.

Rare braking zones/events are preserved rather than forcibly assigned to the normal baseline.

Important limitation now recognized:

Braking-only segmentation does not capture every meaningful style event, especially:

- lift-only corners
- flat high-speed corners
- high-speed steering sequences

However, do **not** respond by building a large manually thresholded maneuver taxonomy before Phase 4.

The revised approach is to prefer continuous maneuver/context descriptors and derive labels later if needed.

---

# 13. Phase 3D — Segment and lap-level feature extraction

## Segment-level features

The pipeline extracts features such as:

- braking distance
- brake strength / peak brake
- braking profile
- brake release smoothness
- minimum speed
- throttle reapplication
- full-throttle recovery
- steering smoothness
- wheel-slip behavior
- G-force behavior
- tyre-related behavior
- input overlap / coasting-related behavior where relevant

Main feature outputs:

```text
segment_features.csv
segment_features.parquet
style_features.csv
style_features.parquet
```

## Lap-level style features

Implemented lap-level summaries include concepts such as:

- brake aggressiveness
- brake peak consistency
- throttle patience
- steering smoothness
- slip frequency
- rear-slip bias
- fuel used
- ERS usage by lap thirds
- tyre temperatures
- tyre wear delta
- coasting
- throttle/brake input overlap

Important decision:

Core driver-control behavior and race/strategy context should initially remain separate.

### Core-driving candidate signals

The first modeling work should focus mainly on:

```text
speed
throttle
brake
steering
gear
RPM
lateral G
longitudinal G
wheel_slip_fl
wheel_slip_fr
wheel_slip_rl
wheel_slip_rr
```

### Context/strategy group

Treat these more carefully/separately initially:

```text
fuel
ERS
tyre wear
tyre temperatures
```

Reason: Time Trial often contains static/non-informative strategy values, which should not contaminate the core style model.

---

# 14. Phase 3E — Feature-quality validation

Phase 3E added non-destructive quality outputs:

```text
style_features_validated.csv
segment_features_validated.csv
braking_zone_quality.csv
feature_quality.csv
feature_quality_report.json
feature_quality_report.md
```

Important quality concepts:

## Style-model-ready

Feature-quality-suitable rows.

## Clean baseline candidate

Strict normal-driving baseline rows suitable for learning the driver's "normal" behavior.

## Full dynamic-context

Rows where dynamic strategy telemetry is genuinely informative and passes quality checks.

## Static strategy context

Usually Time Trial-style data where fuel/ERS/wear/etc. may exist but are essentially constant/non-informative.

## Missing enrichment

Older/incomplete data where some contextual telemetry is absent.

## Rare braking zones

Preserved as rare/anomalous segments rather than discarded.

## Standing/stationary

Rows with stationary behavior that should not be treated as clean normal driving.

---

# 15. G-force impact quality behavior

The validator currently uses a generous plausibility threshold around:

```text
g_force_lat: -15 to +15
g_force_lon: -15 to +15
```

Do not loosen this threshold merely because Monaco produces wall impacts.

Example observed impact lap:

```text
Lat min: -32.768
Lon min: -14.833

outside ±15G:
-19.98 lat / -14.833 lon
-32.197 lat / -4.919 lon
-32.768 lat / -5.858 lon
```

Another lap was game-valid but correctly excluded for an impact-like event:

```text
43 km/h
g_force_lat: -15.526
g_force_lon: -22.293
```

This demonstrated that "game-valid" does not necessarily mean "clean baseline."

Conceptually useful future labels:

```text
invalid_lap
valid_with_impact
clean_baseline
```

Do not delete impact laps.

---

# 16. Phase 3F — Causal modeling contract / few-shot prep

Phase 3F was added after reviewing the planned modeling approach.

Main purpose:

1. Prevent future-information leakage.
2. Separate maneuver context from driver behavior.
3. Prepare session-aware model splits.
4. Provide a streaming/causal segment-state concept for future real-time use.
5. Avoid premature hand-built maneuver taxonomy.

Tool:

```text
tools/prepare_modeling.py
```

There was one import-path bug when run directly.

The fix added the project root to `sys.path` before importing `f1_engineer`:

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

Then the command works from the project root:

```powershell
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

Latest successful Phase 3F run on the current Monaco-processed dataset:

```text
PHASE 3F MODELING CONTRACT + CAUSAL PREP
------------------------------------------
Modeling segment rows:       74
Context-complete rows:       74
k-NN v1 candidate rows:      34
Sessions represented:        2
Tracks represented:          1
Session-ID rows:             74
Context contract:            causal_proxy_v1
Taxonomy:                    deferred (continuous descriptors)
Split rule:                  session-grouped
```

Interpretation:

- Phase 3F is functioning.
- `Tracks represented: 1` is not data loss.
- `data/processed` currently reflects the most recently analyzed track (Monaco).
- Spa and Monza raw data remain safe.
- Before Phase 4 starts, a small aggregation step is required so modeling rows from all target tracks are combined into one multi-track dataset.

This aggregation step has not yet been done.

---

# 17. Critical modeling contract: CONTEXT vs BEHAVIOR

This is one of the most important current design decisions.

A few-shot model must not find neighbors using the same behavior it later claims to predict.

For example, it would be leakage to find "similar corners" using:

```text
minimum speed
peak brake
full-throttle recovery
steering smoothness
```

and then score those same quantities as deviations.

The modeling interface should separate:

## CONTEXT

"What kind of maneuver / local track situation is this?"

Prefer values that are available before or at maneuver entry and ideally are execution-independent.

Potential descriptors:

```text
segment/local path length
corner direction
curvature / heading change
elevation change
approach-speed range
required speed-reduction estimate
local geometry/trajectory descriptors
location within lap (optional/careful)
```

Some current fields are only causal proxies rather than perfect track-geometry descriptors.

## BEHAVIOR

"How did the driver execute it?"

Examples:

```text
brake start
peak brake
brake release
minimum speed
throttle reapplication
steering corrections
wheel slip
coasting
input overlap
```

The model should conceptually do:

```text
current maneuver context
        ↓
find historically similar contexts
        ↓
look at how this driver normally behaved there
        ↓
predict expected behavior
        ↓
current behavior - expected behavior
        ↓
normalized residual/deviation
        ↓
possible coaching event
```

This is the foundation for the revised few-shot approach.

---

# 18. Causal vs offline segmentation

Phase 3 offline extraction is allowed to use the full completed segment.

Offline features such as:

- final peak brake
- minimum-speed point
- full-throttle recovery
- merged brake gaps

look into the completed maneuver.

A live Phase 5 system cannot know the future.

Therefore the runtime path should use a causal state machine conceptually like:

```text
IDLE
  ↓
candidate
  ↓
active segment
  ↓
exit/grace period
  ↓
finalized
```

A small confirmation window (roughly the existing ~0.5 s concept) can confirm braking onset.

During the active segment the live system can update running values causally:

- max brake so far
- min speed so far
- slip peaks
- rolling/online means
- steering variability
- elapsed distance/time
- throttle state

Some outputs, such as full-throttle recovery distance, are only final when the segment exits.

That is acceptable because the coaching design already prefers firing insights on corner exit or the following straight rather than mid-corner.

The existing offline detector should remain as a richer reference/ground-truth implementation.

---

# 19. Dataset strategy

The original plan recommended varied circuits including Spa, Hungaroring, Monza, and Suzuka.

After discussion, the minimal-track/high-variance set was revised to:

```text
Spa
Monza
Monaco
Suzuka
```

Hungaroring was replaced by Monaco for the current minimum-track dataset because Monaco provides a more extreme low-speed/large-steering/traction-limited regime.

## Why these tracks

### Spa

- mixed speeds
- elevation
- long lap
- high-speed + medium-speed behavior
- broad general-purpose circuit

### Monza

- very high speed
- low downforce
- strong straight-line braking
- traction events
- heavy braking contrast

### Monaco

- extreme low speed
- large steering inputs
- rapid direction change
- traction-limited exits
- precision/wall-contact environment

### Suzuka

Planned to add:

- high-speed linked corners
- sustained lateral load
- flowing direction changes
- important non-heavy-braking behavior

This four-track set is intended to maximize driving-regime variance with minimal track count.

---

# 20. Current dataset — detailed known status

Important distinction:

- Raw file count is larger than clean baseline count.
- Invalid/anomalous files remain stored.
- "Clean baseline" is the strictest normal-driving subset.
- "Style-model-ready" is a broader quality-suitable subset.
- Dynamic-context rows are useful later for race/strategy context.

## SPA — complete enough

Latest known summary:

```text
PHASE 3A
Files:                  25
Readable:               25
Valid:                  21
Complete:               22
Core eligible:          18
Training eligible:      17
Lap-time outliers:      1

Exclusions:
invalid_lap                 4
timestamps_not_monotonic    3
all_null_model_columns      1
implausible_values          1
lap_time_outlier            1
model_null_values           1
```

Phase 3B–D:

```text
Eligible:               18
Processed:              18
Failed:                 0
Braking segments:       112
Segment feature rows:   112
Style rows:             18
Complete style rows:    17
Distance grid:          2001
```

Phase 3E:

```text
Style-model-ready:      15
Clean baseline:         14
Full dynamic-context:   4
Canonical brake zones:  6
Rare brake zones:       1
Rare segment rows:      1
Standing/stationary:    3

Context profiles:
dynamic_strategy_context = 6
missing_enrichment        = 1
static_strategy_context   = 11
```

Status:

```text
Spa: DONE / sufficient for current target
```

---

## MONZA — complete enough

The dataset was expanded with a 13-lap race, which substantially improved dynamic strategy context.

Latest known summary:

```text
PHASE 3A
Files found:            30
Readable:               30
Valid:                  26
Complete:               25
Core eligible:          21
Training eligible:      20
Lap-time outliers:      1

Exclusions:
timestamps_not_monotonic    5
invalid_lap                 4
lap_time_outlier            1
```

Phase 3B–D:

```text
Eligible selected:      20
Processed:              20
Failed:                 0
Braking segments:       123
Segment feature rows:   123
Style rows:             20
Complete style rows:    20
Distance grid:          2001
```

Phase 3E:

```text
Style-model-ready:      17
Clean baseline:         16
Full dynamic-context:   13
Canonical brake zones:  6
Rare brake zones:       2
Rare segment rows:      4
Standing/stationary:    0

Context profiles:
dynamic_strategy_context = 15
static_strategy_context  = 5
```

Status:

```text
Monza: DONE / sufficient for current target
```

The 5 non-monotonic laps remain stored and excluded.

---

## MONACO — collection in progress

An early Monaco checkpoint showed:

```text
11 raw files
11 readable
9 valid
10 complete
7 core eligible
7 training eligible

Exclusions:
implausible_values         2
invalid_lap                2
timestamps_not_monotonic   1
```

That early batch produced:

```text
7 eligible / 7 processed
46 braking segments
7 style rows
5 style-model-ready
5 clean baseline
7 canonical zones
0 rare zones
```

After more Time Trial collection, the latest feature summary was:

```text
Eligible laps selected:    11
Processed successfully:    11
Failed:                    0
Braking segments:          74
Segment feature rows:      74
Style feature rows:        11
Complete style rows:       11
Distance grid points:      2001

Style-model-ready rows:    9
Clean baseline candidates: 8
Full dynamic-context rows: 0
Canonical braking zones:   6
Rare braking zones:        1
Rare segment rows:         1
Standing/stationary rows:  0
Context:
static_strategy_context = 11
```

Phase 3F on the latest Monaco processed set:

```text
74 modeling rows
74 context-complete
34 strict k-NN v1 candidates
2 sessions
1 track in current processed set
```

Status:

```text
Monaco: useful but not yet at preferred 12–15 clean baseline laps
Current clean baseline: 8
```

Monaco is intentionally difficult and produces more wall-contact exclusions.

When convenient, collect roughly another 4–6 clean-ish TT attempts, rerun analysis, and stop when clean baseline is around 12+.

It is acceptable to postpone Monaco and collect Suzuka first.

---

## SUZUKA — not yet collected

Status:

```text
Suzuka: pending
```

Plan:

- start with 3–5 valid laps
- run the analyzer
- inspect whether segmentation/quality behavior looks sensible
- if healthy, continue toward roughly 12–15 clean baseline candidates
- ideally add at least some non-TT dynamic context later, but the core driving model does not require dynamic strategy context on every track

---

# 21. Current data target

Original plan:

```text
Minimum ~50 laps across 4 tracks
Target 100+
```

Current practical collection target is more quality-oriented:

```text
roughly 12–15 clean baseline laps per track
across:
Spa
Monza
Monaco
Suzuka
```

This should produce roughly 50+ clean baseline laps total.

Current known clean baseline counts:

```text
Spa:    14
Monza:  16
Monaco: 8
Suzuka: 0
```

Known total so far:

```text
38 clean baseline laps
before Suzuka / additional Monaco collection
```

There are many more raw/model-ready/dynamic-context examples than the clean-baseline count.

---

# 22. Important lesson from Time Trial vs race data

Time Trial is excellent for core style learning:

- braking
- throttle
- steering
- G-force
- wheel slip
- consistency

But Time Trial often gives static strategy context:

- fuel may not evolve meaningfully
- ERS behavior may be static or constrained
- tyre wear may be static/non-informative

This is why Phase 3E distinguishes:

```text
static_strategy_context
dynamic_strategy_context
```

A Time Trial lap is not "bad" merely because strategy fields have zero variance.

NumPy correlation warnings like:

```text
RuntimeWarning: invalid value encountered in divide
```

have occurred when constant columns make correlation undefined.

These warnings did not invalidate the core feature extraction.

---

# 23. Revised Phase 4 modeling approach

The original phase plan proposed a whole-lap LSTM autoencoder:

```text
T × 26 input
LSTM encoder, 128 hidden, 2 layers
32-d style vector
LSTM decoder, 128 hidden, 2 layers
MSE reconstruction
```

This remains a possible later experiment, but it is **no longer the mandatory first model**.

Reasons:

- current dataset is relatively small for a long-sequence neural model
- 2001-point whole-lap sequences are long
- the desired output is local/maneuver coaching
- a whole-lap model can easily learn track identity
- we want few-shot transfer to unfamiliar tracks
- we already have rich engineered segment features
- interpretation matters for coaching

## New Phase 4A

Start with a **k-nearest-neighbor maneuver similarity + residual model**.

Concept:

```text
maneuver context
      ↓
find similar historical maneuver contexts
      ↓
retrieve driver's historical behavior in those contexts
      ↓
weighted expected behavior
      ↓
current behavior - predicted behavior
      ↓
normalized residuals
      ↓
deviation/anomaly score
      ↓
coaching cause
```

This directly tests whether the descriptor representation is transferable.

It is also cheap, interpretable, and well matched to few-shot adaptation.

## Why not immediately build statistical baseline → PCA → Isolation Forest ladder?

Those models can still be useful later, but they should not be built for their own sake.

First prove that:

```text
similar context -> predictable personal behavior
```

If k-NN fails because neighbors are not meaningful, that exposes a representation problem early.

PCA, Isolation Forest, regularized Mahalanobis, etc. should be added only if they address a demonstrated weakness or provide a useful benchmark.

## Important possible later model

A small segment-level sequence encoder/autoencoder remains attractive because scalar engineered features can miss shape.

Example:

Two braking maneuvers can have the same:

- peak brake
- duration
- braking distance

but very different brake-release curves.

A later Phase 4B sequence model could use something like:

```text
64 or 128 resampled samples × telemetry channels
```

per maneuver instead of a 2001-point whole lap.

Whole-lap LSTM should now be treated as an experiment, not an assumption.

---

# 24. Unknown-track / generalization strategy

Do not confuse per-track performance with unknown-track transfer.

A per-track model is useful for validating known-track coaching, but by itself it cannot solve few-shot unseen-track behavior.

Similarly, a track-conditioned model only generalizes if its conditioning representation itself generalizes to new tracks.

The current target is to represent local maneuver context in a shared space rather than using:

```text
track = Spa
corner = 3
```

as the core concept.

Long-term useful descriptors should come from local path/geometry and entry conditions.

Potential F1 Motion-derived local geometry:

- world position
- yaw/heading
- elevation
- distance
- curvature
- corner direction
- heading change
- local path length

If not already stored sufficiently in the current 50-column raw schema, basic geometry telemetry should be verified/added before serious unseen-track work.

Do not destroy old laps if geometry is absent. Existing laps are still valuable for known-track and behavior-model work.

---

# 25. Evaluation strategy for Phase 4

## Never random-split laps across the same session

Laps inside one session share:

- setup
- weather/session conditions
- fuel/tyre progression
- track state
- driver state
- other session signatures

Random lap-level train/test splits could leak session identity.

Phase 3F therefore specifies:

```text
split rule: session-grouped
```

Train/validation/test grouping should respect sessions.

## Leave-one-track-out

With four tracks, perform rotating held-out-track sanity checks:

```text
Train Monza + Monaco + Suzuka -> test Spa
Train Spa + Monaco + Suzuka   -> test Monza
Train Spa + Monza + Suzuka    -> test Monaco
Train Spa + Monza + Monaco    -> test Suzuka
```

Important:

Four folds are only a **sanity check**, not proof of universal generalization.

One unusual held-out session could materially change a result.

Report per-track outcomes, not just one averaged score.

## Few-shot adaptation curve

More important for the actual product goal:

For each held-out track, evaluate approximately:

```text
0 target-track clean laps available
1 target-track clean lap
3 target-track clean laps
5 target-track clean laps
```

This directly measures how quickly the engineer becomes useful on an unfamiliar circuit.

This is more important than claiming "generalization" from one binary held-out-track result.

---

# 26. Per-maneuver anomaly calibration

Do not use one raw global MSE/error threshold for every corner.

A Monaco hairpin naturally has different signal magnitude/variance from a Monza heavy-braking event.

Deviation should be normalized against the expected behavior for similar contexts/local maneuvers.

Conceptually:

```text
observed behavior
    -
predicted/expected behavior
    ↓
normalized residual per behavior feature
    ↓
combined confidence/deviation score
```

Potential coachable residuals:

- brake point later/earlier than expected
- peak brake stronger/weaker
- release less smooth
- minimum speed unusually high/low
- throttle reapplied earlier/later
- rear slip excessive
- steering correction excessive
- coasting unusually long
- throttle/brake overlap unusual

Known-track local calibration can be used as additional evidence, but the transferable system should not rely exclusively on fixed corner IDs.

---

# 27. Coaching/runtime design

The eventual runtime flow should be:

```text
Live telemetry
      ↓
causal local maneuver state
      ↓
context representation
      ↓
expected personal behavior
      ↓
residual/deviation
      ↓
cause classification
      ↓
confidence + priority + cooldown
      ↓
overlay and optional voice
```

Example coaching categories:

```text
late braking
early braking
poor brake release
early throttle
late throttle
wheelspin / rear slip
unusual steering correction
possible understeer-like behavior
possible oversteer-like behavior
high tyre temperature
unusual ERS usage
```

The telemetry/model layer should determine what happened numerically.

Language generation is only presentation.

The runtime should use a pre-generated phrase library rather than requiring a live LLM.

The original plan suggested:

- event classification
- priority queue
- cooldown
- insights mainly on straights or corner exit
- minimum interruption interval around 15 seconds as a starting point
- overlay text card
- optional offline TTS

Those are still sensible future Phase 5/6 ideas.

---

# 27A. Phase 3G — track geometry (NEW)

## Why this exists

Section 29D asked whether the raw Motion / Motion Ex fields preserved enough
information to derive world position, heading, elevation and curvature.

**The answer was no.** The telemetry layer parsed the Motion packets, but the
recorder only pulled `g_force_lat`, `g_force_lon` and the four slip ratios out
of them and discarded the rest. Laps recorded before 2026-08-09 have 50 columns
and contain no geometry at all.

Two consequences made this the blocking issue for Phase 4:

1. **Behaviour-defined segments are circular.** A segment currently starts where
   `brake >= 0.30`. Brake later and the window moves, so a corner has no
   existence independent of how it was driven, and "how did I take this corner
   versus normal" cannot be asked cleanly.
2. **Few-shot on an unseen track was impossible.** Nothing in the feature tables
   describes what a corner *is* — radius, length, elevation change, direction,
   preceding straight. Every feature was keyed to `lap_distance_m` on one
   specific track. Without a geometric description there is no similarity metric
   between corners at different tracks, so the system could only memorise tracks
   it had already driven.

Section 30 says not to assume "corner 3" means anything across tracks. That
still holds. Corner *identity* does not transfer; corner *shape* does. Matching
on geometry is what makes the few-shot goal reachable.

## Recorder format change

`f1_engineer/recording/recorder.py` now writes 19 additional channels, so laps
are 69 columns:

```text
world_position_x/y/z    world_velocity_x/y/z    local_velocity_x/y/z
yaw  pitch  roll  yaw_rate  front_wheels_angle  g_force_vert
wheel_slip_angle_fl/fr/rl/rr
```

This affects future recordings only. Existing laps are untouched and remain
fully valid for behaviour.

## Geometry coverage as of 2026-08-20

```text
track     laps   with geometry   missing
monaco      20               3        17
monza       37               7        30
spa         31               6        25
suzuka       4               4         0
```

Maps are built and validated for all four tracks. Every Suzuka lap carries
geometry, because the recorder change predates the track being driven at all.

## The map

`f1_engineer/analysis/geometry.py` builds a distance-indexed track map and a
corner table from any laps that carry Motion fields:

```powershell
python tools/build_track_geometry.py --audit
python tools/build_track_geometry.py --track monaco
```

Outputs land in a per-track subfolder, `data/track_geometry/<track>/`, holding
`<track>_geometry.parquet`, `<track>_geometry.csv`, `<track>_corners.csv`,
`<track>_summary.json` and a `<track>_map.png` for visual confirmation. Raw laps
are never modified.

How it works, and why each choice matters:

- Heading comes from the **path tangent**, not from yaw, so it is not offset by
  the car's slip angle. Recorded yaw is kept as a cross-check; at Monaco the two
  agree to 1.33 degrees mean difference.
- Everything is computed with **wrapped angular differences and circular
  smoothing**, so the start/finish line is not a special place on the map.
- Laps are combined with a **median**, so one bad lap cannot distort the map.
- Corners are detected by **signed curvature runs split at prominent apexes**,
  not by a single global threshold. A plain threshold chains tight corners
  together — the first attempt found 7 corners at Monaco instead of 19. Sign
  changes split chicanes automatically.
- A split additionally requires the **two apexes to be comparable in size**
  (`corner_split_balance_ratio`, default 0.5). Because the map follows the
  driven line, one long corner taken with a late apex has genuinely uneven
  curvature, and without this rule its decaying exit is split off as a separate
  corner. Parabolica was the case that exposed it: 15 corners at Monza became
  13, and Monaco's hairpin stopped being reported as an entry plus a hairpin.
- Curvature sign is calibrated against **recorded steering input**, so
  left/right labelling is derived from the data rather than assumed.

## Validated results

```text
track    length     corners   official   laps used   heading vs yaw
monaco   3323 m     20         19             3      1.33 deg
spa      7007 m     20         19/20          6      1.04 deg
monza    5798 m     13         11             7      0.76 deg
suzuka   5810 m     17         18             4      0.97 deg
```

Detected counts are geometric features, not FIA turn numbers, so exact agreement
is not the target. Monza's two extras are genuine but very gentle bends either
side of Curva Grande, each turning the car less than 8 degrees on a radius near
300 m. Raising `--corner-merge-gap-m` to 30 absorbs one of them into Curva
Grande if a tighter count is wanted. Suzuka comes in one under the official 18
because T1 and T2 form a single continuous 185 degree sweep on the driven line.

### Suzuka in detail

The closest length match of any track: 5809.7 m against a real 5807 m. The
corner table reads as the circuit does. The Esses appear as a clean alternating
run at 1030-1678 m (left, right, left, right, left) with 12-16 m between them
and climbing 2.6, 5.8, 6.5 m through the sequence. Degner 2 is the tight right
at 2404 m on a 52 m radius. The hairpin is the 23.6 m radius left at 2868 m
turning 173.7 degrees. Spoon is the double-apex left pair at 3754 and 3908 m.
130R is the 182 m radius left at 4924 m preceded by an 816 m straight, which is
the back straight. The Casio Triangle is the right-left pair at 5344 and 5434 m,
and the final corner drops 12.7 m onto the main straight. Overall elevation
range is about 40 m, matching the circuit. The figure-of-eight crossover is
visible on the map.

Measured lengths sit within about 0.5% of the real circuits (Spa 7004 m, Monza
5793 m, Monaco 3337 m), and heading derived from the path agrees with recorded
yaw to around a degree everywhere.

Spa's elevation profile reproduces the circuit correctly: down to -11 m at the
bottom of Eau Rouge around 1050 m, climbing to +68 m at the top of Kemmel around
2650 m, an overall range near 100 m against the real circuit's ~102 m. La Source
appears as the tightest corner, and the Bus Stop chicane splits into a right
then a left. Monza shows the Rettifilo chicane, Curva Grande, Roggia, the
Lesmos, Ascari and Parabolica over a ~12 m elevation range.

### Monaco in detail

20 corners detected against an official 19, at 2.00 m grid spacing over a
3323 m measured lap. The map matches the real circuit: Sainte Dévote at 170 m,
the Massenet left, Casino, Mirabeau, the **Grand Hotel hairpin at 12.7 m minimum
radius** and 210 degrees of heading change, the tunnel, the Nouvelle Chicane
correctly split into a left then a right, Tabac, the swimming pool sequence,
Rascasse, Anthony Noghès. Elevation climbs about 40 m to Casino and drops back
to the tunnel.

## Why old laps do not need re-driving

`lap_distance_m` is measured along the track spline, not along the driven line,
so it is the same key on every lap regardless of racing line. The map therefore
joins onto laps recorded before Motion fields existed:

```python
from f1_engineer.analysis.geometry import attach_track_geometry
enriched = attach_track_geometry(lap, geometry_map)
```

Confirmed on a 50-column Monaco lap from 2026-08-06: it gains curvature, radius,
elevation, gradient, heading and corner_id at 100% coverage.

**This is why only a few geometry laps are needed per track.** Geometry is a
property of the track, not of the lap, and a geometry lap does not need to be
quick or even valid — two of the three Monaco geometry laps are invalid and
contribute normally.

## Related fix

`resample_lap_by_distance` sent every numeric column through `np.interp`. Yaw
wraps at ±pi, so linear interpolation across the wrap produced a sweep through
the entire range instead of a small step, which would have made any derived
heading or curvature garbage. Radian angles (`yaw`, `pitch`, `roll`) are now
interpolated on unwrapped values and wrapped back.

---

# 27B. Phase 3H — geometry corner segments (NEW)

`f1_engineer/analysis/corner_segments.py` replaces brake-threshold segmentation
for modelling. It is additive: Phase 3A-3F still runs unchanged.

```powershell
python tools/analyze_corners.py
```

## What changed and why

**The window no longer depends on the driver.** Segments now come from the
track map, so a corner is the same span of track on every lap. "How did I take
this corner compared with my normal" is finally a well-posed question; with
`brake >= 0.30` defining the window, the window moved whenever the braking did.

**Behaviour is measured on the raw 60 Hz samples.** The distance grid decimates
by a factor inversely proportional to speed, so every derivative computed on it
was smoothed by an amount that depended on how fast the driver was going. Grid
sampling inside corner windows now sits at a median of 60.0 Hz with a standard
deviation of 1.5, and the correlation between sampling rate and corner speed is
-0.08 — effectively zero. The confound is gone.

**Braking point uses a dual threshold.** Detection still uses 0.30, but the
reported braking point is where pressure actually began, so a progressive
application is no longer measured as a late one. Across the dataset this moves
the braking point 4.4 m earlier on average with a 2.1 m spread; that spread was
previously baked into the feature as style-dependent bias.

**Brake release is measured to zero pedal.** The old release window stopped at
the 0.30 crossing, discarding the trail-braking portion. It now runs from the
end of the peak plateau to first-touch pressure.

## Column groups

Thirteen `ctx_` context features, all execution-independent track geometry:
heading change, corner length, minimum/mean/entry/exit radius, radius trend,
elevation change, gradient, the straights either side, and the radii of the
neighbouring corners. Turn direction is deliberately **not** a context feature —
a left and a right corner of the same shape are the same driving problem
mirrored — so it is kept as metadata and matching uses unsigned geometry.

Twenty-eight behaviour targets, including several the old pipeline could not
express: `minimum_speed_offset_from_apex_m`, `trail_brake_past_entry_m`,
`throttle_reapplication_offset_from_apex_m`, `full_throttle_offset_from_exit_m`
and `steering_correction_count`.

This is `context_version: track_geometry_v1`, the geometry-rich context the
Phase 3F contract reserved a slot for. The behaviour-target contract is
unchanged.

## Completeness is split in two

`behavior_complete` covers what every corner has, `braking_complete` covers the
brake-dependent group. A corner taken flat has no brake trace, and that is the
correct answer rather than missing data. `release_truncated` marks rows where
the pedal was still loaded at the window edge — either a genuinely cut-short
release or already braking for the next corner — and those are excluded from
`braking_complete` because their release figures describe a partial event.

## Current output

```text
track     corners   laps   rows   complete   braked   braking complete
monaco         20     14    280        280      177                125
monza          13     33    429        429      215                214
spa            20     22    440        420      166                149
suzuka         17     11    187        187      123                 92

1336 corner rows, 4 tracks, 21 sessions
```

## Checking eligibility

```powershell
python tools/analyze_corners.py --eligibility --track suzuka
```

Prints one line per lap with its time, whether it will be used, and the reason
any excluded lap was dropped. Omit `--track` for every track.

The 20 incomplete Spa rows are one lap that has no `g_force_lat`. Monaco's low
braking-complete count is expected: its corners are close enough together that
you are often already braking for the next one before the release finishes.

Evidence the windows align with the track: at Spa's La Source, constraining the
apex search to the corner cut the lap-to-lap scatter of the minimum-speed point
from 108 m to 14 m, and apex speed from 28.7 to 8.5 km/h.

## Multi-track aggregation — solved

Outputs go to `data/corner_segments/<track>/` plus a combined
`corner_segments_all.csv`/`.parquet` across every track. This is the aggregation
step section 29B asked for; the combined table reports 3 tracks rather than 1.

---

# 27C. Timestamp jitter tolerance (NEW, important)

Validation used to reject any lap whose `timestamp_ms` ever stepped backwards.
Measured across the whole dataset, that rule was discarding **20 of 100 laps**.

Every one of those laps had **exactly one** backward step, and the worst
anywhere was **-16.0 ms — less than a single 60 Hz frame (16.7 ms)**. In all of
them `frame_identifier` and `lap_distance_m` were perfectly monotonic, so the
rows were correctly ordered; the game had simply reported one stale
`currentLapTime`. Several laps glitch at the same point on the track, which
suggests something systematic in the game rather than packet loss.

Do **not** fix this by sorting on `timestamp_ms`. The row order is already
correct and sorting would corrupt it. The value is wrong, not the order.

`ValidationConfig` now tolerates up to `maximum_timestamp_backsteps` (3) steps
no larger than `maximum_timestamp_backstep_ms` (34 ms, two frames). Strict
`timestamps_monotonic` is still recorded, eligibility uses the new
`timestamps_acceptable`, and tolerated laps carry a `timestamp_jitter` quality
warning plus `timestamp_backward_steps` / `timestamp_worst_backward_ms` in the
catalog. A multi-second rewind, which is a flashback or a lap-boundary error,
still disqualifies.

Effect on usable laps:

```text
track     before   after
monaco        12      14
monza         26      33
spa           19      22
suzuka         2      11
rows         992    1336
```

---

# 27D. Phase 4A/4B results (NEW) — read this before designing Phase 4C

```powershell
python tools/run_phase4.py
```

`f1_engineer/modeling/` holds the k-NN, the evaluation protocols and the CLI.
Outputs land in `data/phase4/`.

## What works

**Known track, held out by session** — never by lap, since two laps from one
session share tyres, fuel and conditions:

```text
target                              MAE    baseline   skill
minimum_speed_kmh                  8.638    61.550     0.86
mean_abs_lateral_g                 0.107     0.405     0.74
peak_abs_steering                  0.060     0.194     0.69
minimum_speed_offset_from_apex_m  13.886    28.721     0.50
steering_rate_std_per_s            0.088     0.151     0.42
```

Skill is the fraction of the no-geometry baseline's error removed. Predicting
apex speed to 8.6 km/h against a 61.5 km/h baseline says the corner
representation is sound and the behaviour targets are learnable.

## What does not work yet — zero-shot transfer

On a track held out entirely, corner shape alone is weak:

```text
minimum_speed_kmh                  0.28
peak_abs_steering                  0.29
speed_loss_kmh                     0.24
steering_rate_std_per_s            0.22
peak_brake                         0.20
entry_speed_kmh                    0.13
brake_application_before_entry_m   0.09
mean_abs_lateral_g                 0.07
minimum_speed_offset_from_apex_m  -0.04
brake_release_smoothness_per_s    -0.12
trail_brake_past_entry_m          -0.31
```

Three targets are **worse than predicting the average**. The most
style-idiosyncratic behaviours are exactly the ones the prior gets wrong.

## The trap this evaluation was built to catch

The first version put the practice laps into the k-NN neighbour pool. Skill
looked excellent — 0.84 at one shot on apex speed. It was an artifact: the
identical corner sits at distance zero in geometry space, so with
`1/(d + epsilon)` weighting it swamps every other neighbour and the model
silently collapses into "average your own laps here" while still looking like a
geometry model. Measured against a same-corner-mean baseline the skill was
**exactly 0.000** on every primary target.

The practice laps are now kept out of the prior. `prior_reference_rows` is
recorded per row and a test asserts it does not change with shot count, so this
cannot regress unnoticed.

## Where geometry does earn its place

The prior weight is swept rather than assumed. For most targets the best weight
is **0.00** — once you have your own laps at a corner, the prior adds nothing.
But at weight zero the prior still fills gaps, and there it is worth real money:

```text
target                              gain over own-laps-only
speed_loss_kmh                                       18.4%
entry_speed_kmh                                      15.1%
brake_application_before_entry_m                     11.5%
peak_brake                                            8.1%
```

Those gains come from corners where the practice laps have no braking-complete
row. A geometry-matched prediction beats the driver's generic average there.
That is a genuine few-shot contribution: on a new track, for corners you have
not yet properly attacked, corner shape is better than nothing.

## Phase 4B — within-track normalisation — TESTED AND REJECTED

The hypothesis was that the same shape demands different behaviour at different
tracks (Monza low downforce, Monaco high), so geometry should predict behaviour
*normalised within track* and let the practice laps restore the level and
spread. `TrackNormaliser` implements it and it is scored side by side with the
raw prior on every run.

**It is worse, almost everywhere.**

```text
prior skill on an unseen track      raw    normalised (1 practice lap)
minimum_speed_kmh                  0.28          0.05
peak_abs_steering                  0.29          0.24
steering_rate_std_per_s            0.22          0.12
entry_speed_kmh                    0.13         -0.08
speed_loss_kmh                     0.24          0.12
```

On the full model, 4B is at best equal to 4A and much worse on
`entry_speed_kmh` (-41.6% against own-laps-only). Two reasons, both visible in
the numbers:

* **It destroys the physics that does transfer.** Apex speed against radius is
  roughly `v ~ sqrt(a_lat * r)`, which holds at every track. That relationship
  was carrying the raw prior's best result, and normalising replaces it with
  "rank this corner within its track", which is harder and less physical.
* **The restored level and spread are noisy.** Normalised skill is *higher*
  using pooled statistics (norm@0) than using estimates from one practice lap
  (norm@1), so reconstructing through two estimated quantities costs more than
  the cleaner training signal gains.

The code stays, because the comparison is worth keeping honest on every run,
but the raw prior is the model to build on.

## What the evidence says to do instead

The run now includes a coverage curve: zero-shot skill against the number of
tracks the prior was trained on.

```text
primary targets                  1 track   2 tracks   3 tracks
mean_abs_lateral_g                  0.00       0.03       0.07
minimum_speed_kmh                   0.21       0.24       0.28
minimum_speed_offset_from_apex_m   -0.21      -0.15      -0.04
peak_abs_steering                   0.14       0.27       0.29
steering_rate_std_per_s             0.17       0.20       0.22
mean                                0.06       0.11       0.16

braking targets, mean               0.04       0.04       0.04
```

**Primary targets improve monotonically with every track added, on every
target, with no sign of saturation.** Transfer there is limited by how few
corner shapes the model has seen, not by the method. Collecting a fifth and
sixth track is the highest-value work available.

**Braking targets are flat.** More tracks will not fix them. Braking depends on
things the geometry context does not carry, and the braking rows are sparser.
That group needs different features or a different model, not more laps.

Do not paper over the zero-shot result by tuning until it looks good. It is the
honest state of the representation, and split this way it says exactly where to
spend effort.

---

# 27E. Phase 4C — personal style envelope (NEW) — the differentiated product

```powershell
python tools/build_style_envelope.py
```

`f1_engineer/modeling/style_envelope.py`. Outputs to `data/style_envelope/`,
including `style_envelope.json`, which is the artefact the live overlay loads.

## Why this exists

Phase 4A/4B asked "what should this corner demand". The corner dictates that
(67-86% of variance), so it needs the track known, and it is the part an
in-game ghost already covers.

A variance decomposition asked a different question and split the behaviours
cleanly in two:

```text
behaviour                          corner explains   varies lap to lap
minimum_speed_kmh                             86%                  2%
peak_abs_steering                             86%                 12%
speed_loss_kmh                                85%                  4%
entry_speed_kmh                               73%                  4%
brake_application_before_entry_m              67%                 12%
steering_rate_std_per_s                       60%                 34%
brake_release_smoothness_per_s                21%                 69%
```

The corner-dominated group is where the driver is *already consistent*. The
lap-varying group is execution, and it is mostly the driver rather than the
corner. Knowing which lap a row came from explains a further 17% of release
smoothness and 26% of steering rate beyond the corner, so a lap has a
consistent character — rougher *that lap*, everywhere. That is real driver
state, not sensor noise, and it is the coachable part.

Note that the k-NN failed worst on exactly these targets (release smoothness
-0.12, trail braking -0.31). It was not broken; it was being asked to predict
things that are not properties of the corner.

## What it does

A robust centre and spread per style descriptor, per coarse corner band
(tight / medium / fast by minimum radius), built from every track. Deviation is
reported in robust sigmas, and a lap-level offset is available so a lap that
was rough everywhere is reported once rather than at every corner.

Because it keys only on a band, it applies to a circuit never driven. That is
the thing a ghost structurally cannot do: a ghost needs a reference lap on that
track and reports a time delta, where this reports what you did.

## Calibration result — leave one track out

Not an accuracy test. If the envelope carries, a held-out track sits near
centre 0 and spread 1.

```text
descriptor                       corner%  centre  spread  <2sd  oracle  flag%  worst  pass
brake_build_rate_per_s               33%    0.00    0.98   97%     96%     3%    4.0   4/4
steering_correction_count            68%   -0.06    1.04   97%     97%     3%    4.0   4/4
throttle_brake_overlap_s             15%    0.05    1.05   86%     84%    14%   26.6   4/4
steering_rate_std_per_s              66%    0.07    0.99   90%     91%    10%   10.3   3/4
brake_release_smoothness_per_s       31%    0.16    0.97   87%     91%    13%    6.9   3/4
rear_slip_bias                       45%   -0.00    0.98   86%     88%    14%   18.8   3/4
minimum_speed_offset_from_apex_m     74%    0.06    1.06   85%     93%    15%    9.5   3/4
trail_brake_past_entry_m             64%    0.03    1.04   86%     89%    14%   11.4   2/4
peak_rear_slip_abs                   23%    0.31    1.23   80%     90%    20%   57.5   2/4
coast_distance_m                     45%    0.00    0.87   95%     95%     5%   19.3   1/4
```

**`brake_build_rate_per_s` and `steering_correction_count` are clean.** Centre
and spread land almost exactly where they should, held-out coverage equals the
oracle fitted on that very track, so transfer costs nothing, and a 2-sigma
alert fires on 3% of corners with a worst reading of 4 sigma. These work on
lap 1 of a circuit that has never been driven.

`throttle_brake_overlap_s` passes centre and spread but its worst reading is
26.6 sigma: overlap is usually exactly zero, so the robust spread is tiny and
any overlap looks extreme. It needs a presence/absence treatment, not a sigma.
The same heavy-tail problem affects `peak_rear_slip_abs` (worst 57.5).

Four more descriptors pass on 3 of 4 tracks, including release smoothness. More
tracks would likely settle them.

## Worked example

Coaching speaks only through descriptors that survived the test, which is why
the output is sober rather than a wall of alerts:

```text
lap_20260822_135841_936_valid — using only the 3 descriptors that transfer:
  suzuka:1  (right, medium): more throttle and brake overlap than your normal [+2.0 sigma]
  suzuka:14 (left, fast):    more steering corrections than your normal      [+2.0 sigma]
```

Scored with all ten descriptors the same lap produces +7.3 and -6.6 sigma
readings from the heavy-tailed ones. The calibration gate is what keeps the
output trustworthy; do not remove it to get more to say.

## What this changes strategically

Collecting more tracks improves Phase 4A/4B, which is the ghost-like half.
**Phase 4C needs no new tracks at all** — it is already usable from the 1336
corner rows on disk, and it is the half a ghost cannot replicate.

---

# 28. Current phase status

Use this status, not the old `Context.txt` phase table:

```text
Phase 1 — Native F1 25 telemetry + recorder      COMPLETE
Phase 2 — Dashboard + comparison + overlay       COMPLETE
Phase 3A — Dataset catalog/validation            COMPLETE
Phase 3B — Distance normalization                COMPLETE
Phase 3C — Braking segmentation                  COMPLETE
Phase 3D — Segment/lap feature extraction        COMPLETE
Phase 3E — Feature-quality validation            COMPLETE
Phase 3F — Causal modeling contract/prep         COMPLETE

Phase 3G — track geometry map + corners          COMPLETE, VALIDATED
Phase 3H — geometry corner segments              COMPLETE, 958 ROWS/3 TRACKS
Multi-track aggregation                          COMPLETE (corner_segments_all)

Phase 3 code/architecture overall                COMPLETE
Behaviour dataset collection                     SUFFICIENT (spa/monza)
Geometry lap collection                          MONACO ONLY — SPA/MONZA NEEDED

Phase 4A — k-NN few-shot baseline                COMPLETE, SEE SECTION 27D
           known-track skill 0.42-0.86
           zero-shot transfer WEAK (0.07-0.29)
Phase 4B — within-track normalisation            TESTED, REJECTED (worse)
           coverage curve says: collect more tracks
Phase 4C — personal style envelope               COMPLETE, SEE SECTION 27E
           2 descriptors transfer cleanly to an
           unseen track; needs no new data
Phase 4B — possible segment sequence model       NOT STARTED
Phase 5 — live deviation/trigger logic           NOT STARTED
Phase 6 — final coaching delivery/voice          PARTLY PREBUILT
Phase 7 — integration/polish                     NOT STARTED
```

"Phase 6 partly prebuilt" means the floating driver overlay infrastructure already exists, but actual model-driven coaching and voice delivery are not complete.

---

# 29. Immediate next steps

Do not redesign Phase 3 again unless a concrete blocker appears.

Current order:

## A. Collect track geometry laps — HIGHEST PRIORITY

This is the only collection work currently blocking Phase 4. See section 27A.

Spa and Monza have sufficient **behaviour** data (25 and 30 laps) and must not
be re-collected. What they have zero of is **geometry**, because every one of
those laps predates the recorder format change.

```text
Spa:     DONE — 6 geometry laps, map built and validated
Monza:   DONE — 7 geometry laps, map built and validated
Monaco:  DONE — 3 geometry laps, map built and validated
Suzuka:  geometry DONE (map validated). 12 laps recorded, 11 eligible.
         Only exclusion is one game-invalid lap. Essentially at target.
```

Geometry collection is complete for all four tracks. The only remaining
collection work is **Suzuka behaviour laps**. Because the current recorder
stores Motion on every lap, those laps improve the geometry map as a free side
effect; there is no separate geometry pass any more.

These laps do **not** need to be quick, clean, or valid. Geometry does not care
about lap time — two of the three usable Monaco geometry laps are invalid.
Drive them at a comfortable pace on a normal racing line, staying on track.
Three to five laps is enough to average out line variation.

After driving:

```powershell
python tools/build_track_geometry.py --audit
python tools/build_track_geometry.py --track spa
python tools/build_track_geometry.py --track monza
```

Then open `data/track_geometry/<track>_map.png` and confirm the shape, the
corner count and the elevation profile look like the real circuit before
trusting the map.

Behaviour collection for Suzuka still follows the old pattern:

```powershell
python tools/analyze_laps.py --data data/raw_laps --track suzuka
```

## B. Build multi-track aggregation before Phase 4 — DONE, see section 27B

`python tools/analyze_corners.py` writes per-track folders under
`data/corner_segments/` plus a combined `corner_segments_all.csv` covering every
track in one table. It reports 3 tracks and 20 sessions rather than 1.

The old `data/processed` path still overwrites per track. That is acceptable
while Phase 4 consumes the corner-segment table, but do not mix the two.

## C. Verify session IDs

Phase 4 requires reliable session grouping.

Ensure modeling rows preserve fields like:

```text
session_uid/session_id
track
session_type
lap/source identifier
source_file
```

Exact field names can follow the current codebase; the principle is what matters.

## D. Verify local geometry availability — ANSWERED, see section 27A

The answer was no: the recorder was discarding the Motion fields. It now stores
them, `f1_engineer/analysis/geometry.py` derives world position, heading,
elevation, curvature and turn direction from them, and the result is validated
against Monaco.

The remaining work is not investigation but collection (step A) followed by
re-segmenting on geometry instead of on the brake threshold.

Do not invalidate the existing dataset.

## E. Start Phase 4A

Implement:

```text
k-NN context similarity
weighted expected behavior
normalized residual scoring
session-aware evaluation
known-track test
leave-one-track-out sanity check
0/1/3/5-shot adaptation curves
```

Only then decide whether a more complex model is needed.

---

# 30. Important things NOT to do

Do not:

- delete raw laps to make folders look clean
- redo Spa or Monza without a concrete reason
- loosen G-force plausibility thresholds just to increase clean-lap counts
- treat game-valid as equivalent to clean baseline
- treat Time Trial static strategy context as bad core-driving data
- randomly split laps from the same session between train and test
- use behavior outputs as k-NN context inputs
- build a large explicit maneuver taxonomy before proving it is needed
- force every rare braking event into a canonical zone
- assume "corner 3" means anything shared across tracks
- start with a giant whole-lap LSTM just because the original phase plan said so
- compare laps by timestamp instead of normalized distance
- create a second UDP listener just for the overlay
- move all logic back into `main.py`

---

# 31. Development style / collaboration preferences

When giving code changes:

- Prefer a **single drop-in patch** rather than many alternate downloads.
- Avoid presenting multiple confusing patch choices.
- Give exact commands to run.
- Keep existing raw data safe.
- Preserve modular architecture.
- Do not rewrite the whole project unless absolutely necessary.
- Validate changes against the current pipeline.
- Explain whether a change affects historical data or only future recordings.
- Do not make the project more sophisticated merely for sophistication's sake.
- Prioritize what gets to working few-shot coaching.

The user is comfortable collecting telemetry and running commands, but wants the assistant to keep architectural continuity so they do not have to re-explain every prior decision.

---

# 32. Historical original plan vs current plan

The original `PHASE PLAN.pdf` was a useful starting blueprint.

Original Phase 4:

```text
whole-lap LSTM autoencoder
T × 26
128 hidden × 2 encoder
32-d style vector
128 hidden × 2 decoder
MSE reconstruction
minimum 50 laps across 4 tracks
target 100+
```

The current project has intentionally evolved.

What remains from the original plan:

- collect everything early
- Parquet per lap
- 60 Hz telemetry
- distance alignment
- feature extraction
- braking-event fallback when no corner map
- 120% median lap-time outlier idea
- personalized anomaly/deviation detection
- priority/cooldown coaching
- overlay + optional voice
- 50+ varied clean laps as a sensible data milestone

What changed:

- UI/recording phases were implemented in a more modular architecture
- Phase 3 became much richer and quality-aware
- clean baseline vs dynamic/static context was formalized
- session leakage became an explicit concern
- few-shot unfamiliar-track coaching became the clearer end goal
- whole-lap LSTM is no longer the first required model
- Phase 4A is now k-NN context→expected-behavior residual modeling
- segment-level sequence learning is a later candidate
- continuous context descriptors are preferred over a hand-built maneuver taxonomy
- causal/live compatibility is now part of the modeling contract

---

# 33. One-paragraph project summary for an AI assistant

This is a modular standalone Python F1 25 race-engineer project that already has working native UDP parsing, ~60 Hz enriched per-lap Parquet recording, lap/flashback handling, a live dashboard/reference-delta system, floating driver overlay, dataset validation, 2001-point distance normalization, braking-zone segmentation, segment/lap feature extraction, quality classification, a Phase 3F causal context-vs-behavior modeling contract, and a Phase 3G track geometry map that derives heading, curvature, elevation and a corner table from Motion telemetry (section 27A). Spa and Monza behaviour data are sufficient; Monaco currently has 8 clean baseline laps and Suzuka is still pending. The one outstanding collection task is 3-5 geometry laps each at Spa and Monza, since every existing lap at those tracks predates the recorder format change and carries no Motion fields; those laps need no particular pace or validity, and existing laps do not need re-driving because the map joins onto them by `lap_distance_m`. Raw data must be preserved. The revised goal is few-shot coaching on unfamiliar tracks. Phase 4 should begin with a session-aware k-NN model that matches execution-independent/causal maneuver context to historical similar contexts, predicts this driver's expected behavior, and scores normalized residuals. Evaluate with known-track tests, leave-one-track-out as a sanity check, and especially 0/1/3/5-shot adaptation curves. Do not jump straight to the originally planned whole-lap LSTM; treat segment-level sequence models as later improvements if the simpler interpretable approach proves the representation works.

---

# 34. Current checkpoint

As of the latest working checkpoint:

```text
PHASE 1              COMPLETE
PHASE 2              COMPLETE
PHASE 3A–3F          COMPLETE IN CODE
PHASE 3G geometry    COMPLETE, VALIDATED ON ALL FOUR TRACKS
PHASE 3H corners     COMPLETE — 1336 rows, 4 tracks, 21 sessions
Spa behaviour        SUFFICIENT
Monza behaviour      SUFFICIENT
Monaco behaviour     8 CLEAN BASELINE, OPTIONAL MORE NEEDED
Suzuka behaviour     11 ELIGIBLE LAPS — CLOSE TO TARGET
Spa geometry         6 LAPS, MAP BUILT AND VALIDATED
Monza geometry       7 LAPS, MAP BUILT AND VALIDATED
Monaco geometry      3 LAPS, MAP BUILT AND VALIDATED
Suzuka geometry      4 LAPS, MAP BUILT AND VALIDATED
Multi-track aggregation           DONE (corner_segments_all)
Phase 4A k-NN        DONE — known-track works, zero-shot weak (section 27D)
Phase 4B normalise   DONE — TESTED AND REJECTED, raw prior is better
Phase 4C envelope    DONE — 2 descriptors transfer to an unseen track
Next                 WIRE 4C INTO THE LIVE OVERLAY (needs no new data), and
                     optionally collect more tracks to improve 4A/4B
```

Latest command confirmed working:

```powershell
python tools/prepare_modeling.py --processed data/processed
```

Latest Phase 3F Monaco result:

```text
Modeling segment rows:       74
Context-complete rows:       74
k-NN v1 candidate rows:      34
Sessions represented:        2
Tracks represented:          1
Session-ID rows:             74
Context contract:            causal_proxy_v1
Taxonomy:                    deferred (continuous descriptors)
Split rule:                  session-grouped
```

The next AI assistant should continue from this state rather than restarting the architecture or treating the old Phase Plan's status table as current.
