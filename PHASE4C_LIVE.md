# Phase 4C Live — Corner-by-corner style coaching in the overlay

## Why

Phase 4C (`f1_engineer/modeling/style_envelope.py`, see context.md §27E) learned
what *your* execution normally looks like, per corner band (tight / medium /
fast), from 1336 corner rows across Spa, Monza, Monaco and Suzuka. Three
descriptors passed the leave-one-track-out gate:

| descriptor | what a flag means |
|---|---|
| brake_build_rate_per_s | brake applied harder/faster (or more gradually) than your normal |
| steering_correction_count | more (or fewer) steering corrections than your normal |
| throttle_brake_overlap_s | throttle and brake overlapped (presence-based, see below) |

Today this scoring only runs offline (`tools/build_style_envelope.py`). This
phase runs it live: when you exit a corner, the overlay shows at most one line,
e.g.

    T14 (left, fast) — more steering corrections than your normal  +2.3σ

This is the part a ghost cannot do. It reports *what you did* against *your own
normal*, not a time delta. It needs **no new laps**.

## What is in scope / out of scope

In scope:
- Live scoring on tracks that have a geometry map (spa, monza, monaco, suzuka).
- Only the descriptors listed in `style_summary.json → transferring_targets`.
- Overlay line + Diagnostics status + config switch + `--no-coaching`.

Out of scope (later):
- Tracks with no geometry map (building the corner map live on lap 1).
- Lap-level offset ("you were rough everywhere this lap").
- Voice (Phase 6).
- Changing the envelope itself or the calibration gate.

## Design in one picture

    UDP → UDPReceiver → LapRecorder.try_emit_frame(row)
                              ├─ comparison.append_row(row)      (existing)
                              └─ coach.append_row(row)           (NEW)
          lap/flashback boundary → comparison.start_lap(...)      (existing)
                                 → coach.start_lap(...)           (NEW)

    LiveCornerCoach
      windows  = build_corner_windows(<track>_corners.csv, <track>_geometry)
      envelope = StyleEnvelope.load(style_envelope.json)
      targets  = style_summary.json → transferring_targets
      on row: when lap_distance_m passes the next window_end_m →
              extract_corner_behavior(rows in window, window)
              + ctx_minimum_radius_m → envelope.score() → largest |z| ≥ 2.0
              → CoachingSnapshot.last_message

    AppRuntime.snapshot() → DashboardState.coaching → DriveOverlay / Diagnostics

The same extractor is used offline and live, so a live reading and an offline
reading of the same lap must match. That is the main correctness check.

## Implementation steps

1. **Envelope loading** (`modeling/style_envelope.py`)
   - Add `StyleEnvelope.from_dict(d)` and `StyleEnvelope.load(path)`. These invert
     `to_dict()`: config from band_edges/band_names/deviation_threshold, `bands_`
     keyed by (target, band), and `global_`.
   - Add `load_transferring_targets(summary_path) -> tuple[str, ...]`.
   - Test: fit → to_dict → from_dict gives identical `score()` output.

2. **Coach** (new package `f1_engineer/coaching/`, file `corner_coach.py`)
   - `CoachingConfig` (enabled, envelope_path, summary_path, geometry_root,
     display_seconds=6, minimum_abs_z from envelope threshold).
   - `CoachingSnapshot` frozen dataclass: enabled, status, last_message,
     last_corner_key, last_z, message_time, corners_scored, corners_flagged.
   - `LiveCornerCoach.from_files(track, config)` returns a coach that is
     *disabled with a reason* when geometry, envelope or summary is missing. It
     never raises into the app.
   - `start_lap(lap_num, capture)`: clear rows, reset the next-window pointer.
   - `append_row(row)`: append, and if `lap_distance_m >= windows[next].window_end_m`,
     score that window and advance. This must be cheap. Scoring only happens at
     window ends (~15–20 times per lap) on a few hundred rows.
   - Scoring: DataFrame(rows in window) → `extract_corner_behavior` → add
     `window.context` (gives `ctx_minimum_radius_m`) → `envelope.score()`.
     Restrict to transferring targets. Respect `braking_complete` /
     `behavior_complete` (already enforced by `usable_rows`).
   - Overlap rule: `throttle_brake_overlap_s` flags only when value > 0 and
     z ≥ threshold. The message shows no σ, because its spread is tiny and the σ
     is meaningless (context.md §27E, worst 26.6σ).
   - Thread safety: `threading.Lock` around state. `append_row` runs on the
     receiver thread and `snapshot()` runs on the UI thread.

3. **Plumbing**
   - `recording/recorder.py`: optional `coach` constructor arg. Next to every
     `self.comparison.start_lap(...)` call `self._coach_start_lap(...)`. After
     `self.comparison.append_row(row)` call the coach. A small private helper
     keeps this to one line per site.
   - `runtime.py`: build the coach from config, pass it to `LapRecorder`, and
     add `coaching=` to `DashboardState`.
   - `ui/view_state.py`: `coaching: CoachingSnapshot` field.
   - `ui/drive_overlay.py::_draw`: one text line at the bottom. When the message
     is older than `display_seconds`, show it muted, or show nothing.
   - `ui/diagnostics_view.py`: "Coaching: enabled/disabled (reason), corners
     scored N, flagged M".
   - `config.py` / `config.yaml`: a new `coaching:` section. `main.py`:
     `--no-coaching`. `app.py`: pass it through.

4. **Tests** (`tests/test_corner_coach.py`)
   - Envelope round-trip.
   - Synthetic lap over two synthetic windows. Nothing fires before
     `window_end_m`, there is exactly one score per window, and `start_lap`
     resets.
   - A non-transferring target with huge z never produces a message.
   - Missing files → disabled, status explains why, and no exception.
   - `test_ui_imports` still passes.

5. **Parity check script** (`tools/replay_coaching.py`)
   - Loads a raw Parquet lap, feeds rows one by one into `LiveCornerCoach`, and
     prints every corner's z-scores and message.
   - `--compare` also prints the offline z for the same lap from
     `data/corner_segments/corner_segments_all.csv`, scored with the same
     envelope, so the two can be diffed.

6. **Docs**: update context.md §28 and §34 (4C wired live) and README (the
   coaching section).

## Execute — on the Windows PC

    cd <project>
    git pull
    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -t . -v

    # parity: live replay must equal offline on a known lap
    python tools/replay_coaching.py --track suzuka `
        --lap data/raw_laps/suzuka/lap_20260822_135841_936_valid.parquet --compare

Expected (context.md §27E worked example, transferring descriptors only):

    suzuka:1  (right, medium): throttle and brake overlapped
    suzuka:14 (left, fast):    more steering corrections than your normal  +2.0σ

Then live:

    python main.py --track suzuka

- F1 25 in borderless windowed mode, UDP on port 20777, and nothing else listening on it.
- Drive 2–3 laps. After each corner exit, expect either nothing (normal) or a single line.
- Diagnostics view (key 3): "corners scored" rises by roughly the corner count each lap.
- Flashback mid-lap: no stale message from the discarded segment.
- `python main.py --track suzuka --no-coaching`: the overlay is unchanged from before.

## What "done" looks like

- All tests pass (the old ones plus the new coach tests).
- Replay output matches offline z-scores to about 0.01 on every corner of the reference lap.
- Live laps show sparse, sensible messages. Roughly 3–10% of corners should
  flag, because the calibration says a 2σ alert fires on ~3% for the clean
  descriptors.
- Historical data untouched. This change affects only live display, not recordings.

## Risks to watch

- **Start/finish wrap**: corner 1's approach window may start before 0 m.
  Check how `build_corner_windows` clamps it, and that live rows on the new lap
  begin at ~0 m.
- **Sample rate**: live rows arrive at ~60 Hz, the same rows that end up in
  Parquet. The rate-dependent descriptors (build rate) should match offline as
  long as the coach uses the recorder's rows, not the dashboard buffer.
- **UI thread cost**: scoring must stay on the receiver thread at window end.
  `snapshot()` only copies the latest values.
- **Mac**: the code needs Python ≥3.10. Use `uv venv -p 3.13` to run tests
  there. The game and live test run on Windows only.
