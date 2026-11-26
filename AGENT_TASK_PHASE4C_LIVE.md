# Agent task — Phase 4C live coaching (Windows)

You are continuing an existing, working project. Do not redesign it. Read this
file fully, then `context.md` sections 27E, 28, 30, 31 and 34, and
`PHASE4C_LIVE.md`. The goal, design and acceptance criteria are fixed there.
This file tells you exactly how to execute them.

## Environment
- Windows, PowerShell, project root = the folder containing `main.py`.
- Python ≥ 3.10 (code uses `slots=True`, `X | None`). Install deps:
  `python -m pip install -r requirements.txt`
- Run tests the way this project runs them:
  `$env:PYTHONPATH="."; python -m unittest discover -s tests -t . -q`
- Raw laps live in `data/raw_laps/<track>/*.parquet` (gitignored). **Read only.**
- F1 25 sends UDP to port 20777. Only one listener can bind it. Check with
  `Get-NetUDPEndpoint -LocalPort 20777` before starting the app.

## Hard rules (from context.md §30–31)
- Never delete, move, rewrite or re-save anything under `data/raw_laps/`,
  `data/sessions/`, `data/track_geometry/`, `data/style_envelope/`.
- Do not change `StyleEnvelope.fit/score`, the band edges, the 2.0 threshold,
  or the transferring-target gate. Do not add descriptors to make output
  "more interesting".
- Do not create a second UDP listener. The coach is fed by `LapRecorder`.
- Do not move logic into `main.py`. Keep the modular layout.
- One coherent patch. No alternative variants.
- Explain in your summary whether anything affects historical data (it
  should not).

## Step 0 — baseline
1. `git status` should be clean. Create the branch `phase4c-live`.
2. Run the test suite and record the pass count. If anything already fails,
   stop and report before changing code.
3. `python tools/build_style_envelope.py` should finish and leave
   `data/style_envelope/style_envelope.json` unchanged (check with `git diff --stat data/`).
   If it changes, revert it with `git checkout -- data/style_envelope` and note it.

## Step 1 — envelope loading
File: `f1_engineer/modeling/style_envelope.py`
- Add the classmethod `StyleEnvelope.from_dict(data: dict) -> StyleEnvelope`
  that inverts `to_dict()`:
  - `config = StyleEnvelopeConfig(band_edges=tuple(...), band_names=tuple(...),
    deviation_threshold=float(...))`
  - `targets = tuple(data["targets"])`
  - `bands_[(target, band)] = (centre, spread, rows)` from `data["bands"]`
  - `global_[target] = (centre, spread, rows)` from `data["global"]`
- Add the classmethod `StyleEnvelope.load(path: Path)`.
- Add the module function `load_transferring_targets(summary_path: Path) -> tuple[str, ...]`
  that reads `transferring_targets` and keeps only names in `STYLE_TARGETS`.
- Test in `tests/test_style_envelope.py`: fit on the synthetic rows already used
  there → `from_dict(to_dict())` → `score()` frames are equal
  (`pd.testing.assert_frame_equal`).

## Step 2 — the coach
New: `f1_engineer/coaching/__init__.py`, `f1_engineer/coaching/corner_coach.py`

    @dataclass(frozen=True, slots=True)
    class CoachingConfig:
        enabled: bool = True
        envelope_path: Path = Path("data/style_envelope/style_envelope.json")
        summary_path: Path = Path("data/style_envelope/style_summary.json")
        geometry_root: Path = Path("data/track_geometry")
        display_seconds: float = 6.0

    @dataclass(frozen=True, slots=True)
    class CoachingSnapshot:
        enabled: bool
        status: str               # "active" or why disabled
        last_message: str | None
        last_corner_key: str | None
        last_z: float | None
        message_age_s: float | None
        corners_scored: int
        corners_flagged: int

    class LiveCornerCoach:
        @classmethod
        def from_files(cls, track: str, config: CoachingConfig) -> "LiveCornerCoach"
        def start_lap(self, lap_num: int | None, capture: bool) -> None
        def append_row(self, row: dict[str, Any]) -> None
        def snapshot(self) -> CoachingSnapshot

Implementation notes:
- Load geometry with `_load_track_geometry(geometry_root, track)` from
  `f1_engineer/analysis/corner_pipeline.py`. If importing a private helper feels
  wrong, make a public `load_track_geometry` wrapper there and use it in both
  places. Build windows with `build_corner_windows(corners, grid, track=track)`.
- Any missing file / exception during construction → a disabled coach with
  `status` explaining it (e.g. "no geometry for track 'bahrain'"). It must never
  raise into `AppRuntime`.
- `append_row`: if disabled or `capture` is False, return. Append the row. While
  `self._next < len(windows)` and `row["lap_distance_m"] >= windows[self._next].window_end_m`,
  score that window and then advance `_next`. Ignore rows with a non-finite distance.
- Scoring a window:
  1. `frame = pd.DataFrame(self._rows)` (only rows since lap start; slice by
     distance yourself or let `extract_corner_behavior` select).
  2. `behavior = extract_corner_behavior(frame, window)`
  3. `record = {**window.context, **behavior, "corner_key": window.corner_key,
     "turn_direction": window.direction, "lap_id": "live"}`
  4. `scores = envelope.score(pd.DataFrame([record]))`
  5. Consider only `z_<t>` for `t` in transferring targets. For
     `throttle_brake_overlap_s`, flag only if the value > 0 and z ≥ threshold,
     with the message "throttle and brake overlapped" and no σ. For the others,
     flag if |z| ≥ threshold, phrase with `DEVIATION_PHRASES[t]` (high if z>0
     else low), and format `f"T{corner_id} ({direction}, {band}) — {phrase}  {z:+.1f}σ"`.
  6. Keep the single largest |z| flag for the corner. Store message, z, and
     `time.monotonic()`. Increment the counters.
- `start_lap`: clear rows, `_next = 0`, store `capture`. Do not clear the last
  message (it fades by age).
- Use a `threading.Lock`. `append_row` runs on the receiver thread and
  `snapshot` runs on the Tk thread.
- Performance budget: append is O(1) except at window ends. Measure one
  window score on a real lap. It should be < 20 ms. Report the number.

## Step 3 — plumbing
- `f1_engineer/recording/recorder.py`
  - Add `coach: LiveCornerCoach | None = None` to `__init__`, stored as `self.coach`.
  - Add `def _start_lap(self, lap_num, capture)` that calls
    `self.comparison.start_lap(lap_num, capture=capture)` and then
    `self.coach.start_lap(...)` if a coach is set. Replace each of the existing
    `self.comparison.start_lap(...)` calls (there are 7) with `self._start_lap(...)`.
    Keep the arguments identical.
  - After `self.comparison.append_row(row)` add `if self.coach: self.coach.append_row(row)`.
- `f1_engineer/ui/view_state.py`: add `coaching: CoachingSnapshot | None = None`
  as the last field, so existing constructors in tests still work.
- `f1_engineer/runtime.py`: accept `coaching: CoachingConfig | None`, build
  `LiveCornerCoach.from_files(track, coaching)` when enabled, pass it to
  `LapRecorder`, and add `coaching=self.coach.snapshot() if self.coach else None`.
- `f1_engineer/config.py` + `config.yaml`: a `coaching:` section with
  `enabled`, `envelope_path`, `summary_path`, `geometry_directory`,
  `display_seconds`. Follow the `_nested` / `_as_bool` pattern exactly.
  `main.py`: `--no-coaching`. `app.py` / `ui/dashboard.py`: pass it through to
  `AppRuntime` (follow how `overlay_*` options flow).
- `f1_engineer/ui/drive_overlay.py::_draw`: at the bottom, draw
  `snapshot.last_message`. Use the normal colour when age < display_seconds,
  otherwise draw nothing. Use the existing `theme` colours and "Segoe UI".
  If the coach is disabled, draw nothing (the reason goes to Diagnostics).
- `f1_engineer/ui/diagnostics_view.py`: one row with the status and
  scored/flagged counts.

## Step 4 — tests (`tests/test_corner_coach.py`)
Use synthetic data only. No dependency on gitignored files.
- Build two `CornerWindow`s by hand (one tight, one fast) and a tiny envelope
  via `StyleEnvelope.from_dict`.
- Feed synthetic rows (lap_distance_m increasing, with throttle/brake/steering
  columns that `extract_corner_behavior` needs; check its required columns).
  Assert nothing is scored before `window_end_m` and that exactly one score
  happens per window.
- `start_lap` resets `_next`. A second lap scores both windows again.
- A target not in the transferring set never appears in messages.
- `from_files` with a missing track → `enabled False` and a non-empty `status`.
- Run the full suite. Everything that passed in Step 0 must still pass.

## Step 5 — parity replay (the key correctness check)
New `tools/replay_coaching.py` (follow other `tools/*.py`: thin wrapper that
inserts the project root in `sys.path` and calls a function).
- Args: `--track`, `--lap <parquet>`, `--compare`.
- Reads the Parquet. Feeds rows in order to a fresh coach (`start_lap(1, True)`
  first). Prints one line per corner: key, band, z for each transferring target,
  message.
- `--compare`: loads `data/corner_segments/corner_segments_all.csv`, selects
  the rows for this lap (match on the lap id / source file column; inspect the
  CSV to find it), scores them with the same envelope, and prints
  offline-vs-live z side by side with the max abs diff.
- Run:
      python tools/replay_coaching.py --track suzuka `
        --lap data/raw_laps/suzuka/lap_20260822_135841_936_valid.parquet --compare
  (If that exact file name differs, `Get-ChildItem data/raw_laps/suzuka` and
  pick the matching one.)
- **Pass**: max |live z − offline z| ≤ 0.01 on every corner and target. The
  messages include suzuka:1 overlap and suzuka:14 steering corrections
  (context.md §27E).
- If it does not match, the usual causes are a different row set (start/finish
  wrap on corner 1, flashback rows, a missing column) or the context dict
  not being merged. Fix the coach, not the offline pipeline or the envelope.
  If you cannot get parity, stop and report the diff table.
- Also replay one Spa and one Monza lap and report the flag rate (expected
  roughly 3–10% of corners).

## Step 6 — live run (only if the user is at the PC and the game is running)
- `Get-NetUDPEndpoint -LocalPort 20777` should show the port free.
- `python main.py --track suzuka`. Ask the user to drive 2 laps, including one
  flashback.
- Check that Diagnostics shows coaching active and corners_scored rising, that
  the overlay shows sparse messages, and that no message appears for a corner
  that was rewound by a flashback.
- `python main.py --track suzuka --no-coaching` should run exactly as before.
- Also try `--track bahrain` (no geometry). The app should run, with coaching
  disabled and the reason shown in Diagnostics.

## Step 7 — wrap up
- Update `context.md` §28 (a "Phase 4C live" line) and §34 checkpoint. Add a
  short "Coaching" section to `README.md`.
- Commit on `phase4c-live` with a clear message. Do not push unless the user asks.
- Final report to the user, short:
  - files changed/added
  - test counts before → after
  - the parity table summary (max diff) and the Suzuka messages
  - flag rate on the replayed laps
  - window scoring time
  - confirmation that no data files changed (`git status data/`)
  - anything left undone or uncertain

## Stop and ask the user if
- the baseline tests fail before you start
- parity cannot be reached within 0.01
- anything would require changing raw data, the envelope fit, or the gate
- the recorder's lap/flashback flow behaves differently from what is described here
