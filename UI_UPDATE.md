# Dark UI Update — Installation

This update preserves the verified UDP receiver, recorder, Parquet writer, reference-lap replacement, and health logic. It changes only application orchestration, configuration, and presentation.

## Safest installation

1. Back up your current project folder or create a Git commit.
2. Extract this package beside the current project.
3. Copy your existing `data/` directory into the updated folder. Do not overwrite or delete your Parquet laps.
4. Run the tests:

```powershell
python -m unittest discover -s tests -v
```

5. Start the program:

```powershell
python main.py --track spa
```

## Views

- **Drive:** minimal delta, throttle/brake history, tyre temperatures, and recording state.
- **Analysis:** detailed current/reference telemetry plots.
- **Diagnostics:** health and developer details.

Use the top buttons or keyboard shortcuts `1`, `2`, and `3`.

## Configuration

Edit `config.yaml` to choose the startup view and trace length. Tyre thresholds in this build are presentation warnings only; they are deliberately configurable and can later become compound-aware.

## Old file

The former `f1_engineer/ui/analysis_dashboard.py` is archived as `archive/phase2d_analysis_dashboard.py` and is no longer imported. If installing only selected files over your old project, the old module may remain without affecting execution.
