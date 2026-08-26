# Migration from `phase2d_app.py`

1. Keep your existing `data/` folder unchanged.
2. Extract this project into your F1 Race Engineer folder.
3. Copy your existing `data/raw_laps/` and `data/sessions/` into this project's `data/` folder.
4. Install dependencies with `python -m pip install -r requirements.txt`.
5. Set the track in `config.yaml` or run `python main.py --track spa`.
6. Close every older UDP listener before launching `main.py`.
7. Confirm the dashboard receives telemetry and saves one lap before deleting old scripts.

After the modular build passes a live lap test, archive or delete:

- `phase2_app.py`
- `phase2_app_fixed.py`
- `phase2c_app.py`
- `phase2d_app.py`
- the old placeholder `lap_recorder.py`

Keep one monolithic snapshot until the modular build has completed a full live test.
