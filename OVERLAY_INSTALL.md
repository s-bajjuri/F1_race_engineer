# Floating Driver Overlay — Install and Use

## Install over the existing modular project

1. Close the F1 Race Engineer application.
2. Back up or commit the current project.
3. Extract `f1_driver_overlay_patch.zip` directly into the project root—the folder that already contains `main.py`.
4. Allow Windows to replace the existing files.
5. Do **not** delete or replace the existing `data/` directory.

The patch adds:

```text
f1_engineer/ui/drive_overlay.py
```

and updates:

```text
main.py
config.yaml
f1_engineer/app.py
f1_engineer/config.py
f1_engineer/ui/dashboard.py
tests/test_config.py
tests/test_ui_imports.py
```

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Start the app

```powershell
python main.py --track spa
```

The floating overlay starts automatically. The normal Analysis/Diagnostics window remains separate and can be minimized while the overlay stays above the game.

Use **borderless windowed** or **windowed** display mode in F1. Exclusive fullscreen can prevent ordinary desktop overlays from appearing.

## Toggle the overlay

While the main dashboard has focus:

```text
O = show/hide overlay
```

There is also an **OVERLAY** button at the top of the dashboard.

## Position, size, and appearance

Edit the `overlay:` section in `config.yaml`:

```yaml
overlay:
  enabled: true
  visible_on_start: true
  width: 560
  height: 260
  x: 30
  y: 30
  opacity: 0.92
  click_through: true
  trace_seconds: 7
  refresh_hz: 20
```

- `x`: pixels from the left side of the desktop.
- `y`: pixels from the top of the desktop.
- `width` / `height`: overlay dimensions.
- `opacity`: `1.0` is solid; lower values are more transparent.
- `click_through: true`: mouse clicks go to the game instead of the overlay.
- `trace_seconds`: visible throttle/brake history.

To find a good position, temporarily use:

```yaml
click_through: false
```

The current version is positioned through `x` and `y`, not by dragging. Restore `click_through: true` after placement so the overlay cannot steal game clicks.

## Disable for one run

```powershell
python main.py --track spa --no-overlay
```

Override pedal-history length for one run:

```powershell
python main.py --track spa --overlay-trace 6
```

## If it does not appear

1. Use F1 borderless-windowed mode.
2. Confirm `overlay.enabled: true`.
3. Confirm the overlay coordinates are on the active monitor.
4. Run the application from a normal Windows Python installation with Tkinter available.
5. Look for this message in the terminal:

```text
Overlay disabled: the active Matplotlib backend is not Tk.
```

If that message appears, configure Matplotlib to use the Tk backend on the Windows machine.
