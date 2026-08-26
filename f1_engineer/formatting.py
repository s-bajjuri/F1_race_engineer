from __future__ import annotations

def format_gear(gear: int) -> str:
    if gear == -1:
        return "R"
    if gear == 0:
        return "N"
    return str(gear)

def format_lap_time(milliseconds: int | None) -> str:
    if milliseconds is None:
        return "--:--.---"
    minutes, remainder = divmod(milliseconds, 60_000)
    return f"{minutes}:{remainder / 1000:06.3f}"
