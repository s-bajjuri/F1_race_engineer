from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_engineer.modeling.evaluation import (
    TARGET_GROUPS,
    EvaluationConfig,
    evaluate_all,
)
from f1_engineer.modeling.knn import KNNConfig


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 4A: predict this driver's corner behaviour from track "
            "geometry, and measure how quickly it adapts to an unseen track."
        )
    )
    parser.add_argument(
        "--segments",
        type=Path,
        default=Path("data/corner_segments/corner_segments_all.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/phase4"))
    parser.add_argument("--neighbours", type=int, default=10)
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=[0, 1, 3, 5],
        help="Practice-lap counts for the adaptation curve.",
    )
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip the adaptation-curve PNG."
    )
    return parser


def _load_segments(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path, engine="pyarrow")
        return pd.read_csv(path)
    alternative = path.with_suffix(".csv")
    if alternative.exists():
        return pd.read_csv(alternative)
    raise FileNotFoundError(
        f"no corner segment table at {path}. Build it with:\n"
        "  python tools/analyze_corners.py"
    )


def plot_adaptation_curves(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = sorted(summary["target"].unique())
    tracks = sorted(summary["unseen_track"].unique())
    columns = min(3, max(1, len(targets)))
    rows = int(np.ceil(len(targets) / columns))

    figure, axes = plt.subplots(
        rows, columns, figsize=(5.0 * columns, 3.6 * rows), squeeze=False
    )
    for position, target in enumerate(targets):
        axis = axes[position // columns][position % columns]
        subset = summary[summary["target"] == target]
        for track in tracks:
            track_subset = subset[subset["unseen_track"] == track].sort_values("shots")
            if track_subset.empty:
                continue
            line = axis.plot(
                track_subset["shots"],
                track_subset["blended_normalised_mae"],
                marker="o",
                label=track,
            )[0]
            axis.plot(
                track_subset["shots"],
                track_subset["own_mae"],
                marker="x",
                linestyle="--",
                alpha=0.55,
                color=line.get_color(),
            )
            baseline = track_subset["pool_mean_mae"].mean()
            if np.isfinite(baseline):
                axis.axhline(baseline, linestyle=":", alpha=0.25, color=line.get_color())
        axis.set_title(target, fontsize=9)
        axis.set_xlabel("practice laps on the unseen track")
        axis.set_ylabel("MAE")
        axis.grid(True, alpha=0.25)
    for position in range(len(targets), rows * columns):
        axes[position // columns][position % columns].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=max(1, len(labels)))
    figure.suptitle(f"{title}  (solid = 4B normalised blend, dashed = own laps only)")
    figure.tight_layout(rect=(0, 0.05, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def format_report(results: dict[str, pd.DataFrame], overview: dict[str, Any]) -> str:
    lines = [
        "PHASE 4A — GEOMETRY CONTEXT k-NN",
        "=================================",
        f"Corner rows:        {overview['rows']}",
        f"Tracks:             {overview['tracks']}",
        f"Sessions:           {overview['sessions']}",
        f"Neighbours:         {overview['neighbours']}",
        "",
    ]

    for group in TARGET_GROUPS:
        summary = results.get(f"few_shot_summary_{group}")
        if summary is None or summary.empty or "shots" not in summary.columns:
            continue
        shots = sorted(summary["shots"].unique())
        practice = [shot for shot in shots if shot > 0]

        lines.append(f"PRIOR QUALITY ON AN UNSEEN TRACK — {group} targets")
        lines.append("-" * 78)
        lines.append(
            "Skill = fraction of the no-geometry baseline's error removed; "
            "0 means no better."
        )
        lines.append(
            "raw = geometry predicts behaviour directly.  "
            "norm@n = geometry predicts within-track"
        )
        lines.append(
            "deviation, with level and spread taken from n practice laps "
            "(4B). Neither sees"
        )
        lines.append("the practice laps as neighbours.")
        lines.append("")
        header = f"{'target':36}{'raw':>7}{'norm@0':>9}" + "".join(
            f"{f'norm@{n}':>9}" for n in practice
        )
        lines.append(header)
        for target in sorted(summary["target"].unique()):
            subset = summary[summary["target"] == target]
            zero = subset.loc[subset["shots"] == 0]
            cells = (
                f"{zero['skill_geometry'].mean():>7.2f}" if len(zero) else f"{'-':>7}"
            )
            cells += (
                f"{zero['skill_normalised'].mean():>9.2f}" if len(zero) else f"{'-':>9}"
            )
            for shot in practice:
                row = subset[subset["shots"] == shot]
                cells += (
                    f"{row['skill_normalised'].mean():>9.2f}"
                    if len(row)
                    else f"{'-':>9}"
                )
            lines.append(f"{target[:35]:36}{cells}")
        lines.append("")

        lines.append(f"FULL MODEL — {group} targets")
        lines.append("-" * 78)
        lines.append(
            "own = your own laps at that corner.  "
            "4A = raw prior blended in.  4B = normalised."
        )
        lines.append("")
        header = f"{'target':36}" + "".join(
            f"{f'own@{n}':>8}{f'4A@{n}':>8}{f'4B@{n}':>8}" for n in practice
        )
        lines.append(header)
        for target in sorted(summary["target"].unique()):
            subset = summary[summary["target"] == target]
            cells = ""
            for shot in practice:
                row = subset[subset["shots"] == shot]
                for column in (
                    "skill_own",
                    "skill_blended",
                    "skill_blended_normalised",
                ):
                    cells += (
                        f"{row[column].mean():>8.2f}" if len(row) else f"{'-':>8}"
                    )
            lines.append(f"{target[:35]:36}{cells}")

        chosen = results.get(f"prior_strength_{group}")
        if chosen is not None and not chosen.empty:
            lines.append("")
            lines.append(
                "Best weight for the geometric prior, swept per target "
                "(0 = the prior earns no weight once you have own laps):"
            )
            lines.append(
                f"  {'target':36}{'best prior':>12}{'4A gain':>10}{'4B gain':>10}"
            )
            for _, row in chosen.iterrows():
                lines.append(
                    f"  {str(row['target'])[:35]:36}{row['prior_strength']:>12.2f}"
                    f"{row['gain_over_own']:>10.1%}"
                    f"{row['normalised_gain_over_own']:>10.1%}"
                )
        lines.append("")

    for group in TARGET_GROUPS:
        coverage = results.get(f"track_coverage_{group}")
        if coverage is None or coverage.empty or "training_tracks" not in coverage:
            continue
        lines.append(f"DOES MORE TRACK VARIETY HELP? — {group} targets")
        lines.append("-" * 78)
        lines.append(
            "Zero-shot skill against the number of tracks the prior was "
            "trained on. Rising means"
        )
        lines.append(
            "collecting more tracks will help; flat means the method has to "
            "change instead."
        )
        lines.append("")
        sizes = sorted(coverage["training_tracks"].unique())
        lines.append(
            f"{'target':36}" + "".join(f"{f'{n} track' + ('s' if n > 1 else ''):>10}" for n in sizes)
        )
        for target in sorted(coverage["target"].unique()):
            subset = coverage[coverage["target"] == target]
            cells = ""
            for size in sizes:
                values = subset.loc[subset["training_tracks"] == size, "skill"]
                cells += f"{values.mean():>10.2f}" if len(values) else f"{'-':>10}"
            lines.append(f"{target[:35]:36}{cells}")
        trend = (
            coverage.groupby("training_tracks")["skill"].mean().sort_index()
        )
        if len(trend) >= 2:
            lines.append("")
            lines.append(
                f"Mean skill: {trend.iloc[0]:.2f} at {trend.index[0]} track "
                f"to {trend.iloc[-1]:.2f} at {trend.index[-1]}."
            )
        lines.append("")

    known = results.get("known_track_primary")
    if known is not None and not known.empty and "model_mae" in known.columns:
        scored = known[known["rows_scored"] > 0]
        if not scored.empty:
            lines.append("KNOWN TRACK (held out by session) — primary targets")
            lines.append("-" * 70)
            grouped = (
                scored.groupby("target")[["model_mae", "pool_mean_mae", "skill_vs_pool_mean"]]
                .mean()
                .reset_index()
            )
            lines.append(f"{'target':38}{'MAE':>10}{'baseline':>11}{'skill':>8}")
            for _, row in grouped.iterrows():
                lines.append(
                    f"{str(row['target'])[:37]:38}{row['model_mae']:>10.3f}"
                    f"{row['pool_mean_mae']:>11.3f}{row['skill_vs_pool_mean']:>8.2f}"
                )
            lines.append("")

    lines.append(f"Outputs: {overview['output_root']}")
    return "\n".join(lines)


def run_from_args(args: argparse.Namespace) -> int:
    frame = _load_segments(args.segments)
    config = EvaluationConfig(
        shots=tuple(int(shot) for shot in args.shots),
        repeats=int(args.repeats),
        knn=KNNConfig(neighbours=int(args.neighbours)),
    )

    results = evaluate_all(frame, config=config)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        table.to_csv(output_root / f"{name}.csv", index=False)

    overview: dict[str, Any] = {
        "rows": int(len(frame)),
        "tracks": int(frame["track"].nunique()),
        "sessions": int(frame["session_key"].nunique())
        if "session_key" in frame.columns
        else 0,
        "neighbours": int(args.neighbours),
        "shots": list(config.shots),
        "repeats": config.repeats,
        "output_root": str(output_root),
    }

    if not args.no_plots:
        for group in TARGET_GROUPS:
            summary = results.get(f"few_shot_summary_{group}")
            if summary is None or summary.empty or "shots" not in summary.columns:
                continue
            path = output_root / f"adaptation_{group}.png"
            plot_adaptation_curves(
                summary, path, title=f"Unseen-track adaptation — {group} targets"
            )
            overview[f"adaptation_plot_{group}"] = str(path)

    report = format_report(results, overview)
    (output_root / "phase4_report.txt").write_text(report, encoding="utf-8")
    (output_root / "phase4_summary.json").write_text(
        json.dumps(overview, indent=2), encoding="utf-8"
    )
    print(report)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    return run_from_args(args)