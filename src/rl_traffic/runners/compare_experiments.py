from __future__ import annotations

import argparse
import csv
import fnmatch
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_METRICS = (
    "total_reward",
    "mean_queue",
    "max_queue",
    "mean_waiting_time",
    "throughput",
    "num_safety_overrides",
    "num_switches",
    "mean_queue_event_directions",
    "mean_wait_event_directions",
    "mean_queue_non_event_directions",
    "mean_wait_non_event_directions",
    "event_direction_green_ratio",
    "wasted_green_ratio",
    "tail_queue",
    "active_accidents",
    "active_accident_vehicles",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate DQN experiment metrics.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--pattern", default="*seed*")
    parser.add_argument("--out", default="reports/comparison_dqn_vs_event_dqn.csv")
    parser.add_argument("--metrics", nargs="*", default=list(DEFAULT_METRICS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)

    matched_runs = sorted(
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and fnmatch.fnmatch(path.name, args.pattern)
    )
    eval_runs = [path for path in matched_runs if path.name.startswith("eval_")]
    selected_runs = eval_runs or matched_runs

    for run_dir in selected_runs:
        metrics_path = run_dir / "metrics" / "episode_metrics.csv"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            continue
        row = rows[-1]
        variant = _variant_from_row(row, run_dir.name)
        for metric in args.metrics:
            value = _float_or_none(row.get(metric))
            if value is not None:
                grouped[(variant, metric)].append(value)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("variant", "metric", "mean", "std", "min", "max", "num_runs"),
        )
        writer.writeheader()
        for (variant, metric), values in sorted(grouped.items()):
            writer.writerow(
                {
                    "variant": variant,
                    "metric": metric,
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "num_runs": len(values),
                }
            )
    print(f"Wrote {output}")
    return 0


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _variant_from_run_name(run_name: str) -> str:
    name = run_name.lower()
    if "dqn_no_event" in name:
        return "dqn_no_event"
    if "event_dqn" in name:
        return "event_dqn"
    if "fixed_time" in name:
        return "fixed_time"
    if "actuated" in name:
        return "actuated"
    if "max_pressure" in name:
        return "max_pressure"
    if "random" in name:
        return "random"
    return run_name


def _variant_from_row(row: dict[str, str], run_name: str) -> str:
    controller_name = (row.get("controller_name") or "").lower()
    if controller_name and controller_name != "dqn":
        return _variant_from_run_name(controller_name)
    return row.get("experiment_variant") or _variant_from_run_name(run_name)


if __name__ == "__main__":
    raise SystemExit(main())
