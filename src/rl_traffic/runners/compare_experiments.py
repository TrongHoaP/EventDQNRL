from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


RAW_FIELDS = (
    "variant",
    "seed",
    "run_name",
    "episode",
    "controller_name",
    "experiment_variant",
    "include_capacity",
    "include_event_features",
    "use_effective_metrics",
    "mean_queue",
    "max_queue",
    "mean_waiting_time",
    "throughput",
    "num_switches",
    "num_safety_overrides",
    "event_active_ratio",
    "mean_queue_event_directions",
    "mean_wait_event_directions",
    "event_direction_green_ratio",
    "wasted_green_ratio",
    "mean_queue_non_event_directions",
    "mean_wait_non_event_directions",
    "total_reward",
)

SUMMARY_METRICS = (
    "mean_queue",
    "max_queue",
    "mean_waiting_time",
    "throughput",
    "num_switches",
    "num_safety_overrides",
    "event_active_ratio",
    "mean_queue_event_directions",
    "mean_wait_event_directions",
    "event_direction_green_ratio",
    "wasted_green_ratio",
    "mean_queue_non_event_directions",
    "mean_wait_non_event_directions",
)

LOWER_BETTER = (
    "mean_queue",
    "max_queue",
    "mean_waiting_time",
    "mean_queue_event_directions",
    "mean_wait_event_directions",
    "wasted_green_ratio",
    "mean_queue_non_event_directions",
    "mean_wait_non_event_directions",
    "num_safety_overrides",
)

HIGHER_BETTER = (
    "throughput",
    "event_direction_green_ratio",
)

OPPONENTS = ("dqn_no_event", "fixed_time", "actuated", "max_pressure", "random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate EventDQN comparison metrics.")
    parser.add_argument("--runs-dir", default="runs/compare_event_dqn")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else runs_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _load_eval_rows(runs_dir)
    _write_raw(out_dir / "comparison_raw.csv", raw_rows)
    summary_rows = _build_summary(raw_rows)
    _write_summary(out_dir / "comparison_summary.csv", summary_rows)
    decision = _build_decision(summary_rows)
    _write_decision(out_dir / "comparison_decision.json", decision)

    print(f"Wrote {out_dir / 'comparison_raw.csv'}")
    print(f"Wrote {out_dir / 'comparison_summary.csv'}")
    print(f"Wrote {out_dir / 'comparison_decision.json'}")
    return 0


def _load_eval_rows(runs_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metrics_path in sorted(runs_dir.glob("**/metrics/episode_metrics.csv")):
        relative_parts = metrics_path.relative_to(runs_dir).parts
        if not relative_parts:
            continue
        if relative_parts[0] == "train":
            continue
        run_dir = metrics_path.parent.parent
        run_name = run_dir.name
        if "_incomplete_" in run_name:
            continue
        with metrics_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                normalized = _normalize_row(row, run_name)
                if normalized["variant"]:
                    rows.append(normalized)
    return rows


def _normalize_row(row: dict[str, str], run_name: str) -> dict[str, str]:
    variant = _variant_from_row(row, run_name)
    seed = row.get("seed") or _seed_from_run_name(run_name)
    normalized = {field: "" for field in RAW_FIELDS}
    normalized.update(
        {
            "variant": variant,
            "seed": seed,
            "run_name": run_name,
            "episode": row.get("episode", ""),
            "controller_name": row.get("controller_name", ""),
            "experiment_variant": row.get("experiment_variant", ""),
            "include_capacity": row.get("include_capacity", ""),
            "include_event_features": row.get("include_event_features", ""),
            "use_effective_metrics": row.get("use_effective_metrics", ""),
            "mean_queue": row.get("mean_queue", ""),
            "max_queue": row.get("max_queue", ""),
            "mean_waiting_time": row.get("mean_waiting_time", ""),
            "throughput": row.get("throughput", ""),
            "num_switches": row.get("num_switches", ""),
            "num_safety_overrides": row.get("num_safety_overrides", ""),
            "event_active_ratio": row.get("event_active_ratio", ""),
            "mean_queue_event_directions": row.get("mean_queue_event_directions", ""),
            "mean_wait_event_directions": row.get("mean_wait_event_directions", ""),
            "event_direction_green_ratio": row.get("event_direction_green_ratio", ""),
            "wasted_green_ratio": row.get("wasted_green_ratio", ""),
            "mean_queue_non_event_directions": row.get("mean_queue_non_event_directions", ""),
            "mean_wait_non_event_directions": row.get("mean_wait_non_event_directions", ""),
            "total_reward": row.get("total_reward", ""),
        }
    )
    return normalized


def _write_raw(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(RAW_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
    for row in raw_rows:
        variant = row["variant"]
        seed = row["seed"]
        for metric in SUMMARY_METRICS:
            value = _float_or_none(row.get(metric))
            if value is not None:
                grouped[(variant, metric)].append((value, seed))

    summary_rows: list[dict[str, Any]] = []
    for (variant, metric), values_and_seeds in sorted(grouped.items()):
        values = [value for value, _ in values_and_seeds]
        seeds = {seed for _, seed in values_and_seeds if seed != ""}
        summary_rows.append(
            {
                "variant": variant,
                "metric": metric,
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "num_episodes": len(values),
                "num_seeds": len(seeds),
            }
        )
    return summary_rows


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ("variant", "metric", "mean", "std", "min", "max", "num_episodes", "num_seeds")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_decision(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        (str(row["variant"]), str(row["metric"])): float(row["mean"])
        for row in summary_rows
    }
    decision: dict[str, Any] = {
        "rule": {
            "description": (
                "EventDQN better if it wins >= 60% primary metrics, has >= 2 metrics "
                "improved by >= 5%, throughput drops <= 5%, safety overrides rise <= 10%, "
                "and mean_queue worsens <= 5%. total_reward is debug-only."
            ),
            "lower_better": list(LOWER_BETTER),
            "higher_better": list(HIGHER_BETTER),
            "strong_win_threshold_percent": 5.0,
            "win_rate_threshold": 0.60,
            "max_throughput_drop_percent": 5.0,
            "max_safety_override_increase_percent": 10.0,
            "max_mean_queue_worse_percent": 5.0,
        }
    }
    if not any(variant == "event_dqn" for variant, _ in summary):
        decision["status"] = "missing_event_dqn"
        return decision

    for opponent in OPPONENTS:
        key = f"event_dqn_vs_{opponent}"
        if not any(variant == opponent for variant, _ in summary):
            decision[key] = {"verdict": "missing_opponent"}
            continue
        decision[key] = _compare_variants(summary, "event_dqn", opponent)
    return decision


def _compare_variants(
    summary: dict[tuple[str, str], float],
    candidate: str,
    opponent: str,
) -> dict[str, Any]:
    metric_results = []
    wins = 0
    strong_wins = 0
    primary_metrics = tuple(LOWER_BETTER) + tuple(HIGHER_BETTER)
    for metric in primary_metrics:
        candidate_value = summary.get((candidate, metric))
        opponent_value = summary.get((opponent, metric))
        if candidate_value is None or opponent_value is None:
            continue
        change_percent = _change_percent(candidate_value, opponent_value)
        lower_is_better = metric in LOWER_BETTER
        won = candidate_value < opponent_value if lower_is_better else candidate_value > opponent_value
        strong_win = (
            change_percent <= -5.0 if lower_is_better else change_percent >= 5.0
        )
        wins += int(won)
        strong_wins += int(strong_win)
        metric_results.append(
            {
                "metric": metric,
                "event_dqn": candidate_value,
                "opponent": opponent_value,
                "change_percent": change_percent,
                "direction": "lower_better" if lower_is_better else "higher_better",
                "won": won,
                "strong_win": strong_win,
            }
        )

    compared = len(metric_results)
    win_rate = wins / compared if compared else 0.0
    throughput_change = _metric_change(summary, candidate, opponent, "throughput")
    safety_change = _metric_change(summary, candidate, opponent, "num_safety_overrides")
    mean_queue_change = _metric_change(summary, candidate, opponent, "mean_queue")
    better = (
        compared > 0
        and win_rate >= 0.60
        and strong_wins >= 2
        and throughput_change >= -5.0
        and safety_change <= 10.0
        and mean_queue_change <= 5.0
    )
    return {
        "verdict": "event_dqn_better" if better else "not_enough_evidence",
        "win_rate": win_rate,
        "wins": wins,
        "metrics_compared": compared,
        "strong_wins": strong_wins,
        "throughput_change_percent": throughput_change,
        "safety_override_change_percent": safety_change,
        "mean_queue_change_percent": mean_queue_change,
        "metrics": metric_results,
    }


def _write_decision(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _change_percent(candidate_value: float, opponent_value: float) -> float:
    denominator = abs(opponent_value)
    if denominator <= 1e-9:
        if abs(candidate_value) <= 1e-9:
            return 0.0
        return 100.0 if candidate_value > opponent_value else -100.0
    return ((candidate_value - opponent_value) / denominator) * 100.0


def _metric_change(
    summary: dict[tuple[str, str], float],
    candidate: str,
    opponent: str,
    metric: str,
) -> float:
    candidate_value = summary.get((candidate, metric))
    opponent_value = summary.get((opponent, metric))
    if candidate_value is None or opponent_value is None:
        return 0.0
    return _change_percent(candidate_value, opponent_value)


def _variant_from_row(row: dict[str, str], run_name: str) -> str:
    controller_name = (row.get("controller_name") or "").lower()
    if controller_name and controller_name != "dqn":
        return _variant_from_run_name(controller_name)
    return row.get("experiment_variant") or _variant_from_run_name(run_name)


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


def _seed_from_run_name(run_name: str) -> str:
    match = re.search(r"seed(\d+)", run_name.lower())
    return match.group(1) if match else ""


if __name__ == "__main__":
    raise SystemExit(main())
