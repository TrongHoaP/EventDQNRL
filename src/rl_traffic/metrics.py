from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class EpisodeMetrics:
    controller_name: str
    episode: int
    seed: int
    total_reward: float = 0.0
    steps: int = 0
    mean_queue: float = 0.0
    max_queue: float = 0.0
    mean_waiting_time: float = 0.0
    total_waiting_time: float = 0.0
    mean_speed: float = 0.0
    throughput: float = 0.0
    num_switches: int = 0
    num_safety_overrides: int = 0
    episode_seconds: float = 0.0


class EpisodeMetricAccumulator:
    def __init__(self, controller_name: str, episode: int, seed: int) -> None:
        self.metrics = EpisodeMetrics(controller_name, episode, seed)
        self._queue_sum = 0.0
        self._wait_sum = 0.0
        self._speed_sum = 0.0
        self._speed_count = 0

    def update(self, reward: float, info: dict[str, Any]) -> None:
        queue = float(info.get("total_queue", 0.0))
        wait = float(info.get("total_wait", 0.0))
        speed = float(info.get("mean_speed", 0.0))
        self.metrics.total_reward += float(reward)
        self.metrics.steps += 1
        self.metrics.max_queue = max(self.metrics.max_queue, queue)
        self.metrics.total_waiting_time += wait
        self.metrics.num_switches += int(bool(info.get("switched", False)))
        self.metrics.num_safety_overrides += int(bool(info.get("safety_overridden", False)))
        self.metrics.throughput = float(info.get("arrived_vehicles", self.metrics.throughput))
        self.metrics.episode_seconds = float(info.get("simulation_time", self.metrics.episode_seconds))
        self._queue_sum += queue
        self._wait_sum += wait
        self._speed_sum += speed
        self._speed_count += 1

    def finish(self) -> EpisodeMetrics:
        steps = max(self.metrics.steps, 1)
        self.metrics.mean_queue = self._queue_sum / steps
        self.metrics.mean_waiting_time = self._wait_sum / steps
        self.metrics.mean_speed = self._speed_sum / max(self._speed_count, 1)
        return self.metrics


def append_episode_metrics(path: str | Path, metrics: EpisodeMetrics) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(metrics)
    write_header = not output.exists()
    with output.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_summary(path: str | Path, metrics: list[EpisodeMetrics]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(metric) for metric in metrics]
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")

