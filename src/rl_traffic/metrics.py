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
    effective_controller_name: str = ""
    safety_enabled: bool = False
    total_reward: float = 0.0
    steps: int = 0
    mean_queue: float = 0.0
    max_queue: float = 0.0
    mean_waiting_time: float = 0.0
    total_waiting_time: float = 0.0
    mean_speed: float = 0.0
    throughput: float = 0.0
    interval_arrived_vehicles: float = 0.0
    effective_arrived_delta: float = 0.0
    num_switches: int = 0
    num_safety_overrides: int = 0
    episode_seconds: float = 0.0
    tail_queue: float = 0.0
    active_accidents: int = 0
    active_accident_vehicles: int = 0
    spawned_accident_vehicles: int = 0
    queue_penalty: float = 0.0
    wait_penalty: float = 0.0
    throughput_reward: float = 0.0
    switch_penalty: float = 0.0
    safety_penalty: float = 0.0
    tail_queue_penalty: float = 0.0
    queue_spike_penalty: float = 0.0
    tail_queue_spike_penalty: float = 0.0
    wasted_green_penalty: float = 0.0
    effective_queue: float = 0.0
    effective_wait: float = 0.0
    effective_tail_queue: float = 0.0
    wasted_green_ratio: float = 0.0
    event_direction_green_ratio: float = 0.0
    event_active_ratio: float = 0.0
    wasted_green_when_event_active: float = 0.0
    event_direction_green_when_event_active: float = 0.0
    non_event_queue_when_event_active: float = 0.0
    non_event_wait_when_event_active: float = 0.0
    mean_queue_non_event_directions: float = 0.0
    mean_wait_non_event_directions: float = 0.0
    mean_queue_event_directions: float = 0.0
    mean_wait_event_directions: float = 0.0
    queue_p90: float = 0.0
    queue_p95: float = 0.0
    tail_queue_p90: float = 0.0


class EpisodeMetricAccumulator:
    def __init__(self, controller_name: str, episode: int, seed: int) -> None:
        self.metrics = EpisodeMetrics(controller_name, episode, seed)
        self._queue_sum = 0.0
        self._wait_sum = 0.0
        self._effective_queue_sum = 0.0
        self._effective_wait_sum = 0.0
        self._effective_tail_queue_sum = 0.0
        self._wasted_green_sum = 0.0
        self._event_direction_green_sum = 0.0
        self._event_active_steps = 0
        self._event_active_wasted_green_sum = 0.0
        self._event_active_direction_green_sum = 0.0
        self._event_active_non_event_queue_sum = 0.0
        self._event_active_non_event_wait_sum = 0.0
        self._non_event_queue_sum = 0.0
        self._non_event_wait_sum = 0.0
        self._event_queue_sum = 0.0
        self._event_wait_sum = 0.0
        self._speed_sum = 0.0
        self._speed_count = 0
        self._queue_values: list[float] = []
        self._tail_queue_values: list[float] = []

    def update(self, reward: float, info: dict[str, Any]) -> None:
        queue = float(info.get("total_queue", 0.0))
        wait = float(info.get("total_wait", 0.0))
        speed = float(info.get("mean_speed", 0.0))
        self.metrics.total_reward += float(reward)
        self.metrics.steps += 1
        self.metrics.effective_controller_name = str(
            info.get("effective_controller_name", self.metrics.controller_name)
        )
        self.metrics.safety_enabled = bool(info.get("safety_enabled", False))
        self.metrics.max_queue = max(self.metrics.max_queue, queue)
        self.metrics.total_waiting_time += wait
        self.metrics.num_switches += int(bool(info.get("switched", False)))
        self.metrics.num_safety_overrides += int(bool(info.get("safety_overridden", False)))
        self.metrics.throughput = float(info.get("arrived_vehicles", self.metrics.throughput))
        self.metrics.interval_arrived_vehicles += float(
            info.get("interval_arrived_vehicles", 0.0)
        )
        self.metrics.effective_arrived_delta += float(
            info.get("effective_arrived_delta", 0.0)
        )
        self.metrics.episode_seconds = float(info.get("simulation_time", self.metrics.episode_seconds))
        self.metrics.tail_queue = max(self.metrics.tail_queue, float(info.get("tail_queue", 0.0)))
        self.metrics.active_accidents = max(
            self.metrics.active_accidents,
            int(info.get("active_accidents", 0)),
        )
        self.metrics.active_accident_vehicles = max(
            self.metrics.active_accident_vehicles,
            int(info.get("active_accident_vehicles", 0)),
        )
        self.metrics.spawned_accident_vehicles += int(
            info.get("spawned_accident_vehicles", 0)
        )
        self.metrics.queue_penalty += float(info.get("queue_penalty", 0.0))
        self.metrics.wait_penalty += float(info.get("wait_penalty", 0.0))
        self.metrics.throughput_reward += float(info.get("throughput_reward", 0.0))
        self.metrics.switch_penalty += float(info.get("switch_penalty", 0.0))
        self.metrics.safety_penalty += float(info.get("safety_penalty", 0.0))
        self.metrics.tail_queue_penalty += float(info.get("tail_queue_penalty", 0.0))
        self.metrics.queue_spike_penalty += float(info.get("queue_spike_penalty", 0.0))
        self.metrics.tail_queue_spike_penalty += float(info.get("tail_queue_spike_penalty", 0.0))
        self.metrics.wasted_green_penalty += float(info.get("wasted_green_penalty", 0.0))
        self._queue_sum += queue
        self._wait_sum += wait
        self._effective_queue_sum += float(info.get("effective_queue", queue))
        self._effective_wait_sum += float(info.get("effective_wait", wait))
        self._effective_tail_queue_sum += float(info.get("effective_tail_queue", 0.0))
        self._wasted_green_sum += float(info.get("wasted_green", 0.0))
        self._event_direction_green_sum += float(info.get("event_direction_green", 0.0))
        non_event_queue = float(info.get("non_event_direction_queue", 0.0))
        non_event_wait = float(info.get("non_event_direction_wait", 0.0))
        self._non_event_queue_sum += non_event_queue
        self._non_event_wait_sum += non_event_wait
        self._event_queue_sum += float(info.get("event_direction_queue", 0.0))
        self._event_wait_sum += float(info.get("event_direction_wait", 0.0))
        if bool(info.get("event_active", False)):
            self._event_active_steps += 1
            self._event_active_wasted_green_sum += float(info.get("wasted_green", 0.0))
            self._event_active_direction_green_sum += float(info.get("event_direction_green", 0.0))
            self._event_active_non_event_queue_sum += non_event_queue
            self._event_active_non_event_wait_sum += non_event_wait
        self._speed_sum += speed
        self._speed_count += 1
        self._queue_values.append(queue)
        self._tail_queue_values.append(float(info.get("tail_queue", 0.0)))

    def finish(self) -> EpisodeMetrics:
        steps = max(self.metrics.steps, 1)
        self.metrics.mean_queue = self._queue_sum / steps
        self.metrics.mean_waiting_time = self._wait_sum / steps
        self.metrics.effective_queue = self._effective_queue_sum / steps
        self.metrics.effective_wait = self._effective_wait_sum / steps
        self.metrics.effective_tail_queue = self._effective_tail_queue_sum / steps
        self.metrics.wasted_green_ratio = self._wasted_green_sum / steps
        self.metrics.event_direction_green_ratio = self._event_direction_green_sum / steps
        self.metrics.event_active_ratio = self._event_active_steps / steps
        event_steps = max(self._event_active_steps, 1)
        self.metrics.wasted_green_when_event_active = (
            self._event_active_wasted_green_sum / event_steps
        )
        self.metrics.event_direction_green_when_event_active = (
            self._event_active_direction_green_sum / event_steps
        )
        self.metrics.non_event_queue_when_event_active = (
            self._event_active_non_event_queue_sum / event_steps
        )
        self.metrics.non_event_wait_when_event_active = (
            self._event_active_non_event_wait_sum / event_steps
        )
        self.metrics.mean_queue_non_event_directions = self._non_event_queue_sum / steps
        self.metrics.mean_wait_non_event_directions = self._non_event_wait_sum / steps
        self.metrics.mean_queue_event_directions = self._event_queue_sum / steps
        self.metrics.mean_wait_event_directions = self._event_wait_sum / steps
        self.metrics.mean_speed = self._speed_sum / max(self._speed_count, 1)
        self.metrics.queue_p90 = _percentile(self._queue_values, 0.90)
        self.metrics.queue_p95 = _percentile(self._queue_values, 0.95)
        self.metrics.tail_queue_p90 = _percentile(self._tail_queue_values, 0.90)
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


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * quantile))
    return float(ordered[max(0, min(len(ordered) - 1, index))])
