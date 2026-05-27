from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.rl_traffic.config import repo_path


DIRECTIONS = ("N", "E", "S", "W")


@dataclass(frozen=True)
class CapacityRule:
    edge_id: str
    direction: str
    capacity: float = 0.0


@dataclass(frozen=True)
class CapacityRules:
    default_capacity: float = 1.0
    directions: tuple[str, ...] = DIRECTIONS
    edge_rules: tuple[CapacityRule, ...] = ()
    action_served_directions: dict[int, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class DirectionalValues:
    total_queue: float
    total_wait: float
    total_tail_queue: float
    effective_queue: float
    effective_wait: float
    effective_tail_queue: float
    event_direction_queue: float
    event_direction_wait: float
    non_event_direction_queue: float
    non_event_direction_wait: float


def load_capacity_rules(path: str | Path | None) -> CapacityRules:
    if not path:
        return CapacityRules()
    config_path = repo_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Capacity config must be a JSON object.")
    edge_rules = tuple(
        CapacityRule(
            edge_id=str(item["edge_id"]),
            direction=str(item["direction"]),
            capacity=float(item.get("capacity", 0.0)),
        )
        for item in payload.get("edge_rules", [])
    )
    action_map = {
        int(action): tuple(str(direction) for direction in directions)
        for action, directions in payload.get("action_served_directions", {}).items()
    }
    return CapacityRules(
        default_capacity=float(payload.get("default_capacity", 1.0)),
        directions=tuple(str(item) for item in payload.get("directions", DIRECTIONS)),
        edge_rules=edge_rules,
        action_served_directions=action_map,
    )


class CapacityTracker:
    def __init__(self, rules: CapacityRules, accident_config: str | Path | None) -> None:
        self.rules = rules
        self.events = _load_accident_events(accident_config)

    def capacity_by_direction(self, simulation_time: float) -> dict[str, float]:
        capacity = {
            direction: float(self.rules.default_capacity)
            for direction in self.rules.directions
        }
        active_edges = self._active_edges(simulation_time)
        for rule in self.rules.edge_rules:
            if rule.edge_id in active_edges:
                capacity[rule.direction] = min(
                    capacity.get(rule.direction, self.rules.default_capacity),
                    max(0.0, min(1.0, rule.capacity)),
                )
        return capacity

    def served_directions(self, action: int) -> tuple[str, ...]:
        return self.rules.action_served_directions.get(int(action), ())

    def _active_edges(self, simulation_time: float) -> set[str]:
        active_edges: set[str] = set()
        for event in self.events:
            start = float(event.get("time", 0.0))
            end = start + float(event.get("duration", 0.0))
            if start <= simulation_time < end:
                for vehicle in event.get("vehicles", []):
                    edge_id = vehicle.get("edge_id")
                    if edge_id:
                        active_edges.add(str(edge_id))
        return active_edges


def compute_directional_values(
    queue_by_direction: dict[str, float],
    wait_by_direction: dict[str, float],
    tail_queue_by_direction: dict[str, float],
    capacity_by_direction: dict[str, float],
) -> DirectionalValues:
    total_queue = sum(queue_by_direction.values())
    total_wait = sum(wait_by_direction.values())
    total_tail_queue = sum(tail_queue_by_direction.values())
    effective_queue = 0.0
    effective_wait = 0.0
    effective_tail_queue = 0.0
    event_queue = 0.0
    event_wait = 0.0
    non_event_queue = 0.0
    non_event_wait = 0.0
    directions = set(queue_by_direction) | set(wait_by_direction) | set(tail_queue_by_direction)
    for direction in directions:
        capacity = capacity_by_direction.get(direction, 1.0)
        queue = queue_by_direction.get(direction, 0.0)
        wait = wait_by_direction.get(direction, 0.0)
        tail_queue = tail_queue_by_direction.get(direction, 0.0)
        effective_queue += capacity * queue
        effective_wait += capacity * wait
        effective_tail_queue += capacity * tail_queue
        if capacity <= 0.0:
            event_queue += queue
            event_wait += wait
        else:
            non_event_queue += queue
            non_event_wait += wait
    return DirectionalValues(
        total_queue=float(total_queue),
        total_wait=float(total_wait),
        total_tail_queue=float(total_tail_queue),
        effective_queue=float(effective_queue),
        effective_wait=float(effective_wait),
        effective_tail_queue=float(effective_tail_queue),
        event_direction_queue=float(event_queue),
        event_direction_wait=float(event_wait),
        non_event_direction_queue=float(non_event_queue),
        non_event_direction_wait=float(non_event_wait),
    )


def compute_effective_arrived(arrived_delta: int, capacity_by_direction: dict[str, float]) -> float:
    if all(capacity <= 0.0 for capacity in capacity_by_direction.values()):
        return 0.0
    return float(arrived_delta)


def compute_wasted_green(
    served_directions: tuple[str, ...] | list[str],
    capacity_by_direction: dict[str, float],
) -> float:
    if not served_directions:
        return 0.0
    wasted = sum(
        1.0
        for direction in served_directions
        if capacity_by_direction.get(direction, 1.0) <= 0.0
    )
    return float(wasted / len(served_directions))


def _load_accident_events(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    config_path = repo_path(path)
    if not config_path.exists():
        return []
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    events = payload.get("accident_events", [])
    if not isinstance(events, list):
        raise ValueError("accident_events must be a list.")
    return [event for event in events if isinstance(event, dict)]
