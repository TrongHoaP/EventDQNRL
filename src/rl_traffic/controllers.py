from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class Controller(Protocol):
    name: str

    def act(self, observation: np.ndarray, info: dict[str, Any]) -> int:
        ...

    def observe(self, transition: dict[str, Any]) -> None:
        ...


class RandomAgent:
    name = "random"

    def __init__(self, action_size: int, seed: int = 7) -> None:
        self.action_size = action_size
        self.rng = random.Random(seed)

    def act(self, observation: np.ndarray, info: dict[str, Any]) -> int:
        return self.rng.randrange(self.action_size)

    def observe(self, transition: dict[str, Any]) -> None:
        return None


class FixedTimeController:
    name = "fixed_time"

    def __init__(self, action_size: int, cycle_seconds: int = 42) -> None:
        self.action_size = action_size
        self.cycle_seconds = max(int(cycle_seconds), 1)

    def act(self, observation: np.ndarray, info: dict[str, Any]) -> int:
        slot = int(float(info.get("simulation_time", 0.0)) // self.cycle_seconds)
        return slot % self.action_size

    def observe(self, transition: dict[str, Any]) -> None:
        return None


class ActuatedController:
    name = "actuated"

    def __init__(self, action_size: int, queue_threshold: float = 4.0) -> None:
        self.action_size = action_size
        self.queue_threshold = float(queue_threshold)
        self.last_action = 0

    def act(self, observation: np.ndarray, info: dict[str, Any]) -> int:
        total_queue = float(info.get("total_queue", 0.0))
        if total_queue >= self.queue_threshold:
            self.last_action = (self.last_action + 1) % self.action_size
        return self.last_action

    def observe(self, transition: dict[str, Any]) -> None:
        return None


class MaxPressureController:
    name = "max_pressure"

    def __init__(self, action_size: int) -> None:
        self.action_size = action_size

    def act(self, observation: np.ndarray, info: dict[str, Any]) -> int:
        lane_like_values = observation[2::3] if observation.shape[0] >= 3 else observation
        if lane_like_values.size == 0:
            return 0
        split_scores = np.array_split(lane_like_values, self.action_size)
        scores = [float(np.sum(score)) for score in split_scores]
        return int(np.argmax(scores))

    def observe(self, transition: dict[str, Any]) -> None:
        return None


@dataclass(frozen=True)
class ControllerFactoryConfig:
    action_size: int
    seed: int
    fixed_cycle_seconds: int
    actuated_queue_threshold: float


def build_controller(name: str, config: ControllerFactoryConfig) -> Controller:
    if name == "random":
        return RandomAgent(config.action_size, seed=config.seed)
    if name == "fixed_time":
        return FixedTimeController(config.action_size, config.fixed_cycle_seconds)
    if name == "actuated":
        return ActuatedController(config.action_size, config.actuated_queue_threshold)
    if name == "max_pressure":
        return MaxPressureController(config.action_size)
    raise ValueError(f"Unsupported controller: {name}")

