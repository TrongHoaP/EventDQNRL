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
        return int(self.rng.choice(valid_actions_from_info(info, self.action_size)))

    def observe(self, transition: dict[str, Any]) -> None:
        return None


class FixedTimeController:
    name = "fixed_time"

    def __init__(self, action_size: int, cycle_seconds: int = 42) -> None:
        self.action_size = action_size
        self.cycle_seconds = max(int(cycle_seconds), 1)

    def act(self, observation: np.ndarray, info: dict[str, Any]) -> int:
        slot = int(float(info.get("simulation_time", 0.0)) // self.cycle_seconds)
        candidate = slot % self.action_size
        valid_actions = valid_actions_from_info(info, self.action_size)
        if candidate in valid_actions:
            return candidate
        return valid_actions[0]

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
        candidate = self.last_action
        if total_queue >= self.queue_threshold:
            candidate = (self.last_action + 1) % self.action_size
        valid_actions = valid_actions_from_info(info, self.action_size)
        if candidate not in valid_actions:
            candidate = valid_actions[0]
        self.last_action = candidate
        return candidate

    def observe(self, transition: dict[str, Any]) -> None:
        return None


class MaxPressureController:
    name = "max_pressure"

    def __init__(self, action_size: int) -> None:
        self.action_size = action_size

    def act(self, observation: np.ndarray, info: dict[str, Any]) -> int:
        valid_actions = valid_actions_from_info(info, self.action_size)
        pressures = info.get("phase_pressure")
        if pressures is None:
            return valid_actions[0]

        pressure_array = np.asarray(pressures, dtype=np.float32)
        if pressure_array.size < self.action_size:
            return valid_actions[0]
        pressure_array = pressure_array[: self.action_size].copy()
        mask = np.zeros(self.action_size, dtype=bool)
        mask[np.asarray(valid_actions, dtype=int)] = True
        pressure_array[~mask] = -1e9
        return int(np.argmax(pressure_array))

    def observe(self, transition: dict[str, Any]) -> None:
        return None


@dataclass(frozen=True)
class ControllerFactoryConfig:
    action_size: int
    seed: int
    fixed_cycle_seconds: int
    actuated_queue_threshold: float


def valid_actions_from_info(info: dict[str, Any], action_size: int) -> list[int]:
    mask = info.get("valid_action_mask")
    if mask is None:
        return list(range(action_size))
    mask_array = np.asarray(mask, dtype=bool)
    valid = np.flatnonzero(mask_array).tolist()
    if not valid:
        return list(range(action_size))
    return [int(action) for action in valid]


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
