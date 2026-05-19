from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.rl_traffic.config import RewardConfig


@dataclass(frozen=True)
class RewardInputs:
    total_queue: float
    total_wait: float
    switched: bool
    safety_overridden: bool


class RewardFunction(Protocol):
    def __call__(self, inputs: RewardInputs) -> float:
        ...


class QueueWaitSwitchReward:
    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def __call__(self, inputs: RewardInputs) -> float:
        switch_cost = self.config.switch_weight if inputs.switched else 0.0
        safety_cost = self.config.safety_weight if inputs.safety_overridden else 0.0
        return float(
            -(self.config.queue_weight * inputs.total_queue)
            -(self.config.wait_weight * inputs.total_wait)
            - switch_cost
            - safety_cost
        )


def build_reward(config: RewardConfig) -> RewardFunction:
    if config.name != "queue_wait_switch":
        raise ValueError(f"Unsupported reward function: {config.name}")
    return QueueWaitSwitchReward(config)

