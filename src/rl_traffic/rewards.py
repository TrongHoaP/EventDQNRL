from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.rl_traffic.config import RewardConfig


@dataclass(frozen=True)
class RewardInputs:
    total_queue: float
    total_wait: float
    tail_queue: float
    switched: bool
    safety_overridden: bool


@dataclass(frozen=True)
class RewardResult:
    total: float
    queue_penalty: float
    wait_penalty: float
    switch_penalty: float
    safety_penalty: float
    tail_queue_penalty: float
    queue_spike_penalty: float
    tail_queue_spike_penalty: float


class RewardFunction(Protocol):
    def __call__(self, inputs: RewardInputs) -> RewardResult:
        ...


class QueueWaitSwitchReward:
    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def __call__(self, inputs: RewardInputs) -> RewardResult:
        queue_penalty = -(self.config.queue_weight * inputs.total_queue)
        wait_penalty = -(self.config.wait_weight * inputs.total_wait)
        switch_penalty = -self.config.switch_weight if inputs.switched else 0.0
        safety_penalty = -self.config.safety_weight if inputs.safety_overridden else 0.0
        tail_queue_penalty = -(self.config.tail_queue_weight * inputs.tail_queue)
        queue_spike = max(0.0, inputs.total_queue - self.config.queue_spike_threshold)
        tail_queue_spike = max(0.0, inputs.tail_queue - self.config.tail_queue_spike_threshold)
        queue_spike_penalty = -(self.config.queue_spike_weight * queue_spike)
        tail_queue_spike_penalty = -(self.config.tail_queue_spike_weight * tail_queue_spike)
        total = (
            queue_penalty
            + wait_penalty
            + switch_penalty
            + safety_penalty
            + tail_queue_penalty
            + queue_spike_penalty
            + tail_queue_spike_penalty
        )
        return RewardResult(
            total=float(total),
            queue_penalty=float(queue_penalty),
            wait_penalty=float(wait_penalty),
            switch_penalty=float(switch_penalty),
            safety_penalty=float(safety_penalty),
            tail_queue_penalty=float(tail_queue_penalty),
            queue_spike_penalty=float(queue_spike_penalty),
            tail_queue_spike_penalty=float(tail_queue_spike_penalty),
        )


def build_reward(config: RewardConfig) -> RewardFunction:
    if config.name != "queue_wait_switch":
        raise ValueError(f"Unsupported reward function: {config.name}")
    return QueueWaitSwitchReward(config)
