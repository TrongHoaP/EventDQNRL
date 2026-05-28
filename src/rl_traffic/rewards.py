from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.rl_traffic.config import RewardConfig


@dataclass(frozen=True)
class RewardInputs:
    total_queue: float
    total_wait: float
    arrived_delta: float
    tail_queue: float
    switched: bool
    safety_overridden: bool
    effective_queue: float
    effective_wait: float
    effective_arrived_delta: float
    effective_tail_queue: float
    wasted_green: float


@dataclass(frozen=True)
class RewardResult:
    total: float
    queue_penalty: float
    wait_penalty: float
    throughput_reward: float
    switch_penalty: float
    safety_penalty: float
    tail_queue_penalty: float
    queue_spike_penalty: float
    tail_queue_spike_penalty: float
    wasted_green_penalty: float


class RewardFunction(Protocol):
    def __call__(self, inputs: RewardInputs) -> RewardResult:
        ...


class QueueWaitSwitchReward:
    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def __call__(self, inputs: RewardInputs) -> RewardResult:
        if self.config.use_effective_metrics:
            queue_value = inputs.effective_queue
            wait_value = inputs.effective_wait
            arrived_value = inputs.effective_arrived_delta
            tail_queue_value = inputs.effective_tail_queue
        else:
            queue_value = inputs.total_queue
            wait_value = inputs.total_wait
            arrived_value = inputs.arrived_delta
            tail_queue_value = inputs.tail_queue

        queue_penalty = -(self.config.queue_weight * queue_value)
        wait_penalty = -(self.config.wait_weight * wait_value)
        throughput_reward = self.config.throughput_weight * arrived_value
        switch_penalty = -self.config.switch_weight if inputs.switched else 0.0
        safety_penalty = -self.config.safety_weight if inputs.safety_overridden else 0.0
        tail_queue_penalty = -(self.config.tail_queue_weight * tail_queue_value)
        queue_spike = max(0.0, queue_value - self.config.queue_spike_threshold)
        tail_queue_spike = max(0.0, tail_queue_value - self.config.tail_queue_spike_threshold)
        queue_spike_penalty = -(self.config.queue_spike_weight * queue_spike)
        tail_queue_spike_penalty = -(self.config.tail_queue_spike_weight * tail_queue_spike)
        wasted_green_penalty = -(self.config.wasted_green_weight * inputs.wasted_green)
        total = (
            queue_penalty
            + wait_penalty
            + throughput_reward
            + switch_penalty
            + safety_penalty
            + tail_queue_penalty
            + queue_spike_penalty
            + tail_queue_spike_penalty
            + wasted_green_penalty
        )
        return RewardResult(
            total=float(total),
            queue_penalty=float(queue_penalty),
            wait_penalty=float(wait_penalty),
            throughput_reward=float(throughput_reward),
            switch_penalty=float(switch_penalty),
            safety_penalty=float(safety_penalty),
            tail_queue_penalty=float(tail_queue_penalty),
            queue_spike_penalty=float(queue_spike_penalty),
            tail_queue_spike_penalty=float(tail_queue_spike_penalty),
            wasted_green_penalty=float(wasted_green_penalty),
        )


def build_reward(config: RewardConfig) -> RewardFunction:
    if config.name not in {"queue_wait_switch", "queue_weighted_wait_time"}:
        raise ValueError(f"Unsupported reward function: {config.name}")
    return QueueWaitSwitchReward(config)
