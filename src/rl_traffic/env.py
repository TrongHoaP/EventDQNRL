from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.rl_traffic.config import TrafficRLConfig
from src.rl_traffic.rewards import RewardInputs, RewardResult, build_reward
from src.rl_traffic.safety import SafetyLayer
from src.rl_traffic.sumo_adapter import DemandSpawner, SumoSession
from src.sumo.traffic_light_control import TrafficLightController


class SumoTrafficSignalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: TrafficRLConfig, controller_name: str = "env") -> None:
        super().__init__()
        self.config = config
        self.controller_name = controller_name
        self.session = SumoSession(config.sumo)
        self.spawner = DemandSpawner.from_config(config.sumo)
        self.reward_fn = build_reward(config.reward)
        self.safety = SafetyLayer(config.safety, config.control)
        self.traci: Any | None = None
        self.controller: TrafficLightController | None = None
        self.observation_space: spaces.Box | None = None
        self.action_space: spaces.Discrete | None = None
        self._arrived_total = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.close()
        self.spawner = DemandSpawner.from_config(self.config.sumo, seed=seed)
        self.traci = self.session.start(seed=seed)
        self.controller = TrafficLightController(
            traci_module=self.traci,
            min_green=self.config.control.min_green_seconds,
            yellow_time=self.config.control.yellow_time_seconds,
            decision_interval=self.config.control.decision_interval_seconds,
        )
        observation = self._observation()
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=observation.shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.controller.action_size)
        self._arrived_total = 0
        return observation, self._info(switched=False, safety_decision=None, spawned=0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.traci is None or self.controller is None:
            raise RuntimeError("Environment must be reset before step().")

        safety_decision = self.safety.resolve(int(action), self.controller)
        switched = self.controller.apply_action(safety_decision.action)
        spawned = self._advance_control_interval()
        total_queue, total_wait = self.controller._network_totals()
        tail_queue = self._tail_queue()
        reward_result = self.reward_fn(
            RewardInputs(
                total_queue=total_queue,
                total_wait=total_wait,
                tail_queue=tail_queue,
                switched=switched,
                safety_overridden=safety_decision.overridden,
            )
        )
        terminated = self.traci.simulation.getMinExpectedNumber() == 0
        truncated = self.traci.simulation.getTime() >= self.config.sumo.end_time
        info = self._info(
            switched=switched,
            safety_decision=safety_decision,
            spawned=spawned,
            reward_result=reward_result,
            tail_queue=tail_queue,
        )
        return self._observation(), reward_result.total, bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.session.close()
        self.traci = None
        self.controller = None

    @property
    def action_count(self) -> int:
        if self.controller is None:
            raise RuntimeError("Environment must be reset before action_count is known.")
        return self.controller.action_size

    def _advance_control_interval(self) -> int:
        assert self.traci is not None
        spawned = 0
        step_length = max(float(self.config.sumo.step_length), 1e-6)
        step_count = max(1, int(math.ceil(self.config.control.decision_interval_seconds / step_length)))
        for _ in range(step_count):
            current_time = int(round(self.traci.simulation.getTime()))
            spawned += self.spawner.spawn_due_until(self.traci, current_time)
            self.traci.simulationStep()
        return spawned

    def _observation(self) -> np.ndarray:
        assert self.controller is not None
        if self.config.state.mode == "approach_level":
            return self.controller.get_approach_state_vector()
        if self.config.state.mode != "lane_level":
            raise ValueError(f"Unsupported state mode: {self.config.state.mode}")
        return self.controller.get_state()

    def get_valid_action_mask(self) -> np.ndarray:
        if self.controller is None:
            raise RuntimeError("Environment must be reset before valid action mask is known.")
        return self.safety.valid_action_mask(self.controller)

    def _tail_queue(self) -> float:
        assert self.controller is not None
        approach_metrics = self.controller.get_approach_metrics()
        return max((metrics.queue_max for metrics in approach_metrics.values()), default=0.0)

    def _info(
        self,
        switched: bool,
        safety_decision: Any | None,
        spawned: int,
        reward_result: RewardResult | None = None,
        tail_queue: float | None = None,
    ) -> dict[str, Any]:
        assert self.traci is not None
        assert self.controller is not None
        total_queue, total_wait = self.controller._network_totals()
        approach_metrics = self.controller.get_approach_metrics()
        vehicle_counts = [metric.vehicle_count_total for metric in approach_metrics.values()]
        speed_values = [metric.speed for metric in approach_metrics.values()]
        mean_speed = float(sum(speed_values) / len(speed_values)) if speed_values else 0.0
        arrived = int(self.traci.simulation.getArrivedNumber())
        self._arrived_total += arrived
        if tail_queue is None:
            tail_queue = max((metric.queue_max for metric in approach_metrics.values()), default=0.0)
        if reward_result is None:
            reward_components = {
                "queue_penalty": 0.0,
                "wait_penalty": 0.0,
                "switch_penalty": 0.0,
                "safety_penalty": 0.0,
                "tail_queue_penalty": 0.0,
                "queue_spike_penalty": 0.0,
                "tail_queue_spike_penalty": 0.0,
            }
        else:
            reward_components = {
                "queue_penalty": reward_result.queue_penalty,
                "wait_penalty": reward_result.wait_penalty,
                "switch_penalty": reward_result.switch_penalty,
                "safety_penalty": reward_result.safety_penalty,
                "tail_queue_penalty": reward_result.tail_queue_penalty,
                "queue_spike_penalty": reward_result.queue_spike_penalty,
                "tail_queue_spike_penalty": reward_result.tail_queue_spike_penalty,
            }
        return {
            "controller_name": self.controller_name,
            "simulation_time": float(self.traci.simulation.getTime()),
            "total_queue": float(total_queue),
            "total_wait": float(total_wait),
            "tail_queue": float(tail_queue),
            "mean_speed": mean_speed,
            "vehicle_count": float(sum(vehicle_counts)),
            "arrived_vehicles": self._arrived_total,
            "spawned_vehicles": spawned,
            "switched": switched,
            "raw_action": None if safety_decision is None else safety_decision.raw_action,
            "safe_action": None if safety_decision is None else safety_decision.action,
            "safety_overridden": False if safety_decision is None else safety_decision.overridden,
            "safety_reason": "reset" if safety_decision is None else safety_decision.reason,
            "valid_action_mask": self.get_valid_action_mask(),
            **reward_components,
        }
