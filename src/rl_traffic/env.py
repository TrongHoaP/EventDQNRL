from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.rl_traffic.capacity import (
    CapacityTracker,
    compute_directional_values,
    compute_effective_arrived,
    compute_wasted_green,
    load_capacity_rules,
)
from src.rl_traffic.config import TrafficRLConfig
from src.rl_traffic.rewards import RewardInputs, RewardResult, build_reward
from src.rl_traffic.safety import SafetyLayer
from src.rl_traffic.sumo_adapter import AccidentManager, DemandSpawner, SumoSession
from src.sumo.traffic_light_control import TrafficLightController


class SumoTrafficSignalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: TrafficRLConfig, controller_name: str = "env") -> None:
        super().__init__()
        self.config = config
        self.controller_name = controller_name
        self.session = SumoSession(config.sumo)
        self.spawner = DemandSpawner.from_config(config.sumo)
        self.accidents = AccidentManager.from_config(config.sumo)
        self.capacity_tracker = CapacityTracker(
            load_capacity_rules(config.sumo.capacity_config),
            config.sumo.accident_config,
        )
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
        self.accidents = AccidentManager.from_config(self.config.sumo)
        self.capacity_tracker = CapacityTracker(
            load_capacity_rules(self.config.sumo.capacity_config),
            self.config.sumo.accident_config,
        )
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
        return observation, self._info(
            switched=False,
            safety_decision=None,
            spawned=0,
            spawned_accidents=0,
            arrived_delta=0,
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.traci is None or self.controller is None:
            raise RuntimeError("Environment must be reset before step().")

        safety_decision = self.safety.resolve(int(action), self.controller)
        switched = self.controller.apply_action(safety_decision.action)
        spawned, spawned_accidents, arrived_delta = self._advance_control_interval()
        total_queue, total_wait = self.controller._network_totals()
        tail_queue = self._tail_queue()
        simulation_time = float(self.traci.simulation.getTime())
        capacity_by_direction = self.capacity_tracker.capacity_by_direction(simulation_time)
        directional_values = self._directional_values(capacity_by_direction)
        if all(capacity >= 1.0 for capacity in capacity_by_direction.values()):
            directional_values = directional_values.__class__(
                total_queue=float(total_queue),
                total_wait=float(total_wait),
                total_tail_queue=float(tail_queue),
                effective_queue=float(total_queue),
                effective_wait=float(total_wait),
                effective_tail_queue=float(tail_queue),
                event_direction_queue=0.0,
                event_direction_wait=0.0,
                non_event_direction_queue=float(total_queue),
                non_event_direction_wait=float(total_wait),
            )
        effective_arrived_delta = compute_effective_arrived(
            arrived_delta,
            capacity_by_direction,
        )
        served_directions = self.capacity_tracker.served_directions(safety_decision.action)
        wasted_green = compute_wasted_green(served_directions, capacity_by_direction)
        reward_result = self.reward_fn(
            RewardInputs(
                total_queue=total_queue,
                total_wait=total_wait,
                arrived_delta=float(arrived_delta),
                tail_queue=tail_queue,
                switched=switched,
                safety_overridden=safety_decision.overridden,
                effective_queue=directional_values.effective_queue,
                effective_wait=directional_values.effective_wait,
                effective_arrived_delta=effective_arrived_delta,
                effective_tail_queue=directional_values.effective_tail_queue,
                wasted_green=wasted_green,
            )
        )
        simulation_time = float(self.traci.simulation.getTime())
        terminated = False
        truncated = simulation_time >= self.config.sumo.end_time
        info = self._info(
            switched=switched,
            safety_decision=safety_decision,
            spawned=spawned,
            spawned_accidents=spawned_accidents,
            arrived_delta=arrived_delta,
            reward_result=reward_result,
            tail_queue=tail_queue,
            capacity_by_direction=capacity_by_direction,
            directional_values=directional_values,
            effective_arrived_delta=effective_arrived_delta,
            wasted_green=wasted_green,
            served_directions=served_directions,
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

    def _advance_control_interval(self) -> tuple[int, int, int]:
        assert self.traci is not None
        spawned = 0
        spawned_accidents = 0
        arrived_delta = 0
        step_length = max(float(self.config.sumo.step_length), 1e-6)
        step_count = max(1, int(math.ceil(self.config.control.decision_interval_seconds / step_length)))
        for _ in range(step_count):
            current_time = int(round(self.traci.simulation.getTime()))
            spawned += self.spawner.spawn_due_until(self.traci, current_time)
            spawned_accidents += self.accidents.advance(self.traci, current_time)
            self.traci.simulationStep()
            if self.controller is not None:
                self.controller.finalize_pending_transitions()
            arrived_delta += int(self.traci.simulation.getArrivedNumber())
            self.accidents.pin_active(self.traci)
        return spawned, spawned_accidents, arrived_delta

    def _observation(self) -> np.ndarray:
        assert self.controller is not None
        if self.config.state.mode == "approach_level":
            observation = self.controller.get_approach_state_vector()
        elif self.config.state.mode == "lane_level":
            observation = self.controller.get_state()
        else:
            raise ValueError(f"Unsupported state mode: {self.config.state.mode}")
        observation = self._append_capacity_observation(observation)
        observation = self._append_event_observation(observation)
        return observation.astype(np.float32)

    def get_valid_action_mask(self) -> np.ndarray:
        if self.controller is None:
            raise RuntimeError("Environment must be reset before valid action mask is known.")
        return self.safety.valid_action_mask(self.controller)

    def _tail_queue(self) -> float:
        assert self.controller is not None
        approach_metrics = self.controller.get_approach_metrics()
        return max((metrics.queue_max for metrics in approach_metrics.values()), default=0.0)

    def _append_capacity_observation(self, observation: np.ndarray) -> np.ndarray:
        if not self.config.state.include_capacity:
            return observation
        assert self.traci is not None
        capacity_by_direction = self.capacity_tracker.capacity_by_direction(
            float(self.traci.simulation.getTime())
        )
        features: list[float] = []
        for direction in self.capacity_tracker.rules.directions:
            capacity = float(capacity_by_direction.get(direction, 1.0))
            features.extend([capacity, 1.0 if capacity < 1.0 else 0.0])
        return np.concatenate(
            [observation.astype(np.float32), np.asarray(features, dtype=np.float32)]
        )

    def _append_event_observation(self, observation: np.ndarray) -> np.ndarray:
        if not self.config.state.include_event_features:
            return observation
        return np.concatenate(
            [observation.astype(np.float32), self._event_feature_vector()],
            axis=0,
        )

    def _event_feature_vector(self) -> np.ndarray:
        assert self.traci is not None
        assert self.controller is not None
        simulation_time = float(self.traci.simulation.getTime())
        capacity_by_direction = self.capacity_tracker.capacity_by_direction(simulation_time)
        directional_values = self._directional_values(capacity_by_direction)
        action = self._current_action()
        served_directions = self.capacity_tracker.served_directions(action)
        wasted_green = compute_wasted_green(served_directions, capacity_by_direction)
        event_active = 1.0 if any(capacity < 1.0 for capacity in capacity_by_direction.values()) else 0.0
        return np.asarray(
            [
                event_active,
                float(self.accidents.active_accident_count),
                float(self.accidents.active_vehicle_count),
                directional_values.event_direction_queue,
                directional_values.event_direction_wait,
                directional_values.non_event_direction_queue,
                directional_values.non_event_direction_wait,
                wasted_green,
            ],
            dtype=np.float32,
        )

    def _directional_values(self, capacity_by_direction: dict[str, float]):
        assert self.controller is not None
        approach_metrics = self.controller.get_approach_metrics()
        queue_by_direction = {
            direction: _metric_float(metric, "queue_max", "queue", "queue_total")
            for direction in capacity_by_direction
            if (metric := self._approach_metric_for_direction(direction, approach_metrics)) is not None
        }
        wait_by_direction = {
            direction: _metric_float(
                metric,
                "waiting_time_total",
                "wait_total",
                "waiting_time",
                "wait",
            )
            for direction in capacity_by_direction
            if (metric := self._approach_metric_for_direction(direction, approach_metrics)) is not None
        }
        tail_queue_by_direction = {
            direction: _metric_float(metric, "queue_max", "queue", "queue_total")
            for direction in capacity_by_direction
            if (metric := self._approach_metric_for_direction(direction, approach_metrics)) is not None
        }
        return compute_directional_values(
            queue_by_direction,
            wait_by_direction,
            tail_queue_by_direction,
            capacity_by_direction,
        )

    def _phase_pressure(self) -> list[float]:
        assert self.controller is not None
        approach_metrics = self.controller.get_approach_metrics()
        pressures: list[float] = []
        for action, _ in enumerate(self.controller.action_map):
            incoming_pressure = 0.0
            for direction in self.capacity_tracker.served_directions(action):
                metric = self._approach_metric_for_direction(direction, approach_metrics)
                if metric is None:
                    continue
                incoming_pressure += float(
                    getattr(metric, "queue_max", getattr(metric, "queue", 0.0))
                )
            pressures.append(incoming_pressure)
        return pressures

    def _approach_metric_for_direction(
        self,
        direction: str,
        approach_metrics: dict[str, Any],
    ) -> Any | None:
        for rule in self.capacity_tracker.rules.edge_rules:
            if rule.direction == direction and rule.edge_id in approach_metrics:
                return approach_metrics[rule.edge_id]
        return approach_metrics.get(direction)

    def _current_action(self) -> int:
        assert self.controller is not None
        if (
            self.controller.last_action is not None
            and 0 <= self.controller.last_action < self.controller.action_size
        ):
            return int(self.controller.last_action)
        for index, (traffic_light_id, phase_index) in enumerate(self.controller.action_map):
            if self.controller.traci.trafficlight.getPhase(traffic_light_id) == phase_index:
                return index
        return 0

    def _info(
        self,
        switched: bool,
        safety_decision: Any | None,
        spawned: int,
        spawned_accidents: int,
        arrived_delta: int,
        reward_result: RewardResult | None = None,
        tail_queue: float | None = None,
        capacity_by_direction: dict[str, float] | None = None,
        directional_values: Any | None = None,
        effective_arrived_delta: float = 0.0,
        wasted_green: float = 0.0,
        served_directions: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        assert self.traci is not None
        assert self.controller is not None
        total_queue, total_wait = self.controller._network_totals()
        approach_metrics = self.controller.get_approach_metrics()
        vehicle_counts = [metric.vehicle_count_total for metric in approach_metrics.values()]
        speed_values = [metric.speed for metric in approach_metrics.values()]
        mean_speed = float(sum(speed_values) / len(speed_values)) if speed_values else 0.0
        self._arrived_total += int(arrived_delta)
        if tail_queue is None:
            tail_queue = max((metric.queue_max for metric in approach_metrics.values()), default=0.0)
        if capacity_by_direction is None:
            capacity_by_direction = self.capacity_tracker.capacity_by_direction(
                float(self.traci.simulation.getTime())
            )
        if directional_values is None:
            directional_values = self._directional_values(capacity_by_direction)
            if all(capacity >= 1.0 for capacity in capacity_by_direction.values()):
                directional_values = directional_values.__class__(
                    total_queue=float(total_queue),
                    total_wait=float(total_wait),
                    total_tail_queue=float(tail_queue),
                    effective_queue=float(total_queue),
                    effective_wait=float(total_wait),
                    effective_tail_queue=float(tail_queue),
                    event_direction_queue=0.0,
                    event_direction_wait=0.0,
                    non_event_direction_queue=float(total_queue),
                    non_event_direction_wait=float(total_wait),
                )
        if reward_result is None:
            reward_components = {
                "queue_penalty": 0.0,
                "wait_penalty": 0.0,
                "throughput_reward": 0.0,
                "switch_penalty": 0.0,
                "safety_penalty": 0.0,
                "tail_queue_penalty": 0.0,
                "queue_spike_penalty": 0.0,
                "tail_queue_spike_penalty": 0.0,
                "wasted_green_penalty": 0.0,
            }
        else:
            reward_components = {
                "queue_penalty": reward_result.queue_penalty,
                "wait_penalty": reward_result.wait_penalty,
                "throughput_reward": reward_result.throughput_reward,
                "switch_penalty": reward_result.switch_penalty,
                "safety_penalty": reward_result.safety_penalty,
                "tail_queue_penalty": reward_result.tail_queue_penalty,
                "queue_spike_penalty": reward_result.queue_spike_penalty,
                "tail_queue_spike_penalty": reward_result.tail_queue_spike_penalty,
                "wasted_green_penalty": reward_result.wasted_green_penalty,
            }
        event_green_count = sum(
            1 for direction in served_directions if capacity_by_direction.get(direction, 1.0) <= 0.0
        )
        event_active = any(capacity < 1.0 for capacity in capacity_by_direction.values())
        return {
            "controller_name": self.controller_name,
            "simulation_time": float(self.traci.simulation.getTime()),
            "total_queue": float(total_queue),
            "total_wait": float(total_wait),
            "tail_queue": float(tail_queue),
            "mean_speed": mean_speed,
            "vehicle_count": float(sum(vehicle_counts)),
            "arrived_vehicles": self._arrived_total,
            "interval_arrived_vehicles": int(arrived_delta),
            "effective_queue": directional_values.effective_queue,
            "effective_wait": directional_values.effective_wait,
            "effective_tail_queue": directional_values.effective_tail_queue,
            "effective_arrived_delta": float(effective_arrived_delta),
            "wasted_green": float(wasted_green),
            "event_active": event_active,
            "event_direction_green": 1 if event_green_count > 0 else 0,
            "event_direction_queue": directional_values.event_direction_queue,
            "event_direction_wait": directional_values.event_direction_wait,
            "non_event_direction_queue": directional_values.non_event_direction_queue,
            "non_event_direction_wait": directional_values.non_event_direction_wait,
            "spawned_vehicles": spawned,
            "active_accidents": self.accidents.active_accident_count,
            "active_accident_vehicles": self.accidents.active_vehicle_count,
            "spawned_accident_vehicles": spawned_accidents,
            "switched": switched,
            "raw_action": None if safety_decision is None else safety_decision.raw_action,
            "safe_action": None if safety_decision is None else safety_decision.action,
            "safety_overridden": False if safety_decision is None else safety_decision.overridden,
            "safety_reason": "reset" if safety_decision is None else safety_decision.reason,
            "valid_action_mask": self.get_valid_action_mask(),
            "phase_pressure": self._phase_pressure(),
            "safety_enabled": bool(self.config.safety.enabled),
            "effective_controller_name": f"{self.controller_name}_safety"
            if self.config.safety.enabled
            else self.controller_name,
            **reward_components,
        }


def _metric_float(metric: Any, *names: str) -> float:
    for name in names:
        if hasattr(metric, name):
            return float(getattr(metric, name))
    return 0.0
