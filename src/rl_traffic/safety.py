from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.rl_traffic.config import ControlConfig, SafetyConfig


@dataclass(frozen=True)
class SafetyDecision:
    raw_action: int
    action: int
    overridden: bool
    reason: str


class SafetyLayer:
    def __init__(self, config: SafetyConfig, control: ControlConfig) -> None:
        self.config = config
        self.control = control

    def resolve(self, raw_action: int, controller: Any) -> SafetyDecision:
        if not self.config.enabled:
            return SafetyDecision(raw_action, raw_action, False, "disabled")
        if raw_action < 0 or raw_action >= controller.action_size:
            return self._fallback(raw_action, controller, "invalid_action")

        traffic_light_id, target_green = controller.action_map[raw_action]
        spec = controller.specs[traffic_light_id]
        if self.config.enforce_green_phase_only and target_green not in spec.green_phases:
            return self._fallback(raw_action, controller, "non_green_phase")

        now = float(controller.traci.simulation.getTime())
        current_phase = controller.traci.trafficlight.getPhase(traffic_light_id)
        elapsed = now - controller.last_switch_time.get(traffic_light_id, now)

        if (
            self.config.enforce_min_green
            and current_phase != target_green
            and elapsed < self.control.min_green_seconds
        ):
            return self._fallback(raw_action, controller, "min_green")

        if (
            self.config.enforce_max_green
            and current_phase == target_green
            and elapsed >= self.control.max_green_seconds
        ):
            for index, (candidate_tl_id, candidate_phase) in enumerate(controller.action_map):
                if candidate_tl_id == traffic_light_id and candidate_phase != current_phase:
                    return SafetyDecision(raw_action, index, index != raw_action, "max_green")

        return SafetyDecision(raw_action, raw_action, False, "ok")

    def _fallback(self, raw_action: int, controller: Any, reason: str) -> SafetyDecision:
        if self.config.invalid_action_policy == "first":
            return SafetyDecision(raw_action, 0, raw_action != 0, reason)
        if controller.last_action is not None and 0 <= controller.last_action < controller.action_size:
            return SafetyDecision(raw_action, controller.last_action, raw_action != controller.last_action, reason)
        current_action = self._current_phase_action(controller)
        return SafetyDecision(raw_action, current_action, raw_action != current_action, reason)

    @staticmethod
    def _current_phase_action(controller: Any) -> int:
        for index, (traffic_light_id, phase_index) in enumerate(controller.action_map):
            if controller.traci.trafficlight.getPhase(traffic_light_id) == phase_index:
                return index
        return 0
