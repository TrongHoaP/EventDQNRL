from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import traci as _traci
except ImportError:  # pragma: no cover - reported when the controller is used.
    _traci = None


@dataclass(frozen=True)
class TrafficLightSpec:
    traffic_light_id: str
    controlled_lanes: tuple[str, ...]
    green_phases: tuple[int, ...]
    yellow_after_green: dict[int, int]


@dataclass(frozen=True)
class TrafficLightMetrics:
    queue_length: float
    waiting_time: float
    vehicle_count: float


@dataclass(frozen=True)
class LaneMetrics:
    lane_id: str
    flow_count_5m: float
    queue: float
    waiting: float
    waiting_total: float
    vehicle_count: float
    speed: float
    speed_lane_mean: float
    detector_speed: float
    occupancy: float
    detector_occupancy: float
    jam_veh: float
    detector_jam_veh: float
    lane_length: float

    @property
    def queue_length(self) -> float:
        return self.queue

    @property
    def waiting_time(self) -> float:
        return self.waiting_total

    @property
    def mean_speed(self) -> float:
        return self.speed


@dataclass(frozen=True)
class ApproachMetrics:
    approach_id: str
    lanes: tuple[str, ...]
    flow_count_5m: float
    speed: float
    speed_min: float
    queue: float
    queue_max: float
    waiting: float
    waiting_total: float
    occupancy: float
    occupancy_max: float
    jam_veh: float
    jam_veh_max: float
    vehicle_count_total: float
    has_vehicle: float
    has_queue: float
    speed_lane_mean: float
    detector_speed: float
    detector_occupancy: float
    detector_jam_veh: float
    lane_metrics: tuple[LaneMetrics, ...]

    @property
    def queue_total(self) -> float:
        return self.queue

    @property
    def waiting_time_total(self) -> float:
        return self.waiting_total

    @property
    def waiting_time_mean(self) -> float:
        return self.waiting

    @property
    def speed_mean(self) -> float:
        return self.speed

    @property
    def occupancy_mean(self) -> float:
        return self.occupancy


class TrafficLightController:
    """TraCI traffic-light controller with a DQN-friendly discrete interface."""

    def __init__(
        self,
        traci_module: Any | None = None,
        traffic_light_ids: list[str] | tuple[str, ...] | None = None,
        min_green: float = 5.0,
        yellow_time: float = 3.0,
        decision_interval: int = 10,
        max_queue_per_lane: float = 20.0,
        max_wait_per_lane: float = 300.0,
        max_vehicle_per_lane: float = 30.0,
        max_speed: float = 13.89,
        max_occupancy: float = 100.0,
    ) -> None:
        self.traci = traci_module or _traci
        if self.traci is None:
            raise RuntimeError("traci is required to use TrafficLightController.")
        if decision_interval < 1:
            raise ValueError("decision_interval must be >= 1.")
        if min_green < 0:
            raise ValueError("min_green must be >= 0.")
        if yellow_time < 0:
            raise ValueError("yellow_time must be >= 0.")

        self._configured_ids = tuple(traffic_light_ids) if traffic_light_ids else None
        self.min_green = float(min_green)
        self.yellow_time = float(yellow_time)
        self.decision_interval = int(decision_interval)
        self.max_queue_per_lane = float(max_queue_per_lane)
        self.max_wait_per_lane = float(max_wait_per_lane)
        self.max_vehicle_per_lane = float(max_vehicle_per_lane)
        self.max_speed = float(max_speed)
        self.max_occupancy = float(max_occupancy)

        self.specs: dict[str, TrafficLightSpec] = {}
        self.action_map: list[tuple[str, int]] = []
        self.approach_lanes: dict[str, tuple[str, ...]] = {}
        self.last_switch_time: dict[str, float] = {}
        self.last_action: int | None = None
        self.last_total_queue = 0.0
        self.last_total_wait = 0.0
        self.pending_green_after_yellow: dict[str, int] = {}
        self.pending_yellow_until: dict[str, float] = {}

        self.reset()

    @property
    def traffic_light_ids(self) -> tuple[str, ...]:
        return tuple(self.specs)

    @property
    def action_size(self) -> int:
        return len(self.action_map)

    @property
    def state_size(self) -> int:
        return int(self.get_state().shape[0])

    def reset(self) -> np.ndarray:
        """Refresh SUMO metadata and return the initial state vector."""
        ids = self._configured_ids or tuple(self.traci.trafficlight.getIDList())
        if not ids:
            raise RuntimeError("The SUMO scenario has no traffic lights.")

        self.specs = {}
        self.action_map = []
        for traffic_light_id in ids:
            spec = self._build_spec(traffic_light_id)
            self.specs[traffic_light_id] = spec
            self.action_map.extend(
                (traffic_light_id, phase_index) for phase_index in spec.green_phases
            )

        if not self.action_map:
            raise RuntimeError("No controllable green traffic-light phases were found.")

        self.approach_lanes = self._build_approach_lanes()
        now = self._sim_time()
        self.last_switch_time = {traffic_light_id: now for traffic_light_id in self.specs}
        self.last_action = None
        self.last_total_queue, self.last_total_wait = self._network_totals()
        self.pending_green_after_yellow = {}
        self.pending_yellow_until = {}
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Return a fixed-order state vector suitable for a DQN model."""
        values: list[float] = []
        now = self._sim_time()

        for spec in self.specs.values():
            current_phase = self.traci.trafficlight.getPhase(spec.traffic_light_id)
            values.extend(1.0 if current_phase == phase else 0.0 for phase in spec.green_phases)
            elapsed = now - self.last_switch_time.get(spec.traffic_light_id, now)
            values.append(self._clip01(elapsed / max(self.min_green, 1.0)))

            for lane_id in spec.controlled_lanes:
                queue = self.traci.lane.getLastStepHaltingNumber(lane_id)
                wait = self.traci.lane.getWaitingTime(lane_id)
                count = self.traci.lane.getLastStepVehicleNumber(lane_id)
                values.append(self._clip01(queue / self.max_queue_per_lane))
                values.append(self._clip01(wait / self.max_wait_per_lane))
                values.append(self._clip01(count / self.max_vehicle_per_lane))

        return np.asarray(values, dtype=np.float32)

    def get_approach_metrics(self) -> dict[str, ApproachMetrics]:
        """Aggregate queue, wait, speed, and occupancy by incoming approach."""
        return {
            approach_id: self._metrics_for_approach(approach_id, lanes)
            for approach_id, lanes in self.approach_lanes.items()
        }

    def get_approach_state_vector(self) -> np.ndarray:
        """Return compact approach-level features for RL state construction."""
        values: list[float] = []
        for metrics in self.get_approach_metrics().values():
            lane_count = max(len(metrics.lanes), 1)
            values.extend(
                (
                    self._clip01(metrics.queue_total / (self.max_queue_per_lane * lane_count)),
                    self._clip01(metrics.waiting_time_mean / self.max_wait_per_lane),
                    self._clip01(metrics.speed_mean / self.max_speed),
                    self._clip01(metrics.occupancy_mean / self.max_occupancy),
                    self._clip01(metrics.vehicle_count_total / (self.max_vehicle_per_lane * lane_count)),
                )
            )
        return np.asarray(values, dtype=np.float32)

    def log_approach_metrics(self) -> None:
        time = self._sim_time()
        for metrics in self.get_approach_metrics().values():
            print(
                f"t={time:.0f} {metrics.approach_id} "
                f"flow5m={metrics.flow_count_5m:.1f} queue={metrics.queue:.1f} "
                f"queue_max={metrics.queue_max:.1f} wait={metrics.waiting:.1f} "
                f"wait_total={metrics.waiting_total:.1f} speed={metrics.speed:.2f} "
                f"speed_min={metrics.speed_min:.2f} occupancy={metrics.occupancy:.2f} "
                f"occupancy_max={metrics.occupancy_max:.2f} jam={metrics.jam_veh:.1f} "
                f"jam_max={metrics.jam_veh_max:.1f} veh={metrics.vehicle_count_total:.1f}"
            )

    def apply_action(self, action: int) -> bool:
        """
        Apply a discrete action.

        Returns True when the command changed the selected traffic light's phase.
        """
        if action < 0 or action >= self.action_size:
            raise ValueError(f"Invalid action {action}; expected 0..{self.action_size - 1}.")

        traffic_light_id, target_green = self.action_map[action]
        if traffic_light_id in self.pending_green_after_yellow:
            return False

        current_phase = self.traci.trafficlight.getPhase(traffic_light_id)
        now = self._sim_time()
        elapsed = now - self.last_switch_time.get(traffic_light_id, now)

        if current_phase == target_green:
            self.last_action = action
            return False
        if elapsed < self.min_green:
            self.last_action = action
            return False

        spec = self.specs[traffic_light_id]
        yellow_phase = spec.yellow_after_green.get(current_phase)
        if yellow_phase is not None and self.yellow_time > 0:
            self.traci.trafficlight.setPhase(traffic_light_id, yellow_phase)
            self.traci.trafficlight.setPhaseDuration(traffic_light_id, self.yellow_time)
            self.pending_green_after_yellow[traffic_light_id] = target_green
            self.pending_yellow_until[traffic_light_id] = now + self.yellow_time
            self.last_action = action
            return True

        self.traci.trafficlight.setPhase(traffic_light_id, target_green)
        self.traci.trafficlight.setPhaseDuration(traffic_light_id, self.decision_interval)
        self.last_switch_time[traffic_light_id] = self._sim_time()
        self.last_action = action
        return True

    def finalize_pending_transitions(self) -> None:
        now = self._sim_time()
        for traffic_light_id in list(self.pending_green_after_yellow):
            until = self.pending_yellow_until.get(traffic_light_id, 0.0)
            if now < until:
                continue

            target_green = self.pending_green_after_yellow.pop(traffic_light_id)
            self.pending_yellow_until.pop(traffic_light_id, None)
            self.traci.trafficlight.setPhase(traffic_light_id, target_green)
            self.traci.trafficlight.setPhaseDuration(
                traffic_light_id,
                self.decision_interval,
            )
            self.last_switch_time[traffic_light_id] = now

    def compute_reward(self, switched: bool = False) -> float:
        """Reward lower queues and waiting time, with a small switching penalty."""
        total_queue, total_wait = self._network_totals()
        queue_delta = self.last_total_queue - total_queue
        wait_delta = self.last_total_wait - total_wait

        self.last_total_queue = total_queue
        self.last_total_wait = total_wait

        switch_penalty = 0.1 if switched else 0.0
        return float((0.5 * queue_delta) + (0.01 * wait_delta) - switch_penalty)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        Apply one action, advance SUMO by decision_interval seconds, and return RL data.

        The tuple shape follows Gymnasium's common pattern:
        (next_state, reward, terminated, info).
        """
        switched = self.apply_action(action)
        self._advance_steps(self.decision_interval)
        reward = self.compute_reward(switched=switched)
        terminated = self.traci.simulation.getMinExpectedNumber() == 0
        info = {
            "action": action,
            "switched": switched,
            "simulation_time": self._sim_time(),
            "total_queue": self.last_total_queue,
            "total_wait": self.last_total_wait,
        }
        return self.get_state(), reward, terminated, info

    def metrics(self) -> dict[str, TrafficLightMetrics]:
        return {
            traffic_light_id: self._metrics_for_lanes(spec.controlled_lanes)
            for traffic_light_id, spec in self.specs.items()
        }

    def get_lane_metrics(self, lane_id: str) -> LaneMetrics:
        queue = self._safe_float(self.traci.lane.getLastStepHaltingNumber(lane_id), 0.0)
        detector_jam_veh = self._detector_jam_veh(lane_id)
        jam_veh = queue if detector_jam_veh is None else detector_jam_veh

        vehicle_count = self._safe_float(self.traci.lane.getLastStepVehicleNumber(lane_id), 0.0)

        flow_count_5m = self._detector_flow_count_5m(lane_id)
        if flow_count_5m is None:
            flow_count_5m = 0.0

        if vehicle_count > 0:
            speed_lane_mean = self._safe_float(self.traci.lane.getLastStepMeanSpeed(lane_id), 0.0)
        else:
            speed_lane_mean = 0.0

        detector_speed = self._detector_speed(lane_id)
        speed = speed_lane_mean

        detector_occupancy = self._detector_occupancy(lane_id)
        occupancy = 0.0 if detector_occupancy is None else detector_occupancy

        waiting_total = self._safe_float(self.traci.lane.getWaitingTime(lane_id), 0.0)
        waiting = waiting_total / max(vehicle_count, 1.0)
        return LaneMetrics(
            lane_id=lane_id,
            flow_count_5m=flow_count_5m,
            queue=queue,
            waiting=waiting,
            waiting_total=waiting_total,
            vehicle_count=vehicle_count,
            speed=speed,
            speed_lane_mean=speed_lane_mean,
            detector_speed=0.0 if detector_speed is None else detector_speed,
            occupancy=occupancy,
            detector_occupancy=occupancy,
            jam_veh=jam_veh,
            detector_jam_veh=0.0 if detector_jam_veh is None else detector_jam_veh,
            lane_length=self._safe_float(self.traci.lane.getLength(lane_id), 1.0),
        )

    def _build_spec(self, traffic_light_id: str) -> TrafficLightSpec:
        programs = self.traci.trafficlight.getAllProgramLogics(traffic_light_id)
        if not programs:
            raise RuntimeError(f"Traffic light {traffic_light_id!r} has no programs.")

        phases = programs[0].phases
        green_phases = tuple(
            index for index, phase in enumerate(phases) if self._is_green_phase(phase.state)
        )
        if not green_phases:
            raise RuntimeError(f"Traffic light {traffic_light_id!r} has no green phases.")

        yellow_after_green: dict[int, int] = {}
        for phase_index in green_phases:
            next_index = (phase_index + 1) % len(phases)
            if self._is_yellow_phase(phases[next_index].state):
                yellow_after_green[phase_index] = next_index

        return TrafficLightSpec(
            traffic_light_id=traffic_light_id,
            controlled_lanes=self._unique_lanes(
                self.traci.trafficlight.getControlledLanes(traffic_light_id)
            ),
            green_phases=green_phases,
            yellow_after_green=yellow_after_green,
        )

    def _network_totals(self) -> tuple[float, float]:
        lanes = []
        for spec in self.specs.values():
            lanes.extend(spec.controlled_lanes)
        metrics = self._metrics_for_lanes(tuple(lanes))
        return metrics.queue_length, metrics.waiting_time

    def _metrics_for_lanes(self, lane_ids: tuple[str, ...]) -> TrafficLightMetrics:
        queue = 0.0
        wait = 0.0
        count = 0.0
        for lane_id in lane_ids:
            queue += float(self.traci.lane.getLastStepHaltingNumber(lane_id))
            wait += float(self.traci.lane.getWaitingTime(lane_id))
            count += float(self.traci.lane.getLastStepVehicleNumber(lane_id))
        return TrafficLightMetrics(queue_length=queue, waiting_time=wait, vehicle_count=count)

    def _metrics_for_approach(
        self,
        approach_id: str,
        lane_ids: tuple[str, ...],
    ) -> ApproachMetrics:
        lane_metrics = tuple(self.get_lane_metrics(lane_id) for lane_id in lane_ids)
        flow_count_5m = sum(metric.flow_count_5m for metric in lane_metrics)
        queue = sum(metric.queue for metric in lane_metrics)
        queue_max = max((metric.queue for metric in lane_metrics), default=0.0)
        waiting_total = sum(metric.waiting_total for metric in lane_metrics)
        vehicle_count_total = sum(metric.vehicle_count for metric in lane_metrics)
        speed = self._vehicle_weighted_mean(
            ((metric.speed, metric.vehicle_count) for metric in lane_metrics),
            fallback_values=(metric.speed for metric in lane_metrics),
        )
        speed_min = min(
            (metric.speed for metric in lane_metrics if metric.vehicle_count > 0 and metric.speed >= 0),
            default=0.0,
        )
        waiting = self._vehicle_weighted_mean(
            ((metric.waiting, self._flow_or_queue_weight(metric)) for metric in lane_metrics),
            fallback_values=(metric.waiting for metric in lane_metrics),
        )
        occupancy = self._length_weighted_mean(
            (metric.occupancy, metric.lane_length) for metric in lane_metrics
        )
        occupancy_max = max((metric.occupancy for metric in lane_metrics), default=0.0)
        jam_veh = sum(metric.jam_veh for metric in lane_metrics)
        jam_veh_max = max((metric.jam_veh for metric in lane_metrics), default=0.0)
        has_vehicle = 1.0 if vehicle_count_total > 0 else 0.0
        has_queue = 1.0 if queue > 0 else 0.0
        detector_speed = self._vehicle_weighted_mean(
            ((metric.detector_speed, metric.flow_count_5m) for metric in lane_metrics),
            fallback_values=(metric.detector_speed for metric in lane_metrics if metric.detector_speed > 0),
        )
        detector_occupancy = self._length_weighted_mean(
            (metric.detector_occupancy, metric.lane_length) for metric in lane_metrics
        )
        detector_jam_veh = sum(metric.detector_jam_veh for metric in lane_metrics)
        return ApproachMetrics(
            approach_id=approach_id,
            lanes=lane_ids,
            flow_count_5m=flow_count_5m,
            speed=speed,
            speed_min=speed_min,
            queue=queue,
            queue_max=queue_max,
            waiting=waiting,
            waiting_total=waiting_total,
            occupancy=occupancy,
            occupancy_max=occupancy_max,
            jam_veh=jam_veh,
            jam_veh_max=jam_veh_max,
            vehicle_count_total=vehicle_count_total,
            has_vehicle=has_vehicle,
            has_queue=has_queue,
            speed_lane_mean=speed,
            detector_speed=detector_speed,
            detector_occupancy=detector_occupancy,
            detector_jam_veh=detector_jam_veh,
            lane_metrics=lane_metrics,
        )

    def _build_approach_lanes(self) -> dict[str, tuple[str, ...]]:
        lanes = []
        for spec in self.specs.values():
            lanes.extend(spec.controlled_lanes)
        incoming_lanes = self._unique_lanes([lane for lane in lanes if not lane.startswith(":")])
        grouped: dict[str, list[str]] = {}
        for lane_id in incoming_lanes:
            edge_id, lane_index = self._split_lane_id(lane_id)
            if edge_id is None or lane_index is None:
                continue
            grouped.setdefault(edge_id, []).append(lane_id)

        approach_lanes = {
            edge_id: tuple(sorted(edge_lanes, key=lambda lane: self._split_lane_id(lane)[1] or 0))
            for edge_id, edge_lanes in sorted(grouped.items())
        }
        return approach_lanes

    def _detector_jam_veh(self, detector_id: str) -> float | None:
        if not self._has_detector(self.traci.lanearea, detector_id):
            return None
        return self._safe_positive(self.traci.lanearea.getJamLengthVehicle(detector_id))

    def _detector_flow_count_5m(self, detector_id: str) -> float | None:
        current_time = int(round(self._sim_time()))
        if current_time <= 0 or current_time % 300 != 0:
            return 0.0
        if self._has_detector(self.traci.inductionloop, detector_id):
            flow = self._safe_positive(self.traci.inductionloop.getLastIntervalVehicleNumber(detector_id))
            if flow is not None:
                return flow
            flow = self._safe_positive(self.traci.inductionloop.getIntervalVehicleNumber(detector_id))
            if flow is not None:
                return flow
        if self._has_detector(self.traci.lanearea, detector_id):
            flow = self._safe_positive(self.traci.lanearea.getLastIntervalVehicleNumber(detector_id))
            if flow is not None:
                return flow
            return self._safe_positive(self.traci.lanearea.getIntervalVehicleNumber(detector_id))
        return None

    def _detector_vehicle_count(self, detector_id: str) -> float | None:
        if self._has_detector(self.traci.lanearea, detector_id):
            return self._safe_positive(self.traci.lanearea.getLastStepVehicleNumber(detector_id))
        return None

    def _detector_speed(self, detector_id: str) -> float | None:
        if self._has_detector(self.traci.lanearea, detector_id):
            speed = self._safe_positive(self.traci.lanearea.getLastStepMeanSpeed(detector_id))
            if speed is not None:
                return speed
        if self._has_detector(self.traci.inductionloop, detector_id):
            return self._safe_positive(self.traci.inductionloop.getLastStepMeanSpeed(detector_id))
        return None

    def _detector_occupancy(self, detector_id: str) -> float | None:
        if self._has_detector(self.traci.lanearea, detector_id):
            occupancy = self._safe_positive(self.traci.lanearea.getLastStepOccupancy(detector_id))
            if occupancy is not None:
                return occupancy
        if self._has_detector(self.traci.inductionloop, detector_id):
            return self._safe_positive(self.traci.inductionloop.getLastStepOccupancy(detector_id))
        return None

    @staticmethod
    def _flow_or_vehicle_weight(metric: LaneMetrics) -> float:
        if metric.flow_count_5m > 0:
            return metric.flow_count_5m
        return metric.vehicle_count

    @staticmethod
    def _flow_or_queue_weight(metric: LaneMetrics) -> float:
        if metric.flow_count_5m > 0:
            return metric.flow_count_5m
        return metric.queue

    def _advance_steps(self, seconds: int) -> None:
        if seconds <= 0:
            return
        target_time = self._sim_time() + float(seconds)
        while self._sim_time() < target_time:
            self.traci.simulationStep()

    def _sim_time(self) -> float:
        return float(self.traci.simulation.getTime())

    @staticmethod
    def _is_green_phase(state: str) -> bool:
        return ("G" in state or "g" in state) and "y" not in state.lower()

    @staticmethod
    def _is_yellow_phase(state: str) -> bool:
        return "y" in state.lower()

    @staticmethod
    def _unique_lanes(lane_ids: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        unique: list[str] = []
        for lane_id in lane_ids:
            if lane_id not in unique:
                unique.append(lane_id)
        return tuple(unique)

    @staticmethod
    def _clip01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    @staticmethod
    def _split_lane_id(lane_id: str) -> tuple[str | None, int | None]:
        if "_" not in lane_id:
            return None, None
        edge_id, lane_index = lane_id.rsplit("_", 1)
        if not lane_index.isdigit():
            return None, None
        return edge_id, int(lane_index)

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(value_float):
            return default
        return value_float

    @classmethod
    def _safe_positive(cls, value: Any) -> float | None:
        value_float = cls._safe_float(value, -1.0)
        if value_float < 0:
            return None
        return value_float

    @staticmethod
    def _has_detector(domain: Any, detector_id: str) -> bool:
        try:
            return detector_id in set(domain.getIDList())
        except Exception:
            return False

    @staticmethod
    def _vehicle_weighted_mean(
        values: Any,
        fallback_values: Any,
    ) -> float:
        weighted_sum = 0.0
        weight_sum = 0.0
        for value, weight in values:
            if value < 0:
                continue
            if weight > 0:
                weighted_sum += value * weight
                weight_sum += weight
        if weight_sum > 0:
            return float(weighted_sum / weight_sum)

        fallback = [value for value in fallback_values if value >= 0]
        if not fallback:
            return 0.0
        return float(sum(fallback) / len(fallback))

    @staticmethod
    def _length_weighted_mean(values: Any) -> float:
        weighted_sum = 0.0
        weight_sum = 0.0
        for value, weight in values:
            if value < 0 or weight <= 0:
                continue
            weighted_sum += value * weight
            weight_sum += weight
        if weight_sum <= 0:
            return 0.0
        return float(weighted_sum / weight_sum)
