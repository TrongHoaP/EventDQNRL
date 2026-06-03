from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.rl_traffic.config import SumoConfig, repo_path
from src.sumo.sumo_simulation import (
    AccidentEvent,
    build_spawn_schedule,
    get_route_lane_counts,
    get_sumo_input_files,
    load_or_create_accident_schedule,
    load_simulation_config,
)

try:
    import sumolib
    import traci
except ImportError:  # pragma: no cover - reported at runtime.
    sumolib = None
    traci = None


class DemandSpawner:
    def __init__(self, events: list[Any]) -> None:
        self.events = sorted(events, key=lambda event: int(event.time))
        self.next_event_index = 0
        self.events_by_time: dict[int, list[Any]] = defaultdict(list)
        for event in self.events:
            self.events_by_time[int(event.time)].append(event)

    @classmethod
    def from_config(cls, config: SumoConfig, seed: int | None = None) -> "DemandSpawner":
        simulation_config = load_simulation_config(config.spawn_config)
        route_file, net_file = get_sumo_input_files(config.sumocfg)
        route_lane_counts = get_route_lane_counts(route_file, net_file)
        events = build_spawn_schedule(
            simulation_config,
            route_lane_counts,
            demand_scale=config.demand_scale,
        )
        if config.randomize_demand:
            events = _jitter_spawn_times(
                events,
                seed=config.seed if seed is None else seed,
                jitter_seconds=config.demand_time_jitter_seconds,
                end_time=config.end_time,
            )
        return cls(events)

    def reset(self) -> None:
        self.next_event_index = 0

    def spawn_for_time(self, traci_module: Any, current_time: int) -> int:
        spawned = 0
        for event in self.events_by_time.get(current_time, ()):
            spawned += self._spawn_event(traci_module, event, current_time)
        return spawned

    def spawn_due_until(self, traci_module: Any, current_time: int) -> int:
        spawned = 0
        while self.next_event_index < len(self.events):
            event = self.events[self.next_event_index]
            if int(event.time) > current_time:
                break
            spawned += self._spawn_event(traci_module, event, current_time)
            self.next_event_index += 1
        return spawned

    @staticmethod
    def _spawn_event(traci_module: Any, event: Any, current_time: int) -> int:
        try:
            traci_module.vehicle.add(
                vehID=event.vehicle_id,
                routeID=event.route_id,
                typeID=event.vehicle_type,
                depart=str(current_time),
                departLane=str(event.lane_index),
            )
            return 1
        except traci_module.TraCIException:
            return 0


class AccidentManager:
    def __init__(self, events: list[AccidentEvent]) -> None:
        self.events = sorted(events, key=lambda event: int(event.time))
        self.next_event_index = 0
        self.active_accidents: dict[str, AccidentEvent] = {}

    @classmethod
    def from_config(cls, config: SumoConfig) -> "AccidentManager":
        if config.disable_accidents:
            return cls([])
        route_file, net_file = get_sumo_input_files(config.sumocfg)
        events = load_or_create_accident_schedule(
            config.accident_config,
            route_file,
            net_file,
        )
        return cls(events)

    def reset(self) -> None:
        self.next_event_index = 0
        self.active_accidents.clear()

    def advance(self, traci_module: Any, current_time: int) -> int:
        spawned_vehicles = 0
        self._despawn_due(traci_module, current_time)
        while self.next_event_index < len(self.events):
            event = self.events[self.next_event_index]
            if int(event.time) > current_time:
                break
            spawned_vehicles += self._spawn_event(traci_module, event, current_time)
            self.active_accidents[event.event_id] = event
            self.next_event_index += 1
        self.pin_active(traci_module)
        return spawned_vehicles

    def pin_active(self, traci_module: Any) -> None:
        for event in self.active_accidents.values():
            for vehicle in event.vehicles:
                try:
                    lane_id = f"{vehicle.edge_id}_{vehicle.lane_index}"
                    if vehicle.vehicle_id not in traci_module.vehicle.getIDList():
                        continue
                    if traci_module.vehicle.getLaneID(vehicle.vehicle_id) != lane_id:
                        traci_module.vehicle.moveTo(
                            vehicle.vehicle_id,
                            lane_id,
                            vehicle.position,
                        )
                    elif (
                        abs(
                            traci_module.vehicle.getLanePosition(vehicle.vehicle_id)
                            - vehicle.position
                        )
                        > 0.5
                    ):
                        traci_module.vehicle.moveTo(
                            vehicle.vehicle_id,
                            lane_id,
                            vehicle.position,
                        )
                    traci_module.vehicle.setSpeed(vehicle.vehicle_id, 0.0)
                except traci_module.TraCIException:
                    continue

    @property
    def active_accident_count(self) -> int:
        return len(self.active_accidents)

    @property
    def active_vehicle_count(self) -> int:
        return sum(len(event.vehicles) for event in self.active_accidents.values())

    def _despawn_due(self, traci_module: Any, current_time: int) -> None:
        expired = [
            event_id
            for event_id, event in self.active_accidents.items()
            if event.despawn_time <= current_time
        ]
        for event_id in expired:
            event = self.active_accidents.pop(event_id)
            for vehicle in event.vehicles:
                try:
                    traci_module.vehicle.remove(vehicle.vehicle_id)
                except traci_module.TraCIException:
                    continue

    @staticmethod
    def _spawn_event(
        traci_module: Any,
        event: AccidentEvent,
        current_time: int,
    ) -> int:
        spawned = 0
        for vehicle in event.vehicles:
            try:
                traci_module.vehicle.add(
                    vehID=vehicle.vehicle_id,
                    routeID=vehicle.route_id,
                    typeID=vehicle.vehicle_type,
                    depart=str(current_time),
                    departLane=str(vehicle.lane_index),
                    departPos=str(vehicle.position),
                    departSpeed="0",
                )
                traci_module.vehicle.setLaneChangeMode(vehicle.vehicle_id, 0)
                traci_module.vehicle.setSpeed(vehicle.vehicle_id, 0.0)
                traci_module.vehicle.setColor(vehicle.vehicle_id, (255, 0, 0, 255))
                spawned += 1
            except traci_module.TraCIException:
                continue
        return spawned


class SumoSession:
    def __init__(self, config: SumoConfig) -> None:
        self.config = config
        self.traci = traci
        self._started = False
        self._output_prefix = ""

    def start(self, seed: int | None = None) -> Any:
        if sumolib is None or traci is None:
            raise RuntimeError("sumolib and traci are required to run SUMO.")
        if self._started:
            self.close()
        self._output_prefix = f".sumo_rl_{uuid4().hex}_"
        sumo_seed = self.config.seed if seed is None else seed
        sumo_binary = sumolib.checkBinary("sumo-gui" if self.config.gui else "sumo")
        command = [
            sumo_binary,
            "-c",
            str(repo_path(self.config.sumocfg)),
            "--seed",
            str(sumo_seed),
            "--step-length",
            str(self.config.step_length),
            "--end",
            str(self.config.end_time),
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
            "--time-to-teleport",
            "-1",
            "--xml-validation",
            "never",
            "--output-prefix",
            self._output_prefix,
        ]
        try:
            traci.start(command)
        except Exception:
            self._cleanup_output_files()
            raise
        self._started = True
        return traci

    def close(self) -> None:
        if not self._started or traci is None:
            self._cleanup_output_files()
            return
        try:
            traci.close()
        finally:
            self._started = False
            self._cleanup_output_files()

    def _cleanup_output_files(self) -> None:
        if not self._output_prefix:
            return
        output_root = repo_path(self.config.sumocfg).parent
        for path in output_root.rglob(f"{self._output_prefix}*"):
            if path.is_file():
                path.unlink(missing_ok=True)
        self._output_prefix = ""


def output_dir(run_name: str) -> Path:
    return repo_path(Path("runs") / run_name)


def _jitter_spawn_times(
    events: list[Any],
    seed: int,
    jitter_seconds: int,
    end_time: int,
) -> list[Any]:
    jitter = max(0, int(jitter_seconds))
    if jitter == 0:
        return list(events)
    rng = random.Random(seed)
    latest_depart = max(0, int(end_time) - 1)
    randomized = []
    for event in events:
        depart_time = int(event.time) + rng.randint(-jitter, jitter)
        randomized.append(replace(event, time=max(0, min(latest_depart, depart_time))))
    randomized.sort(key=lambda event: (int(event.time), event.vehicle_id))
    return randomized
