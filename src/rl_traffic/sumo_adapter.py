from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.rl_traffic.config import SumoConfig, repo_path
from src.sumo.sumo_simulation import (
    build_spawn_schedule,
    get_route_lane_counts,
    get_sumo_input_files,
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
        self.events_by_time: dict[int, list[Any]] = defaultdict(list)
        for event in events:
            self.events_by_time[int(event.time)].append(event)

    @classmethod
    def from_config(cls, config: SumoConfig) -> "DemandSpawner":
        simulation_config = load_simulation_config(config.spawn_config)
        route_file, net_file = get_sumo_input_files(config.sumocfg)
        route_lane_counts = get_route_lane_counts(route_file, net_file)
        events = build_spawn_schedule(
            simulation_config,
            route_lane_counts,
            demand_scale=config.demand_scale,
        )
        return cls(events)

    def spawn_for_time(self, traci_module: Any, current_time: int) -> int:
        spawned = 0
        for event in self.events_by_time.get(current_time, ()):
            try:
                traci_module.vehicle.add(
                    vehID=event.vehicle_id,
                    routeID=event.route_id,
                    typeID=event.vehicle_type,
                    depart=str(current_time),
                    departLane=str(event.lane_index),
                )
                spawned += 1
            except traci_module.TraCIException:
                continue
        return spawned


class SumoSession:
    def __init__(self, config: SumoConfig) -> None:
        self.config = config
        self.traci = traci
        self._started = False

    def start(self) -> Any:
        if sumolib is None or traci is None:
            raise RuntimeError("sumolib and traci are required to run SUMO.")
        if self._started:
            self.close()
        sumo_binary = sumolib.checkBinary("sumo-gui" if self.config.gui else "sumo")
        command = [
            sumo_binary,
            "-c",
            str(repo_path(self.config.sumocfg)),
            "--seed",
            str(self.config.seed),
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
        ]
        traci.start(command)
        self._started = True
        return traci

    def close(self) -> None:
        if not self._started or traci is None:
            return
        try:
            traci.close()
        finally:
            self._started = False


def output_dir(run_name: str) -> Path:
    return repo_path(Path("runs") / run_name)
