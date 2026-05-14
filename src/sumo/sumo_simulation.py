from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from src.sumo.traffic_light_control import TrafficLightController
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from traffic_light_control import TrafficLightController

try:
    import sumolib
    import traci
except ImportError:  # pragma: no cover - handled at runtime for dry-run usage
    sumolib = None
    traci = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "src" / "configs" / "simulation_density_spawn_config.json"
DEFAULT_ACCIDENT_CONFIG = REPO_ROOT / "src" / "configs" / "accident_spawn_events.json"
DEFAULT_SUMOCFG = REPO_ROOT / "scenarios" / "grid" / "grid.sumocfg"
DEFAULT_SUMO_SEED = 7
DEFAULT_ACCIDENT_HOURS = (4, 8, 12, 17, 20, 23)
DEFAULT_ACCIDENT_DURATION = 1800
DEFAULT_ACCIDENT_DISTANCE_METERS = 30.0
ACCIDENT_TYPE_PRIORITY = ("car", "minibus", "bus", "bike", "bicycle")

ROAD_ROUTE_MAPPING = {
    "road_3_lane": ("routeWE", "routeEW"),
    "road_2_lane": ("routeSN", "routeNS"),
}

ROUTE_APPROACH_MAPPING = {
    "routeWE": "A1B1",
    "routeEW": "C1B1",
    "routeSN": "B0B1",
    "routeNS": "B2B1",
}


@dataclass(frozen=True)
class SpawnEvent:
    time: int
    route_id: str
    lane_index: int
    vehicle_type: str
    vehicle_id: str


@dataclass(frozen=True)
class AccidentVehicle:
    vehicle_id: str
    route_id: str
    edge_id: str
    lane_index: int
    vehicle_type: str
    position: float


@dataclass(frozen=True)
class AccidentEvent:
    event_id: str
    time: int
    duration: int
    vehicles: tuple[AccidentVehicle, ...]

    @property
    def despawn_time(self) -> int:
        return self.time + self.duration


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def load_simulation_config(config_path: str | Path) -> dict[str, Any]:
    path = _resolve_path(config_path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    simulation_config = payload.get("simulation_config")
    if not isinstance(simulation_config, dict):
        raise ValueError("Missing object: simulation_config")

    road_density = simulation_config.get("densitybyhour_by_road")
    density_by_time = simulation_config.get("density_by_time")
    if not isinstance(road_density, dict) or not road_density:
        raise ValueError("Missing object: simulation_config.densitybyhour_by_road")
    if not isinstance(density_by_time, list) or not density_by_time:
        raise ValueError("Missing list: simulation_config.density_by_time")

    return simulation_config


def get_route_lane_counts(route_file: str | Path, net_file: str | Path) -> dict[str, int]:
    routes = _load_route_edges(route_file)
    edge_lane_counts = _load_edge_lane_counts(net_file)
    route_lane_counts: dict[str, int] = {}

    for route_id, edges in routes.items():
        if not edges:
            continue
        first_edge = edges[0]
        route_lane_counts[route_id] = edge_lane_counts.get(first_edge, 0)

    return route_lane_counts


def get_sumo_input_files(sumocfg_path: str | Path) -> tuple[Path, Path]:
    path = _resolve_path(sumocfg_path)
    tree = ET.parse(path)
    input_node = tree.getroot().find("input")
    if input_node is None:
        raise ValueError(f"Missing <input> block in {path}")

    net_file_node = input_node.find("net-file")
    route_file_node = input_node.find("route-files")
    if net_file_node is None or route_file_node is None:
        raise ValueError(f"Missing net-file or route-files in {path}")

    net_file = net_file_node.attrib.get("value")
    route_file = route_file_node.attrib.get("value")
    if not net_file or not route_file:
        raise ValueError(f"Empty net-file or route-files value in {path}")

    base_dir = path.parent
    return base_dir / route_file, base_dir / net_file


def build_spawn_schedule(
    simulation_config: dict[str, Any],
    route_lane_counts: dict[str, int],
    demand_scale: float = 1.0,
) -> list[SpawnEvent]:
    if demand_scale < 0:
        raise ValueError("demand_scale must be >= 0.")
    road_density = simulation_config["densitybyhour_by_road"]
    density_by_time = sorted(
        simulation_config["density_by_time"],
        key=lambda interval: int(interval["start_time"]),
    )
    counters: defaultdict[str, int] = defaultdict(int)
    remainders: defaultdict[tuple[str, str, str, int], float] = defaultdict(float)
    events: list[SpawnEvent] = []

    for interval in density_by_time:
        start_time = int(interval["start_time"])
        end_time = int(interval["end_time"])
        density = float(interval["density"]) * demand_scale
        if end_time <= start_time:
            raise ValueError(f"Invalid density interval: {interval}")

        interval_seconds = end_time - start_time
        for road_id, route_ids in ROAD_ROUTE_MAPPING.items():
            if road_id not in road_density:
                raise ValueError(f"Missing road density config for {road_id}")
            lane_config = road_density[road_id]
            for route_id in route_ids:
                route_lane_count = route_lane_counts.get(route_id)
                if not route_lane_count:
                    print(f"Warning: route {route_id} not found in route/net files; skipping.")
                    continue
                for lane_name, vehicle_counts in lane_config.items():
                    lane_index = _parse_lane_index(lane_name)
                    if lane_index >= route_lane_count:
                        print(
                            f"Warning: {route_id} has {route_lane_count} lanes; "
                            f"skipping {lane_name}."
                        )
                        continue
                    for vehicle_type, base_hourly_count in vehicle_counts.items():
                        expected_count = (
                            float(base_hourly_count) * density * interval_seconds / 3600.0
                        )
                        key = (road_id, route_id, vehicle_type, lane_index)
                        total_count = expected_count + remainders[key]
                        spawn_count = int(math.floor(total_count))
                        remainders[key] = total_count - spawn_count
                        if spawn_count <= 0:
                            continue
                        for depart_time in _even_depart_times(
                            start_time,
                            end_time,
                            spawn_count,
                        ):
                            counter_key = f"{route_id}_{lane_index}_{vehicle_type}"
                            counters[counter_key] += 1
                            events.append(
                                SpawnEvent(
                                    time=depart_time,
                                    route_id=route_id,
                                    lane_index=lane_index,
                                    vehicle_type=vehicle_type,
                                    vehicle_id=(
                                        f"{route_id}_lane{lane_index + 1}_"
                                        f"{vehicle_type}_{counters[counter_key]}"
                                    ),
                                )
                            )

    events.sort(key=lambda event: event.time)
    return events


def load_or_create_accident_schedule(
    accident_config_path: str | Path,
    route_file: str | Path,
    net_file: str | Path,
) -> list[AccidentEvent]:
    path = _resolve_path(accident_config_path)
    if path.exists():
        events = _load_accident_schedule(path)
        print(f"Loaded {len(events)} accident events from {path}.")
        return events

    events = _build_accident_schedule(route_file, net_file)
    _save_accident_schedule(path, events)
    print(f"Created {len(events)} accident events at {path}.")
    return events


def run_simulation(
    sumocfg_path: str | Path,
    events: list[SpawnEvent],
    accident_events: list[AccidentEvent],
    density_by_time: list[dict[str, Any]],
    demand_scale: float,
    gui: bool,
    end_time: int,
    step_length: float,
    log_approach_metrics: bool = False,
    log_interval: int = 300,
    approach_metrics_csv: str | Path | None = None,
) -> None:
    if sumolib is None or traci is None:
        raise RuntimeError("sumolib and traci are required to run SUMO simulation.")
    if log_interval < 1:
        raise ValueError("log_interval must be >= 1.")

    sumo_binary = sumolib.checkBinary("sumo-gui" if gui else "sumo")
    command = [
        sumo_binary,
        "-c",
        str(_resolve_path(sumocfg_path)),
        "--seed",
        str(DEFAULT_SUMO_SEED),
        "--step-length",
        str(step_length),
        "--end",
        str(end_time),
        "--no-step-log",
        "true",
        "--time-to-teleport",
        "-1",
        "--xml-validation",
        "never",
    ]

    events_by_time: defaultdict[int, list[SpawnEvent]] = defaultdict(list)
    for event in events:
        events_by_time[event.time].append(event)
    spawn_index = _build_spawn_index(events)

    accidents_by_spawn_time: defaultdict[int, list[AccidentEvent]] = defaultdict(list)
    accidents_by_despawn_time: defaultdict[int, list[AccidentEvent]] = defaultdict(list)
    for event in accident_events:
        accidents_by_spawn_time[event.time].append(event)
        accidents_by_despawn_time[event.despawn_time].append(event)

    traci.start(command)
    csv_file = None
    csv_writer = None
    try:
        traffic_light_controller = None
        next_metrics_log_time = log_interval
        if approach_metrics_csv is not None:
            csv_path = _resolve_path(approach_metrics_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = csv_path.open("w", encoding="utf-8", newline="")
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "time",
                    "approach_id",
                    "lanes",
                    "density_factor",
                    "spawn_count_5m",
                    "spawn_count_hour",
                    "flow_count_5m",
                    "detector_flow_5m",
                    "speed",
                    "speed_min",
                    "speed_lane_mean",
                    "detector_speed",
                    "queue",
                    "queue_max",
                    "waiting",
                    "waiting_total",
                    "occupancy",
                    "detector_occupancy",
                    "occupancy_max",
                    "jam_veh",
                    "detector_jam_veh",
                    "jam_veh_max",
                    "vehicle_count_total",
                    "has_vehicle",
                    "has_queue",
                ],
            )
            csv_writer.writeheader()
            print(f"Saving approach metrics CSV to {csv_path}.")

        if log_approach_metrics or csv_writer is not None:
            traffic_light_controller = TrafficLightController(traci_module=traci)
            _record_approach_metrics(
                traffic_light_controller,
                log_to_console=log_approach_metrics,
                csv_writer=csv_writer,
                spawn_index=spawn_index,
                density_by_time=density_by_time,
                demand_scale=demand_scale,
            )

        current_time = 0
        active_accidents: dict[str, AccidentEvent] = {}
        while current_time <= end_time:
            for accident in accidents_by_despawn_time.get(current_time, ()):
                _despawn_accident(accident)
                active_accidents.pop(accident.event_id, None)
            for event in events_by_time.get(current_time, ()):
                try:
                    traci.vehicle.add(
                        vehID=event.vehicle_id,
                        routeID=event.route_id,
                        typeID=event.vehicle_type,
                        depart=str(current_time),
                        departLane=str(event.lane_index),
                    )
                except traci.TraCIException as exc:
                    print(f"Warning: failed to add {event.vehicle_id}: {exc}")
            for accident in accidents_by_spawn_time.get(current_time, ()):
                _spawn_accident(accident, current_time)
                active_accidents[accident.event_id] = accident
            for accident in active_accidents.values():
                _pin_accident(accident)
            traci.simulationStep()
            for accident in active_accidents.values():
                _pin_accident(accident)
            current_time = int(traci.simulation.getTime())
            if (
                traffic_light_controller is not None
                and current_time >= next_metrics_log_time
            ):
                _record_approach_metrics(
                    traffic_light_controller,
                    log_to_console=log_approach_metrics,
                    csv_writer=csv_writer,
                    spawn_index=spawn_index,
                    density_by_time=density_by_time,
                    demand_scale=demand_scale,
                )
                while next_metrics_log_time <= current_time:
                    next_metrics_log_time += log_interval
            if not gui and current_time > end_time and traci.simulation.getMinExpectedNumber() == 0:
                break
    finally:
        if csv_file is not None:
            csv_file.close()
        traci.close()


def summarize_schedule(events: list[SpawnEvent]) -> None:
    by_route: defaultdict[str, int] = defaultdict(int)
    by_type: defaultdict[str, int] = defaultdict(int)
    for event in events:
        by_route[event.route_id] += 1
        by_type[event.vehicle_type] += 1

    print(f"Prepared {len(events)} spawn events.")
    print("By route:")
    for route_id in sorted(by_route):
        print(f"  {route_id}: {by_route[route_id]}")
    print("By vehicle type:")
    for vehicle_type in sorted(by_type):
        print(f"  {vehicle_type}: {by_type[vehicle_type]}")


def summarize_accident_schedule(events: list[AccidentEvent]) -> None:
    print(f"Prepared {len(events)} accident events.")
    for event in events:
        print(
            f"  {event.event_id}: spawn={event.time}, despawn={event.despawn_time}, "
            f"duration={event.duration}"
        )
        for vehicle in event.vehicles:
            print(
                f"    {vehicle.vehicle_id}: route={vehicle.route_id}, edge={vehicle.edge_id}, "
                f"lane={vehicle.lane_index}, type={vehicle.vehicle_type}, "
                f"pos={vehicle.position:.2f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SUMO simulation with density-based spawns.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to spawn config JSON.")
    parser.add_argument(
        "--accident-config",
        default=str(DEFAULT_ACCIDENT_CONFIG),
        help="Path to persistent accident schedule JSON.",
    )
    parser.add_argument(
        "--disable-accidents",
        action="store_true",
        help="Run without accident spawn/despawn events.",
    )
    parser.add_argument("--sumocfg", default=str(DEFAULT_SUMOCFG), help="Path to SUMO config file.")
    parser.add_argument("--gui", action="store_true", help="Run with sumo-gui instead of sumo.")
    parser.add_argument("--end-time", type=int, default=None, help="Override simulation end time.")
    parser.add_argument("--step-length", type=float, default=1.0, help="SUMO simulation step length.")
    parser.add_argument(
        "--demand-scale",
        type=float,
        default=1.0,
        help="Runtime multiplier for all configured density factors.",
    )
    parser.add_argument(
        "--log-approach-metrics",
        action="store_true",
        help="Log aggregated approach metrics while the simulation runs.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=300,
        help="Seconds between approach metric logs when --log-approach-metrics is enabled.",
    )
    parser.add_argument(
        "--approach-metrics-csv",
        default=None,
        help="Optional CSV path for aggregated approach metrics.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only build and summarize spawn schedule.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    simulation_config = load_simulation_config(args.config)
    route_file, net_file = get_sumo_input_files(args.sumocfg)
    route_lane_counts = get_route_lane_counts(route_file, net_file)
    events = build_spawn_schedule(simulation_config, route_lane_counts, args.demand_scale)
    accident_events = []
    if not args.disable_accidents:
        accident_events = load_or_create_accident_schedule(
            args.accident_config,
            route_file,
            net_file,
        )
    end_time = args.end_time or _config_end_time(simulation_config)

    summarize_schedule(events)
    summarize_accident_schedule(accident_events)
    if args.dry_run:
        return 0

    run_simulation(
        args.sumocfg,
        events,
        accident_events,
        simulation_config["density_by_time"],
        args.demand_scale,
        args.gui,
        end_time,
        args.step_length,
        args.log_approach_metrics,
        args.log_interval,
        args.approach_metrics_csv,
    )
    return 0


def _config_end_time(simulation_config: dict[str, Any]) -> int:
    return max(int(interval["end_time"]) for interval in simulation_config["density_by_time"])


def _even_depart_times(
    start_time: int,
    end_time: int,
    count: int,
) -> list[int]:
    if count == 1:
        return [start_time]

    interval_seconds = end_time - start_time
    spacing = interval_seconds / count
    departures = []
    for index in range(count):
        depart_time = start_time + int(index * spacing)
        departures.append(min(depart_time, end_time - 1))
    return departures


def _load_route_edges(route_file: str | Path) -> dict[str, list[str]]:
    tree = ET.parse(route_file)
    routes: dict[str, list[str]] = {}
    for route in tree.getroot().findall("route"):
        route_id = route.attrib.get("id")
        edges = route.attrib.get("edges", "")
        if route_id:
            routes[route_id] = edges.split()
    return routes


def _load_vehicle_classes(route_file: str | Path) -> dict[str, str]:
    tree = ET.parse(route_file)
    vehicle_classes: dict[str, str] = {}
    for vehicle_type in tree.getroot().findall("vType"):
        type_id = vehicle_type.attrib.get("id")
        vehicle_class = vehicle_type.attrib.get("vClass", "passenger")
        if type_id:
            vehicle_classes[type_id] = vehicle_class
    return vehicle_classes


def _load_edge_lane_specs(net_file: str | Path) -> dict[str, list[dict[str, Any]]]:
    tree = ET.parse(net_file)
    lane_specs: dict[str, list[dict[str, Any]]] = {}
    for edge in tree.getroot().findall("edge"):
        edge_id = edge.attrib.get("id")
        if not edge_id or edge.attrib.get("function") == "internal":
            continue
        specs = []
        for lane in edge.findall("lane"):
            specs.append(
                {
                    "id": lane.attrib["id"],
                    "index": int(lane.attrib["index"]),
                    "length": float(lane.attrib["length"]),
                    "allow": set(lane.attrib.get("allow", "").split()),
                    "disallow": set(lane.attrib.get("disallow", "").split()),
                }
            )
        specs.sort(key=lambda spec: int(spec["index"]))
        lane_specs[edge_id] = specs
    return lane_specs


def _load_edge_lane_counts(net_file: str | Path) -> dict[str, int]:
    tree = ET.parse(net_file)
    lane_counts: dict[str, int] = {}
    for edge in tree.getroot().findall("edge"):
        edge_id = edge.attrib.get("id")
        if edge_id and edge.attrib.get("function") != "internal":
            lane_counts[edge_id] = len(edge.findall("lane"))
    return lane_counts


def _build_accident_schedule(route_file: str | Path, net_file: str | Path) -> list[AccidentEvent]:
    routes = _load_route_edges(route_file)
    edge_to_route = _route_by_first_edge(routes)
    vehicle_classes = _load_vehicle_classes(route_file)
    edge_lane_specs = _load_edge_lane_specs(net_file)
    incoming_edges = sorted(edge for edge in edge_to_route if edge in edge_lane_specs)
    candidates = _accident_candidates(incoming_edges, edge_lane_specs, vehicle_classes, edge_to_route)
    if not candidates:
        raise ValueError("No valid accident lane pairs found.")

    rng = random.Random(DEFAULT_SUMO_SEED)
    events = []
    for hour in DEFAULT_ACCIDENT_HOURS:
        candidate = rng.choice(candidates)
        event_time = hour * 3600
        vehicles = []
        for vehicle_number, lane_spec in enumerate(candidate["lanes"], start=1):
            position = _accident_position(float(lane_spec["length"]), rng)
            vehicle_type = rng.choice(candidate["vehicle_types_by_lane"][lane_spec["index"]])
            vehicles.append(
                AccidentVehicle(
                    vehicle_id=f"accident_{hour:02d}h_{vehicle_number}",
                    route_id=candidate["route_id"],
                    edge_id=candidate["edge_id"],
                    lane_index=int(lane_spec["index"]),
                    vehicle_type=vehicle_type,
                    position=position,
                )
            )
        events.append(
            AccidentEvent(
                event_id=f"accident_{hour:02d}h",
                time=event_time,
                duration=DEFAULT_ACCIDENT_DURATION,
                vehicles=tuple(vehicles),
            )
        )
    return events


def _accident_candidates(
    incoming_edges: list[str],
    edge_lane_specs: dict[str, list[dict[str, Any]]],
    vehicle_classes: dict[str, str],
    edge_to_route: dict[str, str],
) -> list[dict[str, Any]]:
    candidates = []
    for edge_id in incoming_edges:
        lanes = edge_lane_specs[edge_id]
        for first, second in zip(lanes, lanes[1:]):
            if int(second["index"]) != int(first["index"]) + 1:
                continue
            first_types = _allowed_vehicle_types(first, vehicle_classes)
            second_types = _allowed_vehicle_types(second, vehicle_classes)
            if not first_types or not second_types:
                continue
            candidates.append(
                {
                    "edge_id": edge_id,
                    "route_id": edge_to_route[edge_id],
                    "lanes": (first, second),
                    "vehicle_types_by_lane": {
                        int(first["index"]): first_types,
                        int(second["index"]): second_types,
                    },
                }
            )
    return candidates


def _allowed_vehicle_types(
    lane_spec: dict[str, Any],
    vehicle_classes: dict[str, str],
) -> list[str]:
    allow = lane_spec["allow"]
    disallow = lane_spec["disallow"]
    allowed = []
    for type_id in ACCIDENT_TYPE_PRIORITY:
        vehicle_class = vehicle_classes.get(type_id)
        if vehicle_class is None:
            continue
        if allow and vehicle_class not in allow:
            continue
        if disallow and vehicle_class in disallow:
            continue
        allowed.append(type_id)
    return allowed


def _accident_position(lane_length: float, rng: random.Random) -> float:
    max_distance = min(DEFAULT_ACCIDENT_DISTANCE_METERS, lane_length - 1.0)
    distance_to_junction = rng.uniform(1.0, max_distance)
    return round(lane_length - distance_to_junction, 2)


def _route_by_first_edge(routes: dict[str, list[str]]) -> dict[str, str]:
    edge_to_route = {}
    for route_id, edges in routes.items():
        if edges:
            edge_to_route[edges[0]] = route_id
    return edge_to_route


def _load_accident_schedule(path: Path) -> list[AccidentEvent]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    events_payload = payload.get("accident_events")
    if not isinstance(events_payload, list):
        raise ValueError(f"Missing list: {path}.accident_events")

    events = []
    for event_payload in events_payload:
        vehicles = tuple(
            AccidentVehicle(
                vehicle_id=str(vehicle["vehicle_id"]),
                route_id=str(vehicle["route_id"]),
                edge_id=str(vehicle["edge_id"]),
                lane_index=int(vehicle["lane_index"]),
                vehicle_type=str(vehicle["vehicle_type"]),
                position=float(vehicle["position"]),
            )
            for vehicle in event_payload["vehicles"]
        )
        events.append(
            AccidentEvent(
                event_id=str(event_payload["event_id"]),
                time=int(event_payload["time"]),
                duration=int(event_payload["duration"]),
                vehicles=vehicles,
            )
        )
    return events


def _save_accident_schedule(path: Path, events: list[AccidentEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": DEFAULT_SUMO_SEED,
        "description": (
            "Persistent accident spawn schedule. Delete this file to regenerate "
            "lane, position, and vehicle-type choices."
        ),
        "accident_events": [
            {
                "event_id": event.event_id,
                "time": event.time,
                "duration": event.duration,
                "vehicles": [
                    {
                        "vehicle_id": vehicle.vehicle_id,
                        "route_id": vehicle.route_id,
                        "edge_id": vehicle.edge_id,
                        "lane_index": vehicle.lane_index,
                        "vehicle_type": vehicle.vehicle_type,
                        "position": vehicle.position,
                    }
                    for vehicle in event.vehicles
                ],
            }
            for event in events
        ],
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)
        file.write("\n")


def _spawn_accident(accident: AccidentEvent, current_time: int) -> None:
    print(
        f"[ACCIDENT SPAWN] {accident.event_id} at t={current_time}, "
        f"despawn={accident.despawn_time}"
    )
    for vehicle in accident.vehicles:
        try:
            traci.vehicle.add(
                vehID=vehicle.vehicle_id,
                routeID=vehicle.route_id,
                typeID=vehicle.vehicle_type,
                depart=str(current_time),
                departLane=str(vehicle.lane_index),
                departPos=str(vehicle.position),
                departSpeed="0",
            )
            traci.vehicle.setLaneChangeMode(vehicle.vehicle_id, 0)
            traci.vehicle.setSpeed(vehicle.vehicle_id, 0.0)
            traci.vehicle.setColor(vehicle.vehicle_id, (255, 0, 0, 255))
            print(
                f"  spawned {vehicle.vehicle_id}: route={vehicle.route_id}, "
                f"edge={vehicle.edge_id}, lane={vehicle.lane_index}, "
                f"type={vehicle.vehicle_type}, pos={vehicle.position:.2f}, speed=0"
            )
        except traci.TraCIException as exc:
            print(f"Warning: failed to spawn accident vehicle {vehicle.vehicle_id}: {exc}")


def _despawn_accident(accident: AccidentEvent) -> None:
    print(f"[ACCIDENT DESPAWN] {accident.event_id} at t={accident.despawn_time}")
    for vehicle in accident.vehicles:
        try:
            traci.vehicle.remove(vehicle.vehicle_id)
            print(f"  removed {vehicle.vehicle_id}")
        except traci.TraCIException as exc:
            print(f"Warning: failed to remove accident vehicle {vehicle.vehicle_id}: {exc}")


def _record_approach_metrics(
    controller: TrafficLightController,
    log_to_console: bool,
    csv_writer: csv.DictWriter | None,
    spawn_index: dict[str, list[int]],
    density_by_time: list[dict[str, Any]],
    demand_scale: float,
) -> None:
    time = controller.traci.simulation.getTime()
    time_int = int(time)
    density_factor = _density_factor_at(time_int, density_by_time, demand_scale)
    if log_to_console:
        print(f"[APPROACH METRICS] t={time:.0f}")
    for approach_id, metrics in controller.get_approach_metrics().items():
        spawn_count_5m = _spawn_count_in_window(spawn_index, approach_id, time_int - 300, time_int)
        spawn_count_hour = _spawn_count_in_window(
            spawn_index,
            approach_id,
            (time_int // 3600) * 3600,
            time_int,
        )
        if log_to_console:
            print(
                f"  {approach_id}: "
                f"density={density_factor:.2f}, spawn5m={spawn_count_5m}, "
                f"spawn_hour={spawn_count_hour}, "
                f"flow_count_5m={metrics.flow_count_5m:.1f}, "
                f"speed={metrics.speed:.2f}, speed_min={metrics.speed_min:.2f}, "
                f"queue={metrics.queue:.1f}, queue_max={metrics.queue_max:.1f}, "
                f"waiting={metrics.waiting:.2f}, waiting_total={metrics.waiting_total:.1f}, "
                f"occupancy={metrics.occupancy:.2f}, occupancy_max={metrics.occupancy_max:.2f}, "
                f"jam_veh={metrics.jam_veh:.1f}, jam_veh_max={metrics.jam_veh_max:.1f}, "
                f"vehicle_count_total={metrics.vehicle_count_total:.1f}, "
                f"has_vehicle={metrics.has_vehicle:.0f}, has_queue={metrics.has_queue:.0f}"
            )
        if csv_writer is not None:
            csv_writer.writerow(
                {
                    "time": time_int,
                    "approach_id": approach_id,
                    "lanes": " ".join(metrics.lanes),
                    "density_factor": density_factor,
                    "spawn_count_5m": spawn_count_5m,
                    "spawn_count_hour": spawn_count_hour,
                    "flow_count_5m": metrics.flow_count_5m,
                    "detector_flow_5m": metrics.flow_count_5m,
                    "speed": metrics.speed,
                    "speed_min": metrics.speed_min,
                    "speed_lane_mean": metrics.speed_lane_mean,
                    "detector_speed": metrics.detector_speed,
                    "queue": metrics.queue,
                    "queue_max": metrics.queue_max,
                    "waiting": metrics.waiting,
                    "waiting_total": metrics.waiting_total,
                    "occupancy": metrics.occupancy,
                    "detector_occupancy": metrics.detector_occupancy,
                    "occupancy_max": metrics.occupancy_max,
                    "jam_veh": metrics.jam_veh,
                    "detector_jam_veh": metrics.detector_jam_veh,
                    "jam_veh_max": metrics.jam_veh_max,
                    "vehicle_count_total": metrics.vehicle_count_total,
                    "has_vehicle": metrics.has_vehicle,
                    "has_queue": metrics.has_queue,
                }
            )


def _build_spawn_index(events: list[SpawnEvent]) -> dict[str, list[int]]:
    max_time = max((event.time for event in events), default=0)
    counts_by_approach = {
        approach_id: [0] * (max_time + 2) for approach_id in set(ROUTE_APPROACH_MAPPING.values())
    }
    for event in events:
        approach_id = ROUTE_APPROACH_MAPPING.get(event.route_id)
        if approach_id is None:
            continue
        if approach_id not in counts_by_approach:
            counts_by_approach[approach_id] = [0] * (max_time + 2)
        counts_by_approach[approach_id][event.time + 1] += 1

    for counts in counts_by_approach.values():
        for index in range(1, len(counts)):
            counts[index] += counts[index - 1]
    return counts_by_approach


def _spawn_count_in_window(
    spawn_index: dict[str, list[int]],
    approach_id: str,
    start_time: int,
    end_time: int,
) -> int:
    if end_time <= 0 or end_time < start_time:
        return 0
    counts = spawn_index.get(approach_id)
    if not counts:
        return 0
    start_index = max(start_time, 0)
    end_index = min(end_time + 1, len(counts) - 1)
    if end_index < start_index:
        return 0
    return counts[end_index] - counts[start_index]


def _density_factor_at(
    time: int,
    density_by_time: list[dict[str, Any]],
    demand_scale: float,
) -> float:
    for interval in density_by_time:
        start_time = int(interval["start_time"])
        end_time = int(interval["end_time"])
        if start_time <= time < end_time:
            return float(interval["density"]) * demand_scale
    return 0.0


def _pin_accident(accident: AccidentEvent) -> None:
    for vehicle in accident.vehicles:
        try:
            lane_id = f"{vehicle.edge_id}_{vehicle.lane_index}"
            if vehicle.vehicle_id not in traci.vehicle.getIDList():
                continue
            if traci.vehicle.getLaneID(vehicle.vehicle_id) != lane_id:
                traci.vehicle.moveTo(vehicle.vehicle_id, lane_id, vehicle.position)
            elif abs(traci.vehicle.getLanePosition(vehicle.vehicle_id) - vehicle.position) > 0.5:
                traci.vehicle.moveTo(vehicle.vehicle_id, lane_id, vehicle.position)
            traci.vehicle.setSpeed(vehicle.vehicle_id, 0.0)
        except traci.TraCIException as exc:
            print(f"Warning: failed to pin accident vehicle {vehicle.vehicle_id}: {exc}")


def _parse_lane_index(lane_name: str) -> int:
    prefix = "lane_"
    if not lane_name.startswith(prefix):
        raise ValueError(f"Invalid lane key {lane_name!r}; expected format lane_1")
    lane_number = int(lane_name[len(prefix) :])
    if lane_number < 1:
        raise ValueError(f"Invalid lane key {lane_name!r}; lane number must be >= 1")
    return lane_number - 1


if __name__ == "__main__":
    sys.exit(main())
