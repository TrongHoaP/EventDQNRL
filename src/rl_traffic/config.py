from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SumoConfig:
    sumocfg: str = "scenarios/grid/grid.sumocfg"
    spawn_config: str = "src/configs/simulation_density_spawn_config.json"
    accident_config: str = "src/configs/accident_spawn_events.json"
    gui: bool = False
    seed: int = 7
    step_length: float = 1.0
    end_time: int = 3600
    demand_scale: float = 0.2
    disable_accidents: bool = True
    randomize_demand: bool = False
    demand_time_jitter_seconds: int = 0


@dataclass(frozen=True)
class ControlConfig:
    decision_interval_seconds: int = 10
    yellow_time_seconds: float = 3.0
    min_green_seconds: float = 10.0
    max_green_seconds: float = 60.0


@dataclass(frozen=True)
class SafetyConfig:
    enabled: bool = True
    invalid_action_policy: str = "hold"
    enforce_min_green: bool = True
    enforce_max_green: bool = True
    enforce_green_phase_only: bool = True
    override_penalty: float = 0.05


@dataclass(frozen=True)
class StateConfig:
    mode: str = "lane_level"


@dataclass(frozen=True)
class RewardConfig:
    name: str = "queue_wait_switch"
    queue_weight: float = 1.0
    wait_weight: float = 0.01
    switch_weight: float = 0.1
    safety_weight: float = 0.05
    tail_queue_weight: float = 0.0
    queue_spike_threshold: float = 0.0
    queue_spike_weight: float = 0.0
    tail_queue_spike_threshold: float = 0.0
    tail_queue_spike_weight: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 5
    batch_size: int = 64
    learning_rate: float = 0.001
    gamma: float = 0.99
    replay_size: int = 20000
    min_replay_size: int = 256
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    target_update_steps: int = 500
    hidden_size: int = 128
    device: str = "auto"
    run_name: str = "dqn"
    eval_epsilon: float = 0.0
    checkpoint_window: int = 1
    validation_interval: int = 0
    validation_episodes: int = 0
    validation_wait_target: float = 4.46
    validation_throughput_target: float = 1100.0
    validation_wait_penalty_weight: float = 20.0
    validation_throughput_penalty_weight: float = 0.2
    validation_safety_penalty_weight: float = 20.0
    validation_max_queue_target: float = 10.65
    validation_tail_queue_target: float = 4.0
    validation_max_queue_penalty_weight: float = 20.0
    validation_tail_queue_penalty_weight: float = 10.0


@dataclass(frozen=True)
class EvaluationConfig:
    controllers: tuple[str, ...] = (
        "fixed_time",
        "actuated",
        "max_pressure",
        "random",
        "dqn",
    )
    episodes: int = 1
    fixed_cycle_seconds: int = 42
    actuated_queue_threshold: float = 4.0


@dataclass(frozen=True)
class TrafficRLConfig:
    sumo: SumoConfig = field(default_factory=SumoConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    state: StateConfig = field(default_factory=StateConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def load_config(path: str | Path) -> TrafficRLConfig:
    config_path = repo_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("RL config must be a JSON object.")
    return TrafficRLConfig(
        sumo=_build_dataclass(SumoConfig, payload.get("sumo", {})),
        control=_build_dataclass(ControlConfig, payload.get("control", {})),
        safety=_build_dataclass(SafetyConfig, payload.get("safety", {})),
        state=_build_dataclass(StateConfig, payload.get("state", {})),
        reward=_build_dataclass(RewardConfig, payload.get("reward", {})),
        training=_build_dataclass(TrainingConfig, payload.get("training", {})),
        evaluation=_build_dataclass(EvaluationConfig, payload.get("evaluation", {})),
    )


def _build_dataclass(cls: type[Any], values: Any) -> Any:
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} config must be an object.")
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {unknown}")
    return cls(**values)
