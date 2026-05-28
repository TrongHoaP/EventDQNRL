from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.rl_traffic.config import TrafficRLConfig, load_config
from src.rl_traffic.env import SumoTrafficSignalEnv
from src.rl_traffic.metrics import EpisodeMetricAccumulator, EpisodeMetrics


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="src/configs/rl_traffic_control.json",
        help="Path to RL traffic-control config JSON.",
    )
    parser.add_argument("--run-name", default=None, help="Override run output directory name.")
    parser.add_argument("--seed", type=int, default=None, help="Override SUMO/training seed.")


def load_runner_config(args: argparse.Namespace) -> TrafficRLConfig:
    config = load_config(args.config)
    if getattr(args, "seed", None) is not None:
        config = replace(config, sumo=replace(config.sumo, seed=int(args.seed)))
    return config


def run_episode(
    env: SumoTrafficSignalEnv,
    controller: Any,
    episode: int,
    seed: int,
    metadata: dict[str, Any] | None = None,
) -> EpisodeMetrics:
    observation, info = env.reset(seed=seed)
    accumulator = EpisodeMetricAccumulator(controller.name, episode, seed)
    if metadata:
        accumulator.set_metadata(metadata)
    terminated = False
    truncated = False
    while not terminated and not truncated:
        state = observation
        action = int(controller.act(observation, info))
        observation, reward, terminated, truncated, next_info = env.step(action)
        controller.observe(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": observation,
                "done": terminated or truncated,
                "info": next_info,
                "next_valid_action_mask": next_info.get("valid_action_mask"),
            }
        )
        accumulator.update(reward, next_info)
        info = next_info
    return accumulator.finish()


def run_dir(run_name: str) -> Path:
    return Path("runs") / run_name


def experiment_metadata(config: TrafficRLConfig) -> dict[str, Any]:
    variant = config.training.experiment_variant or config.training.run_name
    return {
        "experiment_variant": variant,
        "include_capacity": config.state.include_capacity,
        "include_event_features": config.state.include_event_features,
        "use_effective_metrics": config.reward.use_effective_metrics,
    }
