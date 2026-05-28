from __future__ import annotations

import argparse
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


def load_runner_config(args: argparse.Namespace) -> TrafficRLConfig:
    return load_config(args.config)


def run_episode(
    env: SumoTrafficSignalEnv,
    controller: Any,
    episode: int,
    seed: int,
) -> EpisodeMetrics:
    observation, info = env.reset(seed=seed)
    accumulator = EpisodeMetricAccumulator(controller.name, episode, seed)
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
