from __future__ import annotations

import argparse
import sys

from src.rl_traffic.controllers import ControllerFactoryConfig, build_controller
from src.rl_traffic.env import SumoTrafficSignalEnv
from src.rl_traffic.metrics import append_episode_metrics
from src.rl_traffic.runners.common import add_common_args, load_runner_config, run_dir, run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one non-learning baseline controller.")
    add_common_args(parser)
    parser.add_argument(
        "--controller",
        required=True,
        choices=("fixed_time", "actuated", "max_pressure", "random"),
    )
    parser.add_argument("--episodes", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_runner_config(args)
    name = args.run_name or args.controller
    env = SumoTrafficSignalEnv(config, controller_name=args.controller)
    try:
        env.reset(seed=config.sumo.seed)
        factory_config = ControllerFactoryConfig(
            action_size=env.action_count,
            seed=config.sumo.seed,
            fixed_cycle_seconds=config.evaluation.fixed_cycle_seconds,
            actuated_queue_threshold=config.evaluation.actuated_queue_threshold,
        )
        env.close()
        controller = build_controller(args.controller, factory_config)
        for episode in range(args.episodes):
            env = SumoTrafficSignalEnv(config, controller_name=controller.name)
            metrics = run_episode(env, controller, episode=episode, seed=config.sumo.seed + episode)
            append_episode_metrics(run_dir(name) / "metrics" / "episode_metrics.csv", metrics)
            env.close()
            print(f"{controller.name} episode {episode}: reward={metrics.total_reward:.3f}, queue={metrics.mean_queue:.3f}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())

