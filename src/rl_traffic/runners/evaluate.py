from __future__ import annotations

import argparse
import sys

from src.rl_traffic.controllers import ControllerFactoryConfig, build_controller
from src.rl_traffic.dqn import DQNAgent
from src.rl_traffic.env import SumoTrafficSignalEnv
from src.rl_traffic.metrics import append_episode_metrics, write_summary
from src.rl_traffic.runners.common import add_common_args, load_runner_config, run_dir, run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baselines and optional DQN checkpoint.")
    add_common_args(parser)
    parser.add_argument("--checkpoint", default=None, help="DQN checkpoint path.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override evaluation episodes per controller.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_runner_config(args)
    name = args.run_name or "evaluation"
    episodes = args.episodes or config.evaluation.episodes
    output = run_dir(name)
    summaries = []
    probe_env = SumoTrafficSignalEnv(config)
    try:
        observation, _ = probe_env.reset(seed=config.sumo.seed)
        factory_config = ControllerFactoryConfig(
            action_size=probe_env.action_count,
            seed=config.sumo.seed,
            fixed_cycle_seconds=config.evaluation.fixed_cycle_seconds,
            actuated_queue_threshold=config.evaluation.actuated_queue_threshold,
        )
        action_size = probe_env.action_count
        state_dim = observation.shape[0]
    finally:
        probe_env.close()

    for controller_name in config.evaluation.controllers:
        if controller_name == "dqn":
            if args.checkpoint is None:
                continue
            controller = DQNAgent.load(
                args.checkpoint,
                config.training,
                seed=config.sumo.seed,
                trainable=False,
            )
        else:
            controller = build_controller(controller_name, factory_config)
        for episode in range(episodes):
            env = SumoTrafficSignalEnv(config, controller_name=controller.name)
            metrics = run_episode(env, controller, episode=episode, seed=config.sumo.seed + episode)
            append_episode_metrics(output / "metrics" / "episode_metrics.csv", metrics)
            summaries.append(metrics)
            env.close()
            print(f"{controller.name} episode {episode}: reward={metrics.total_reward:.3f}")
    write_summary(output / "metrics" / "evaluation_summary.json", summaries)
    print(f"Evaluation complete: state_dim={state_dim}, actions={action_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
