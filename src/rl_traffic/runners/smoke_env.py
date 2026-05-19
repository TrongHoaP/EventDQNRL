from __future__ import annotations

import argparse
import sys

from src.rl_traffic.controllers import RandomAgent
from src.rl_traffic.env import SumoTrafficSignalEnv
from src.rl_traffic.metrics import append_episode_metrics
from src.rl_traffic.runners.common import add_common_args, load_runner_config, run_dir, run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the SUMO Gymnasium environment.")
    add_common_args(parser)
    parser.add_argument("--episode-seconds", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_runner_config(args)
    config = config.__class__(
        sumo=config.sumo.__class__(**{**config.sumo.__dict__, "end_time": args.episode_seconds}),
        control=config.control,
        safety=config.safety,
        state=config.state,
        reward=config.reward,
        training=config.training,
        evaluation=config.evaluation,
    )
    name = args.run_name or "smoke-env"
    env = SumoTrafficSignalEnv(config, controller_name="random")
    try:
        observation, info = env.reset(seed=config.sumo.seed)
        agent = RandomAgent(env.action_count, seed=config.sumo.seed)
        env.close()
        env = SumoTrafficSignalEnv(config, controller_name=agent.name)
        metrics = run_episode(env, agent, episode=0, seed=config.sumo.seed)
        append_episode_metrics(run_dir(name) / "metrics" / "episode_metrics.csv", metrics)
        print(f"Smoke OK: obs_dim={observation.shape[0]}, actions={agent.action_size}, reward={metrics.total_reward:.3f}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())

