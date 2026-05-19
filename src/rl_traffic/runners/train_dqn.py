from __future__ import annotations

import argparse
import sys

from src.rl_traffic.dqn import DQNAgent
from src.rl_traffic.env import SumoTrafficSignalEnv
from src.rl_traffic.metrics import append_episode_metrics
from src.rl_traffic.runners.common import add_common_args, load_runner_config, run_dir, run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN traffic-signal controller.")
    add_common_args(parser)
    parser.add_argument("--episodes", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_runner_config(args)
    episodes = args.episodes or config.training.episodes
    name = args.run_name or config.training.run_name
    env = SumoTrafficSignalEnv(config, controller_name="dqn")
    try:
        observation, _ = env.reset(seed=config.sumo.seed)
        agent = DQNAgent(
            state_dim=observation.shape[0],
            action_dim=env.action_count,
            config=config.training,
            seed=config.sumo.seed,
        )
        env.close()
        best_reward = None
        for episode in range(episodes):
            env = SumoTrafficSignalEnv(config, controller_name=agent.name)
            metrics = run_episode(env, agent, episode=episode, seed=config.sumo.seed + episode)
            append_episode_metrics(run_dir(name) / "metrics" / "episode_metrics.csv", metrics)
            agent.save(run_dir(name) / "checkpoints" / "latest.pt")
            if best_reward is None or metrics.total_reward > best_reward:
                best_reward = metrics.total_reward
                agent.save(run_dir(name) / "checkpoints" / "best.pt")
            env.close()
            print(
                f"dqn episode {episode}: reward={metrics.total_reward:.3f}, "
                f"epsilon={agent.epsilon:.3f}, replay={len(agent.replay)}"
            )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())

