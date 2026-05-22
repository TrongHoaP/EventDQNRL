from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean

from src.rl_traffic.dqn import DQNAgent
from src.rl_traffic.env import SumoTrafficSignalEnv
from src.rl_traffic.metrics import EpisodeMetrics, append_episode_metrics
from src.rl_traffic.runners.common import add_common_args, load_runner_config, run_dir, run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN traffic-signal controller.")
    add_common_args(parser)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--checkpoint-window", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--validation-episodes", type=int, default=None)
    return parser.parse_args()


def validation_score(metrics: list[EpisodeMetrics], config) -> float:
    reward = mean(item.total_reward for item in metrics)
    wait = mean(item.mean_waiting_time for item in metrics)
    throughput = mean(item.throughput for item in metrics)
    max_queue = mean(item.max_queue for item in metrics)
    tail_queue = mean(item.tail_queue for item in metrics)
    safety_overrides = sum(item.num_safety_overrides for item in metrics)
    return (
        reward
        - config.validation_wait_penalty_weight * max(0.0, wait - config.validation_wait_target)
        - config.validation_throughput_penalty_weight
        * max(0.0, config.validation_throughput_target - throughput)
        - config.validation_safety_penalty_weight * safety_overrides
        - config.validation_max_queue_penalty_weight
        * max(0.0, max_queue - config.validation_max_queue_target)
        - config.validation_tail_queue_penalty_weight
        * max(0.0, tail_queue - config.validation_tail_queue_target)
    )


def append_checkpoint_metrics(path: str | Path, row: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists()
    with output.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_validation(
    config,
    agent: DQNAgent,
    run_name: str,
    train_episode: int,
) -> list[EpisodeMetrics]:
    validation_agent = agent.clone_for_eval()
    validation_metrics = []
    for validation_episode in range(config.training.validation_episodes):
        env = SumoTrafficSignalEnv(config, controller_name=validation_agent.name)
        try:
            seed = config.sumo.seed + 100_000 + train_episode * 100 + validation_episode
            metrics = run_episode(
                env,
                validation_agent,
                episode=train_episode,
                seed=seed,
            )
            row = asdict(metrics)
            row["train_episode"] = train_episode
            row["validation_episode"] = validation_episode
            append_checkpoint_metrics(run_dir(run_name) / "metrics" / "validation_metrics.csv", row)
            validation_metrics.append(metrics)
        finally:
            env.close()
    return validation_metrics


def main() -> int:
    args = parse_args()
    config = load_runner_config(args)
    training_overrides = {}
    if args.checkpoint_window is not None:
        training_overrides["checkpoint_window"] = args.checkpoint_window
    if args.validation_interval is not None:
        training_overrides["validation_interval"] = args.validation_interval
    if args.validation_episodes is not None:
        training_overrides["validation_episodes"] = args.validation_episodes
    if training_overrides:
        config = replace(config, training=replace(config.training, **training_overrides))
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
        best_stable_score = None
        best_validation_score = None
        reward_window: deque[float] = deque(maxlen=max(1, config.training.checkpoint_window))
        for episode in range(episodes):
            env = SumoTrafficSignalEnv(config, controller_name=agent.name)
            try:
                metrics = run_episode(env, agent, episode=episode, seed=config.sumo.seed + episode)
            finally:
                env.close()
            append_episode_metrics(run_dir(name) / "metrics" / "episode_metrics.csv", metrics)
            agent.save(run_dir(name) / "checkpoints" / "latest.pt")
            reward_window.append(metrics.total_reward)
            stable_score = mean(reward_window)
            is_best_train = best_reward is None or metrics.total_reward > best_reward
            is_best_stable = best_stable_score is None or stable_score > best_stable_score
            if best_reward is None or metrics.total_reward > best_reward:
                best_reward = metrics.total_reward
                agent.save(run_dir(name) / "checkpoints" / "best.pt")
                agent.save(run_dir(name) / "checkpoints" / "best_train.pt")
            if is_best_stable:
                best_stable_score = stable_score
                agent.save(run_dir(name) / "checkpoints" / "best_stable.pt")
            validation_metric = None
            if (
                config.training.validation_interval > 0
                and config.training.validation_episodes > 0
                and (episode + 1) % config.training.validation_interval == 0
            ):
                validation_metrics = run_validation(config, agent, name, episode)
                validation_metric = validation_score(validation_metrics, config.training)
                if best_validation_score is None or validation_metric > best_validation_score:
                    best_validation_score = validation_metric
                    agent.save(run_dir(name) / "checkpoints" / "best_validation.pt")
            append_checkpoint_metrics(
                run_dir(name) / "metrics" / "checkpoint_metrics.csv",
                {
                    "episode": episode,
                    "seed": config.sumo.seed + episode,
                    "reward": metrics.total_reward,
                    "stable_score": stable_score,
                    "checkpoint_window": len(reward_window),
                    "best_reward": best_reward,
                    "best_stable_score": best_stable_score,
                    "validation_score": validation_metric,
                    "best_validation_score": best_validation_score,
                    "is_best_train": is_best_train,
                    "is_best_stable": is_best_stable,
                },
            )
            print(
                f"dqn episode {episode}: reward={metrics.total_reward:.3f}, "
                f"stable={stable_score:.3f}, epsilon={agent.epsilon:.3f}, "
                f"replay={len(agent.replay)}"
            )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
