from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from gymnasium.vector import AsyncVectorEnv

from src.rl_traffic.dqn import DQNAgent
from src.rl_traffic.env import SumoTrafficSignalEnv
from src.rl_traffic.metrics import EpisodeMetricAccumulator, EpisodeMetrics, append_episode_metrics
from src.rl_traffic.runners.common import (
    add_common_args,
    experiment_metadata,
    load_runner_config,
    run_dir,
    run_episode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN traffic-signal controller.")
    add_common_args(parser)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--checkpoint-window", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--validation-episodes", type=int, default=None)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel SUMO environments for DQN training.",
    )
    return parser.parse_args()


def validation_score(metrics: list[EpisodeMetrics], config) -> float:
    reward = mean(item.total_reward for item in metrics)
    wait = mean(item.mean_waiting_time for item in metrics)
    throughput = mean(item.throughput for item in metrics)
    max_queue = mean(item.max_queue for item in metrics)
    tail_queue = mean(item.tail_queue for item in metrics)
    safety_overrides = sum(item.num_safety_overrides for item in metrics)
    wait_target = config.validation_mean_wait_target or config.validation_wait_target
    throughput_target = (
        config.validation_min_throughput_target or config.validation_throughput_target
    )
    wait_penalty_weight = (
        config.validation_mean_wait_penalty_weight
        or config.validation_wait_penalty_weight
    )
    throughput_penalty_weight = (
        config.validation_min_throughput_penalty_weight
        or config.validation_throughput_penalty_weight
    )
    return (
        reward
        - wait_penalty_weight * max(0.0, wait - wait_target)
        - throughput_penalty_weight * max(0.0, throughput_target - throughput)
        - config.validation_safety_penalty_weight * safety_overrides
        - config.validation_max_queue_penalty_weight
        * max(0.0, max_queue - config.validation_max_queue_target)
        - config.validation_tail_queue_penalty_weight
        * max(0.0, tail_queue - config.validation_tail_queue_target)
    )


def capacity_score(metrics: EpisodeMetrics, config) -> float:
    return (
        metrics.total_reward
        - config.capacity_score_wasted_green_weight * metrics.wasted_green_ratio
        - config.capacity_score_event_green_weight * metrics.event_direction_green_ratio
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


def update_training_checkpoints(
    config,
    agent: DQNAgent,
    run_name: str,
    metrics: EpisodeMetrics,
    reward_window: deque[float],
    best_state: dict[str, float | None],
) -> None:
    append_episode_metrics(run_dir(run_name) / "metrics" / "episode_metrics.csv", metrics)
    agent.save(run_dir(run_name) / "checkpoints" / "latest.pt")
    reward_window.append(metrics.total_reward)
    stable_score = mean(reward_window)
    capacity_metric = capacity_score(metrics, config.training)
    is_best_train = (
        best_state["best_reward"] is None
        or metrics.total_reward > float(best_state["best_reward"])
    )
    is_best_stable = (
        best_state["best_stable_score"] is None
        or stable_score > float(best_state["best_stable_score"])
    )
    is_best_capacity = (
        best_state["best_capacity_score"] is None
        or capacity_metric > float(best_state["best_capacity_score"])
    )
    if is_best_train:
        best_state["best_reward"] = metrics.total_reward
        agent.save(run_dir(run_name) / "checkpoints" / "best.pt")
        agent.save(run_dir(run_name) / "checkpoints" / "best_train.pt")
    if is_best_stable:
        best_state["best_stable_score"] = stable_score
        agent.save(run_dir(run_name) / "checkpoints" / "best_stable.pt")
    if is_best_capacity:
        best_state["best_capacity_score"] = capacity_metric
        agent.save(run_dir(run_name) / "checkpoints" / "best_capacity.pt")
    append_checkpoint_metrics(
        run_dir(run_name) / "metrics" / "checkpoint_metrics.csv",
        {
            "episode": metrics.episode,
            "seed": metrics.seed,
            "reward": metrics.total_reward,
            "stable_score": stable_score,
            "capacity_score": capacity_metric,
            "checkpoint_window": len(reward_window),
            "best_reward": best_state["best_reward"],
            "best_stable_score": best_state["best_stable_score"],
            "best_capacity_score": best_state["best_capacity_score"],
            "validation_score": None,
            "best_validation_score": best_state["best_validation_score"],
            "is_best_train": is_best_train,
            "is_best_stable": is_best_stable,
            "is_best_capacity": is_best_capacity,
        },
    )
    print(
        f"dqn episode {metrics.episode}: reward={metrics.total_reward:.3f}, "
        f"stable={stable_score:.3f}, epsilon={agent.epsilon:.3f}, "
        f"replay={len(agent.replay)}"
    )


def run_validation(
    config,
    agent: DQNAgent,
    run_name: str,
    train_episode: int,
) -> list[EpisodeMetrics]:
    validation_agent = agent.clone_for_eval()
    validation_metrics = []
    metadata = experiment_metadata(config)
    for validation_episode in range(config.training.validation_episodes):
        env = SumoTrafficSignalEnv(config, controller_name=validation_agent.name)
        try:
            seed = config.sumo.seed + 100_000 + train_episode * 100 + validation_episode
            metrics = run_episode(
                env,
                validation_agent,
                episode=train_episode,
                seed=seed,
                metadata=metadata,
            )
            row = asdict(metrics)
            row["train_episode"] = train_episode
            row["validation_episode"] = validation_episode
            append_checkpoint_metrics(run_dir(run_name) / "metrics" / "validation_metrics.csv", row)
            validation_metrics.append(metrics)
        finally:
            env.close()
    return validation_metrics


def make_primed_env(config, controller_name: str, seed: int) -> SumoTrafficSignalEnv:
    env = SumoTrafficSignalEnv(config, controller_name=controller_name)
    env.reset(seed=seed)
    env.close()
    return env


def vector_info_at(infos: dict[str, Any], index: int) -> dict[str, Any]:
    item: dict[str, Any] = {}
    for key, value in infos.items():
        if key.startswith("_"):
            continue
        mask = infos.get(f"_{key}")
        if mask is not None:
            try:
                if not bool(mask[index]):
                    continue
            except (IndexError, TypeError):
                pass
        try:
            value_at_index = value[index]
        except (IndexError, KeyError, TypeError):
            value_at_index = value
        if isinstance(value_at_index, np.generic):
            value_at_index = value_at_index.item()
        item[key] = value_at_index
    return item


def run_vector_episode_batch(
    config,
    agent: DQNAgent,
    start_episode: int,
    batch_size: int,
    metadata: dict[str, Any],
) -> list[EpisodeMetrics]:
    seeds = [config.sumo.seed + start_episode + offset for offset in range(batch_size)]
    episodes = [start_episode + offset for offset in range(batch_size)]
    env_fns = [
        (lambda seed=seed: make_primed_env(config, agent.name, seed))
        for seed in seeds
    ]
    vector_env = AsyncVectorEnv(
        env_fns,
        shared_memory=False,
        daemon=False,
    )
    try:
        observations, infos = vector_env.reset(seed=seeds)
        accumulators = [
            EpisodeMetricAccumulator(agent.name, episode, seed)
            for episode, seed in zip(episodes, seeds, strict=True)
        ]
        for accumulator in accumulators:
            accumulator.set_metadata(metadata)
        done = np.zeros(batch_size, dtype=bool)
        while not bool(done.all()):
            actions = []
            step_infos = [vector_info_at(infos, index) for index in range(batch_size)]
            for index in range(batch_size):
                actions.append(int(agent.act(observations[index], step_infos[index])))
            next_observations, rewards, terminated, truncated, next_infos = vector_env.step(
                np.asarray(actions, dtype=np.int64)
            )
            done_now = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
            if bool(done_now.any()) and not bool(done_now.all()):
                raise RuntimeError(
                    "Vectorized SUMO episodes ended at different times; "
                    "use equal end_time/step settings or fall back to --num-envs 1."
                )
            for index in range(batch_size):
                info_i = vector_info_at(next_infos, index)
                agent.observe(
                    {
                        "state": observations[index],
                        "action": actions[index],
                        "reward": float(rewards[index]),
                        "next_state": next_observations[index],
                        "done": bool(done_now[index]),
                        "info": info_i,
                        "next_valid_action_mask": info_i.get("valid_action_mask"),
                    }
                )
                accumulators[index].update(float(rewards[index]), info_i)
            observations = next_observations
            infos = next_infos
            done = done_now
        return [accumulator.finish() for accumulator in accumulators]
    finally:
        vector_env.close()


def run_serial_training(
    config,
    agent: DQNAgent,
    episodes: int,
    run_name: str,
    metadata: dict[str, Any],
) -> None:
    best_state: dict[str, float | None] = {
        "best_reward": None,
        "best_stable_score": None,
        "best_capacity_score": None,
        "best_validation_score": None,
    }
    reward_window: deque[float] = deque(maxlen=max(1, config.training.checkpoint_window))
    for episode in range(episodes):
        env = SumoTrafficSignalEnv(config, controller_name=agent.name)
        try:
            metrics = run_episode(
                env,
                agent,
                episode=episode,
                seed=config.sumo.seed + episode,
                metadata=metadata,
            )
        finally:
            env.close()
        update_training_checkpoints(
            config,
            agent,
            run_name,
            metrics,
            reward_window,
            best_state,
        )
        if (
            config.training.validation_interval > 0
            and config.training.validation_episodes > 0
            and (episode + 1) % config.training.validation_interval == 0
        ):
            validation_metrics = run_validation(config, agent, run_name, episode)
            validation_metric = validation_score(validation_metrics, config.training)
            if (
                best_state["best_validation_score"] is None
                or validation_metric > float(best_state["best_validation_score"])
            ):
                best_state["best_validation_score"] = validation_metric
                agent.save(run_dir(run_name) / "checkpoints" / "best_validation.pt")


def run_vector_training(
    config,
    agent: DQNAgent,
    episodes: int,
    run_name: str,
    metadata: dict[str, Any],
    num_envs: int,
) -> None:
    best_state: dict[str, float | None] = {
        "best_reward": None,
        "best_stable_score": None,
        "best_capacity_score": None,
        "best_validation_score": None,
    }
    reward_window: deque[float] = deque(maxlen=max(1, config.training.checkpoint_window))
    episode = 0
    while episode < episodes:
        batch_size = min(num_envs, episodes - episode)
        batch_metrics = run_vector_episode_batch(
            config,
            agent,
            start_episode=episode,
            batch_size=batch_size,
            metadata=metadata,
        )
        for metrics in sorted(batch_metrics, key=lambda item: item.episode):
            update_training_checkpoints(
                config,
                agent,
                run_name,
                metrics,
                reward_window,
                best_state,
            )
        episode += batch_size


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
    num_envs = max(1, int(args.num_envs))
    name = args.run_name or config.training.run_name
    metadata = experiment_metadata(config)
    env = SumoTrafficSignalEnv(config, controller_name="dqn")
    try:
        observation, _ = env.reset(seed=config.sumo.seed)
        agent = DQNAgent(
            state_dim=observation.shape[0],
            action_dim=env.action_count,
            config=config.training,
            seed=config.sumo.seed,
        )
        print(
            f"train config: variant={metadata['experiment_variant']}, "
            f"obs_dim={observation.shape[0]}, "
            f"include_capacity={metadata['include_capacity']}, "
            f"include_event_features={metadata['include_event_features']}, "
            f"use_effective_metrics={metadata['use_effective_metrics']}, "
            f"num_envs={num_envs}"
        )
        env.close()
        if num_envs == 1:
            run_serial_training(config, agent, episodes, name, metadata)
        else:
            run_vector_training(config, agent, episodes, name, metadata, num_envs)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
