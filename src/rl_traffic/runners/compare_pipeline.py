from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


BASELINES = ("fixed_time", "actuated", "max_pressure", "random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EventDQN comparison pipeline.")
    parser.add_argument("--run-group", required=True)
    parser.add_argument("--seeds", nargs="+", default=["7", "17", "27", "37", "47"])
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--dqn-config", default="src/configs/dqn_no_event.json")
    parser.add_argument("--event-dqn-config", default="src/configs/event_dqn.json")
    parser.add_argument("--baseline-config", default="src/configs/dqn_no_event.json")
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip completed runs and archive incomplete run directories before retrying.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    run_group = args.run_group.strip("/\\")
    if not run_group:
        raise ValueError("--run-group must not be empty.")

    run_root = Path("runs") / run_group
    (run_root / "train").mkdir(parents=True, exist_ok=True)
    (run_root / "eval").mkdir(parents=True, exist_ok=True)
    (run_root / "baselines").mkdir(parents=True, exist_ok=True)
    (run_root / "reports").mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        dqn_train_name = f"{run_group}/train/dqn_no_event_seed{seed}"
        event_train_name = f"{run_group}/train/event_dqn_seed{seed}"

        if not args.skip_train:
            _train_variant(
                config=args.dqn_config,
                run_name=dqn_train_name,
                seed=seed,
                episodes=args.train_episodes,
                resume=args.resume,
                checkpoint_name=args.checkpoint_name,
            )
            _train_variant(
                config=args.event_dqn_config,
                run_name=event_train_name,
                seed=seed,
                episodes=args.train_episodes,
                resume=args.resume,
                checkpoint_name=args.checkpoint_name,
            )

        if not args.skip_eval:
            _evaluate_variant(
                config=args.dqn_config,
                checkpoint=str(Path("runs") / dqn_train_name / "checkpoints" / args.checkpoint_name),
                run_name=f"{run_group}/eval/eval_dqn_no_event_seed{seed}",
                seed=seed,
                episodes=args.eval_episodes,
                resume=args.resume,
            )
            _evaluate_variant(
                config=args.event_dqn_config,
                checkpoint=str(Path("runs") / event_train_name / "checkpoints" / args.checkpoint_name),
                run_name=f"{run_group}/eval/eval_event_dqn_seed{seed}",
                seed=seed,
                episodes=args.eval_episodes,
                resume=args.resume,
            )

        if args.include_baselines and not args.skip_baselines:
            for baseline in BASELINES:
                _run_baseline(
                    config=args.baseline_config,
                    controller=baseline,
                    run_name=f"{run_group}/baselines/{baseline}_seed{seed}",
                    seed=seed,
                    episodes=args.eval_episodes,
                    resume=args.resume,
                )

    _run(
        [
            sys.executable,
            "-m",
            "src.rl_traffic.runners.compare_experiments",
            "--runs-dir",
            str(run_root),
            "--out-dir",
            str(run_root / "reports"),
        ]
    )
    return 0


def _parse_seeds(raw_seeds: list[str]) -> list[int]:
    seeds: list[int] = []
    for item in raw_seeds:
        for part in item.split(","):
            part = part.strip()
            if part:
                seeds.append(int(part))
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed.")
    return seeds


def _train_variant(
    config: str,
    run_name: str,
    seed: int,
    episodes: int,
    resume: bool,
    checkpoint_name: str,
) -> None:
    if resume and _completed_run(run_name, episodes):
        checkpoint = Path("runs") / run_name / "checkpoints" / checkpoint_name
        if checkpoint.exists():
            print(f"skip completed train {run_name}", flush=True)
            return
    if resume:
        _archive_incomplete_run(run_name, episodes)
    _run(
        [
            sys.executable,
            "-m",
            "src.rl_traffic.runners.train_dqn",
            "--config",
            config,
            "--episodes",
            str(episodes),
            "--seed",
            str(seed),
            "--run-name",
            run_name,
        ]
    )


def _evaluate_variant(
    config: str,
    checkpoint: str,
    run_name: str,
    seed: int,
    episodes: int,
    resume: bool,
) -> None:
    if resume and _completed_run(run_name, episodes):
        print(f"skip completed eval {run_name}", flush=True)
        return
    if resume:
        _archive_incomplete_run(run_name, episodes)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for evaluation: {checkpoint_path}")
    _run(
        [
            sys.executable,
            "-m",
            "src.rl_traffic.runners.evaluate",
            "--config",
            config,
            "--checkpoint",
            checkpoint,
            "--episodes",
            str(episodes),
            "--seed",
            str(seed),
            "--run-name",
            run_name,
        ]
    )


def _run_baseline(
    config: str,
    controller: str,
    run_name: str,
    seed: int,
    episodes: int,
    resume: bool,
) -> None:
    if resume and _completed_run(run_name, episodes):
        print(f"skip completed baseline {run_name}", flush=True)
        return
    if resume:
        _archive_incomplete_run(run_name, episodes)
    _run(
        [
            sys.executable,
            "-m",
            "src.rl_traffic.runners.run_baseline",
            "--config",
            config,
            "--controller",
            controller,
            "--episodes",
            str(episodes),
            "--seed",
            str(seed),
            "--run-name",
            run_name,
        ]
    )


def _run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def _completed_run(run_name: str, expected_episodes: int) -> bool:
    metrics_path = Path("runs") / run_name / "metrics" / "episode_metrics.csv"
    return _episode_row_count(metrics_path) >= expected_episodes


def _archive_incomplete_run(run_name: str, expected_episodes: int) -> None:
    run_path = Path("runs") / run_name
    metrics_path = run_path / "metrics" / "episode_metrics.csv"
    row_count = _episode_row_count(metrics_path)
    if row_count <= 0 or row_count >= expected_episodes:
        return
    archive_path = run_path.with_name(
        f"{run_path.name}_incomplete_{int(time.time())}_{row_count}eps"
    )
    print(
        f"archive incomplete run {run_name}: {row_count}/{expected_episodes} episodes -> {archive_path}",
        flush=True,
    )
    run_path.rename(archive_path)


def _episode_row_count(metrics_path: Path) -> int:
    if not metrics_path.exists():
        return 0
    with metrics_path.open("r", encoding="utf-8", newline="") as file:
        line_count = sum(1 for _ in file)
    return max(0, line_count - 1)


if __name__ == "__main__":
    raise SystemExit(main())
