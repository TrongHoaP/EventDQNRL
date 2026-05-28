from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASELINES = ("fixed_time", "actuated", "max_pressure", "random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DQN vs EventDQN comparison pipeline.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27, 37, 47])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--out", default="reports/comparison_dqn_vs_event_dqn.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for seed in args.seeds:
        _train_variant(
            config="src/configs/dqn_no_event.json",
            run_name=f"dqn_no_event_seed{seed}",
            seed=seed,
            episodes=args.episodes,
            skip_existing=args.skip_existing,
        )
        _train_variant(
            config="src/configs/event_dqn.json",
            run_name=f"event_dqn_seed{seed}",
            seed=seed,
            episodes=args.episodes,
            skip_existing=args.skip_existing,
        )
        _evaluate_variant(
            config="src/configs/dqn_no_event.json",
            checkpoint=f"runs/dqn_no_event_seed{seed}/checkpoints/{args.checkpoint}",
            run_name=f"eval_dqn_no_event_seed{seed}",
            seed=seed,
            episodes=args.eval_episodes,
            skip_existing=args.skip_existing,
        )
        _evaluate_variant(
            config="src/configs/event_dqn.json",
            checkpoint=f"runs/event_dqn_seed{seed}/checkpoints/{args.checkpoint}",
            run_name=f"eval_event_dqn_seed{seed}",
            seed=seed,
            episodes=args.eval_episodes,
            skip_existing=args.skip_existing,
        )
        if args.include_baselines:
            for baseline in BASELINES:
                _run_baseline(
                    controller=baseline,
                    run_name=f"eval_{baseline}_seed{seed}",
                    seed=seed,
                    episodes=args.eval_episodes,
                    skip_existing=args.skip_existing,
                )

    _run(
        [
            sys.executable,
            "-m",
            "src.rl_traffic.runners.compare_experiments",
            "--runs-dir",
            "runs",
            "--pattern",
            "*seed*",
            "--out",
            args.out,
        ]
    )
    return 0


def _train_variant(
    config: str,
    run_name: str,
    seed: int,
    episodes: int,
    skip_existing: bool,
) -> None:
    if skip_existing and _has_episode_metrics(run_name):
        print(f"skip train {run_name}")
        return
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
    skip_existing: bool,
) -> None:
    if skip_existing and _has_episode_metrics(run_name):
        print(f"skip eval {run_name}")
        return
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
    controller: str,
    run_name: str,
    seed: int,
    episodes: int,
    skip_existing: bool,
) -> None:
    if skip_existing and _has_episode_metrics(run_name):
        print(f"skip baseline {run_name}")
        return
    _run(
        [
            sys.executable,
            "-m",
            "src.rl_traffic.runners.run_baseline",
            "--config",
            "src/configs/dqn_no_event.json",
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


def _has_episode_metrics(run_name: str) -> bool:
    return (Path("runs") / run_name / "metrics" / "episode_metrics.csv").exists()


def _run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
