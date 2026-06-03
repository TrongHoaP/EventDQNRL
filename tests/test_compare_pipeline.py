from __future__ import annotations

import sys
import threading
import time
import unittest
from argparse import Namespace
from unittest.mock import patch

from src.rl_traffic.runners import compare_pipeline


def pipeline_args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "run_group": "test-group",
        "seeds": ["7", "17", "27", "37", "47"],
        "train_episodes": 10,
        "eval_episodes": 2,
        "dqn_config": "src/configs/dqn_no_event.json",
        "event_dqn_config": "src/configs/event_dqn.json",
        "baseline_config": "src/configs/dqn_no_event.json",
        "include_baselines": False,
        "skip_train": False,
        "skip_eval": False,
        "skip_baselines": False,
        "checkpoint_name": "best.pt",
        "num_envs": 5,
        "seed_workers": 5,
        "variant_workers": 2,
        "resume": True,
    }
    values.update(overrides)
    return Namespace(**values)


class ComparePipelineTests(unittest.TestCase):
    def test_parse_args_defaults_to_five_seed_workers_and_five_envs(self) -> None:
        with patch.object(sys, "argv", ["compare_pipeline", "--run-group", "test-group"]):
            args = compare_pipeline.parse_args()

        self.assertEqual(args.seeds, ["7", "17", "27", "37", "47"])
        self.assertEqual(args.seed_workers, 5)
        self.assertEqual(args.variant_workers, 2)
        self.assertEqual(args.num_envs, 5)

    def test_validate_parallel_args_rejects_invalid_values_and_duplicate_seeds(self) -> None:
        for name in ("num_envs", "seed_workers", "variant_workers"):
            with self.subTest(name=name):
                args = pipeline_args(**{name: 0})
                with self.assertRaisesRegex(ValueError, name.replace("_", "-")):
                    compare_pipeline._validate_parallel_args(args, [7, 17])

        with self.assertRaisesRegex(ValueError, "duplicates"):
            compare_pipeline._validate_parallel_args(pipeline_args(), [7, 17, 7])

    def test_parallel_training_runs_two_variants_for_each_seed_concurrently(self) -> None:
        args = pipeline_args(seed_workers=2, variant_workers=2, num_envs=5)
        barrier = threading.Barrier(4)
        lock = threading.Lock()
        calls: list[dict[str, object]] = []
        active_by_seed: dict[int, int] = {}
        max_active_tasks = 0
        max_active_seeds = 0
        max_variants_per_seed = 0

        def fake_train_variant(**kwargs: object) -> None:
            nonlocal max_active_tasks, max_active_seeds, max_variants_per_seed
            seed = int(kwargs["seed"])
            with lock:
                calls.append(kwargs)
                active_by_seed[seed] = active_by_seed.get(seed, 0) + 1
                active_tasks = sum(active_by_seed.values())
                active_seeds = sum(value > 0 for value in active_by_seed.values())
                max_active_tasks = max(max_active_tasks, active_tasks)
                max_active_seeds = max(max_active_seeds, active_seeds)
                max_variants_per_seed = max(
                    max_variants_per_seed,
                    active_by_seed[seed],
                )
            barrier.wait(timeout=3)
            time.sleep(0.01)
            with lock:
                active_by_seed[seed] -= 1

        with patch.object(compare_pipeline, "_train_variant", side_effect=fake_train_variant):
            compare_pipeline._run_parallel_training(args, [7, 17], "test-group")

        self.assertEqual(len(calls), 4)
        self.assertEqual(max_active_tasks, 4)
        self.assertEqual(max_active_seeds, 2)
        self.assertEqual(max_variants_per_seed, 2)
        self.assertTrue(all(call["num_envs"] == 5 for call in calls))

    def test_parallel_training_reports_all_variant_failures(self) -> None:
        args = pipeline_args(seed_workers=2, variant_workers=2)
        calls: list[str] = []

        def fake_train_variant(**kwargs: object) -> None:
            run_name = str(kwargs["run_name"])
            calls.append(run_name)
            if "event_dqn" in run_name:
                raise RuntimeError("train failed")

        with patch.object(compare_pipeline, "_train_variant", side_effect=fake_train_variant):
            with self.assertRaisesRegex(RuntimeError, "seed=7 variant=event_dqn") as error:
                compare_pipeline._run_parallel_training(args, [7, 17], "test-group")

        self.assertIn("seed=17 variant=event_dqn", str(error.exception))
        self.assertEqual(len(calls), 4)

    def test_train_failure_prevents_eval_and_report(self) -> None:
        args = pipeline_args()
        with (
            patch.object(compare_pipeline, "parse_args", return_value=args),
            patch.object(compare_pipeline.Path, "mkdir"),
            patch.object(
                compare_pipeline,
                "_run_parallel_training",
                side_effect=RuntimeError("train failed"),
            ),
            patch.object(compare_pipeline, "_evaluate_variant") as evaluate,
            patch.object(compare_pipeline, "_run") as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "train failed"):
                compare_pipeline.main()

        evaluate.assert_not_called()
        run.assert_not_called()

    def test_train_variant_passes_num_envs_to_child_command(self) -> None:
        with (
            patch.object(compare_pipeline, "_completed_run", return_value=False),
            patch.object(compare_pipeline, "_archive_incomplete_run"),
            patch.object(compare_pipeline, "_run") as run,
        ):
            compare_pipeline._train_variant(
                config="config.json",
                run_name="test-group/train/dqn_no_event_seed7",
                seed=7,
                episodes=10,
                resume=True,
                checkpoint_name="best.pt",
                num_envs=5,
            )

        command = run.call_args.args[0]
        self.assertEqual(command[-2:], ["--num-envs", "5"])

    def test_train_variant_resume_skips_completed_run_with_checkpoint(self) -> None:
        with (
            patch.object(compare_pipeline, "_completed_run", return_value=True),
            patch.object(compare_pipeline.Path, "exists", return_value=True),
            patch.object(compare_pipeline, "_archive_incomplete_run") as archive,
            patch.object(compare_pipeline, "_run") as run,
        ):
            compare_pipeline._train_variant(
                config="config.json",
                run_name="test-group/train/dqn_no_event_seed7",
                seed=7,
                episodes=10,
                resume=True,
                checkpoint_name="best.pt",
                num_envs=5,
            )

        archive.assert_not_called()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
