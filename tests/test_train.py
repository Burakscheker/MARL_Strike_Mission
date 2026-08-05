import math
import tempfile
import unittest
from pathlib import Path

import torch

from agents.ppo import HAPPOTrainer, MAPPOTrainer
from config import OBS_DIM
from env.strike_env import StrikeMissionEnv
from eval.evaluate import evaluate, evaluate_checkpoint
from train import collect_rollouts, train


class TestRolloutCollection(unittest.TestCase):
    def test_two_short_episodes_produce_finite_joint_batch(self):
        env = StrikeMissionEnv(max_steps=8)
        trainer = MAPPOTrainer(seed=0)

        batch, episodes = collect_rollouts(env, trainer, episode_count=2, seed=10)

        self.assertEqual(batch.obs.shape, (16, 2, OBS_DIM))
        self.assertEqual(batch.rewards.shape, (16,))
        self.assertEqual(len(episodes), 2)
        self.assertTrue(torch.isfinite(batch.advantages).all())
        self.assertTrue(torch.isfinite(batch.returns).all())
        self.assertTrue(all(episode["truncated"] for episode in episodes))

    def test_short_rollout_updates_both_algorithms_with_finite_losses(self):
        for trainer_class in (MAPPOTrainer, HAPPOTrainer):
            trainer = trainer_class(seed=1)
            batch, _ = collect_rollouts(
                StrikeMissionEnv(max_steps=4), trainer, episode_count=2, seed=20
            )

            metrics = trainer.update(batch)

            self.assertTrue(math.isfinite(metrics["actor_loss"]))
            self.assertTrue(math.isfinite(metrics["critic_loss"]))


class TestTrainingRun(unittest.TestCase):
    def test_training_writes_checkpoint_config_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            result = train(
                algorithm="mappo",
                episodes=2,
                rollout_episodes=1,
                seed=0,
                device="cpu",
                output_dir=output,
                max_steps=4,
            )

            self.assertEqual(result["episodes"], 2)
            self.assertTrue((output / "checkpoint.pt").is_file())
            self.assertTrue((output / "config.json").is_file())
            self.assertTrue((output / "metrics.csv").is_file())
            self.assertGreater((output / "metrics.csv").stat().st_size, 0)


class TestEvaluation(unittest.TestCase):
    def test_deterministic_evaluation_reports_required_metrics(self):
        rows, summary = evaluate(
            MAPPOTrainer(seed=0), episodes=2, seed=100, max_steps=3
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["episodes"], 2)
        self.assertIn("team_success_rate", summary)
        self.assertIn("both_reached_rate", summary)
        self.assertIn("route_overlap_mean", summary)
        self.assertTrue(all(row["truncated"] for row in rows))

    def test_checkpoint_evaluation_writes_csv_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "train"
            report = root / "eval"
            train("happo", 1, 1, 0, "cpu", training, max_steps=2)

            summary = evaluate_checkpoint(
                training / "checkpoint.pt", report, episodes=2, seed=5, max_steps=2
            )

            self.assertEqual(summary["algorithm"], "happo")
            self.assertTrue((report / "episodes.csv").is_file())
            self.assertTrue((report / "summary.json").is_file())
            self.assertTrue((report / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
