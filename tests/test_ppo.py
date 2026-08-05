import math
import tempfile
import unittest
from pathlib import Path

import torch

from agents.ppo import (
    Actor,
    CentralCritic,
    HAPPOTrainer,
    MAPPOTrainer,
    RolloutBatch,
    clipped_policy_loss,
    compute_gae,
    load_checkpoint,
    masked_categorical,
    save_checkpoint,
)
from config import N_ACTIONS, OBS_DIM, STATE_DIM


def deterministic_batch(trainer, steps=16):
    generator = torch.Generator().manual_seed(123)
    obs = torch.randn(steps, 2, OBS_DIM, generator=generator)
    states = obs.reshape(steps, STATE_DIM)
    masks = torch.ones(steps, 2, N_ACTIONS, dtype=torch.bool)
    masks[:, :, 4] = False
    actions = torch.ones(steps, 2, dtype=torch.long)
    with torch.no_grad():
        old_logp = torch.stack(
            [
                masked_categorical(trainer.actors[agent](obs[:, agent]), masks[:, agent])
                .log_prob(actions[:, agent])
                for agent in range(2)
            ],
            dim=1,
        )
        values = trainer.critic(states)
    advantages = torch.linspace(-1.0, 1.0, steps)
    return RolloutBatch(
        obs=obs,
        states=states,
        masks=masks,
        actions=actions,
        old_logp=old_logp,
        rewards=torch.zeros(steps),
        values=values,
        terminated=torch.zeros(steps, dtype=torch.bool),
        advantages=advantages,
        returns=values + 0.5,
        alive=torch.ones(steps, 2, dtype=torch.bool),
    )


class TestNetworks(unittest.TestCase):
    def test_masked_distribution_assigns_zero_probability_to_illegal_actions(self):
        logits = torch.tensor([[100.0, 0.0, -1.0, 2.0, 50.0]])
        mask = torch.tensor([[False, True, True, True, False]])

        distribution = masked_categorical(logits, mask)

        self.assertEqual(float(distribution.probs[0, 0]), 0.0)
        self.assertEqual(float(distribution.probs[0, 4]), 0.0)
        self.assertEqual(int(distribution.probs.argmax(dim=-1).item()), 3)

    def test_empty_action_mask_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no legal action"):
            masked_categorical(torch.zeros(1, N_ACTIONS), torch.zeros(1, N_ACTIONS, dtype=torch.bool))

    def test_actor_and_critic_shapes_match_environment(self):
        actor = Actor()
        critic = CentralCritic()

        self.assertEqual(actor(torch.zeros(3, OBS_DIM)).shape, (3, N_ACTIONS))
        self.assertEqual(critic(torch.zeros(3, STATE_DIM)).shape, (3,))


class TestGAE(unittest.TestCase):
    def test_terminal_transition_does_not_bootstrap(self):
        advantages, returns = compute_gae(
            rewards=torch.tensor([1.0]),
            values=torch.tensor([0.5]),
            terminated=torch.tensor([True]),
            bootstrap_value=torch.tensor(9.0),
            gamma=0.99,
            gae_lambda=0.95,
        )

        torch.testing.assert_close(advantages, torch.tensor([0.5]))
        torch.testing.assert_close(returns, torch.tensor([1.0]))

    def test_truncated_transition_bootstraps(self):
        advantages, returns = compute_gae(
            rewards=torch.tensor([1.0]),
            values=torch.tensor([0.5]),
            terminated=torch.tensor([False]),
            bootstrap_value=torch.tensor(2.0),
            gamma=0.99,
            gae_lambda=0.95,
        )

        torch.testing.assert_close(advantages, torch.tensor([2.48]))
        torch.testing.assert_close(returns, torch.tensor([2.98]))

    def test_two_step_terminal_return_propagates_backwards(self):
        advantages, returns = compute_gae(
            rewards=torch.tensor([0.0, 1.0]),
            values=torch.tensor([0.0, 0.0]),
            terminated=torch.tensor([False, True]),
            bootstrap_value=torch.tensor(7.0),
            gamma=1.0,
            gae_lambda=1.0,
        )

        torch.testing.assert_close(advantages, torch.tensor([1.0, 1.0]))
        torch.testing.assert_close(returns, torch.tensor([1.0, 1.0]))


class TestRolloutBatch(unittest.TestCase):
    def test_to_moves_every_tensor_and_preserves_shapes(self):
        batch = RolloutBatch(
            obs=torch.zeros(4, 2, OBS_DIM),
            states=torch.zeros(4, STATE_DIM),
            masks=torch.ones(4, 2, N_ACTIONS, dtype=torch.bool),
            actions=torch.zeros(4, 2, dtype=torch.long),
            old_logp=torch.zeros(4, 2),
            rewards=torch.zeros(4),
            values=torch.zeros(4),
            terminated=torch.zeros(4, dtype=torch.bool),
            advantages=torch.ones(4),
            returns=torch.ones(4),
            alive=torch.ones(4, 2, dtype=torch.bool),
        )

        moved = batch.to("cpu")

        self.assertEqual(moved.obs.shape, (4, 2, OBS_DIM))
        self.assertEqual(moved.masks.dtype, torch.bool)
        self.assertTrue(all(tensor.device.type == "cpu" for tensor in moved.tensors()))


class TestMAPPO(unittest.TestCase):
    def test_clipped_surrogate_uses_probability_ratio(self):
        loss = clipped_policy_loss(
            new_logp=torch.log(torch.tensor([0.75, 0.25])),
            old_logp=torch.log(torch.tensor([0.50, 0.50])),
            advantages=torch.ones(2),
            clip_coef=0.2,
        )

        torch.testing.assert_close(loss, -torch.tensor([1.2, 0.5]).mean())

    def test_update_changes_both_actors_and_critic(self):
        trainer = MAPPOTrainer(seed=0)
        batch = deterministic_batch(trainer)
        actor_before = [
            [parameter.detach().clone() for parameter in actor.parameters()]
            for actor in trainer.actors
        ]
        critic_before = [parameter.detach().clone() for parameter in trainer.critic.parameters()]

        metrics = trainer.update(batch)

        for before, actor in zip(actor_before, trainer.actors):
            self.assertTrue(
                any(not torch.equal(old, new) for old, new in zip(before, actor.parameters()))
            )
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(critic_before, trainer.critic.parameters())
            )
        )
        self.assertTrue(math.isfinite(metrics["actor_loss"]))
        self.assertTrue(math.isfinite(metrics["critic_loss"]))


class TestHAPPO(unittest.TestCase):
    def test_seed_controls_sequential_order_and_second_actor_factor(self):
        trainer = HAPPOTrainer(seed=7)
        batch = deterministic_batch(trainer)

        trainer.update(batch)

        self.assertEqual(set(trainer.last_update_order), {0, 1})
        first_actor = trainer.last_update_order[0]
        with torch.no_grad():
            new_logp = masked_categorical(
                trainer.actors[first_actor](batch.obs[:, first_actor]),
                batch.masks[:, first_actor],
            ).log_prob(batch.actions[:, first_actor])
        expected = (new_logp - batch.old_logp[:, first_actor]).exp()
        torch.testing.assert_close(trainer.last_factor, expected, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.isfinite(trainer.last_factor).all())


class TestCheckpoint(unittest.TestCase):
    def test_round_trip_preserves_deterministic_actions_and_config(self):
        trainer = MAPPOTrainer(seed=3)
        observations = torch.zeros(2, OBS_DIM)
        masks = torch.ones(2, N_ACTIONS, dtype=torch.bool)
        masks[:, 4] = False
        before, _, _ = trainer.act(observations, masks, deterministic=True)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, trainer, {"episodes": 10})
            loaded, saved_config = load_checkpoint(path)

        after, _, _ = loaded.act(observations, masks, deterministic=True)
        self.assertEqual(saved_config, {"episodes": 10})
        self.assertEqual(loaded.algorithm, "mappo")
        self.assertListEqual(before.tolist(), after.tolist())


if __name__ == "__main__":
    unittest.main()
