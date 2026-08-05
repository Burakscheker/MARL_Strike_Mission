import unittest

import torch

from agents.ppo import (
    Actor,
    CentralCritic,
    RolloutBatch,
    compute_gae,
    masked_categorical,
)
from config import N_ACTIONS, OBS_DIM, STATE_DIM


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


if __name__ == "__main__":
    unittest.main()
