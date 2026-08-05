import unittest

import numpy as np

from config import DOWN, GOAL, LEFT, NOOP, RIGHT, START, UP
from env.strike_env import RADARS, StrikeMissionEnv, zone_for_radar


class SequenceRNG:
    def __init__(self, values):
        self._values = iter(values)
        self.calls = 0

    def random(self):
        self.calls += 1
        return next(self._values)


class TestGeometry(unittest.TestCase):
    def test_half_open_square_boundaries_are_exact(self):
        radar = RADARS[0]

        self.assertEqual(zone_for_radar((radar.x - 110, radar.y), radar), 1)
        self.assertEqual(zone_for_radar((radar.x + 109, radar.y), radar), 1)
        self.assertEqual(zone_for_radar((radar.x + 110, radar.y), radar), 0)
        self.assertEqual(zone_for_radar((radar.x - 70, radar.y), radar), 2)
        self.assertEqual(zone_for_radar((radar.x + 69, radar.y), radar), 2)
        self.assertEqual(zone_for_radar((radar.x + 70, radar.y), radar), 1)

    def test_square_cell_counts_are_220_and_140_per_axis(self):
        radar = RADARS[0]
        outer_axis = sum(
            zone_for_radar((x, radar.y), radar) > 0
            for x in range(-500, 500)
        )
        inner_axis = sum(
            zone_for_radar((x, radar.y), radar) == 2
            for x in range(-500, 500)
        )

        self.assertEqual(outer_axis, 220)
        self.assertEqual(inner_axis, 140)


class TestMovementAndObservation(unittest.TestCase):
    def test_actions_are_simultaneous_and_overlap_is_allowed(self):
        env = StrikeMissionEnv()
        env.reset(seed=0, options={"positions": [(-1, 0), (1, 0)]})

        _, _, terminated, truncated, _ = env.step([RIGHT, LEFT])

        np.testing.assert_array_equal(env.positions, [[0, 0], [0, 0]])
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_live_masks_allow_only_in_bounds_movement(self):
        env = StrikeMissionEnv()
        env.reset(seed=0)

        masks = env.action_masks()

        np.testing.assert_array_equal(masks[0], [True, True, True, False, False])
        np.testing.assert_array_equal(masks[1], [True, True, True, False, False])

    def test_observation_and_state_are_finite_and_compact(self):
        env = StrikeMissionEnv()

        obs, _ = env.reset(seed=0)

        self.assertEqual(obs.shape, (2, 23))
        self.assertEqual(env.state().shape, (46,))
        self.assertTrue(np.isfinite(obs).all())
        self.assertTrue(np.isfinite(env.state()).all())

    def test_timeout_is_truncation_not_termination(self):
        env = StrikeMissionEnv(max_steps=1)
        env.reset(seed=0)

        _, _, terminated, truncated, _ = env.step([RIGHT, RIGHT])

        self.assertFalse(terminated)
        self.assertTrue(truncated)


class TestRisk(unittest.TestCase):
    def test_staying_in_outer_zone_rolls_once(self):
        env = StrikeMissionEnv()
        env.reset(seed=0, options={"positions": [(-391, 220), START]})
        env.rng = SequenceRNG([0.99])

        env.step([RIGHT, RIGHT])
        env.step([RIGHT, RIGHT])

        self.assertEqual(env.rng.calls, 1)
        self.assertTrue(env.alive[0])

    def test_leaving_and_reentering_outer_zone_rolls_again(self):
        env = StrikeMissionEnv()
        env.reset(seed=0, options={"positions": [(-391, 220), START]})
        env.rng = SequenceRNG([0.99, 0.99])

        env.step([RIGHT, RIGHT])
        env.step([LEFT, RIGHT])
        env.step([RIGHT, RIGHT])

        self.assertEqual(env.rng.calls, 2)
        self.assertTrue(env.alive[0])

    def test_outer_then_inner_entry_uses_two_independent_rolls(self):
        env = StrikeMissionEnv()
        env.reset(seed=0, options={"positions": [(-391, 220), START]})
        env.rng = SequenceRNG([0.20, 0.90])

        env.step([RIGHT, RIGHT])
        for _ in range(40):
            env.step([RIGHT, RIGHT])

        self.assertEqual(tuple(env.positions[0]), (-350, 220))
        self.assertEqual(env.rng.calls, 2)
        self.assertTrue(env.alive[0])

    def test_aircraft_death_does_not_end_survivors_mission(self):
        env = StrikeMissionEnv()
        env.reset(seed=0, options={"positions": [(-391, 220), START]})
        env.rng = SequenceRNG([0.0])

        _, reward, terminated, truncated, info = env.step([RIGHT, RIGHT])

        self.assertFalse(env.alive[0])
        self.assertTrue(env.alive[1])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["deaths"], 1)
        self.assertAlmostEqual(reward, -24.981)
        np.testing.assert_array_equal(
            env.action_masks()[0], [False, False, False, False, True]
        )


class TestRewardsAndTerminal(unittest.TestCase):
    def test_first_goal_arrival_ends_episode_with_full_team_reward(self):
        env = StrikeMissionEnv()
        env.reset(seed=0, options={"positions": [(498, -494), START]})

        _, reward, terminated, truncated, info = env.step([RIGHT, RIGHT])

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])
        self.assertFalse(info["both_reached"])
        self.assertAlmostEqual(reward, 100.019)

    def test_both_reaching_together_still_awards_one_success_bonus(self):
        env = StrikeMissionEnv()
        env.reset(seed=0, options={"positions": [(498, -494), (498, -494)]})

        _, reward, terminated, _, info = env.step([RIGHT, RIGHT])

        self.assertTrue(terminated)
        self.assertTrue(info["both_reached"])
        self.assertAlmostEqual(reward, 100.019)

    def test_invalid_live_noop_is_rejected(self):
        env = StrikeMissionEnv()
        env.reset(seed=0)

        with self.assertRaisesRegex(ValueError, "illegal action"):
            env.step([NOOP, RIGHT])

    def test_goal_constant_matches_the_1000_cell_grid(self):
        self.assertEqual(GOAL, (499, -494))
        self.assertEqual((UP, RIGHT, DOWN, LEFT, NOOP), (0, 1, 2, 3, 4))


if __name__ == "__main__":
    unittest.main()
