from dataclasses import dataclass

import numpy as np

from config import (
    DOWN,
    GOAL,
    GRID_MAX,
    GRID_MIN,
    INNER_HALF_SIZE,
    LEFT,
    MAX_STEPS,
    NOOP,
    OBS_DIM,
    OUTER_HALF_SIZE,
    P_INNER_DEATH,
    P_OUTER_DEATH,
    RADAR_CENTERS,
    REWARD_DEATH,
    REWARD_PROGRESS,
    REWARD_STEP,
    REWARD_SUCCESS,
    RIGHT,
    START,
    UP,
)


@dataclass(frozen=True)
class Radar:
    x: int
    y: int


RADARS = tuple(Radar(*center) for center in RADAR_CENTERS)
MOVES = np.asarray(((0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)), dtype=np.int32)


def zone_for_radar(position, radar):
    x, y = map(int, position)
    in_outer = (
        radar.x - OUTER_HALF_SIZE <= x < radar.x + OUTER_HALF_SIZE
        and radar.y - OUTER_HALF_SIZE <= y < radar.y + OUTER_HALF_SIZE
    )
    in_inner = (
        radar.x - INNER_HALF_SIZE <= x < radar.x + INNER_HALF_SIZE
        and radar.y - INNER_HALF_SIZE <= y < radar.y + INNER_HALF_SIZE
    )
    return 2 if in_inner else 1 if in_outer else 0


def _manhattan(position):
    return abs(int(position[0]) - GOAL[0]) + abs(int(position[1]) - GOAL[1])


class StrikeMissionEnv:
    def __init__(self, max_steps=MAX_STEPS):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng()
        self._finished = True

    def reset(self, seed=None, options=None):
        self.rng = np.random.default_rng(seed)
        options = options or {}
        positions = options.get("positions", (START, START))
        self.positions = np.asarray(positions, dtype=np.int32).copy()
        if self.positions.shape != (2, 2):
            raise ValueError("positions must have shape (2, 2)")
        if np.any(self.positions < GRID_MIN) or np.any(self.positions > GRID_MAX):
            raise ValueError("positions must be inside the grid")

        self.alive = np.ones(2, dtype=bool)
        self.reached = np.zeros(2, dtype=bool)
        self.steps = 0
        self.zones = self._zones_at(self.positions)
        self.routes = [[tuple(position)] for position in self.positions]
        self._finished = False
        return self._observations(), {"seed": seed}

    def _zones_at(self, positions):
        return np.asarray(
            [[zone_for_radar(position, radar) for radar in RADARS] for position in positions],
            dtype=np.int8,
        )

    def action_masks(self):
        masks = np.zeros((2, 5), dtype=bool)
        for agent, ((x, y), alive, reached) in enumerate(
            zip(self.positions, self.alive, self.reached)
        ):
            if not alive or reached:
                masks[agent, NOOP] = True
                continue
            masks[agent, UP] = y < GRID_MAX
            masks[agent, RIGHT] = x < GRID_MAX
            masks[agent, DOWN] = y > GRID_MIN
            masks[agent, LEFT] = x > GRID_MIN
        return masks

    def state(self):
        return self._observations().reshape(-1)

    def _observations(self):
        observations = np.empty((2, OBS_DIM), dtype=np.float32)
        time = self.steps / self.max_steps
        for agent in range(2):
            other = 1 - agent
            own = self.positions[agent]
            teammate = self.positions[other]
            values = [
                own[0] / 999.0,
                own[1] / 999.0,
                teammate[0] / 999.0,
                teammate[1] / 999.0,
                float(self.alive[agent]),
                float(self.alive[other]),
                (GOAL[0] - own[0]) / 999.0,
                (GOAL[1] - own[1]) / 999.0,
                (GOAL[0] - teammate[0]) / 999.0,
                (GOAL[1] - teammate[1]) / 999.0,
                time,
            ]
            for radar_index, radar in enumerate(RADARS):
                values.extend(
                    (
                        (radar.x - own[0]) / 999.0,
                        (radar.y - own[1]) / 999.0,
                        self.zones[agent, radar_index] / 2.0,
                        self.zones[other, radar_index] / 2.0,
                    )
                )
            observations[agent] = values
        return observations

    def _roll_entry_risk(self, agent, old_zones, new_zones):
        entries = 0
        for old_zone, new_zone in zip(old_zones, new_zones):
            if not self.alive[agent] or old_zone == new_zone or new_zone == 0:
                continue
            probabilities = []
            if new_zone == 1:
                probabilities.append(P_OUTER_DEATH)
            elif new_zone == 2:
                if old_zone == 0:
                    probabilities.append(P_OUTER_DEATH)
                probabilities.append(P_INNER_DEATH)
            for probability in probabilities:
                entries += 1
                if self.rng.random() < probability:
                    self.alive[agent] = False
                    break
        return entries

    def step(self, actions):
        if self._finished:
            raise RuntimeError("episode is finished; call reset")
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (2,):
            raise ValueError("actions must have shape (2,)")
        masks = self.action_masks()
        for agent, action in enumerate(actions):
            if action < 0 or action >= 5 or not masks[agent, action]:
                raise ValueError("illegal action for agent {}".format(agent))

        was_alive = self.alive.copy()
        old_positions = self.positions.copy()
        old_distances = np.asarray([_manhattan(position) for position in old_positions])
        self.positions = self.positions + MOVES[actions]
        for agent in range(2):
            self.routes[agent].append(tuple(self.positions[agent]))

        new_zones = self._zones_at(self.positions)
        radar_entries = 0
        for agent in range(2):
            if was_alive[agent]:
                radar_entries += self._roll_entry_risk(agent, self.zones[agent], new_zones[agent])
        self.zones = new_zones

        reward = REWARD_STEP
        new_distances = np.asarray([_manhattan(position) for position in self.positions])
        reward += REWARD_PROGRESS * float(
            (old_distances[was_alive] - new_distances[was_alive]).sum()
        )

        deaths = int(np.count_nonzero(was_alive & ~self.alive))
        reward += REWARD_DEATH * deaths
        for agent in range(2):
            if self.alive[agent] and tuple(self.positions[agent]) == GOAL:
                self.reached[agent] = True

        success = bool(self.reached.any())
        if success:
            reward += REWARD_SUCCESS

        self.steps += 1
        terminated = success or not bool(self.alive.any())
        truncated = self.steps >= self.max_steps and not terminated
        self._finished = terminated or truncated
        info = {
            "success": success,
            "both_reached": bool(self.reached.all()),
            "deaths": deaths,
            "radar_entries": radar_entries,
            "steps": self.steps,
        }
        return self._observations(), float(reward), terminated, truncated, info
