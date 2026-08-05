# MAPPO/HAPPO Strike Mission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, testable two-aircraft 1000×1000 grid environment and train directly comparable MAPPO and HAPPO policies on it.

**Architecture:** A custom NumPy environment advances both aircraft simultaneously and exposes compact 23-value observations plus action masks. Two separate PyTorch actor MLPs use local shared-state observations, while one centralized critic consumes the flattened joint observation; MAPPO and HAPPO share rollout, GAE, checkpoint, and evaluation infrastructure.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, Matplotlib, standard-library `unittest`, CSV/JSON.

## Global Constraints

- Grid coordinates are integer `x,y ∈ [-500, 499]`; start is `(-500, 495)` and goal is `(499, -494)`.
- Both aircraft move simultaneously by exactly one unit in four directions and may occupy the same cell.
- Action `4=NOOP` is internal: masked for living aircraft and the only legal action for dead aircraft.
- Radar centers are `(-280,220)`, `(200,100)`, and `(-100,-280)`.
- Outer squares use half-open bounds `[center-110, center+110)`; inner squares use `[center-70, center+70)`.
- Death is rolled once on every zone entry: outer `0.20`, inner `0.90`; staying in a zone never rerolls.
- Entering outer then inner yields `0.80 × 0.10 = 0.08` survival; aircraft rolls are independent.
- First target arrival terminates successfully with `+100`; each aircraft death gives `-25`.
- Signed Manhattan progress is `±0.01` per unit per living aircraft and global step cost is `-0.001`.
- `MAX_STEPS=3000`; timeout is truncation, not termination.
- Use compact MLP observations, separate actors, one centralized critic, and no CNN.
- Compare MAPPO and HAPPO only, with identical capacity, seeds, and training budget.
- Add no dependency beyond the existing NumPy, PyTorch, and Matplotlib requirements.

## File Structure

- `config.py`: immutable geometry, reward, PPO, and path constants.
- `env/strike_env.py`: geometry helpers and complete simultaneous environment.
- `agents/ppo.py`: networks, masked distribution, rollout batch, GAE, MAPPO, HAPPO, checkpointing.
- `train.py`: seed setup, rollout collection, training CLI, CSV logging.
- `eval/evaluate.py`: deterministic checkpoint evaluation and JSON/Markdown summaries.
- `viz/plot_map.py`: exact 1-unit geometry and optional trajectory rendering.
- `tests/test_env.py`: geometry, transition, risk, reward, observation, reproducibility tests.
- `tests/test_ppo.py`: masking, GAE, PPO update, HAPPO factor, checkpoint tests.
- `tests/test_train.py`: short end-to-end rollout/training/evaluation smoke tests.

---

### Task 1: Implement the 1-Unit Simultaneous Environment

**Files:**
- Create: `config.py`
- Create: `env/strike_env.py`
- Create: `tests/test_env.py`
- Delete: `baselines/map_check.py`

**Interfaces:**
- Produces: `Radar`, `zone_for_radar(position, radar) -> int`, and `StrikeMissionEnv`.
- Produces: `StrikeMissionEnv(max_steps: int = MAX_STEPS)` so tests can use short truncation horizons.
- Produces: `reset(seed=None, options=None) -> tuple[np.ndarray, dict]`.
- Produces: `step(actions) -> tuple[np.ndarray, float, bool, bool, dict]`.
- Produces: `action_masks() -> np.ndarray` with shape `(2, 5)` and `state() -> np.ndarray`.
- Observation shape: `(2, OBS_DIM)` where `OBS_DIM=23`; state is the flattened observations with shape `(46,)`.

- [ ] **Step 1: Write failing geometry, mask, and simultaneous-motion tests**

```python
class TestGeometry(unittest.TestCase):
    def test_square_sizes_are_exact(self):
        radar = RADARS[0]
        outer = sum(zone_for_radar((x, y), radar) > 0
                    for x in range(radar.x - 110, radar.x + 110)
                    for y in range(radar.y - 110, radar.y + 110))
        inner = sum(zone_for_radar((x, y), radar) == 2
                    for x in range(radar.x - 70, radar.x + 70)
                    for y in range(radar.y - 70, radar.y + 70))
        self.assertEqual(outer, 220 * 220)
        self.assertEqual(inner, 140 * 140)

class TestMovement(unittest.TestCase):
    def test_actions_are_simultaneous_and_overlap_is_allowed(self):
        env = StrikeMissionEnv()
        env.reset(seed=0, options={"positions": [(-1, 0), (1, 0)]})
        _, _, terminated, truncated, _ = env.step([RIGHT, LEFT])
        np.testing.assert_array_equal(env.positions, [[0, 0], [0, 0]])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m unittest tests.test_env -v`

Expected: FAIL because `config` and `env.strike_env` do not exist.

- [ ] **Step 3: Implement constants and exact half-open zone geometry**

```python
@dataclass(frozen=True)
class Radar:
    x: int
    y: int

def zone_for_radar(position, radar):
    x, y = map(int, position)
    in_outer = radar.x - 110 <= x < radar.x + 110 and radar.y - 110 <= y < radar.y + 110
    in_inner = radar.x - 70 <= x < radar.x + 70 and radar.y - 70 <= y < radar.y + 70
    return 2 if in_inner else 1 if in_outer else 0
```

Define `UP, RIGHT, DOWN, LEFT, NOOP = range(5)`, `GRID_MIN=-500`,
`GRID_MAX=499`, `START=(-500,495)`, `GOAL=(499,-494)`, and the exact reward,
risk, network, and PPO constants from `Strike_Mission.md` in `config.py`.

- [ ] **Step 4: Implement reset, action masks, simultaneous movement, and observations**

`StrikeMissionEnv.reset` must initialize `positions`, `alive`, `reached`,
`steps`, the NumPy generator, and per-aircraft/per-radar `zones`. Build each
23-value observation as:

```text
self_xy(2), other_xy(2), alive_flags(2),
self_goal_delta(2), other_goal_delta(2), time(1),
3 × [self_radar_delta_x, self_radar_delta_y, self_zone, other_zone](12)
```

Normalize positions and deltas by `999`, time by `3000`, and zones by `2`.

- [ ] **Step 5: Write failing one-roll, re-entry, cumulative-risk, and reward tests**

```python
class SequenceRNG:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def random(self):
        self.calls += 1
        return next(self.values)

def test_staying_in_outer_zone_rolls_once(self):
    env = StrikeMissionEnv()
    env.reset(seed=0, options={"positions": [(-391, 220), START]})
    env.rng = SequenceRNG([0.99])
    env.step([RIGHT, RIGHT])
    env.step([RIGHT, RIGHT])
    self.assertEqual(env.rng.calls, 1)

def test_outer_then_inner_survival_is_eight_percent(self):
    env = StrikeMissionEnv()
    env.reset(seed=0, options={"positions": [(-391, 220), START]})
    env.rng = SequenceRNG([0.20, 0.90])
    env.step([RIGHT, RIGHT])
    for _ in range(40):
        env.step([RIGHT, RIGHT])
    self.assertTrue(env.alive[0])
    self.assertEqual(env.rng.calls, 2)
```

- [ ] **Step 6: Run risk tests and verify RED**

Run: `python3 -m unittest tests.test_env.TestRisk -v`

Expected: FAIL because zone-transition risk is not implemented.

- [ ] **Step 7: Implement entry-only risk and exact rewards**

For each living aircraft compare the old and new zone for every radar. Roll
`0.20` when the new zone is outer and differs from the old zone. Roll `0.90`
when the new zone is inner; if a future transition jumps directly from safe to
inner, roll outer then inner. Stop processing that aircraft after death.
Reward begins at `-0.001`, adds signed distance deltas for aircraft alive at
the start of the step, subtracts `25` per death, and adds `100` once when any
surviving aircraft reaches `GOAL`.

- [ ] **Step 8: Run the complete environment suite**

Run: `python3 -m unittest tests.test_env -v`

Expected: all tests pass, including fixed-seed reproducibility and observation
shape/finite-value checks.

- [ ] **Step 9: Commit the environment**

```bash
git add config.py env/strike_env.py tests/test_env.py baselines/map_check.py
git commit -m "feat(env): add entry-risk strike grid"
```

---

### Task 2: Add PPO Networks, Masking, Rollouts, and GAE

**Files:**
- Create: `agents/ppo.py`
- Create: `tests/test_ppo.py`

**Interfaces:**
- Consumes: `OBS_DIM=23`, `STATE_DIM=46`, `N_ACTIONS=5`, PPO constants.
- Produces: `Actor`, `CentralCritic`, `masked_categorical`, `RolloutBatch`, and `compute_gae`.
- `Actor.forward(obs) -> logits[...,5]`; `CentralCritic.forward(state) -> values[...]`.

- [ ] **Step 1: Write failing mask and network shape tests**

```python
def test_invalid_action_probability_is_zero(self):
    logits = torch.tensor([[100.0, 0.0, -1.0, 2.0, 50.0]])
    mask = torch.tensor([[False, True, True, True, False]])
    dist = masked_categorical(logits, mask)
    self.assertEqual(float(dist.probs[0, 0]), 0.0)
    self.assertEqual(float(dist.probs[0, 4]), 0.0)

def test_network_shapes(self):
    self.assertEqual(Actor()(torch.zeros(3, OBS_DIM)).shape, (3, 5))
    self.assertEqual(CentralCritic()(torch.zeros(3, STATE_DIM)).shape, (3,))
```

- [ ] **Step 2: Run primitive tests and verify RED**

Run: `python3 -m unittest tests.test_ppo.TestNetworks -v`

Expected: import failure because `agents.ppo` does not exist.

- [ ] **Step 3: Implement minimal MLPs and masked categorical**

```python
def masked_categorical(logits, mask):
    if (~mask).all(dim=-1).any():
        raise ValueError("action mask has no legal action")
    return Categorical(logits=logits.masked_fill(~mask, -torch.inf))
```

Use two `Tanh` hidden layers of 128 units for actors and 256 units for the
critic, with orthogonal initialization and a `0.01` actor output gain.

- [ ] **Step 4: Write failing terminal/truncation GAE tests**

```python
def test_terminal_does_not_bootstrap(self):
    adv, returns = compute_gae(
        torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([True]),
        torch.tensor(9.0), gamma=0.99, gae_lambda=0.95)
    torch.testing.assert_close(adv, torch.tensor([0.5]))
    torch.testing.assert_close(returns, torch.tensor([1.0]))

def test_truncation_bootstraps(self):
    adv, returns = compute_gae(
        torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([False]),
        torch.tensor(2.0), gamma=0.99, gae_lambda=0.95)
    torch.testing.assert_close(adv, torch.tensor([2.48]))
    torch.testing.assert_close(returns, torch.tensor([2.98]))
```

- [ ] **Step 5: Implement reverse-time GAE and RolloutBatch**

Implement `advantages[t] = delta + gamma * gae_lambda * nonterminal * next_gae`
in reverse order. `RolloutBatch` stores tensors for observations `(T,2,23)`,
states `(T,46)`, masks `(T,2,5)`, actions/log-probs `(T,2)`, rewards,
values, terminated flags, advantages, returns, and alive masks `(T,2)`.

- [ ] **Step 6: Run PPO primitive tests**

Run: `python3 -m unittest tests.test_ppo.TestNetworks tests.test_ppo.TestGAE -v`

Expected: all tests pass.

- [ ] **Step 7: Commit PPO primitives**

```bash
git add agents/ppo.py tests/test_ppo.py
git commit -m "feat(ppo): add models rollout and GAE"
```

---

### Task 3: Implement MAPPO and HAPPO Updates

**Files:**
- Modify: `agents/ppo.py`
- Modify: `tests/test_ppo.py`

**Interfaces:**
- Produces: `MAPPOTrainer.act`, `MAPPOTrainer.update`, `HAPPOTrainer.update`.
- Produces: `save_checkpoint(path, trainer, config)` and `load_checkpoint(path, algorithm)`.
- Both trainers own `actors: ModuleList`, one critic, actor optimizers, and a critic optimizer.

- [ ] **Step 1: Write failing clipped-objective and MAPPO update tests**

```python
def test_clipped_surrogate_uses_ratio(self):
    loss = clipped_policy_loss(
        torch.log(torch.tensor([0.75, 0.25])),
        torch.log(torch.tensor([0.50, 0.50])),
        torch.ones(2), 0.2)
    torch.testing.assert_close(loss, -torch.tensor([1.2, 0.5]).mean())

def test_mappo_update_changes_actor_and_critic(self):
    trainer = MAPPOTrainer(seed=0)
    batch = deterministic_batch()
    actor_before = [p.detach().clone() for p in trainer.actors[0].parameters()]
    critic_before = [p.detach().clone() for p in trainer.critic.parameters()]
    metrics = trainer.update(batch)
    self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor_before, trainer.actors[0].parameters())))
    self.assertTrue(any(not torch.equal(a, b) for a, b in zip(critic_before, trainer.critic.parameters())))
    self.assertTrue(math.isfinite(metrics["actor_loss"]))
```

- [ ] **Step 2: Run MAPPO tests and verify RED**

Run: `python3 -m unittest tests.test_ppo.TestMAPPO -v`

Expected: missing `clipped_policy_loss` or `MAPPOTrainer`.

- [ ] **Step 3: Implement shared critic and MAPPO actor updates**

Use frozen old log-probabilities, normalized advantages over each actor's alive
samples, five PPO epochs, seed-shuffled minibatches, entropy regularization,
value clipping, and gradient norm clipping. Reject non-finite losses and
gradients with `FloatingPointError`.

- [ ] **Step 4: Write failing HAPPO sequential-factor test**

```python
def test_happo_factor_contains_first_actor_ratio(self):
    trainer = HAPPOTrainer(seed=7)
    batch = deterministic_batch()
    trainer.update(batch)
    self.assertEqual(set(trainer.last_update_order), {0, 1})
    self.assertEqual(trainer.last_factor.shape, batch.advantages.shape)
    self.assertTrue(torch.isfinite(trainer.last_factor).all())
```

- [ ] **Step 5: Implement seeded sequential HAPPO updates**

Shuffle actor order once per update. Begin `factor=torch.ones(T)`. After each
actor update, recompute that actor's new log-probability on the full frozen
rollout and multiply `factor` by `exp(new_logp-old_logp)` only where that actor
was alive. Detach the factor before the next actor objective.

- [ ] **Step 6: Add checkpoint round-trip test and implementation**

Save algorithm name, actor/critic state dicts, optimizer state dicts, config,
and seed. Load with `map_location="cpu"`; assert the same deterministic action
before and after loading.

- [ ] **Step 7: Run all PPO tests**

Run: `python3 -m unittest tests.test_ppo -v`

Expected: all tests pass with finite losses.

- [ ] **Step 8: Commit trainers**

```bash
git add agents/ppo.py tests/test_ppo.py
git commit -m "feat: add MAPPO and HAPPO trainers"
```

---

### Task 4: Add Rollout Collection, Training, and Evaluation

**Files:**
- Create: `train.py`
- Create: `eval/evaluate.py`
- Create: `tests/test_train.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `set_seed(seed)`, `collect_rollouts(env, trainer, episodes) -> RolloutBatch`.
- Produces: CLI `python3 train.py --algo {mappo,happo} --episodes N --seed N`.
- Produces: CLI `python3 -m eval.evaluate --checkpoint PATH --episodes N`.

- [ ] **Step 1: Write failing rollout smoke test**

```python
def test_collect_rollouts_returns_finite_batch(self):
    env = StrikeMissionEnv(max_steps=8)
    trainer = MAPPOTrainer(seed=0)
    batch = collect_rollouts(env, trainer, episodes=2)
    self.assertEqual(batch.obs.shape[1:], (2, OBS_DIM))
    self.assertTrue(torch.isfinite(batch.rewards).all())
    self.assertTrue(torch.isfinite(batch.advantages).all())
```

- [ ] **Step 2: Run smoke test and verify RED**

Run: `python3 -m unittest tests.test_train -v`

Expected: import failure because `train.py` does not exist.

- [ ] **Step 3: Implement deterministic rollout collection**

Collect complete episodes, retaining the final critic bootstrap only for
truncation. Keep one common timeline and write one batch tensor per field.
Expose a small `max_steps` constructor override solely for smoke tests and
debugging; production default remains `3000`.

- [ ] **Step 4: Implement training CLI and logs**

Parse only algorithm, episodes, rollout episodes, seed, device, and output
directory. Write `metrics.csv` after each update and `checkpoint.pt` at the end.
Store the exact runtime config beside the checkpoint as JSON.

- [ ] **Step 5: Implement deterministic evaluation**

Use actor argmax under masks. Record team success, both reached, per-aircraft
death/reach, radar entries, timeout, episode steps, return, route overlap, and
wall time. Write raw CSV plus `summary.json` and `report.md`.

- [ ] **Step 6: Run end-to-end smoke tests**

Run: `python3 -m unittest tests.test_env tests.test_ppo tests.test_train -v`

Expected: all tests pass; a two-episode MAPPO and HAPPO update has finite losses.

- [ ] **Step 7: Commit training and evaluation**

```bash
git add train.py eval/evaluate.py tests/test_train.py .gitignore
git commit -m "feat: add training and evaluation pipeline"
```

---

### Task 5: Update the Exact Map Renderer and Usage Docs

**Files:**
- Modify: `viz/plot_map.py`
- Modify: `README.md`
- Test: `tests/test_env.py`

**Interfaces:**
- Consumes: all geometry from `config.py`; embeds no duplicate coordinates.
- Produces: `python3 -m viz.plot_map --output runs/map.png`.

- [ ] **Step 1: Replace the obsolete 51×51 renderer**

Draw `[-500,499]` bounds, exact 220/140 half-open rectangles, both aircraft at
`(-500,495)`, target `(499,-494)`, and optional trajectory arrays. Import every
coordinate and dimension from `config.py`.

- [ ] **Step 2: Add a non-interactive render smoke test**

Call `plot_map(output=temp_path)` under Matplotlib `Agg`; assert the PNG exists
and has non-zero size.

- [ ] **Step 3: Update README commands**

Document environment test, smoke training, full MAPPO/HAPPO training,
evaluation, and map rendering commands. Label long 1-unit runs explicitly.

- [ ] **Step 4: Run full verification**

Run:

```bash
python3 -m unittest discover -s tests -v
python3 -m viz.plot_map --output runs/map.png
git diff --check
```

Expected: zero test failures, renderer exit code 0, non-empty `runs/map.png`,
and no whitespace errors.

- [ ] **Step 5: Commit documentation and renderer**

```bash
git add viz/plot_map.py README.md tests/test_env.py
git commit -m "docs: add MAPPO and HAPPO workflow"
```
