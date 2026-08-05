# Reliable Policy Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the strongest deterministic mission policy while keeping PPO exploration metrics separate from mission evaluation.

**Architecture:** Reuse the existing deterministic evaluator inside `train.py`. Rank evaluation summaries with one small pure function, save best and last checkpoints separately, and keep `checkpoint.pt` as the backward-compatible best-model path.

**Tech Stack:** Python 3.12, PyTorch, NumPy, standard-library `unittest` and `csv`.

## Global Constraints

- Do not change the grid, radar, reward, risk, or terminal rules.
- Training action collection remains stochastic.
- Checkpoint comparison and mission evaluation use deterministic argmax actions.
- Do not add dependencies.

---

### Task 1: Deterministic checkpoint ranking

**Files:**
- Modify: `train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: evaluation summary dictionaries returned by `eval.evaluate.evaluate`.
- Produces: `_evaluation_score(summary: dict) -> tuple[float, ...]`.

- [ ] **Step 1: Write the failing ranking test**

```python
def test_evaluation_score_prefers_success_then_safety_then_speed(self):
    safe = {
        "team_success_rate": 1.0,
        "aircraft_0_death_rate": 0.0,
        "aircraft_1_death_rate": 0.0,
        "mean_radar_entries": 0.0,
        "mean_steps": 1988.0,
        "mean_return": 127.0,
    }
    unsafe = dict(safe, aircraft_0_death_rate=0.1, mean_steps=1900.0)
    assert _evaluation_score(safe) > _evaluation_score(unsafe)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.venv/bin/python -m unittest tests.test_train.TestTrainingRun.test_evaluation_score_prefers_success_then_safety_then_speed -v`

Expected: import failure because `_evaluation_score` does not exist.

- [ ] **Step 3: Implement the pure ranking function**

```python
def _evaluation_score(summary):
    return (
        summary["team_success_rate"],
        -(summary["aircraft_0_death_rate"] + summary["aircraft_1_death_rate"]),
        -summary["mean_radar_entries"],
        -summary["mean_steps"],
        summary["mean_return"],
    )
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `.venv/bin/python -m unittest tests.test_train.TestTrainingRun.test_evaluation_score_prefers_success_then_safety_then_speed -v`

Expected: PASS.

---

### Task 2: Best and last checkpoint lifecycle

**Files:**
- Modify: `train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `evaluate(trainer, episodes, seed, max_steps)` and `_evaluation_score`.
- Produces: `checkpoint_best.pt`, `checkpoint_last.pt`, and backward-compatible `checkpoint.pt`.

- [ ] **Step 1: Extend the existing real training-run test**

```python
result = train(
    "mappo",
    episodes=1,
    rollout_episodes=1,
    seed=9,
    output_dir=output,
    max_steps=2,
    eval_episodes=1,
)
self.assertTrue((output / "checkpoint_best.pt").is_file())
self.assertTrue((output / "checkpoint_last.pt").is_file())
self.assertTrue((output / "checkpoint.pt").is_file())
```

Also assert that `metrics.csv` contains `sampled_success_rate`,
`eval_success_rate`, `eval_mean_deaths`, `eval_mean_radar_entries`,
`eval_mean_steps`, and `is_best`.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.venv/bin/python -m unittest tests.test_train.TestTrainingRun.test_training_writes_checkpoint_config_and_metrics -v`

Expected: failure because `train` does not accept `eval_episodes` and the new files do not exist.

- [ ] **Step 3: Implement initial and per-update deterministic evaluation**

Add `eval_episodes=4` and `eval_seed=10_000` parameters. Evaluate before the
first rollout, save the initial best policy, then evaluate after every update.
Replace the best files only when `_evaluation_score(current) > best_score`.
Always save the final update as `checkpoint_last.pt`.

- [ ] **Step 4: Separate CSV metric names**

Write sampled rollout success to `sampled_success_rate`; write deterministic
evaluation values to the `eval_*` columns and record whether the update replaced
the best checkpoint.

- [ ] **Step 5: Add CLI flags**

```python
parser.add_argument("--eval-episodes", type=int, default=4)
parser.add_argument("--eval-seed", type=int, default=10_000)
```

- [ ] **Step 6: Run focused and full tests**

Run: `.venv/bin/python -m unittest tests.test_train -v`

Expected: PASS.

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: all tests PASS.

---

### Task 3: Train and verify selected checkpoints

**Files:**
- Runtime outputs: `runs/mappo_seed0_reliable/`
- Runtime outputs: `runs/happo_seed0_reliable/`

**Interfaces:**
- Consumes: the updated `train.py` CLI and deterministic evaluator.
- Produces: selected MAPPO/HAPPO best checkpoints and fresh evaluation reports.

- [ ] **Step 1: Run one 2,000-episode seed for each algorithm**

```bash
.venv/bin/python train.py --algo mappo --episodes 2000 --rollout-episodes 32 --seed 0 --eval-episodes 4 --output runs/mappo_seed0_reliable
.venv/bin/python train.py --algo happo --episodes 2000 --rollout-episodes 32 --seed 0 --eval-episodes 4 --output runs/happo_seed0_reliable
```

- [ ] **Step 2: Evaluate each best checkpoint for 100 episodes**

```bash
.venv/bin/python -m eval.evaluate --checkpoint runs/mappo_seed0_reliable/checkpoint_best.pt --episodes 100 --seed 30000 --output runs/mappo_seed0_reliable/eval_best_100
.venv/bin/python -m eval.evaluate --checkpoint runs/happo_seed0_reliable/checkpoint_best.pt --episodes 100 --seed 30000 --output runs/happo_seed0_reliable/eval_best_100
```

- [ ] **Step 3: Enforce the acceptance gate**

Both summaries must report `team_success_rate == 1.0`, `timeout_rate == 0.0`,
both death rates `== 0.0`, and `mean_radar_entries == 0.0`. If either fails,
do not present routes; return to root-cause analysis.

- [ ] **Step 4: Generate ten deterministic grid-audit routes**

Use the selected best checkpoint with `deterministic=True`. Validate every
consecutive route transition has Manhattan distance at most one and diagonal
movement count zero before rendering full-map plus one-cell zoom panels.
