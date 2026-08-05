# Reliable Policy Selection Design

## Problem

The trained actors have a safe deterministic route, but training-mode action sampling
keeps substantial entropy. Across the deterministic route, the mean probability of the
most likely action is only 0.53–0.69. Sampling thousands of actions therefore creates
detours, timeouts, and radar exposure even when the argmax policy succeeds.

The training loop also saves only the final update. A later unstable PPO update can
overwrite a better intermediate policy, and the CSV currently mixes sampled rollout
success with evaluation success.

## Design

Training remains stochastic because PPO needs exploration. Mission execution,
checkpoint comparison, evaluation, and route visualization use deterministic argmax
actions.

The training loop evaluates the initial policy and every completed PPO update on a
fixed evaluation seed set. It ranks policies lexicographically by:

1. higher team success rate;
2. lower combined aircraft death rate;
3. fewer radar entries;
4. fewer steps;
5. higher mean return.

The best-ranked policy is saved as `checkpoint_best.pt` and mirrored to
`checkpoint.pt` for backward compatibility. The final update is saved separately as
`checkpoint_last.pt`. Training metrics explicitly distinguish
`sampled_success_rate` from deterministic evaluation metrics.

## Failure Semantics

One aircraft dying does not fail the mission while the teammate remains alive. A
mission fails only when both aircraft die or no aircraft reaches the goal before the
3,000-step limit. No grid, radar, reward, or terminal rule changes are part of this
fix.

## Verification

Automated tests must prove that:

- the initial policy is eligible to become the best checkpoint;
- a worse later policy cannot overwrite the best checkpoint;
- the last checkpoint is still retained separately;
- metrics label sampled and deterministic success separately;
- evaluation and visualization default to deterministic actions.

For the current fixed map, the selected checkpoint must then pass a fresh 100-episode
deterministic evaluation with zero deaths, zero radar entries, and no timeouts before
new route visuals are presented.
