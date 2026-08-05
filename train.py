import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from agents.ppo import (
    HAPPOTrainer,
    MAPPOTrainer,
    RolloutBatch,
    compute_gae,
    save_checkpoint,
)
from config import GAE_LAMBDA, GAMMA, MAX_STEPS, ROLLOUT_EPISODES
from env.strike_env import StrikeMissionEnv


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_rollouts(env, trainer, episode_count, seed=0):
    fields = {
        name: []
        for name in (
            "obs",
            "states",
            "masks",
            "actions",
            "old_logp",
            "rewards",
            "values",
            "terminated",
            "advantages",
            "returns",
            "alive",
        )
    }
    episode_metrics = []

    for episode_index in range(episode_count):
        observations, _ = env.reset(seed=seed + episode_index)
        local = {name: [] for name in fields if name not in ("advantages", "returns")}
        episode_return = 0.0
        total_deaths = 0
        total_entries = 0
        terminated = truncated = False

        while not (terminated or truncated):
            state = env.state()
            masks = env.action_masks()
            alive = env.alive.copy()
            actions, log_probabilities, value = trainer.act(
                observations, masks, state=state
            )
            next_observations, reward, terminated, truncated, info = env.step(actions)

            local["obs"].append(observations.copy())
            local["states"].append(state.copy())
            local["masks"].append(masks.copy())
            local["actions"].append(actions.copy())
            local["old_logp"].append(log_probabilities.copy())
            local["rewards"].append(reward)
            local["values"].append(value)
            local["terminated"].append(terminated)
            local["alive"].append(alive)

            observations = next_observations
            episode_return += reward
            total_deaths += info["deaths"]
            total_entries += info["radar_entries"]

        rewards = torch.as_tensor(local["rewards"], dtype=torch.float32)
        values = torch.as_tensor(local["values"], dtype=torch.float32)
        terminal_flags = torch.as_tensor(local["terminated"], dtype=torch.bool)
        if truncated:
            with torch.no_grad():
                final_state = torch.as_tensor(
                    env.state(), dtype=torch.float32, device=trainer.device
                )
                bootstrap = trainer.critic(final_state).detach().cpu()
        else:
            bootstrap = torch.tensor(0.0)
        advantages, returns = compute_gae(
            rewards,
            values,
            terminal_flags,
            bootstrap,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
        )

        for name, values_list in local.items():
            fields[name].extend(values_list)
        fields["advantages"].extend(advantages.tolist())
        fields["returns"].extend(returns.tolist())
        episode_metrics.append(
            {
                "return": episode_return,
                "steps": info["steps"],
                "success": info["success"],
                "both_reached": info["both_reached"],
                "deaths": total_deaths,
                "radar_entries": total_entries,
                "terminated": terminated,
                "truncated": truncated,
            }
        )

    batch = RolloutBatch(
        obs=torch.as_tensor(np.asarray(fields["obs"]), dtype=torch.float32),
        states=torch.as_tensor(np.asarray(fields["states"]), dtype=torch.float32),
        masks=torch.as_tensor(np.asarray(fields["masks"]), dtype=torch.bool),
        actions=torch.as_tensor(np.asarray(fields["actions"]), dtype=torch.long),
        old_logp=torch.as_tensor(np.asarray(fields["old_logp"]), dtype=torch.float32),
        rewards=torch.as_tensor(fields["rewards"], dtype=torch.float32),
        values=torch.as_tensor(fields["values"], dtype=torch.float32),
        terminated=torch.as_tensor(fields["terminated"], dtype=torch.bool),
        advantages=torch.as_tensor(fields["advantages"], dtype=torch.float32),
        returns=torch.as_tensor(fields["returns"], dtype=torch.float32),
        alive=torch.as_tensor(np.asarray(fields["alive"]), dtype=torch.bool),
    )
    return batch, episode_metrics


def train(
    algorithm,
    episodes,
    rollout_episodes=ROLLOUT_EPISODES,
    seed=0,
    device="cpu",
    output_dir="runs/ppo",
    max_steps=MAX_STEPS,
):
    trainers = {"mappo": MAPPOTrainer, "happo": HAPPOTrainer}
    if algorithm not in trainers:
        raise ValueError("algorithm must be mappo or happo")
    if episodes <= 0 or rollout_episodes <= 0:
        raise ValueError("episode counts must be positive")

    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = trainers[algorithm](seed=seed, device=device)
    env = StrikeMissionEnv(max_steps=max_steps)
    config = {
        "algorithm": algorithm,
        "episodes": int(episodes),
        "rollout_episodes": int(rollout_episodes),
        "seed": int(seed),
        "device": str(device),
        "max_steps": int(max_steps),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    completed = 0
    updates = 0
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "update",
                "episodes",
                "actor_loss",
                "critic_loss",
                "mean_return",
                "success_rate",
            ),
        )
        writer.writeheader()
        while completed < episodes:
            count = min(rollout_episodes, episodes - completed)
            batch, episode_rows = collect_rollouts(
                env, trainer, episode_count=count, seed=seed + completed
            )
            losses = trainer.update(batch)
            completed += count
            updates += 1
            writer.writerow(
                {
                    "update": updates,
                    "episodes": completed,
                    "actor_loss": losses["actor_loss"],
                    "critic_loss": losses["critic_loss"],
                    "mean_return": float(np.mean([row["return"] for row in episode_rows])),
                    "success_rate": float(np.mean([row["success"] for row in episode_rows])),
                }
            )
            stream.flush()

    save_checkpoint(output_dir / "checkpoint.pt", trainer, config)
    return {"episodes": completed, "updates": updates, "output_dir": str(output_dir)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train MAPPO or HAPPO on Strike Mission")
    parser.add_argument("--algo", choices=("mappo", "happo"), required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--rollout-episodes", type=int, default=ROLLOUT_EPISODES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="runs/ppo")
    args = parser.parse_args(argv)
    result = train(
        algorithm=args.algo,
        episodes=args.episodes,
        rollout_episodes=args.rollout_episodes,
        seed=args.seed,
        device=args.device,
        output_dir=args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
