import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from agents.ppo import load_checkpoint
from config import MAX_STEPS
from env.strike_env import StrikeMissionEnv


def _route_overlap(routes):
    first, second = map(set, routes)
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def evaluate(trainer, episodes, seed=10_000, max_steps=MAX_STEPS):
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    env = StrikeMissionEnv(max_steps=max_steps)
    rows = []
    started = time.perf_counter()

    for episode in range(episodes):
        observations, _ = env.reset(seed=seed + episode)
        terminated = truncated = False
        episode_return = 0.0
        deaths = 0
        radar_entries = 0
        while not (terminated or truncated):
            actions, _, _ = trainer.act(
                observations,
                env.action_masks(),
                state=env.state(),
                deterministic=True,
            )
            observations, reward, terminated, truncated, info = env.step(actions)
            episode_return += reward
            deaths += info["deaths"]
            radar_entries += info["radar_entries"]

        rows.append(
            {
                "episode": episode,
                "seed": seed + episode,
                "team_success": info["success"],
                "both_reached": info["both_reached"],
                "aircraft_0_reached": bool(env.reached[0]),
                "aircraft_1_reached": bool(env.reached[1]),
                "aircraft_0_dead": not bool(env.alive[0]),
                "aircraft_1_dead": not bool(env.alive[1]),
                "deaths": deaths,
                "radar_entries": radar_entries,
                "steps": info["steps"],
                "return": episode_return,
                "route_overlap": _route_overlap(env.routes),
                "terminated": terminated,
                "truncated": truncated,
            }
        )

    summary = {
        "algorithm": trainer.algorithm,
        "episodes": episodes,
        "team_success_rate": float(np.mean([row["team_success"] for row in rows])),
        "both_reached_rate": float(np.mean([row["both_reached"] for row in rows])),
        "aircraft_0_reach_rate": float(
            np.mean([row["aircraft_0_reached"] for row in rows])
        ),
        "aircraft_1_reach_rate": float(
            np.mean([row["aircraft_1_reached"] for row in rows])
        ),
        "aircraft_0_death_rate": float(
            np.mean([row["aircraft_0_dead"] for row in rows])
        ),
        "aircraft_1_death_rate": float(
            np.mean([row["aircraft_1_dead"] for row in rows])
        ),
        "timeout_rate": float(np.mean([row["truncated"] for row in rows])),
        "mean_radar_entries": float(np.mean([row["radar_entries"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "route_overlap_mean": float(np.mean([row["route_overlap"] for row in rows])),
        "wall_seconds": time.perf_counter() - started,
    }
    return rows, summary


def evaluate_checkpoint(
    checkpoint_path,
    output_dir,
    episodes,
    seed=10_000,
    max_steps=None,
    device="cpu",
):
    trainer, training_config = load_checkpoint(checkpoint_path, device=device)
    horizon = training_config.get("max_steps", MAX_STEPS) if max_steps is None else max_steps
    rows, summary = evaluate(trainer, episodes, seed=seed, max_steps=horizon)
    summary["training_seed"] = training_config.get("seed")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    report_lines = [
        "# Strike Mission Evaluation",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    report_lines.extend(
        "| {} | {:.6g} |".format(name.replace("_", " "), value)
        for name, value in summary.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    (output_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a Strike Mission checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="runs/eval")
    args = parser.parse_args(argv)
    summary = evaluate_checkpoint(
        args.checkpoint,
        args.output,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
