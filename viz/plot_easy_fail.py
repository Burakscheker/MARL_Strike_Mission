"""'COK KOLAY' (oracle_surv>=0.9) haritalarda ajanin neden basarisiz oldugunu
gorsellestirir — belirli map_seed listesi verilir, plot_failures.py'nin
cizim mantigini yeniden kullanir.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

import config as C
from baselines.risk_oracle import direction_costs, greedy_path, survival_prob
from env.strike_env import StrikeMissionEnv
from env.two_agent import play_episode_ppo, play_episode_qmix, play_episode_vdn

RUNNER = {"mappo": play_episode_ppo, "happo": play_episode_ppo,
         "vdn": play_episode_vdn, "qmix": play_episode_qmix}
BG = "#0d1117"
plt.style.use("dark_background")


def man(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def collect_seeds(agent, algo, seeds, max_steps):
    env = StrikeMissionEnv(seed=12345, radar_random=True, n_radar=C.N_RADAR,
                           max_steps=max_steps, death_enabled=False)
    runner = RUNNER[algo]
    out = []
    for ms in seeds:
        info, _ = runner(env, agent, train=False,
                         reset_kwargs={"map_seed": ms, "n_radar": C.N_RADAR})
        z = env.zone.copy()
        orc = greedy_path(C.START, C.GOAL, env.dist,
                          direction_costs(z, mode=C.HAZARD_MODE))
        out.append({
            "seed": ms, "zone": z, "oracle": orc,
            "p1": list(env.path[C.AGENT_1]), "p2": list(env.path[C.AGENT_2]),
            "reached": bool(info["reached1"] or info["reached2"]),
            "steps": int(info["steps"]),
            "orc_s": survival_prob(orc, z, C.HAZARD_MODE),
            "orc_len": len(orc) - 1,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="vdn", choices=("mappo", "happo", "vdn", "qmix"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--max-steps", type=int, default=C.MAX_STEPS)
    ap.add_argument("--out", default="runs/fig_easy_fail.png")
    args = ap.parse_args()

    from train import build_agent
    agent = build_agent(args.algo, 0, "cpu")
    agent.load(args.ckpt)

    data = collect_seeds(agent, args.algo, args.seeds, args.max_steps)

    cols = min(3, len(data))
    rows = (len(data) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 4.9 * rows))
    fig.patch.set_facecolor(BG)
    cmap = matplotlib.colors.ListedColormap(["#0d1117", "#4a3a12", "#5c1f1f"])

    for ax, e in zip(np.atleast_1d(axes).ravel(), data):
        ax.imshow(e["zone"], cmap=cmap, origin="upper", interpolation="nearest")
        oy, ox = zip(*e["oracle"])
        ax.plot(ox, oy, color="#8b949e", lw=1.5, ls="--", label="oracle")
        notes = []
        for pk, c, lb in (("p1", "#3ddc97", "ucak 1"), ("p2", "#58a6ff", "ucak 2")):
            if not e[pk]:
                continue
            py, px = zip(*e[pk])
            ax.plot(px, py, color=c, lw=1.7, label=lb)
            end = e[pk][-1]
            ax.plot(end[1], end[0], "x", color=c, ms=11, mew=2.5)
            notes.append(f"{lb}: kalan {man(end, C.GOAL)}")
        ax.plot(0, 0, "o", color="#58a6ff", ms=8)
        ax.plot(C.GOAL[1], C.GOAL[0], "*", color="#ff7b72", ms=15)
        capped = e["steps"] >= args.max_steps
        ax.set_title(
            f"{e['seed']}   adim {e['steps']}/{args.max_steps}"
            f"{'  BUTCE BITTI' if capped else '  (limite dayanmadi)'}\n"
            + "   ".join(notes)
            + f"\noracle {e['orc_len']} adimda varirdi (surv={e['orc_s']:.2f})",
            fontsize=8.5, color="#ff7b72" if capped else "#e3b341")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in np.atleast_1d(axes).ravel()[len(data):]:
        ax.axis("off")

    np.atleast_1d(axes).ravel()[0].legend(fontsize=7, framealpha=0.25, loc="lower left")
    fig.suptitle(
        f"COK KOLAY haritalarda (oracle risksiz gecebiliyor) NEDEN BASARISIZ — "
        f"{args.algo.upper()}, zar KAPALI\n"
        f"X = rotanin bittigi yer   koyu sari: dis halka   koyu kirmizi: ic halka",
        fontsize=12, y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("yazildi:", args.out)


if __name__ == "__main__":
    main()
