"""Neden hedefe varamadilar? — BASARISIZ rotalarin teshis figuru.

fig4 basari/basarisizlik karisik gosteriyor. Bu figur SADECE varamayanlari
alir ve her panelde "nerede kaldi, neden durdu" sorusunu cevaplayacak
bilgiyi yazar:

  son konum  : rota nerede bitti (X ile isaretli)
  kalan       : oradan hedefe manhattan mesafe (0 = varmis)
  adim        : kullanilan adim / limit  -> limite dayandiysa BUTCE bitti,
                dayanmadiysa ajan takildi/dolandi demektir

ZAR KAPALI kosuluyor: ucak olmuyor, 4000 adim serbestce dolasabiliyor.
Yani buradaki her basarisizlik sansizlik degil, POLITIKANIN kendi hatasi.
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
from env.sampler import eval_map_seeds
from env.strike_env import StrikeMissionEnv
from env.two_agent import play_episode_ppo, play_episode_qmix, play_episode_vdn

RUNNER = {"mappo": play_episode_ppo, "happo": play_episode_ppo,
         "vdn": play_episode_vdn, "qmix": play_episode_qmix}
BG = "#0d1117"
plt.style.use("dark_background")


def man(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def collect(agent, algo, n_scan, max_steps, dice_on=False, seed=12345):
    env = StrikeMissionEnv(seed=seed, radar_random=True, n_radar=C.N_RADAR,
                           max_steps=max_steps, death_enabled=dice_on)
    if dice_on:
        env.rng = np.random.default_rng(seed)
    runner = RUNNER[algo]
    out = []
    for ms in eval_map_seeds(n_scan):
        info, _ = runner(env, agent, train=False,
                         reset_kwargs={"map_seed": ms, "n_radar": C.N_RADAR})
        z = env.zone.copy()
        orc = greedy_path(C.START, C.GOAL, env.dist,
                          direction_costs(z, mode=C.HAZARD_MODE))
        out.append({
            "seed": ms, "zone": z, "oracle": orc,
            "p1": list(env.path[C.AGENT_1]), "p2": list(env.path[C.AGENT_2]),
            "dead1": bool(dice_on and not env.alive[C.AGENT_1]),
            "dead2": bool(dice_on and not env.alive[C.AGENT_2]),
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
    ap.add_argument("--scan", type=int, default=20, help="taranacak harita")
    ap.add_argument("--panels", type=int, default=6)
    ap.add_argument("--mode", default="fail", choices=("fail", "risky"),
                    help="fail = varamayanlar; risky = VARAN ama rotasi "
                         "oracle'a gore en tehlikeli olanlar")
    ap.add_argument("--max-steps", type=int, default=C.MAX_STEPS)
    ap.add_argument("--dice-on", action="store_true",
                    help="zar ACIK kosuluyor (gercek olum riski); "
                         "varsayilan zar KAPALI (rotanin kendisini teshis eder)")
    ap.add_argument("--seed", type=int, default=12345,
                    help="zar-acik modda hangi zar dizisi (tekrarlanabilirlik)")
    ap.add_argument("--out", default="runs/fig_failures.png")
    args = ap.parse_args()

    from train import build_agent
    agent = build_agent(args.algo, 0, "cpu")
    agent.load(args.ckpt)

    data = collect(agent, args.algo, args.scan, args.max_steps,
                   dice_on=args.dice_on, seed=args.seed)
    if args.mode == "fail":
        bad = [d for d in data if not d["reached"]][:args.panels]
    else:
        # VARAN ama tehlikeli: rota guvenligi oracle'in ne kadar altinda
        ok = [d for d in data if d["reached"]]
        ok.sort(key=lambda d: survival_prob(d["p1"], d["zone"], C.HAZARD_MODE))
        bad = ok[:args.panels]
    n_ok = sum(d["reached"] for d in data)
    print(f"{len(data)} harita tarandi, {n_ok} varis, {len(data)-n_ok} basarisiz")
    if not bad:
        print("basarisiz harita yok"); return

    cols = min(3, len(bad))
    rows = (len(bad) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 4.9 * rows))
    fig.patch.set_facecolor(BG)
    cmap = matplotlib.colors.ListedColormap(["#0d1117", "#4a3a12", "#5c1f1f"])

    for ax, e in zip(np.atleast_1d(axes).ravel(), bad):
        ax.imshow(e["zone"], cmap=cmap, origin="upper", interpolation="nearest")
        oy, ox = zip(*e["oracle"])
        ax.plot(ox, oy, color="#8b949e", lw=1.5, ls="--", label="oracle")
        notes = []
        for pk, dk, c, lb in (("p1", "dead1", "#3ddc97", "ucak 1"),
                              ("p2", "dead2", "#58a6ff", "ucak 2")):
            if not e[pk]:
                continue
            py, px = zip(*e[pk])
            ax.plot(px, py, color=c, lw=1.7, label=lb)
            end = e[pk][-1]
            died = e.get(dk, False)
            marker = "X" if died else "x"
            ax.plot(end[1], end[0], marker, color=c, ms=13 if died else 11,
                    mew=2.5, mec="#ff7b72" if died else c)
            tag = " OLDU" if died else ""
            notes.append(f"{lb}{tag}: kalan {man(end, C.GOAL)}")
        ax.plot(0, 0, "o", color="#58a6ff", ms=8)
        ax.plot(C.GOAL[1], C.GOAL[0], "*", color="#ff7b72", ms=15)
        capped = e["steps"] >= args.max_steps
        ax.set_title(
            f"{e['seed']}   adim {e['steps']}/{args.max_steps}"
            f"{'  BUTCE BITTI' if capped else '  (limite dayanmadi)'}\n"
            + "   ".join(notes)
            + f"\noracle {e['orc_len']} adimda varirdi",
            fontsize=8.5, color="#ff7b72" if capped else "#e3b341")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in np.atleast_1d(axes).ravel()[len(bad):]:
        ax.axis("off")

    np.atleast_1d(axes).ravel()[0].legend(fontsize=7, framealpha=0.25,
                                          loc="lower left")
    zar_txt = "zar ACIK (gercek olum riski)" if args.dice_on else "zar KAPALI (ucak olmuyor)"
    x_txt = "buyuk X (kirmizi kenarli) = OLDU   " if args.dice_on else ""
    fig.suptitle(
        f"NEDEN VARAMADILAR — {args.algo.upper()}, {zar_txt}\n"
        f"{x_txt}kucuk x = rotanin bittigi yer   koyu sari: dis halka   koyu kirmizi: ic halka",
        fontsize=12, y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("yazildi:", args.out)


if __name__ == "__main__":
    main()
