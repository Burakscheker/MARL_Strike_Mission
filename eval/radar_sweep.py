"""Radar sayisina gore VARIS orani — "sorun navigasyon mu, radar mi?" teshisi.

Burak'in sorusu (2026-08-08): "sorunumuz hedefe gitmekse, 10 radarli curriculum
basinda da benzer basari almamiz gerekmiyor mu?"

Dogru teshis bunun bir adim otesi: SIFIR radarda ne oluyor? Bos haritada
B'den H'ye gidemiyorsa radarlarin hicbir onemi yok — sorun radar kacinma
degil, saf navigasyon demektir.

Olculen: route_reached (rotasi hedefe VARIYOR mu), ZAR KAPALI. Ucak olmuyor,
2800 adim serbestce dolasabiliyor. Yani bu sayi sansla ilgili degil, tamamen
politikanin hedefi bulup bulamadigini olcer.
"""
from __future__ import annotations

import argparse

import numpy as np

import config as C
from env.sampler import eval_map_seeds
from env.strike_env import StrikeMissionEnv
from env.two_agent import play_episode_ppo, play_episode_qmix, play_episode_vdn

RUNNER = {"mappo": play_episode_ppo, "happo": play_episode_ppo,
         "vdn": play_episode_vdn, "qmix": play_episode_qmix}


def sweep(agent, algo, radar_counts, n_maps=20, seed=12345):
    # death_enabled=False: sadece rotayi olcuyoruz, olum karismasin.
    env = StrikeMissionEnv(seed=seed, radar_random=True, n_radar=C.N_RADAR,
                           death_enabled=False)
    runner = RUNNER[algo]
    rows = []
    for nr in radar_counts:
        reached, steps, mindist = 0, [], []
        for ms in eval_map_seeds(n_maps):
            env.rng = np.random.default_rng(seed)
            info, _ = runner(env, agent, train=False,
                             reset_kwargs={"map_seed": ms, "n_radar": nr})
            if info["reached1"] or info["reached2"]:
                reached += 1
            steps.append(info["steps"])
            # Hedefe EN COK ne kadar yaklasabildi (varamasa bile ilerleme
            # olcusu — "hic kimildamiyor" ile "az kaldi" ayrilsin).
            d = min(abs(p[0] - C.GOAL[0]) + abs(p[1] - C.GOAL[1])
                    for a in (C.AGENT_1, C.AGENT_2) for p in env.path[a])
            mindist.append(d)
        rows.append((nr, reached / n_maps, np.mean(steps), np.mean(mindist)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=("mappo", "happo", "vdn", "qmix"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--maps", type=int, default=20)
    ap.add_argument("--radars", default="0,5,10,20,30")
    args = ap.parse_args()

    from train import build_agent
    agent = build_agent(args.algo, 0, "cpu")
    agent.load(args.ckpt)
    print(f"model: {args.ckpt}   ({args.maps} harita, ZAR KAPALI)\n")

    opt = 2 * (C.GRID_N - 1)
    print(f"{'radar':>6}{'VARIS':>9}{'adim':>8}{'hedefe en yakin':>17}")
    print(f"{'':>6}{'':>9}{'':>8}{'(optimal 0, baslangic ' + str(opt) + ')':>17}")
    for nr, rate, st, md in sweep(agent, args.algo,
                                  [int(x) for x in args.radars.split(",")],
                                  args.maps):
        print(f"{nr:>6}{100*rate:>8.0f}%{st:>8.0f}{md:>17.0f}")

    print("\nOKUMA: 0 radarda VARIS dusukse sorun RADAR KACINMA DEGIL, saf")
    print("navigasyon. 'hedefe en yakin' baslangictaki 1998'e yakinsa ajan")
    print("haritada ilerlemiyor demektir.")


if __name__ == "__main__":
    main()
