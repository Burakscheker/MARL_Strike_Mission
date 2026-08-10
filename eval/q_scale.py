"""Q degerleri uzun egitimde SISIYOR MU? — Strike_Mission.md §11.14.

SORU: 3000 episode'luk kosuda hem VDN hem QMIX ep1000 civarinda tepe yapip
sonra bozuldu (VDN surv_ratio 0.0423 -> 0.0280, QMIX 0.0000'da kaldi; egitim
egrisinde olu 1.1 -> 1.9, adim 2270 -> 920). Kesif gurultusu DEGIL: epsilon
DUSTUKCE olum ARTTI. Suphe: gamma=0.9998 ile 2800 adimlik episode'larda
bootstrap hatalari birikiyor ve Q degerleri gercek getiriden kopuyor.

BU BETIK TAHMIN ETMEZ, OLCER. Ayni sabit gozlem kumesi uzerinde ep1000 ve
ep3000 checkpoint'lerinin Q'lari karsilastirilir.

REFERANS ARALIK (odul fonksiyonundan turetildi, §11.8):
    en iyi durum  : hedefe varis  ~ +50 - 1998*0.01        = +30
    en kotu durum : ikisi de olur ~ -30 -50 -2800*0.01     = -108
Yani SAGLIKLI bir Q, kabaca [-108, +30] araliginda olmali. Bunun disina
cikmasi (ozellikle buyuk pozitif) sisme demektir: ag, ulasilamayacak bir
getiri vaat ediyordur ve politika o vaade gore secim yapar.
"""
from __future__ import annotations

import os

import numpy as np
import torch

import config as C
from agents.qmix import QMixAgent
from agents.vdn import VDNAgent
from env.sampler import eval_map_seeds
from env.strike_env import StrikeMissionEnv

# Odulden turetilen saglikli aralik (yukaridaki not)
Q_LO = 2 * C.R_DEATH + C.R_ALL_DEAD + C.MAX_STEPS * C.R_STEP
Q_HI = C.R_FIRST_GOAL + C.R_SECOND_GOAL

CKPTS = (
    ("VDN  ep1000", "vdn", "runs/ckpt/s0_vdn_last.pt", 0.0423),
    ("VDN  ep3000", "vdn", "runs/ckpt/long_vdn_last.pt", 0.0280),
    ("QMIX ep1000", "qmix", "runs/ckpt/s0_qmix_last.pt", 0.0000),
    ("QMIX ep3000", "qmix", "runs/ckpt/long_qmix_last.pt", 0.0000),
)


def collect_obs(n_maps=3, per_map=250, seed=7):
    """SABIT gozlem kumesi: oracle yolu boyunca yurunerek toplanir.

    Neden oracle yolu: tum checkpoint'ler AYNI durumlarda karsilastirilmali,
    yoksa "kotu politika kotu durumlara gider, oradaki Q farkli olur" diye
    bir karistirici girer. Oracle yolu politikadan BAGIMSIZ bir referans.
    """
    from baselines.risk_oracle import RISK_W, direction_costs
    from train_bc import oracle_action

    env = StrikeMissionEnv(seed=seed, radar_random=True,
                           n_radar=C.N_RADAR, death_enabled=False)
    obs_list = []
    for ms in eval_map_seeds(n_maps):
        obs = env.reset(map_seed=ms, n_radar=C.N_RADAR)
        cost = direction_costs(env.zone, RISK_W, env.hazard_mode)
        for t in range(per_map):
            if env.done:
                break
            obs_list.append(obs[C.AGENT_1])
            acts = {a: oracle_action(env.pos[a], env.dist, cost, env.n)
                    for a in (C.AGENT_1, C.AGENT_2)}
            obs, _r, _d, _i = env.step(acts)
    return torch.as_tensor(np.asarray(obs_list, dtype=np.float32))


def main():
    X = collect_obs()
    print(f"sabit gozlem kumesi: {tuple(X.shape)} (oracle yolu boyunca)")
    print(f"odulden turetilen saglikli Q araligi: [{Q_LO:.0f}, {Q_HI:.0f}]\n")

    print(f"{'model':<13}{'ort Q':>10}{'max Q':>10}{'min Q':>10}"
          f"{'aksiyon bosl.':>15}{'aralik disi %':>15}{'surv_ratio':>12}")
    for label, algo, path, sr in CKPTS:
        if not os.path.exists(path):
            print(f"{label:<13}  (dosya yok: {path})")
            continue
        agent = (VDNAgent if algo == "vdn" else QMixAgent)(seed=0, device="cpu")
        agent.load(path)
        with torch.no_grad():
            Q = agent.online[C.AGENT_1](X)[:, :4]      # NOOP haric
        top2 = Q.topk(2, dim=1).values
        gap = (top2[:, 0] - top2[:, 1]).mean()
        out_of_range = ((Q < Q_LO) | (Q > Q_HI)).float().mean()
        print(f"{label:<13}{Q.mean():>10.2f}{Q.max():>10.2f}{Q.min():>10.2f}"
              f"{gap:>15.4f}{100*out_of_range:>14.1f}%{sr:>12.4f}")

    print("\nOKUMA: 'ort Q' ep1000'den ep3000'e BUYUYORSA sisme var demektir.")
    print("'aksiyon boslugu' kucukse ag aksiyonlari birbirinden ayiramiyor.")
    print("'aralik disi %' 0'dan buyukse Q, odulun MUMKUN kildigi degerlerin")
    print("disinda — bu kesin bir sisme kanitidir, yorum gerektirmez.")


if __name__ == "__main__":
    main()
