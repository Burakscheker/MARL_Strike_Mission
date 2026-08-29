"""Oracle davranis klonlama (BC) on-egitimi — Strike_Mission.md §11.12.

NEDEN: 50 held-out haritada olculdu (vdn_v3.pt, 1000 episode RL):
    oracle tavani      0.3561   (takim %45.8)
    ajanin rotasi      0.0071   -> oracle'in %2.8'i
    rotasi hedefe variyor %46
Yani ajan navige etmeyi kismen ogrendi ama radarlardan KACMAYI ogrenemedi.

Kritik gozlem: DOGRU CEVAP ZATEN GOZLEMDE. observe()'un 12-15. skalarlari
    d_own - d_komsu   (4 komsu icin, risk-farkindalı mesafe haritasindan)
ve bu 4 sayinin argmax'ini alan politika TAM oracle yolunu cizer (greedy_path
bunu yapiyor). Yani bilgi mevcut, aksiyon uzayinda tek adimlik bir karar —
ama ag 200k gradyan adiminda bunu kullanmayi ogrenemedi. Bu bir ODUL degil
OPTIMIZASYON problemi: seyrek odulle 898 boyutlu girdinin icindeki 4 sayiya
"dikkat etmeyi" kesfetmek zor.

COZUM: once GOZETIMLI olarak oracle aksiyonunu taklit ettir (capraz entropi),
sonra RL ile ince ayar yap. Haritayi KOLAYLASTIRMAZ — ayni 40 radar, ayni
kurallar; sadece ogrenmenin baslangic noktasi degisir.

DAgger tarzi veri toplama: uzman yolunu takip ederken epsilon kadar RASTGELE
sapma yapilir, boylece ag "yoldan ciktim, nasil donerim" durumlarini da
gorur. Sadece temiz uzman yolunda egitilen politika ilk hatada dagilir.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

import config as C
from agents.vdn import VDNAgent
from baselines.risk_oracle import RISK_W, direction_costs, oracle_action
from env.sampler import curriculum_n_radar
from env.strike_env import StrikeMissionEnv


def collect_episode(env, eps: float, rng, stride: int, n_radar: int,
                    map_seed: int):
    """Uzmani epsilon-gurultuyle takip et, (gozlem, uzman_aksiyonu) topla."""
    obs = env.reset(map_seed=map_seed, n_radar=n_radar)
    cost = direction_costs(env.zone, RISK_W, env.hazard_mode)
    X, Y = [], []
    step = 0
    while not env.done:
        acts = {}
        for a in (C.AGENT_1, C.AGENT_2):
            exp_a = oracle_action(env.pos[a], env.dist, cost, env.n)
            if step % stride == 0 and not env.terminal(a):
                X.append(obs[a])
                Y.append(exp_a)
            # DAgger: bazen kasten sap ki toparlanma durumlari da gorulsun
            acts[a] = (int(rng.integers(0, 4)) if rng.random() < eps else exp_a)
        obs, _r, _done, _info = env.step(acts)
        step += 1
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=120,
                    help="uzman episode sayisi (veri toplama)")
    ap.add_argument("--eps", type=float, default=0.15,
                    help="DAgger sapma orani")
    ap.add_argument("--stride", type=int, default=4,
                    help="her N adimda bir ornek al (bellek icin)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-radar", type=int, default=None,
                    help="sabitle; verilmezse curriculum 10->40")
    ap.add_argument("--out", default="runs/ckpt/vdn_bc.pt")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = StrikeMissionEnv(seed=args.seed, radar_random=True,
                           n_radar=C.N_RADAR, death_enabled=False)
    # death_enabled=False: uzman verisi toplarken ucagin olmesi episode'u
    # yarida keser ve yolun GERI KALANINI hic gormeyiz. Uzmani sonuna kadar
    # izlemek istiyoruz; olum dinamigini RL asamasi ogretecek.

    agent = VDNAgent(seed=args.seed, device="cpu")
    params = (list(agent.online[C.AGENT_1].parameters())
              + list(agent.online[C.AGENT_2].parameters()))
    opt = torch.optim.Adam(params, lr=args.lr)

    print(f"BC on-egitimi: {args.episodes} uzman episode, eps={args.eps}, "
          f"stride={args.stride}, {args.epochs} epoch")
    t0 = time.perf_counter()
    X, Y = [], []
    for ep in range(1, args.episodes + 1):
        nr = args.n_radar or curriculum_n_radar(ep, args.episodes)
        ms = int(rng.integers(0, C.TRAIN_SEED_MAX))
        xs, ys = collect_episode(env, args.eps, rng, args.stride, nr, ms)
        X.extend(xs); Y.extend(ys)
        if ep % 5 == 0:
            print(f"  veri toplama ep{ep:>4}/{args.episodes}  radar={nr:>2}  "
                  f"ornek={len(X):>7}  {time.perf_counter()-t0:.0f}s", flush=True)

    X = torch.as_tensor(np.asarray(X, dtype=np.float32))
    Y = torch.as_tensor(np.asarray(Y, dtype=np.int64))
    print(f"veri: {tuple(X.shape)}  uzman aksiyon dagilimi="
          f"{np.bincount(Y.numpy(), minlength=5).tolist()}")

    # Q degerlerini logit gibi kullanip capraz entropi: argmax Q = uzman
    # aksiyonu olsun. Buyukluk RL asamasinda yeniden oturur.
    lossf = nn.CrossEntropyLoss()
    n = len(X)
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n)
        tot, cor, nb = 0.0, 0, 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = X[idx], Y[idx]
            loss = 0.0
            for a in (C.AGENT_1, C.AGENT_2):
                q = agent.online[a](xb)
                loss = loss + lossf(q, yb)
                if a == C.AGENT_1:
                    cor += int((q.argmax(1) == yb).sum())
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, C.GRAD_CLIP)
            opt.step()
            tot += loss.item(); nb += 1
        print(f"epoch {epoch}: loss={tot/nb:.4f}  uzman-eslesmesi="
              f"%{100*cor/n:.1f}", flush=True)

    for a in (C.AGENT_1, C.AGENT_2):
        agent.target[a].load_state_dict(agent.online[a].state_dict())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    agent.save(args.out)
    print(f"\nyazildi: {args.out}  ({time.perf_counter()-t0:.0f}s)")
    print("Sonraki adim: train.py --algo vdn --resume-from "
          f"{args.out} --eps-start 0.1")


if __name__ == "__main__":
    main()
