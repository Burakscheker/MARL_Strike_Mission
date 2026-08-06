"""Odul BUYUKLUGU ogrenmeye yansiyor mu? Huber doygunlugu hipotezi.

R_ALL_DEAD'i -25 / -80 / -5000 yapip AYNI tohumla kisa egitim kosuyoruz.
Trajektoriler ayni cikarsa odul buyuklugu gradyana hic girmiyor demektir
(smooth_l1_loss |hata|>1'de sadece ISARETI tasir).
"""
import numpy as np
import torch

import config as C
import env.strike_env as SE
from env.sampler import curriculum_n_radar
from env.two_agent import play_episode_vdn
from train import build_agent

EPISODES = 25


def run(all_dead_value):
    SE.R_ALL_DEAD = all_dead_value          # modul seviyesinde ezildi
    torch.manual_seed(0)
    np.random.seed(0)
    agent = build_agent("vdn", 0, "cpu")
    agent.load("runs/ckpt/vdn_fixed.pt")
    env = SE.StrikeMissionEnv(seed=0, radar_random=True)
    sig = []
    for ep in range(1, EPISODES + 1):
        agent.set_eps_progress(min(1.0, ep / (EPISODES * C.EPS_FLOOR_FRAC)))
        info, _ = play_episode_vdn(
            env, agent, train=True,
            reset_kwargs={"n_radar": curriculum_n_radar(ep, EPISODES)})
        sig.append((info["steps"], info["n_dead"], info["outer_total"]))
    return sig


base = run(-25.0)
for v in (-80.0, -5000.0):
    s = run(v)
    same = (s == base)
    print(f"R_ALL_DEAD={v:>9.0f} -> trajektori {'AYNI' if same else 'FARKLI'}")
    if not same:
        d = next(i for i, (a, b) in enumerate(zip(base, s)) if a != b)
        print(f"   ilk fark episode {d+1}: {base[d]} vs {s[d]}")

print(f"\nreferans (-25) ilk 5 episode: {base[:5]}")
print("\nHIPOTEZ: hepsi AYNI ise odul BUYUKLUGU gradyana girmiyor —")
print("smooth_l1_loss |TD hatasi|>1'de sabit egimli, sadece isaret tasiniyor.")
