"""Ajan vs oracle: kac AYRI bolgeye giriyor? Reward shaping'in dogru olcusu."""
import numpy as np, torch
import config as C
from baselines.risk_oracle import (RISK_W, direction_costs, entries,
                                   greedy_path, survival_prob)
from env.sampler import eval_map_seeds
from env.strike_env import StrikeMissionEnv
from env.two_agent import play_episode_vdn
from train import build_agent

agent = build_agent("vdn", 0, "cpu"); agent.load("runs/ckpt/vdn_v2.pt")
# zar KAPALI: ajanin niyet ettigi TAM rota gorulsun
renv = StrikeMissionEnv(seed=1, radar_random=True, death_enabled=False)
oenv = StrikeMissionEnv(seed=1, radar_random=True)

rows = []
for ms in eval_map_seeds(20):
    oenv.reset(map_seed=ms, n_radar=C.N_RADAR)
    z = oenv.zone
    orc = greedy_path(C.START, C.GOAL, oenv.dist,
                      direction_costs(z, RISK_W, C.HAZARD_MODE))
    oo, oi = entries(orc, z)
    info, _ = play_episode_vdn(renv, agent, train=False,
                              reset_kwargs={"map_seed": ms, "n_radar": C.N_RADAR})
    p1 = list(info["path1"])
    ao, ai = entries(p1, z)
    rows.append((oo, oi, survival_prob(orc, z), len(orc)-1,
                 ao, ai, survival_prob(p1, z), len(p1)-1,
                 bool(info["reached1"] or info["reached2"])))

a = np.array([r[:8] for r in rows], dtype=float)
reach = sum(r[8] for r in rows)
print(f"{'':<12}{'dis giris':>11}{'ic giris':>10}{'hayatta':>10}{'adim':>8}")
print(f"{'ORACLE':<12}{a[:,0].mean():>11.1f}{a[:,1].mean():>10.1f}"
      f"{a[:,2].mean():>10.3f}{a[:,3].mean():>8.0f}")
print(f"{'AJAN':<12}{a[:,4].mean():>11.1f}{a[:,5].mean():>10.1f}"
      f"{a[:,6].mean():>10.3f}{a[:,7].mean():>8.0f}")
print(f"\nAjanin rotasi hedefe variyor: {reach}/20")
print(f"Ajan yolu / optimal uzunluk: {a[:,7].mean()/1998:.2f}x")
