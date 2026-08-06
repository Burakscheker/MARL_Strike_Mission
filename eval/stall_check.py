"""Takildigi noktada odul ne diyor: engeli GEC mi, KAL mi?"""
import numpy as np
import config as C
from env.sampler import eval_map_seeds
from env.strike_env import StrikeMissionEnv
from env.two_agent import play_episode_vdn
from train import build_agent

agent = build_agent("vdn", 0, "cpu"); agent.load("runs/ckpt/vdn_v2.pt")
env = StrikeMissionEnv(seed=1, radar_random=True, death_enabled=False)

print("Takilma noktasinda TEK ADIMLIK odul (shaping + risk + adim):")
print(f"{'harita':>7}{'kaldigi z':>10}{'en iyi komsu':>14}{'GECMEK':>9}{'KALMAK':>9}")
gec, kal = [], []
for ms in eval_map_seeds(6):
    info, _ = play_episode_vdn(env, agent, train=False,
                               reset_kwargs={"map_seed": ms, "n_radar": C.N_RADAR})
    p = list(info["path1"])
    stall = p[-1]                      # takildigi yer
    d, sc, z = env.dist, env.dist_scale, env.zone
    phi = lambda c: 1.0 - min(1.0, float(d[c]) / sc)
    best, best_v = None, -1e9
    for dr, dc in C.DIRS:
        nb = (stall[0]+dr, stall[1]+dc)
        if not (0 <= nb[0] < C.GRID_N and 0 <= nb[1] < C.GRID_N):
            continue
        zt, zf = int(z[nb]), int(z[stall])
        p_ent = (C.P_INNER_TOTAL if zt == 2 else C.P_OUTER_TOTAL) if zt > zf else 0.0
        r = C.R_STEP + C.SHAPING_COEF*(C.GAMMA*phi(nb) - phi(stall)) - C.R_RISK_COEF*p_ent
        if r > best_v:
            best_v, best = r, (nb, zt, r)
    # KALMAK: yerinde salinim -> Phi degismez, sadece drag + adim maliyeti
    stay = C.R_STEP + C.SHAPING_COEF*(C.GAMMA - 1.0)*phi(stall)
    gec.append(best_v); kal.append(stay)
    print(f"{ms-C.EVAL_SEED_BASE:>7}{int(z[stall]):>10}{f'z={best[1]}':>14}"
          f"{best_v:>9.3f}{stay:>9.3f}")
print(f"\nortalama  GECMEK {np.mean(gec):+.3f}   KALMAK {np.mean(kal):+.3f}")
print("GECMEK daha yuksekse odul dogru, sorun OGRENMEDE.")
print("KALMAK daha yuksekse odul yanlis, reward shaping gerekli.")
