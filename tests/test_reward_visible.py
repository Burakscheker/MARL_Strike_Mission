"""Odul BUYUKLUGU ogrenmeye yansiyor mu? — Strike_Mission.md §11.11

SORU: R_ALL_DEAD'i buyutunce ajanin davranisi degisiyor mu?
Degismiyorsa odul fonksiyonunun o kismi OLU demektir — kalibrasyon
(kapilar, R_TIMEOUT, R_ALL_DEAD) sadece kagit uzerinde dogru olur.

Bulgu (2026-08-06): olceklenmemis odul + varsayilan Huber ile R_ALL_DEAD'i
-25'ten -5000'e cikarmak (200 KAT) trajektoriyi HIC degistirmiyordu.
Iki etki ust uste biniyordu:
  1) smooth_l1_loss |hata| > beta'da gradyani +-1'e sabitler (buyukluk degil
     ISARET tasinir); terminal oduller -50..-5000, daima doygun bolgede.
  2) LR 3e-5 x gradyan tavani 1 -> Q tek guncellemede 3e-5 hareket eder,
     hedefe (~+-100) hic yaklasamaz.

Bu betik dort konfigurasyonu AYNI tohumla karsilastirir ve hangisinin
odulu gorunur kildigini OLCER.

CLI:
    python -m tests.test_reward_visible            # tam matris
    python -m tests.test_reward_visible --check    # sadece aktif config
                                                   # (regresyon testi)
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

import config as C
import env.strike_env as SE
from env.sampler import curriculum_n_radar
from env.two_agent import play_episode_vdn
from train import build_agent

EPISODES = 25
CKPT = "runs/ckpt/vdn_fixed.pt"


def run(all_dead: float, reward_scale: float, huber_beta: float,
        episodes: int = EPISODES) -> list:
    """Kisa egitim kosusu; donen imza (adim, olu, maruziyet) dizisi."""
    import agents.vdn as VDN
    SE.R_ALL_DEAD = all_dead
    SE.REWARD_SCALE = reward_scale
    VDN.HUBER_BETA = huber_beta

    torch.manual_seed(0)
    np.random.seed(0)
    agent = build_agent("vdn", 0, "cpu")
    # N_SCALARS 16->18 (son-hareket ozelligi) sabit CKPT'yi ESKI sekle
    # kilitledi — KATI agent.load() artik SEKIL UYUSMAZLIGI ile patlar.
    # load_matching() UYANLARI yukler, uymayan 2 katmani (scalar_enc,
    # head ilk katmanlari) rasgele baslatilmis birakir — bu testin amaci
    # icin (trajektori DEGISIYOR mu, mutlak performans degil) yeterli.
    import agents.transfer as transfer
    ckpt = torch.load(CKPT, map_location="cpu")
    transfer.load_matching(agent.online[C.AGENT_1], ckpt["online1"], "A1")
    transfer.load_matching(agent.online[C.AGENT_2], ckpt["online2"], "A2")
    for a in (C.AGENT_1, C.AGENT_2):
        agent.target[a].load_state_dict(agent.online[a].state_dict())
    env = SE.StrikeMissionEnv(seed=0, radar_random=True)
    sig = []
    for ep in range(1, episodes + 1):
        agent.set_eps_progress(min(1.0, ep / (episodes * C.EPS_FLOOR_FRAC)))
        info, _ = play_episode_vdn(
            env, agent, train=True,
            reset_kwargs={"n_radar": curriculum_n_radar(ep, episodes)})
        sig.append((info["steps"], info["n_dead"], info["outer_total"]))
    return sig


def visible(scale: float, beta: float, episodes: int = EPISODES) -> bool:
    """R_ALL_DEAD'i 4 katina cikarmak trajektoriyi DEGISTIRIYOR mu?"""
    a = run(-25.0, scale, beta, episodes)
    b = run(-100.0, scale, beta, episodes)
    return a != b


def matrix():
    print(f"{EPISODES} episode, ayni tohum. 'GORUNUR' = R_ALL_DEAD -25 -> -100 "
          f"trajektoriyi degistiriyor.\n")
    print(f"{'REWARD_SCALE':>14}{'HUBER_BETA':>12}{'odul gorunur mu':>18}")
    for scale, beta in ((1.0, 1.0), (1.0, 50.0), (0.05, 1.0), (0.05, 50.0),
                       (1.0, 2.0), (1.0, 5.0), (1.0, 10.0), (1.0, 25.0)):
        v = visible(scale, beta)
        print(f"{scale:>14}{beta:>12}{('EVET' if v else 'hayir'):>18}")
    print("\nSecim bu tabloya gore yapilir; config.REWARD_SCALE / HUBER_BETA.")


def check():
    """Regresyon: AKTIF config odulu gorunur kiliyor mu?"""
    v = visible(C.REWARD_SCALE, C.HUBER_BETA)
    status = "GECTI" if v else "KALDI"
    print(f"  [{status}] odul buyuklugu ogrenmeye yansiyor "
          f"(REWARD_SCALE={C.REWARD_SCALE}, HUBER_BETA={C.HUBER_BETA})")
    if not v:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    check() if args.check else matrix()
