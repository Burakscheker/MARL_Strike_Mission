"""Scripted baseline politikalari + "egitimsiz ag" kontrolu.

Kosum: python -m baselines.policies

BU MODULUN VAROLUS SEBEBI (olculdu, tahmin degil): ilk duman testinde 4
episode'luk VDN kosusu eval'de %100 takim basarisi, TAM 1998 adim ve SIFIR
radar maruziyeti verdi. Bu ogrenme olamazdi. Hipotez: rastgele baslatilmis
bir ag neredeyse SABIT Q degerleri uretir, argmax tek bir aksiyonu (ornegin
SAG) secer, ucak sag kenara dayanip maske SAG'i kapatinca ASAGI'ya doner —
yani "once saga sonra asagi" L yolunu cizer. O yol da bu haritada tam olarak
SIFIR RISKLI OPTIMAL yoldur.

Yani sabit haritada TRIVIAL bir politika mukemmel skor aliyor. Bu modul o
hipotezi dogrudan test eder; sonuc Strike_Mission.md §0'a islenir.
"""
from __future__ import annotations

import numpy as np

import config as C
from baselines.risk_oracle import exposure, greedy_path, risk_distance_map, survival_prob
from env.strike_env import StrikeMissionEnv


class ConstantPolicy:
    """Her zaman AYNI aksiyonu dener; maskeliyse listedeki bir sonrakini.

    "Egitimsiz/dejenere ag" davranisinin scripted karsiligi.
    """

    def __init__(self, order=(C.RIGHT, C.DOWN, C.UP, C.LEFT)):
        self.order = order

    def act(self, agent_id, obs, mask, eps=None):
        for a in self.order:
            if mask[a] > 0:
                return a
        return C.NOOP


class OraclePolicy:
    """Risk-mesafe haritasinda tepe inisi — en guvenli yol."""

    def __init__(self):
        self.d = risk_distance_map()

    def act(self, agent_id, obs, mask, eps=None):
        raise NotImplementedError("scripted yol icin greedy_path kullan")


class RandomMonotonePolicy:
    """Hedefe dogru rastgele monoton adim (radar koru) — ASIL baseline.
    'random walk' DEGIL (bkz. Strike_Mission.md §5)."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, agent_id, obs, mask, eps=None):
        opts = [a for a in (C.RIGHT, C.DOWN) if mask[a] > 0]
        if not opts:
            opts = [a for a in range(4) if mask[a] > 0]
        return int(self.rng.choice(opts))


def run(policy, episodes=20, seed=0, alert=False):
    from env.two_agent import play_episode_vdn
    # SABIT harita: bu betik §0.3'teki referans sayilari (merdiven / L yolu /
    # oracle / rastgele monoton) yeniden uretiyor, onlar sabit haritaya ait.
    # Rastgele haritadaki baseline'lar eval/evaluate.py'nin isi (ortak harita
    # seti gerekiyor, bkz. env/sampler.eval_map_seeds).
    env = StrikeMissionEnv(seed=seed, alert_enabled=alert, radar_random=False)
    acc = {}
    for i in range(episodes):
        env.rng = np.random.default_rng(10_000 + i)
        info, _ = play_episode_vdn(env, policy, train=False)
        for k in ("team_success", "both_reached", "n_dead", "steps",
                  "outer_total", "inner_total", "analytic_surv_team",
                  "route_overlap"):
            acc[k] = acc.get(k, 0.0) + float(info[k])
    return {k: v / episodes for k, v in acc.items()}


def untrained_net(seed=0):
    """Hic egitilmemis VDNAgent — dejenere politika hipotezinin dogrudan testi."""
    from agents.vdn import VDNAgent
    return VDNAgent(seed=seed)


def main():
    print(f"{'politika':<28}{'takim':>8}{'ikisi':>8}{'olu':>6}{'adim':>7}"
          f"{'dis':>7}{'ic':>6}{'analitik':>10}{'ortusme':>9}")
    rows = [
        ("SABIT (sag->asagi)", ConstantPolicy()),
        ("SABIT (asagi->sag)", ConstantPolicy((C.DOWN, C.RIGHT, C.UP, C.LEFT))),
        ("rastgele monoton", RandomMonotonePolicy(seed=0)),
        ("EGITILMEMIS ag (seed0)", untrained_net(0)),
        ("EGITILMEMIS ag (seed7)", untrained_net(7)),
    ]
    for name, pol in rows:
        m = run(pol, episodes=10)
        print(f"{name:<28}{m['team_success']*100:7.1f}%{m['both_reached']*100:7.1f}%"
              f"{m['n_dead']:6.2f}{m['steps']:7.0f}{m['outer_total']:7.0f}"
              f"{m['inner_total']:6.0f}{m['analytic_surv_team']:10.4f}"
              f"{m['route_overlap']:9.2f}")

    d = risk_distance_map()
    p = greedy_path(C.START, C.GOAL, d)
    o, i = exposure(p)
    print(f"{'Dijkstra oracle (yol)':<28}{100.0:7.1f}%{100.0:7.1f}%{0.0:6.2f}"
          f"{len(p)-1:7d}{o:7d}{i:6d}{survival_prob(p):10.4f}{1.0:9.2f}")


if __name__ == "__main__":
    main()
