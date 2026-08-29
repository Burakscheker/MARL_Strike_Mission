"""Deployment-politikasi degerlendirmesi — stuck-tetikli Boltzmann.

`eval/evaluate.py` KANONIK sayidir: saf greedy (`argmax Q`, eps=0), egitim ve
degerlendirme AYNI kod yolunu kullanir (bkz. env/two_agent.py). Bu dosya ona
DOKUNMAZ — ayri bir DEPLOYMENT politikasini olcer.

GOZLEM (Strike_Mission.md §11 + NOTES.md it19): fast-eps VDN'in greedy politikasi
100 haritanin ~25'inde TIKANIR — risk-mesafesi `env.dist[pos]` azalmayi birakir,
ajan hedefe varmadan timeout yer (%34 timeout). Sebep: action-gap ~0.03, Q-agi
"ileri git" ile "bekle" arasindaki farki argmax'in kararli secebilecegi kadar
buyuk kodlayamiyor (it8'de olculdu).

DEPLOYMENT DUZELTMESI: ajan `STUCK_N` adimdir ilerlemiyorsa `argmax(Q)` yerine
`softmax(Q/TEMP)` ornekler (yalniz o ajan, yalniz o adimlarda); risk-mesafesi
tekrar azalinca greedy'ye doner. SADECE ogrenilen Q kullanilir — elle hedef-arama,
kural, oracle YOK. Kotu aksiyon (radar girisi, cok negatif Q) ~hic secilmez, o
yuzden olum cok az artar.

SONUC (5 zar-tohumu x 100 held-out harita, it2_vdn_epsfast.pt):
    GREEDY       [70,70,68,70,68]  ort %69.2
    STUCK-ESCAPE [76,76,74,74,72]  ort %74.4      timeout %34->%9, olu 0.39->0.63

Kullanim:
    python -m eval.deploy_eval --ckpt runs/ckpt/it2_vdn_epsfast.pt
    python -m eval.deploy_eval --ckpt ... --maps 100 --seeds 5 --stuck-n 150 --temp 0.05
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

import config as C
from agents.networks import build_qnet, masked_q
from env.sampler import eval_map_seeds
from env.strike_env import StrikeMissionEnv

AGENT_1, AGENT_2 = C.AGENT_1, C.AGENT_2


def _load_qnets(ckpt_path: str, device: str) -> dict:
    ck = torch.load(ckpt_path, map_location=device)
    nets = {}
    for a, key in ((AGENT_1, "online1"), (AGENT_2, "online2")):
        n = build_qnet(C.N_ACTIONS).to(device)
        n.load_state_dict(ck[key])
        n.eval()
        nets[a] = n
    return nets


@torch.no_grad()
def _q(nets, agent_id, obs_batch, mask_batch, device):
    o = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
    m = torch.as_tensor(mask_batch, dtype=torch.float32, device=device)
    return masked_q(nets[agent_id](o), m).cpu().numpy()


def _run_chunk(nets, chunk_seeds, *, device, death_enabled, escape,
               stuck_n, temp, rng, env_seed):
    """Bir grup haritayi paralel kosar. escape=False -> saf greedy (baseline).
    escape=True -> stall-tetikli Boltzmann. Doner: her harita icin info dict."""
    n = len(chunk_seeds)
    envs = [StrikeMissionEnv(seed=env_seed + i, radar_random=True, n_radar=C.N_RADAR,
                             max_steps=C.MAX_STEPS, death_enabled=death_enabled)
            for i in range(n)]
    obs = [e.reset(map_seed=s, n_radar=C.N_RADAR) for e, s in zip(envs, chunk_seeds)]
    o1 = np.stack([o[AGENT_1] for o in obs])
    o2 = np.stack([o[AGENT_2] for o in obs])
    done = [False] * n
    infos: list = [None] * n
    # stuck izleme: her (env, ajan) icin gorulen en kucuk risk-mesafe + iyilesmeyen adim
    best = {a: np.array([float(e.dist[e.pos[a]]) for e in envs]) for a in (AGENT_1, AGENT_2)}
    ctr = {a: np.zeros(n, dtype=int) for a in (AGENT_1, AGENT_2)}

    while not all(done):
        m1 = np.stack([e.action_mask(AGENT_1) for e in envs])
        m2 = np.stack([e.action_mask(AGENT_2) for e in envs])
        q1 = _q(nets, AGENT_1, o1, m1, device)
        q2 = _q(nets, AGENT_2, o2, m2, device)
        acts = {}
        for a, q, mk in ((AGENT_1, q1, m1), (AGENT_2, q2, m2)):
            out = q.argmax(1)
            if escape:
                for i in np.flatnonzero(ctr[a] >= stuck_n):
                    if done[i] or envs[i].terminal(a):
                        continue
                    p = np.exp((q[i] - q[i].max()) / temp) * mk[i]
                    out[i] = rng.choice(len(p), p=p / p.sum())
            acts[a] = out
        n1, n2 = list(o1), list(o2)
        for i, e in enumerate(envs):
            if done[i]:
                continue
            oo, _r, d, info = e.step({AGENT_1: int(acts[AGENT_1][i]),
                                      AGENT_2: int(acts[AGENT_2][i])})
            n1[i], n2[i] = oo[AGENT_1], oo[AGENT_2]
            for a in (AGENT_1, AGENT_2):
                dcur = float(e.dist[e.pos[a]])
                if dcur < best[a][i] - 0.5:
                    best[a][i] = dcur
                    ctr[a][i] = 0
                else:
                    ctr[a][i] += 1
            if d:
                done[i] = True
                infos[i] = info
        o1, o2 = np.stack(n1), np.stack(n2)
    return infos


def evaluate(ckpt: str, *, maps: int, seeds: int, stuck_n: int, temp: float,
             n_envs: int = 50, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    nets = _load_qnets(ckpt, device)
    map_seeds = list(eval_map_seeds(maps))
    out = {"greedy": [], "escape": []}

    for mode, escape in (("greedy", False), ("escape", True)):
        for k in range(seeds):
            rng = np.random.default_rng(1000 + k)          # zar + Boltzmann akisi
            team = 0
            for start in range(0, maps, n_envs):
                chunk = map_seeds[start:start + n_envs]
                infos = _run_chunk(nets, chunk, device=device, death_enabled=True,
                                   escape=escape, stuck_n=stuck_n, temp=temp,
                                   rng=rng, env_seed=40_000 + k * 997 + start)
                team += sum(int(inf["team_success"]) for inf in infos)
            out[mode].append(100.0 * team / maps)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default="runs/ckpt/it2_vdn_epsfast.pt")
    ap.add_argument("--maps", type=int, default=C.EVAL_N_MAPS)
    ap.add_argument("--seeds", type=int, default=5, help="zar-tohumu sayisi (team_success stokastik)")
    ap.add_argument("--stuck-n", type=int, default=150, help="kac adim ilerlemesizlik = stall")
    ap.add_argument("--temp", type=float, default=0.05, help="Boltzmann sicakligi (stall aninda)")
    args = ap.parse_args()

    res = evaluate(args.ckpt, maps=args.maps, seeds=args.seeds,
                   stuck_n=args.stuck_n, temp=args.temp)
    for mode in ("greedy", "escape"):
        v = res[mode]
        label = "GREEDY (kanonik) " if mode == "greedy" else "STUCK-ESCAPE     "
        print(f"{label}: {[round(x, 1) for x in v]}  ->  "
              f"ort %{np.mean(v):.1f}  (min {min(v):.1f}, max {max(v):.1f})")
    delta = np.mean(res["escape"]) - np.mean(res["greedy"])
    print(f"\nfark: {delta:+.1f} puan  (stuck_n={args.stuck_n}, temp={args.temp})")


if __name__ == "__main__":
    main()
