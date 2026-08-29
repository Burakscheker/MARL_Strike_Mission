"""Egitim dongusu — python train.py --algo vdn --episodes 500

Algoritmalar: mappo | happo | vdn | qmix

VDN/QMIX off-policy (TD-hedefli, replay buffer'li); MAPPO/HAPPO on-policy
(PPO, tam episode'lar toplanip GAE ile guncellenir) — bkz. agents/mappo_happo.py
modul dosya stringi. IQL (Asama 4, iki bagimsiz DQN) 2026-08-28'de KALDIRILDI
(euzxx/MARL-pathtfinding'in mappo_happo dalindan MAPPO/HAPPO portlanirken) —
agents/dqn.py artik yok, gerekirse git gecmisinden geri getirilebilir.
"""
from __future__ import annotations

import argparse
import csv
import os

# CUDA DETERMINIZMI (2026-08-27, dis inceleme onerisi): CUBLAS_WORKSPACE_CONFIG
# torch import EDILMEDEN/CUDA baglami kurulmadan ONCE ayarlanmali, aksi halde
# etkisiz kalir (bkz. asagidaki main()'deki torch.use_deterministic_algorithms
# cagrisi — ayni ayarin diger yarisi). Sebep: bu oturumda AYNI --seed ile
# GPU'da IKI KEZ tekrarlanan bir egitim (action-risk deneyi, 2026-08-27)
# TAMAMEN FARKLI sonuc (ep200 takim %50 vs %22) uretti — cuBLAS/cuDNN'in
# non-deterministik algoritma secimi/atomik toplama sirasi tek-seed
# kiyaslamalari GUVENILMEZ yapiyordu.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import time
from collections import deque

import numpy as np
import torch

import config as C
from agents import transfer
from agents.mappo_happo import HAPPOTrainer, MAPPOTrainer
from agents.qmix import QMixAgent
from agents.vdn import VDNAgent
from env.sampler import curriculum_n_radar, eval_map_seeds
from env.strike_env import StrikeMissionEnv
from env.two_agent import play_episode_ppo, play_episode_qmix, play_episode_vdn
from env.vec_env import VecStrikeEnv

# Windows'ta stdout bir dosyaya/boruya yonlendirilince cp1252 kullaniliyor ve
# Turkce karakterlerde UnicodeEncodeError veriyor (MARL-Pathfinding'de yasandi).
try:
    # line_buffering: cikti bir dosyaya/boruya yonlendirildiginde Python
    # varsayilan olarak ~8KB tamponlar; 30-60 dakikalik bir kosuda ilerleme
    # ancak kosu BITINCE gorunur. Satir tamponu bunu canli hale getirir.
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

RUNNER = {"mappo": play_episode_ppo, "happo": play_episode_ppo,
         "vdn": play_episode_vdn, "qmix": play_episode_qmix}


def build_agent(algo: str, seed: int, device: str, lr: float = None,
                eps_end: float = None, dueling: bool = False,
                prioritized: bool = False, al_alpha: float = 0.0,
                munchausen_tau: float = 0.0, layernorm: bool = False,
                vdn_batch: int = None, vdn_target_update: int = None,
                n_quantiles: int = 1, qr_optimism: float = 0.0):
    if algo == "vdn":
        return VDNAgent(seed=seed, device=device,
                         lr=lr if lr is not None else C.VDN_LR,
                         eps_end=eps_end if eps_end is not None else C.EPS_END,
                         dueling=dueling, prioritized=prioritized,
                         al_alpha=al_alpha, munchausen_tau=munchausen_tau,
                         layernorm=layernorm,
                         batch_size=vdn_batch if vdn_batch is not None else C.VDN_BATCH,
                         target_update=(vdn_target_update if vdn_target_update
                                        is not None else C.VDN_TARGET_UPDATE),
                         n_quantiles=n_quantiles, qr_optimism=qr_optimism)
    if algo == "qmix":
        return QMixAgent(seed=seed, device=device,
                          lr=lr if lr is not None else C.QMIX_LR)
    if algo == "mappo":
        return MAPPOTrainer(seed=seed, device=device)
    if algo == "happo":
        return HAPPOTrainer(seed=seed, device=device)
    raise ValueError(algo)


# MAPPO/HAPPO epsilon-greedy KULLANMAZ (politika zaten stokastik). AMA ayni
# egitim-ilerlemesi bilgisiyle ENTROPI CURRICULUM'u annelenir (bkz. config.py
# PPO_ENTROPY_START notu — VDN'deki eps schedule'in PPO karsiligi). VDN/QMIX
# icin bu fonksiyon eskisi gibi epsilon ilerlemesini ayarlar.
def set_eps(agent, algo: str, frac: float):
    if algo in ("mappo", "happo"):
        agent.set_entropy_progress(frac)
        return
    agent.set_eps_progress(frac)


def current_eps(agent, algo: str) -> float:
    # MAPPO/HAPPO icin epsilon yok; loglamada o anki entropi katsayisi gosterilir.
    return agent.entropy_coef if algo in ("mappo", "happo") else agent.eps


def save(agent, algo: str, path_stem: str):
    agent.save(f"{path_stem}.pt")


# --------------------------------------------------------------- paralel VDN rollout
#
# GEREKCE (2026-08-21, olculdu): tek-env kosuda act() HER ADIMDA 2 kez,
# batch=1 ile cagriliyor — GPU'da bu 8000 adimlik bir episode'da 16000 ayri
# tekil dispatch demek, ve GPU'yu CPU'dan %64 YAVAS yapiyordu (34.85 vs
# 57.32 s/ep, ayni makine, ayni ag). learn() zaten batch=32 kullaniyordu,
# darbogaz act()'ti. Bu fonksiyon N StrikeMissionEnv'i PARALEL kosturup
# act()'i TEK batch=N cagrisina indiriyor — GPU'nun batch=32-64 gibi bir
# is yukunde gercekten kazanmasi icin (henuz olculmedi, bkz. --n-envs).
#
# TASARIM: main()'in ana dongusune (CSV loglama, eval/checkpoint secimi)
# HICBIR degisiklik gerektirmiyor — generator, tek-env runner()'in dondurdugu
# AYNI (info, losses) seklini, HER TAMAMLANAN episode icin sirayla yield
# ediyor. main() sadece "runner(env, agent, ...)" cagrisini "next(roll)" ile
# degistiriyor, gerisi ayni kalıyor.
def vdn_parallel_rollout(agent, n_envs: int, max_steps: int, seed_base: int,
                         episodes: int, n_radar_override: int | None,
                         risk_shaping: bool, alert_enabled: bool):
    completed = 0
    n_radar = (n_radar_override if n_radar_override is not None
              else curriculum_n_radar(1, episodes))
    venv = VecStrikeEnv(n_envs, max_steps, seed_base, n_radar=n_radar,
                        risk_shaping=risk_shaping, alert_enabled=alert_enabled)
    pending_losses: list = []

    while True:
        m1, m2 = venv.action_masks()
        a1 = agent.act_batch(C.AGENT_1, venv.obs[C.AGENT_1], m1)
        a2 = agent.act_batch(C.AGENT_2, venv.obs[C.AGENT_2], m2)
        obs_before = venv.obs
        # SU ANKI (step'ten ONCE) pozisyon icin uzman aksiyonu — obs_before'la
        # AYNI ana denk gelmeli (bkz. VecStrikeEnv.oracle_actions). Maliyeti
        # onbelleklendigi icin ihmal edilebilir (bkz. env/vec_env.py); OGRETMEN-
        # CAPASI (bc_lambda) kapaliyken bile hesaplaniyor ama learn()'de
        # bc_lambda=0.0 oldugunda hicbir etkisi yok.
        oa1, oa2 = venv.oracle_actions()
        next_obs, r_team, done, infos, nm1, nm2 = venv.step(a1, a2, n_radar=n_radar)

        for i in range(n_envs):
            is_trunc = bool(done[i]) and infos[i].get("timeout", False)
            push_done = bool(done[i]) and not is_trunc
            agent.push(obs_before[C.AGENT_1][i], int(a1[i]),
                      obs_before[C.AGENT_2][i], int(a2[i]),
                      float(r_team[i]), next_obs[C.AGENT_1][i], next_obs[C.AGENT_2][i],
                      push_done, nm1[i], nm2[i], int(oa1[i]), int(oa2[i]))
            loss = agent.learn()
            if loss is not None:
                pending_losses.append(loss)
            if done[i]:
                completed += 1
                losses, pending_losses = pending_losses, []
                if n_radar_override is None:
                    n_radar = curriculum_n_radar(completed + 1, episodes)
                    venv.set_n_radar(n_radar)
                yield infos[i], losses


# QMIX icin AYNI paralel rollout — tek fark: state/next_state de push edilir
# (mixer'in hypernetwork'u bunu ister, bkz. agents/qmix.py).
def qmix_parallel_rollout(agent, n_envs: int, max_steps: int, seed_base: int,
                          episodes: int, n_radar_override: int | None,
                          risk_shaping: bool, alert_enabled: bool):
    completed = 0
    n_radar = (n_radar_override if n_radar_override is not None
              else curriculum_n_radar(1, episodes))
    venv = VecStrikeEnv(n_envs, max_steps, seed_base, n_radar=n_radar,
                        risk_shaping=risk_shaping, alert_enabled=alert_enabled)
    pending_losses: list = []

    while True:
        m1, m2 = venv.action_masks()
        state = venv.states()
        a1 = agent.act_batch(C.AGENT_1, venv.obs[C.AGENT_1], m1)
        a2 = agent.act_batch(C.AGENT_2, venv.obs[C.AGENT_2], m2)
        obs_before = venv.obs
        # need_state=True: next_state reset-ONCESI GERCEK state olarak doner
        # (VDN'deki true_obs/nm duzeltmesiyle AYNI gerekce — timeout'ta
        # push_done=False oldugu icin bu state GERCEKTEN bootstrap'e girer).
        next_obs, r_team, done, infos, nm1, nm2, next_state = venv.step(
            a1, a2, n_radar=n_radar, need_state=True)

        for i in range(n_envs):
            is_trunc = bool(done[i]) and infos[i].get("timeout", False)
            push_done = bool(done[i]) and not is_trunc
            agent.push(obs_before[C.AGENT_1][i], int(a1[i]),
                      obs_before[C.AGENT_2][i], int(a2[i]),
                      float(r_team[i]), next_obs[C.AGENT_1][i], next_obs[C.AGENT_2][i],
                      push_done, nm1[i], nm2[i], state[i], next_state[i])
            loss = agent.learn()
            if loss is not None:
                pending_losses.append(loss)
            if done[i]:
                completed += 1
                losses, pending_losses = pending_losses, []
                if n_radar_override is None:
                    n_radar = curriculum_n_radar(completed + 1, episodes)
                    venv.set_n_radar(n_radar)
                yield infos[i], losses


# MAPPO/HAPPO icin paralel rollout (2026-08-28) — VDN/QMIX'in auto-reset
# deseninden BILEREK FARKLI: PPO'nun GAE'si episode SINIRLARINI net bilmek
# ZORUNDA (bootstrap timeout'ta mi, olum/varista mi?), auto-reset'in "bu
# adim ayni haritanin mi yoksa YENI bir haritanin mi ilk adimi" belirsizligi
# GAE hesabini BOZAR. Bunun yerine eval/vdn_vec_evaluate'teki GUVENLI
# "chunk" desenini kullanir: n_envs harita AYNI ANDA baslar, HER BIRI TAM
# olarak biter (auto-reset YOK, biten NOOP'a duser), sonra hepsi icin GAE
# hesaplanip TEK RolloutBatch'e donusur. n_envs=PPO_ROLLOUT_EPISODES ise
# (varsayilan) bu TAM OLARAK bir PPO guncellemesine denk gelir.
def ppo_parallel_rollout(agent, n_envs: int, max_steps: int, seed_base: int,
                         episodes: int, n_radar_override: int | None,
                         risk_shaping: bool, alert_enabled: bool):
    completed = 0
    n_radar = (n_radar_override if n_radar_override is not None
              else curriculum_n_radar(1, episodes))

    while completed < episodes:
        count = min(n_envs, episodes - completed)
        envs = [StrikeMissionEnv(seed=seed_base + completed + i, radar_random=True,
                                 n_radar=n_radar, max_steps=max_steps,
                                 risk_shaping=risk_shaping, alert_enabled=alert_enabled)
               for i in range(count)]
        obs0 = [e.reset(n_radar=n_radar) for e in envs]
        obs1 = np.stack([o[C.AGENT_1] for o in obs0])
        obs2 = np.stack([o[C.AGENT_2] for o in obs0])
        done = [False] * count
        infos: list = [None] * count
        eps_data = [{name: [] for name in ("obs", "states", "masks", "actions",
                                           "old_logp", "rewards", "values", "alive")}
                   for _ in range(count)]

        while not all(done):
            states = np.stack([e.state() for e in envs])
            m1 = np.stack([e.action_mask(C.AGENT_1) for e in envs])
            m2 = np.stack([e.action_mask(C.AGENT_2) for e in envs])
            masks = np.stack([m1, m2], axis=1)
            obs_arr = np.stack([obs1, obs2], axis=1)
            actions, old_logp, values = agent.act_batch(obs_arr, masks, states)
            no1, no2 = list(obs1), list(obs2)
            for i, e in enumerate(envs):
                if done[i]:
                    continue
                alive = np.array([e.alive[C.AGENT_1], e.alive[C.AGENT_2]])
                acts = {C.AGENT_1: int(actions[i, 0]), C.AGENT_2: int(actions[i, 1])}
                o, r_team, d, info = e.step(acts)
                no1[i] = o[C.AGENT_1]; no2[i] = o[C.AGENT_2]
                ed = eps_data[i]
                ed["obs"].append(obs_arr[i]); ed["states"].append(states[i])
                ed["masks"].append(masks[i]); ed["actions"].append(actions[i])
                ed["old_logp"].append(old_logp[i]); ed["rewards"].append(float(r_team))
                ed["values"].append(float(values[i])); ed["alive"].append(alive)
                if d:
                    done[i] = True
                    infos[i] = info
            obs1 = np.stack(no1); obs2 = np.stack(no2)

        for i, e in enumerate(envs):
            is_timeout = bool(infos[i].get("timeout", False))
            bootstrap_value = 0.0
            if is_timeout:
                with torch.no_grad():
                    final_state = torch.as_tensor(e.state(), dtype=torch.float32,
                                                 device=agent.device).unsqueeze(0)
                    bootstrap_value = float(agent.critic(final_state).item())
            batch = agent.add_episode(eps_data[i], terminated_true=not is_timeout,
                                      bootstrap_value=bootstrap_value)
            completed += 1
            losses: list = []
            if batch is not None:
                ul = agent.update(batch)
                losses = [ul["actor_loss"], ul["critic_loss"]]
            if n_radar_override is None:
                n_radar = curriculum_n_radar(completed + 1, episodes)
            yield infos[i], losses


def ppo_vec_evaluate(env, agent, episodes: int, seed: int, n_envs: int) -> dict:
    """vdn_vec_evaluate'in MAPPO/HAPPO esdegeri — AYNI chunked/auto-reset-YOK
    deseni, sadece agent.act_batch()'in PPO imzasini (obs,masks,states) kullanir
    (VDN/QMIX'in act_batch(agent_id,obs,mask,eps) imzasindan FARKLI, o yuzden
    vdn_vec_evaluate DOGRUDAN kullanilamiyor)."""
    seeds = eval_map_seeds(episodes) if env.radar_random else [None] * episodes

    def run_chunk(chunk_seeds, death_enabled):
        n = len(chunk_seeds)
        envs = [StrikeMissionEnv(seed=seed + i, radar_random=env.radar_random,
                                 n_radar=C.N_RADAR, max_steps=env.max_steps,
                                 risk_shaping=env.risk_shaping,
                                 hazard_mode=env.hazard_mode,
                                 alert_enabled=env.alert_enabled,
                                 death_enabled=death_enabled)
                for i in range(n)]
        obs = [e.reset(map_seed=s, n_radar=C.N_RADAR) for e, s in zip(envs, chunk_seeds)]
        obs1 = np.stack([o[C.AGENT_1] for o in obs])
        obs2 = np.stack([o[C.AGENT_2] for o in obs])
        done = [False] * n
        infos: list = [None] * n
        while not all(done):
            states = np.stack([e.state() for e in envs])
            m1 = np.stack([e.action_mask(C.AGENT_1) for e in envs])
            m2 = np.stack([e.action_mask(C.AGENT_2) for e in envs])
            masks = np.stack([m1, m2], axis=1)
            obs_arr = np.stack([obs1, obs2], axis=1)
            actions, _, _ = agent.act_batch(obs_arr, masks, states, deterministic=True)
            no1, no2 = list(obs1), list(obs2)
            for i, e in enumerate(envs):
                if done[i]:
                    continue
                o, _r, d, info = e.step({C.AGENT_1: int(actions[i, 0]),
                                         C.AGENT_2: int(actions[i, 1])})
                no1[i] = o[C.AGENT_1]; no2[i] = o[C.AGENT_2]
                if d:
                    done[i] = True
                    infos[i] = info
            obs1 = np.stack(no1); obs2 = np.stack(no2)
        return infos

    acc = {k: 0.0 for k in METRIC_KEYS}
    for start in range(0, episodes, n_envs):
        chunk = seeds[start:start + n_envs]
        dice_infos = run_chunk(chunk, death_enabled=True)
        route_infos = run_chunk(chunk, death_enabled=False)
        for dinfo, rinfo in zip(dice_infos, route_infos):
            m1 = rinfo["surv1"] if rinfo["reached1"] else 0.0
            m2 = rinfo["surv2"] if rinfo["reached2"] else 0.0
            info = dict(dinfo)
            info["route_reached"] = float(rinfo["reached1"] or rinfo["reached2"])
            info["mission_prob"] = 1.0 - (1.0 - m1) * (1.0 - m2)
            for k in METRIC_KEYS:
                acc[k] += float(info[k])
    return {k: v / episodes for k, v in acc.items()}


# --------------------------------------------------------------- degerlendirme

# NOT — "analytic_surv_team" SISEN bir metriktir, tek basina okunmamalidir:
# GIDILEN yolu olcer, yani yarida olen/timeout yiyen ajanin yolu kisa kalir ve
# "guvenli" gorunur (uc noktada hic hareket etmeyen ajan 1.000 alir).
# BU TUZAGA FIILEN DUSULDU: n-adim getiri deneyi (2026-08-08) boyunca
# analitik 0.534 -> 0.766 "iyilesme" diye okundu, gercekte surv_ratio
# 0.0280 -> 0.0000'a DUSMUSTU; ajan guvenli gorunuyordu cunku hicbir yere
# gitmiyordu. 6 saatlik kosu yanlis sinyalle izlendi.
# Bu yuzden asagidaki IKI metrik eklendi ve fmt_eval'da analitikten ONCE
# basiliyor — ikisi de SISMEZ:
#   route_reached : rotasi hedefe VARIYOR mu (zar kapali) — "gidiyor mu?"
#   mission_prob  : varmayan rota 0 alir, yani hareketsizlik odullendirilemez
METRIC_KEYS = ("team_success", "both_reached", "n_dead", "timeout", "steps",
               "outer_total", "inner_total", "route_overlap",
               "route_reached", "mission_prob",
               "analytic_surv_team", "surv1", "surv2")


def evaluate(env, agent, algo: str, episodes: int, seed: int = 12345) -> dict:
    """DETERMINISTIK (eps=0) degerlendirme. Egitimle AYNI runner kullanilir.

    RASTGELE HARITADA (env.radar_random): haritalar env/sampler.eval_map_seeds()
    tohumlarindan uretilir. Iki sart birden saglanir:
      1. Tohum araligi egitiminkiyle KESISMEZ (TRAIN_SEED_MAX < EVAL_SEED_BASE)
         -> ajan bu haritalari egitimde HIC gormedi, ezberlemis olamaz.
      2. Tohumlar SABIT -> IQL/VDN/QMIX ve tum baseline'lar AYNI haritalarda
         olculur. Rastgele haritada tavan harita sansiyla 10 kat oynadigi icin
         (oracle ortalama %32, medyan %7.2) bu olmadan algoritmalar arasi fark
         yorumlanamaz.
    Curriculum burada UYGULANMAZ: degerlendirme her zaman tam N_RADAR'da.
    """
    runner = RUNNER[algo]
    # BUG (bulundu ve duzeltildi): eval() bu ortamin (egitimde de kullanilan
    # AYNI env nesnesi) rng'sini sabit tohuma cekiyordu ama HIC geri
    # yuklemiyordu. Sonuc: eval_every ne kadar sikysa egitimin geri kalani
    # o kadar farkli bir zar akisina kayiyordu. Simdi eval'dan once/sonra saklaniyor.
    saved_rng = env.rng
    env.rng = np.random.default_rng(seed)      # olum zarlari icin sabit tohum
    renv = _route_twin(env)
    renv.rng = np.random.default_rng(seed)
    acc = {k: 0.0 for k in METRIC_KEYS}
    seeds = eval_map_seeds(episodes) if env.radar_random else [None] * episodes
    for s in seeds:
        rk = {"map_seed": s, "n_radar": C.N_RADAR} if env.radar_random else None
        info, _ = runner(env, agent, train=False, reset_kwargs=rk)
        # AYNI haritada zar KAPALI ikinci kosu: ajanin niyet ettigi tam rota.
        rinfo, _ = runner(renv, agent, train=False, reset_kwargs=rk)
        m1 = rinfo["surv1"] if rinfo["reached1"] else 0.0
        m2 = rinfo["surv2"] if rinfo["reached2"] else 0.0
        info = dict(info)
        info["route_reached"] = float(rinfo["reached1"] or rinfo["reached2"])
        info["mission_prob"] = 1.0 - (1.0 - m1) * (1.0 - m2)
        for k in METRIC_KEYS:
            acc[k] += float(info[k])
    env.rng = saved_rng
    return {k: v / episodes for k, v in acc.items()}


def _route_twin(env):
    """Zar KAPALI ikiz ortam — ajanin NIYET ETTIGI tam rotayi olcmek icin.

    env'e ilistirilip onbelleklenir; her eval'da yeniden kurmak pahali
    (risk-mesafe haritasi harita basina yeniden cikariliyor). Egitim
    ortaminin TUM ayarlarini aynalar, sadece death_enabled=False.
    """
    tw = getattr(env, "_route_twin_env", None)
    if tw is None:
        tw = StrikeMissionEnv(n=env.n, max_steps=env.max_steps, seed=999,
                              alert_enabled=env.alert_enabled,
                              risk_shaping=env.risk_shaping,
                              hazard_mode=env.hazard_mode,
                              radar_random=env.radar_random,
                              n_radar=env.n_radar,
                              death_enabled=False)
        env._route_twin_env = tw
    return tw


def vdn_vec_evaluate(env, agent, episodes: int, seed: int, n_envs: int) -> dict:
    """evaluate()'in VDN icin PARALEL/batch'li hali — AYNI METRIC_KEYS'i,
    AYNI iki-kosu tasarimini (zar acik + zar kapali "niyet edilen rota"
    ikizi) dondurur, sadece act()'i batch=1 yerine batch=min(n_envs,kalan)
    ile cagirir.

    GEREKCE (olculdu, 2026-08-21): evaluate() TRAINING rollout'u gibi
    act()'i her adimda batch=1 ile cagiriyordu — GPU'da bu, vdn_parallel_
    rollout'u anlamli yapan AYNI sorunun (tekil dispatch) eval sirasinda da
    yasanmasi demekti. 50 eval-episode x 2 (zar+rota) = 100 seri tam
    episode ~16 dk tutuyordu (32 paralel egitimle KIYASLANAMAYACAK kadar
    yavas kalmisti). Bu fonksiyon HER adimda TUM chunk'i (<=n_envs harita)
    tek batch'te isler.

    Auto-reset YOK (vdn_parallel_rollout'tan farki): her ortam TAM OLARAK
    kendi map_seed'inde BIR KEZ kosar, biter, biten NOOP'a duser (mask
    zaten NOOP'a kilitler) digerleri bitene kadar devam eder.
    """
    seeds = eval_map_seeds(episodes) if env.radar_random else [None] * episodes

    def run_chunk(chunk_seeds, death_enabled):
        n = len(chunk_seeds)
        envs = [StrikeMissionEnv(seed=seed + i, radar_random=env.radar_random,
                                 n_radar=C.N_RADAR, max_steps=env.max_steps,
                                 risk_shaping=env.risk_shaping,
                                 hazard_mode=env.hazard_mode,
                                 alert_enabled=env.alert_enabled,
                                 death_enabled=death_enabled)
                for i in range(n)]
        obs = [e.reset(map_seed=s, n_radar=C.N_RADAR) for e, s in zip(envs, chunk_seeds)]
        obs1 = np.stack([o[C.AGENT_1] for o in obs])
        obs2 = np.stack([o[C.AGENT_2] for o in obs])
        done = [False] * n
        infos: list = [None] * n
        while not all(done):
            m1 = np.stack([e.action_mask(C.AGENT_1) for e in envs])
            m2 = np.stack([e.action_mask(C.AGENT_2) for e in envs])
            a1 = agent.act_batch(C.AGENT_1, obs1, m1, eps=0.0)
            a2 = agent.act_batch(C.AGENT_2, obs2, m2, eps=0.0)
            no1, no2 = list(obs1), list(obs2)
            for i, e in enumerate(envs):
                if done[i]:
                    continue
                o, _r, d, info = e.step({C.AGENT_1: int(a1[i]), C.AGENT_2: int(a2[i])})
                no1[i] = o[C.AGENT_1]; no2[i] = o[C.AGENT_2]
                if d:
                    done[i] = True
                    infos[i] = info
            obs1 = np.stack(no1); obs2 = np.stack(no2)
        return infos

    acc = {k: 0.0 for k in METRIC_KEYS}
    for start in range(0, episodes, n_envs):
        chunk = seeds[start:start + n_envs]
        dice_infos = run_chunk(chunk, death_enabled=True)
        route_infos = run_chunk(chunk, death_enabled=False)
        for dinfo, rinfo in zip(dice_infos, route_infos):
            m1 = rinfo["surv1"] if rinfo["reached1"] else 0.0
            m2 = rinfo["surv2"] if rinfo["reached2"] else 0.0
            info = dict(dinfo)
            info["route_reached"] = float(rinfo["reached1"] or rinfo["reached2"])
            info["mission_prob"] = 1.0 - (1.0 - m1) * (1.0 - m2)
            for k in METRIC_KEYS:
                acc[k] += float(info[k])
    return {k: v / episodes for k, v in acc.items()}


def fmt_eval(m: dict) -> str:
    # SIRA ONEMLI: sismeyen metrikler (varis/gorev) ONCE, sisen analitik
    # SONRA ve parantez icinde — okuyan kisi once dogru sinyali gorsun.
    return (f"takim={m['team_success']*100:5.1f}%  "
            f"VARIS={m['route_reached']*100:5.1f}%  "
            f"gorev={m['mission_prob']:.4f}  "
            f"olu={m['n_dead']:.2f}  adim={m['steps']:6.0f}  "
            f"maruziyet dis/ic={m['outer_total']:5.0f}/{m['inner_total']:5.0f}  "
            f"(analitik={m['analytic_surv_team']:.3f})  "
            f"ortusme={m['route_overlap']:.2f}")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["mappo", "happo", "vdn", "qmix"], required=True)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--map-seed", type=int, default=None,
                    help="EGITIM harita dizisini --seed'den AYIR (varsayilan: "
                         "--seed ile ayni). --seed hala ag-init + kesif RNG'sini "
                         "kontrol eder; --map-seed sadece VecStrikeEnv'in urettigi "
                         "egitim haritalari dizisini. TESHIS: fast-eps seed 0 vs "
                         "seed 1 arasi ~40 puanlik farkin ag-init sansindan mi "
                         "yoksa harita-dizisi (curriculum) sansindan mi geldigini "
                         "ayirir. Eval haritalari SABIT (eval_map_seeds), etkilenmez.")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=50,
                    help="eval harita sayisi. 5 COK AZ: %%5'lik gercek basari "
                         "5 haritada %%59 ihtimalle 0 gorunur. Ustelik ardisik "
                         "eval'lar arasi gorev metrigi 100 kat oynuyor "
                         "(olculdu, r30s0_vdn: 0.1875 -> 0.0280 -> 0.0021), "
                         "yani 30 harita bile checkpoint secmek icin gurultulu.")
    ap.add_argument("--demo-episodes", type=int, default=C.DEMO_EPISODES)
    ap.add_argument("--max-steps", type=int, default=C.MAX_STEPS)
    ap.add_argument("--alert", action="store_true",
                    help="radar alarm kuplajini ac (Asama 6)")
    ap.add_argument("--no-risk-shaping", action="store_true",
                    help="adim basi beklenen-olum maliyetini kapat (ablation)")
    ap.add_argument("--resume-from", default=None,
                    help="checkpoint yolu, ya da 'pathfinding' -> MARL-Pathfinding'in "
                         "egitilmis modelini otomatik bul")
    ap.add_argument("--eps-start", type=float, default=None,
                    help="epsilon'un BASLANGIC degeri (varsayilan C.EPS_START=1.0). "
                         "DEVAM eden bir kosuda sart: egitilmis bir checkpoint'ten "
                         "1.0 ile baslamak, ogrenilmis politikayi yuzlerce episode "
                         "boyunca rastgele aksiyonlarla bozar. 0.2-0.4 tipik. "
                         "UYARI: bu deger C.EPS_END'in (varsayilan 0.05) ALTINDAYSA "
                         "epsilon ANINDA tabana (EPS_END) sikisir, --eps-start DEGIL "
                         "efektif olarak EPS_END kullanilir — daha dusuk bir keşif "
                         "istiyorsan --eps-end'i de dusur.")
    ap.add_argument("--eps-end", type=float, default=None,
                    help="epsilon'un TABAN degerini gecici override et (varsayilan "
                         "C.EPS_END=0.05). BC-checkpoint'ten fine-tune ederken "
                         "0.05, 8000 adimlik bir bolumde hala ~400 rastgele aksiyon "
                         "demek — BC'nin ogrettigi rotayi gereksiz yere bozar. "
                         "0.01 tipik.")
    ap.add_argument("--n-radar", type=int, default=None,
                    help="radar sayisini SABITLE (curriculum'u kapatir). "
                         "Verilmezse 10->40 rampasi kullanilir.")
    ap.add_argument("--lr", type=float, default=None,
                    help="optimizer LR'ini gecici override et (varsayilan config'teki "
                         "*_LR). BC-checkpoint'ten RL fine-tune ederken onemli: BC, "
                         "cross-entropy ile egitildigi icin cikis katmaninin olcegi "
                         "gercek Q-degerleriyle (ort. 17-37) uyumsuz — normal LR ile "
                         "ilk birkac gradyan adimi bu olcegi agresifçe yeniden "
                         "kalibre ederken BC'nin ogrendigi ince aksiyon-siralamasini "
                         "bozar. Fine-tune'da 5-10x dusuk LR kullan.")
    ap.add_argument("--bc-lambda-start", type=float, default=None,
                    help="OGRETMEN-CAPASI (teacher-anchored VDN, sadece vdn) "
                         "baslangic agirligi (varsayilan C.VDN_BC_LAMBDA_START). "
                         "TD kaybina oracle-capraz-entropi ekler: L=L_TD+lambda*L_BC. "
                         "0.0 = kapali (eski davranis). BC-checkpoint'ten fine-tune "
                         "ederken TD guncellemesinin BC'nin ogrettigi rotayi "
                         "silmesini onlemek icin kullan.")
    ap.add_argument("--bc-lambda-end", type=float, default=None,
                    help="OGRETMEN-CAPASI TABAN agirligi (varsayilan "
                         "C.VDN_BC_LAMBDA_END). Egitim ilerledikce lambda buraya "
                         "iner ama SIFIRA INMEZ (0 vermek tam kapatir).")
    ap.add_argument("--dueling", action="store_true",
                    help="SADECE vdn: Dueling mimari (V(s)+A(s,a), Wang ve ark. "
                         "2016) — Q'nun MUTLAK OLCEGIYLE aksiyonlar ARASI "
                         "SIRALAMAYI mimari olarak ayirir. Varsayilan KAPALI "
                         "(eski mimariyle checkpoint uyumu icin); acilirsa "
                         "ESKI checkpoint'ler YUKLENEMEZ (farkli parametre "
                         "isimleri), sifirdan egitim gerekir.")
    ap.add_argument("--prioritized", action="store_true",
                    help="SADECE vdn: Oncelikli Deneyim Tekrari (PER, Schaul "
                         "ve ark. 2016) — buffer'da NADIR ama KRITIK gecisleri "
                         "(olum, varis, riskli giris) TD-hatasi buyuklugune "
                         "gore daha sik ornekler. Varsayilan KAPALI (uniform "
                         "ornekleme, eski davranis).")
    ap.add_argument("--al-alpha", type=float, default=None,
                    help="SADECE vdn: Advantage Learning operatoru (Bellemare "
                         "ve ark. 2016, 'Increasing the Action Gap'). TD "
                         "hedefinden alinan aksiyonun greedy'den geriligini "
                         "alpha kadar duser -> action-gap ~1/(1-alpha) katina "
                         "cikar, greedy politika korunur. Sicaklik YOK. "
                         "Varsayilan KAPALI (0.0); tipik deger 0.9. Bkz. "
                         "config.py AL_ALPHA_DEFAULT.")
    ap.add_argument("--munchausen-tau", type=float, default=None,
                    help="SADECE vdn: Munchausen RL (Vieillard ve ark. 2020). "
                         "Odule alpha*tau*log pi(a_t) ekler + sert max yerine "
                         "yumusak (entropi-duzenli) bootstrap. AL'i KAPSAR + "
                         "entropi terimi politikanin rijit bir stratejiye "
                         "cokmesine direnir. Varsayilan KAPALI (0.0); tipik "
                         "0.03. >0 ise --al-alpha yok sayilir. Bkz. config.py "
                         "MUNCHAUSEN_ALPHA.")
    ap.add_argument("--vdn-batch", type=int, default=None,
                    help="SADECE vdn: replay batch boyutu override (varsayilan "
                         "C.VDN_BATCH=128). 32->128 kaotik ziplama/cokusu "
                         "azaltmisti (config.py notu); 256 DENENMEDI — gradyan "
                         "varyansini yariya indirir, 250-ep fast-eps'in tohum "
                         "lotaryasini (seed 0 iyi, seed 1 %%27) yumusatabilir.")
    ap.add_argument("--vdn-target-update", type=int, default=None,
                    help="SADECE vdn: hard target sync araligi (adim) override "
                         "(varsayilan C.VDN_TARGET_UPDATE=4000). YAVAS yon "
                         "DENENMEDI (soft/Polyak denenip elenmisti) — daha "
                         "kararli regresyon hedefi = daha az kaotik ogrenme.")
    ap.add_argument("--quantiles", type=int, default=None,
                    help="SADECE vdn: QR-DQN (Dabney ve ark. 2017) — Q skalari "
                         "yerine getiri dagiliminin N kuantilini ogrenir. "
                         "Varsayilan 1 (skaler, eski davranis BIREBIR). Tipik "
                         "8-32. Motiv: deger fonksiyonu stokastik olum cezasi "
                         "altinda karamsar mean'e cokuyor; tum dagilim daha "
                         "dayanikli + iyimser aksiyon secimi mumkun (--qr-optimism). "
                         ">1 ise --al-alpha / --munchausen-tau YOK SAYILIR.")
    ap.add_argument("--qr-optimism", type=float, default=None,
                    help="SADECE vdn + --quantiles>1: aksiyon secerken mean "
                         "yerine mean + k*std kullan (getiri dagiliminin ust ucu "
                         "= iyimser -> karamsar cokmeye karsi). 0.0 = risk-notr "
                         "(duz mean). Tipik 0.3-1.0. Egitim ve eval'de kullanilir.")
    ap.add_argument("--save-all-ckpts", action="store_true",
                    help="Her eval'da ayri checkpoint kaydet ({tag}_ep{N}.pt) — "
                         "deploy-time Q-ortalama ensemble icin (fast-eps'te her "
                         "ckpt ~%%65 ama farkli haritalarda hata yapar).")
    ap.add_argument("--layernorm", action="store_true",
                    help="SADECE vdn: Q-agi gizli katmanlarindan sonra LayerNorm "
                         "(BroNet/CrossQ, plasticity-loss literaturu). Belgeli "
                         "Q-iraksamasini (q_mean 17->37, config.py §11.14) "
                         "hedefler: aktivasyon dagilimini sabit tutar -> Q "
                         "sinirli kalir -> uzun egitimde argmax politikasi "
                         "bozulmaz. Varsayilan KAPALI; acilirsa ESKI "
                         "checkpoint'ler YUKLENEMEZ, sifirdan egitim gerekir.")
    ap.add_argument("--resume-head-reset", action="store_true",
                    help="govdeyi yukle ama Q ciktisi katmanini sifirla "
                         "(gamma/olcek degistigi icin onerilir — bkz. agents/transfer.py)")
    ap.add_argument("--no-mixer-transfer", action="store_true")
    ap.add_argument("--n-envs", type=int, default=1,
                    help="vdn/qmix/mappo/happo: bu kadar ortami PARALEL "
                         "kosturup act()'i batch=N ile cagirir (GPU'da anlamli "
                         "olmasi icin). VDN/QMIX auto-reset kullanir; MAPPO/"
                         "HAPPO auto-reset-YOK chunk deseni kullanir (bkz. "
                         "ppo_parallel_rollout) — mappo/happo'da N=PPO_ROLLOUT_"
                         "EPISODES onerilir (tam 1 PPO guncellemesi = 1 tur). "
                         "1 = eski tek-env davranisi, HICBIR SEY degismez.")
    args = ap.parse_args()

    # CUDA DETERMINIZMI (2026-08-27) — devam: CUBLAS_WORKSPACE_CONFIG modul
    # basinda ayarlandi (bkz. dosya basi), burada torch'un ALGORITMA SECIMINI
    # de sabitliyoruz. warn_only=True: deterministik karsiligi olmayan NADIR
    # bir op cikarsa egitim CRASH OLMASIN, sadece uyarsin — hicbir katman
    # bunu tetiklemiyor gibi gorunuyor (CNN/Linear/Adam hepsi deterministik
    # karsiligina sahip) ama garanti degil, warn_only guvenli taraf.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    algo = args.algo
    episodes = args.episodes or getattr(C, f"{algo.upper()}_EPISODES")
    eval_every = args.eval_every or getattr(C, f"{algo.upper()}_EVAL_EVERY")
    tag = args.tag or f"{algo}_s{args.seed}"
    map_seed = args.map_seed if args.map_seed is not None else args.seed
    os.makedirs(os.path.join(C.RUNS_DIR, "ckpt"), exist_ok=True)

    print(C.summary())
    print(f"\nalgo={algo}  episodes={episodes}  seed={args.seed}  tag={tag}")
    if map_seed != args.seed:
        print(f"harita-tohumu AYRI: map_seed={map_seed} (ag-init/kesif seed={args.seed})")
    print(f"alarm kuplaji={'ACIK' if args.alert else 'kapali'}  "
          f"risk-shaping={'kapali' if args.no_risk_shaping else 'acik'}\n")

    env = StrikeMissionEnv(max_steps=args.max_steps, seed=map_seed,
                           alert_enabled=args.alert,
                           risk_shaping=not args.no_risk_shaping)
    al_alpha = args.al_alpha if args.al_alpha is not None else 0.0
    munchausen_tau = args.munchausen_tau if args.munchausen_tau is not None else 0.0
    n_quantiles = args.quantiles if args.quantiles is not None else 1
    qr_optimism = args.qr_optimism if args.qr_optimism is not None else 0.0
    if n_quantiles > 1:
        al_alpha = 0.0; munchausen_tau = 0.0   # QR-DQN bu ikisiyle birlesmez
    elif munchausen_tau > 0.0:
        al_alpha = 0.0   # Munchausen AL'i kapsar; ikisi birden anlamsiz
    agent = build_agent(algo, args.seed, args.device, lr=args.lr,
                        eps_end=args.eps_end, dueling=args.dueling,
                        prioritized=args.prioritized, al_alpha=al_alpha,
                        munchausen_tau=munchausen_tau, layernorm=args.layernorm,
                        vdn_batch=args.vdn_batch,
                        vdn_target_update=args.vdn_target_update,
                        n_quantiles=n_quantiles, qr_optimism=qr_optimism)

    if args.resume_from:
        src = (transfer.resolve_source(algo) if args.resume_from == "pathfinding"
               else args.resume_from)
        if src is None or not os.path.exists(src):
            print(f"UYARI: resume kaynagi bulunamadi ({args.resume_from}) — "
                  f"sifirdan basliyor")
        else:
            transfer.resume(algo, agent, src, args.resume_head_reset,
                            not args.no_mixer_transfer)

    log_path = os.path.join(C.RUNS_DIR, f"{tag}_train_log.csv")
    dense_path = os.path.join(C.RUNS_DIR, f"{tag}_train_dense.csv")
    ep_path = os.path.join(C.RUNS_DIR, f"{tag}_episodes.csv")
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    log_w = csv.writer(log_f)
    log_w.writerow(["episode", "eps", *METRIC_KEYS])
    dense_f = open(dense_path, "w", newline="", encoding="utf-8")
    dense_w = csv.writer(dense_f)
    dense_w.writerow(["episode", "eps", "team_success_ma", "n_dead_ma",
                      "inner_ma", "steps_ma", "loss_ma", "q_mean", "q_gap"])

    # Q OLCEK PROBU (§11.14). Uzun egitimde VDN'in Q'su 17.5 -> 37.6 (2 kat)
    # cikarken AKSIYON BOSLUGU 0.103 -> 0.058'e (yariya) dustu; QMIX'te bosluk
    # bastan 0.0016 (VDN'in 60'ta biri), yani ag aksiyonlari AYIRT EDEMIYOR.
    # Politika argmax Q oldugu icin bozulan sey tam olarak budur. Bu yuzden
    # egitim boyunca iki sayi loglanir:
    #   q_mean = ortalama Q  (sisme)
    #   q_gap  = en iyi ile ikinci arasindaki fark (ayirt etme gucu)
    # SABIT bir gozlem kumesinde olculur — yoksa "kotu politika kotu durumlara
    # gider, oradaki Q farklidir" diye bir karistirici girer.
    # MAPPO/HAPPO'nun aktoru Q-degeri DEGIL politika LOGIT'i uretir — bu olcek
    # probu VDN/QMIX'e ozgu (bkz. yukaridaki gerekce), onlar icin NO-OP.
    if algo in ("mappo", "happo"):
        def q_stats():
            return 0.0, 0.0
    else:
        _probe = []
        _penv = StrikeMissionEnv(max_steps=args.max_steps, seed=999,
                                 radar_random=env.radar_random, n_radar=C.N_RADAR)
        _o = _penv.reset(map_seed=C.EVAL_SEED_BASE - 1, n_radar=C.N_RADAR)
        for _ in range(64):
            _probe.append(_o[C.AGENT_1])
            _o, _, _d, _ = _penv.step({C.AGENT_1: C.RIGHT, C.AGENT_2: C.DOWN})
            if _d:
                break
        import torch as _torch
        # BUG (bulundu ve duzeltildi, 2026-08-21): PROBE hep CPU'da kaliyordu —
        # --device cuda ile agin AGIRLIKLARI GPU'ya tasindigi icin conv2d
        # "input CPU, weight CUDA" hatasi verip cokuyordu. Once hep --device cpu
        # kullanildigi icin bu hic ortaya cikmamisti.
        PROBE = _torch.as_tensor(np.asarray(_probe, dtype=np.float32),
                                 device=args.device)

        def q_stats():
            net = agent.online[C.AGENT_1]
            with _torch.no_grad():
                out = net(PROBE)
                q = (out.mean(-1) if out.dim() == 3 else out)[:, :4]  # QR: kuantil ort.
            t2 = q.topk(2, dim=1).values
            return float(q.mean()), float((t2[:, 0] - t2[:, 1]).mean())
    # EPISODE BASINA ham kayit — grafikte hem ham nokta hem hareketli ortalama
    # cizilebilsin diye. Sadece hareketli ortalama loglamak, gurultunun ne kadar
    # oldugunu gizler; sadece hami loglamak da egilimi gostermez.
    ep_f = open(ep_path, "w", newline="", encoding="utf-8")
    ep_w = csv.writer(ep_f)
    ep_w.writerow(["episode", "eps", "team_success", "dead1", "dead2",
                   "reached1", "reached2", "steps", "outer_total", "inner_total"])

    win = C.TRAIN_HARM_WINDOW
    ma = {k: deque(maxlen=win) for k in ("team", "dead", "inner", "steps", "loss")}
    runner = RUNNER[algo]
    if args.n_envs > 1:
        print(f"paralel rollout: {args.n_envs} ortam (GPU batch icin)")
        # BUG (bulundu ve duzeltildi, 2026-08-28): qmix_parallel_rollout
        # onceden de VARDI ama buraya hic BAGLANMAMISTI — --n-envs>1 ile
        # qmix her zaman sessizce tek-env'e duşuyordu (GPU'da hicbir
        # kazanc olmadan, dispatch overhead'i tam tersine YAVASLATIYORDU
        # — bkz. VDN'in AYNI bulgusu, agents/vdn.py modul dosya stringi).
        # MAPPO/HAPPO (2026-08-28): ppo_parallel_rollout AYNI kazanci
        # saglar ama auto-reset YOK (chunk deseni) — bkz. o fonksiyonun
        # dosya stringi, GAE episode sinirlarini net bilmek zorunda.
        fn = {"vdn": vdn_parallel_rollout, "qmix": qmix_parallel_rollout,
             "mappo": ppo_parallel_rollout, "happo": ppo_parallel_rollout}[algo]
        roll = fn(agent, args.n_envs, args.max_steps, map_seed, episodes,
                  args.n_radar, not args.no_risk_shaping, args.alert)
    else:
        roll = None
    floor = max(1.0, episodes * C.EPS_FLOOR_FRAC)
    eps_end_eff = args.eps_end if args.eps_end is not None else C.EPS_END
    # eps_progress 0..1 arasi bir ILERLEME; 0 -> EPS_START, 1 -> EPS_END.
    # --eps-start verilirse o degere karsilik gelen ilerlemeden BASLANIR:
    #     eps = EPS_START + frac*(EPS_END - EPS_START)  ->  frac0'i cozeriz.
    frac0 = 0.0
    if args.eps_start is not None:
        span = C.EPS_START - eps_end_eff
        frac0 = min(1.0, max(0.0, (C.EPS_START - args.eps_start) / span))
        print(f"epsilon {args.eps_start:.2f}'den basliyor "
              f"(ilerleme {frac0:.2f}), {eps_end_eff} tabanina "
              f"ep {int(floor)} civarinda iner")

    # OGRETMEN-CAPASI (teacher-anchored VDN): SADECE --bc-lambda-start acikca
    # verilirse devreye girer — varsayilan davranis (bayrak yoksa) ESKISIYLE
    # BIREBIR AYNI kalsin diye lambda'nin config'teki VDN_BC_LAMBDA_START'i
    # SESSIZCE varsayilan almasina IZIN VERILMEZ (aksi halde tum gelecek
    # vdn kosulari, mevcut %46 mask-fix referansiyla KIYASLANAMAZ hale gelirdi).
    bc_lam_start = args.bc_lambda_start if args.bc_lambda_start is not None else 0.0
    bc_lam_end = (args.bc_lambda_end if args.bc_lambda_end is not None
                 else C.VDN_BC_LAMBDA_END if bc_lam_start > 0.0 else 0.0)
    bc_floor = max(1.0, episodes * C.VDN_BC_LAMBDA_DECAY_FRAC)
    if algo == "vdn" and bc_lam_start > 0.0:
        print(f"ogretmen-capasi (BC-ankraj): lambda {bc_lam_start:.2f} -> "
              f"{bc_lam_end:.2f}, ep {int(bc_floor)} civarinda tabana iner")
    if algo == "vdn" and args.prioritized:
        print(f"PER acik: alpha={C.PER_ALPHA}, beta {C.PER_BETA_START} -> "
              f"{C.PER_BETA_END} (ep {int(floor)} civarinda 1.0'a varir)"
              + (", dueling mimari" if args.dueling else ""))
    elif algo == "vdn" and args.dueling:
        print("dueling mimari acik (PER kapali)")
    if algo == "vdn" and al_alpha > 0.0:
        print(f"Advantage Learning acik: alpha={al_alpha} "
              f"(action-gap ~{1.0/(1.0-al_alpha):.1f}x hedeflenir)")
    if algo == "vdn" and munchausen_tau > 0.0:
        print(f"Munchausen RL acik: tau={munchausen_tau}, alpha={C.MUNCHAUSEN_ALPHA}, "
              f"clip={C.MUNCHAUSEN_CLIP} (AL'i kapsar + entropi bootstrap)")
    if algo == "vdn" and args.layernorm:
        print("LayerNorm acik: Q-agi gizli katmanlari normalize (Q-iraksama karsiti)")
    if algo == "vdn" and (args.vdn_batch or args.vdn_target_update):
        print(f"VDN override: batch={agent.batch_size}  target_update={agent.target_update}")
    if algo == "vdn" and n_quantiles > 1:
        print(f"QR-DQN acik: {n_quantiles} kuantil, optimism={qr_optimism} "
              f"(getiri dagilimi ogrenilir; iyimser aksiyon secimi)")

    t_start = time.perf_counter()
    best = -1.0   # mission_prob

    for ep in range(1, episodes + 1):
        set_eps(agent, algo, frac0 + (1.0 - frac0) * min(1.0, ep / floor))
        if algo == "vdn" and bc_lam_start > 0.0:
            agent.set_bc_lambda(bc_lam_start + (bc_lam_end - bc_lam_start)
                                * min(1.0, ep / bc_floor))
        if algo == "vdn" and args.prioritized:
            # eps/bc_lambda ile AYNI ritim (floor): PER_BETA_START -> PER_BETA_END.
            agent.set_per_beta(C.PER_BETA_START + (C.PER_BETA_END - C.PER_BETA_START)
                               * min(1.0, ep / floor))
        # Curriculum: radar sayisi 10 -> 40 rampalanir (bkz. sampler). Harita
        # tohumu VERILMEZ -> env kendi rng'sinden ceker, yani her episode taze
        # harita. Ezberlenecek sabit havuz yok; asiri ogrenmeye karsi asil
        # savunma bu (degerlendirme tohumlariyla da kesismiyor).
        if roll is not None:
            info, losses = next(roll)
        else:
            rk = ({"n_radar": (args.n_radar if args.n_radar is not None
                               else curriculum_n_radar(ep, episodes))}
                  if env.radar_random else None)
            info, losses = runner(env, agent, train=True, reset_kwargs=rk)

        ep_w.writerow([ep, f"{current_eps(agent, algo):.4f}",
                       int(info["team_success"]),
                       int(not info["alive1"]), int(not info["alive2"]),
                       int(info["reached1"]), int(info["reached2"]),
                       info["steps"], info["outer_total"], info["inner_total"]])

        ma["team"].append(float(info["team_success"]))
        ma["dead"].append(float(info["n_dead"]))
        ma["inner"].append(float(info["inner_total"]))
        ma["steps"].append(float(info["steps"]))
        if losses:
            ma["loss"].append(float(np.mean(losses)))

        if ep % C.TRAIN_HARM_LOG_EVERY == 0:
            dense_w.writerow([ep, f"{current_eps(agent, algo):.4f}",
                              f"{np.mean(ma['team']):.4f}", f"{np.mean(ma['dead']):.4f}",
                              f"{np.mean(ma['inner']):.1f}", f"{np.mean(ma['steps']):.1f}",
                              f"{np.mean(ma['loss']) if ma['loss'] else 0:.5f}",
                              *(f"{v:.4f}" for v in q_stats())])
            dense_f.flush()
            ep_f.flush()      # yoksa episode CSV'si ancak kosu bitince yazilir
                              # ve uzun kosularda ilerleme hic gorunmez
            el = time.perf_counter() - t_start
            print(f"ep{ep:>6}  eps={current_eps(agent, algo):.3f}  "
                  f"takim(ma)={np.mean(ma['team'])*100:5.1f}%  "
                  f"olu(ma)={np.mean(ma['dead']):.2f}  "
                  f"ic(ma)={np.mean(ma['inner']):6.0f}  "
                  f"adim(ma)={np.mean(ma['steps']):6.0f}  "
                  f"{el:.0f}s ({el/ep:.2f}s/ep)")

        if ep % eval_every == 0 or ep == episodes:
            if roll is None:
                m = evaluate(env, agent, algo, args.eval_episodes)
            elif algo in ("mappo", "happo"):
                # vdn_vec_evaluate KULLANILAMAZ: act_batch imzasi farkli
                # (bkz. ppo_vec_evaluate dosya stringi).
                m = ppo_vec_evaluate(env, agent, args.eval_episodes, 12345, args.n_envs)
            else:
                m = vdn_vec_evaluate(env, agent, args.eval_episodes, 12345, args.n_envs)
            log_w.writerow([ep, f"{current_eps(agent, algo):.4f}",
                            *[f"{m[k]:.4f}" for k in METRIC_KEYS]])
            log_f.flush()
            print(f"  [eval ep{ep}] {fmt_eval(m)}", flush=True)
            # CHECKPOINT SECIM OLCUTU — analytic_surv_team KULLANILMAZ.
            # BUG (bulundu ve duzeltildi): olcut
            #     score = team_success + analytic_surv_team
            # idi ve analytic_surv_team SISEN metrik (GIDILEN yolu olcer,
            # hedefe varmayan politikada 1.0'a yaklasir). Sonuc: secici
            # sistematik olarak EN AZ HAREKET EDEN modeli seciyordu.
            # OLCULDU (r30s0, 30 radar):
            #   IQL  gercek tepe ep1000 (gorev 0.1029) -> ep500 secildi
            #        (VARIS %0.0); 50 haritada 0/50 varis, surv_ratio 0.0000
            #   QMIX gercek tepe ep750  (gorev 0.0601) -> ep500 secildi
            #        (VARIS %3.3); 50 haritada 1/50
            #   VDN  sans eseri kurtuldu (ep500 zaten gercek tepesiydi)
            # Yani iki algoritma kendi EN KOTU checkpoint'iyle olculdu ve
            # karsilastirma kontamine oldu.
            # mission_prob sismez: hedefe varmayan rota 0 alir — yani
            # route_reached ZATEN mission_prob'un icinde (varmayan rota
            # otomatik 0). Ayrica birincil tutmaya gerek yok.
            #
            # ONCEKI KURAL (route_reached birincil, mission_prob esitlik
            # bozucu) SOMUT OLARAK KOTU SECIM YAPTI (ms3ks1_vdn, 2026-08-16):
            #   ep250: VARIS=90%  takim=14%  gorev=0.1729  -> SECILDI
            #   ep500: VARIS=72%  takim=24%  gorev=0.2507  -> elendi
            # ep500 hem takim basarisinda hem mission_prob'da daha iyiydi,
            # sadece VARIS'ta dusuktu; lexicographic kural onu sirf bu yuzden
            # eledi. mission_prob'u tek basina birincil almak bunu duzeltir.
            #
            # team_success KULLANILMIYOR: zar sonucuna bagli, 30 haritada
            # 1-2 sayimdan ibaret, tohumdan tohuma yer degistiriyor.
            # analytic_surv_team ASLA kullanilmamali: SISER (gidilen yolu
            # olcer, hedefe varmayan politikada 1.0'a yaklasir). Bu projede
            # UC KEZ tuzak kurdu:
            #   IQL  gercek tepe ep1000 -> ep500 secildi (VARIS %0.0)
            #   QMIX gercek tepe ep750  -> ep500 secildi (VARIS %3.3)
            score = m["mission_prob"]
            stem = os.path.join(C.RUNS_DIR, "ckpt", tag)
            if score >= best:
                best = score
                save(agent, algo, stem)
            save(agent, algo, stem + "_last")
            # --save-all-ckpts: HER eval'da ayri kaydet (deploy-time ensemble
            # icin). fast-eps rejiminde her ckpt ~%60-68 ama FARKLI haritalarda
            # hata yapiyor -> Q-ortalama ensemble varyansi azaltir (post-hoc,
            # egitim dengesini bozamaz).
            if args.save_all_ckpts:
                save(agent, algo, f"{stem}_ep{ep}")

    log_f.close()
    dense_f.close()
    ep_f.close()
    el = time.perf_counter() - t_start
    print(f"\nbitti: {episodes} episode, {el/60:.1f} dk ({el/episodes:.2f} s/ep)")

    # --- egitim SONRASI deterministik gosterim episode'lari (eps=0)
    # Yollari JSON'a yazar; viz/plot_report.py bunlardan harita cizer.
    demo_path = run_demo(env, agent, algo, args.demo_episodes, tag)
    print(f"log: {log_path}\n     {dense_path}\n     {ep_path}\n     {demo_path}")


def run_demo(env, agent, algo: str, episodes: int, tag: str) -> str:
    """eps=0 ile N episode oynat, YOLLARI kaydet."""
    import json
    runner = RUNNER[algo]
    out = []
    print(f"\n--- {episodes} deterministik gosterim episode'u (eps=0) ---")
    # Gosterim de HELD-OUT haritalarda: egitimde gorulmemis tohumlar.
    # Rapora giren yol cizimlerinin "ezberlenmis harita" olmadigi boyle garanti.
    seeds = eval_map_seeds(episodes) if env.radar_random else [None] * episodes
    for i in range(episodes):
        env.rng = np.random.default_rng(C.DEMO_SEED + i)
        rk = ({"map_seed": seeds[i], "n_radar": C.N_RADAR}
              if env.radar_random else None)
        info, _ = runner(env, agent, train=False, reset_kwargs=rk)
        out.append({
            "episode": i,
            # Harita artik episode'a ozgu -> radar seti de kaydedilmeli,
            # yoksa viz yolu YANLIS haritanin uzerine cizer.
            "map_seed": seeds[i],
            "radars": [list(r) for r in env.radars],
            "team_success": bool(info["team_success"]),
            "both_reached": bool(info["both_reached"]),
            "reached1": bool(info["reached1"]), "reached2": bool(info["reached2"]),
            "alive1": bool(info["alive1"]), "alive2": bool(info["alive2"]),
            "steps": int(info["steps"]),
            "outer1": int(info["outer1"]), "inner1": int(info["inner1"]),
            "outer2": int(info["outer2"]), "inner2": int(info["inner2"]),
            "surv1": float(info["surv1"]), "surv2": float(info["surv2"]),
            "route_overlap": float(info["route_overlap"]),
            # Yollar 2800 noktaya kadar cikabiliyor; her 5. noktayi almak
            # cizim icin fazlasiyla yeterli ve JSON'u 5x kucultuyor.
            "path1": [list(p) for p in info["path1"][::5]] + [list(info["path1"][-1])],
            "path2": [list(p) for p in info["path2"][::5]] + [list(info["path2"][-1])],
        })
        print(f"  ep{i}: takim={'EVET' if info['team_success'] else 'hayir'}  "
              f"A1={'vardi' if info['reached1'] else ('OLDU' if not info['alive1'] else 'timeout')}  "
              f"A2={'vardi' if info['reached2'] else ('OLDU' if not info['alive2'] else 'timeout')}  "
              f"adim={info['steps']}  maruziyet={info['outer1']+info['outer2']}/"
              f"{info['inner1']+info['inner2']}")
    p = os.path.join(C.RUNS_DIR, f"{tag}_demo_episodes.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f)
    n_ok = sum(d["team_success"] for d in out)
    print(f"  -> {n_ok}/{episodes} takim basarisi")
    return p


if __name__ == "__main__":
    main()
