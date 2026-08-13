"""Egitim dongusu — python train.py --algo vdn --episodes 500

Algoritmalar: iql | vdn | qmix

DQN (Asama 3, tek ajan sanity check) AYRI bir modul olarak yazilmadi, cunku
ALARM KUPLAJI KAPALIYKEN (varsayilan) iki ucak gercekten bagimsiz: ortak odul
yok, carpisma yok, olum zarlari bagimsiz. Yani `--algo iql` zaten iki paralel
tek-ajan DQN kosusudur ve Asama 3'un kabul kriterini (tek ucak hedefe
guvenle varabiliyor mu) dogrudan olcer.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import deque

import numpy as np

import config as C
from agents import transfer
from agents.dqn import DQNAgent
from agents.qmix import QMixAgent
from agents.vdn import VDNAgent
from env.sampler import curriculum_n_radar, eval_map_seeds
from env.strike_env import StrikeMissionEnv
from env.two_agent import play_episode, play_episode_qmix, play_episode_vdn

# Windows'ta stdout bir dosyaya/boruya yonlendirilince cp1252 kullaniliyor ve
# Turkce karakterlerde UnicodeEncodeError veriyor (MARL-Pathfinding'de yasandi).
try:
    # line_buffering: cikti bir dosyaya/boruya yonlendirildiginde Python
    # varsayilan olarak ~8KB tamponlar; 30-60 dakikalik bir kosuda ilerleme
    # ancak kosu BITINCE gorunur. Satir tamponu bunu canli hale getirir.
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

RUNNER = {"iql": play_episode, "vdn": play_episode_vdn, "qmix": play_episode_qmix}


def build_agent(algo: str, seed: int, device: str):
    if algo == "iql":
        return {C.AGENT_1: DQNAgent(seed=seed, device=device,
                                    buffer_size=C.IQL_BUFFER, batch_size=C.IQL_BATCH,
                                    lr=C.IQL_LR, eps_decay_steps=C.IQL_EPS_DECAY_STEPS,
                                    learn_start=C.IQL_LEARN_START,
                                    target_update=C.IQL_TARGET_UPDATE),
                C.AGENT_2: DQNAgent(seed=seed + 1, device=device,
                                    buffer_size=C.IQL_BUFFER, batch_size=C.IQL_BATCH,
                                    lr=C.IQL_LR, eps_decay_steps=C.IQL_EPS_DECAY_STEPS,
                                    learn_start=C.IQL_LEARN_START,
                                    target_update=C.IQL_TARGET_UPDATE)}
    if algo == "vdn":
        return VDNAgent(seed=seed, device=device)
    if algo == "qmix":
        return QMixAgent(seed=seed, device=device)
    raise ValueError(algo)


def set_eps(agent, algo: str, frac: float):
    if algo == "iql":
        for a in agent.values():
            a.set_eps_progress(frac)
    else:
        agent.set_eps_progress(frac)


def current_eps(agent, algo: str) -> float:
    return agent[C.AGENT_1].eps if algo == "iql" else agent.eps


def save(agent, algo: str, path_stem: str):
    if algo == "iql":
        agent[C.AGENT_1].save(f"{path_stem}_agent1.pt")
        agent[C.AGENT_2].save(f"{path_stem}_agent2.pt")
    else:
        agent.save(f"{path_stem}.pt")


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
    ap.add_argument("--algo", choices=["iql", "vdn", "qmix"], required=True)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=5)
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
                         "boyunca rastgele aksiyonlarla bozar. 0.2-0.4 tipik.")
    ap.add_argument("--n-radar", type=int, default=None,
                    help="radar sayisini SABITLE (curriculum'u kapatir). "
                         "Verilmezse 10->40 rampasi kullanilir.")
    ap.add_argument("--resume-head-reset", action="store_true",
                    help="govdeyi yukle ama Q ciktisi katmanini sifirla "
                         "(gamma/olcek degistigi icin onerilir — bkz. agents/transfer.py)")
    ap.add_argument("--n-step", type=int, default=None,
                    help="n-adim getiri ufku (varsayilan config.N_STEP=1). "
                         ">1 bootstrap payini azaltir; uzun egitimdeki "
                         "bozulma hipotezinin testi (bkz. agents/nstep.py).")
    ap.add_argument("--no-mixer-transfer", action="store_true")
    args = ap.parse_args()

    if args.n_step is not None:          # ajanlar kurulmadan ONCE ezilmeli
        C.N_STEP = args.n_step
        import agents.dqn, agents.vdn, agents.qmix
        for m in (agents.dqn, agents.vdn, agents.qmix):
            m.N_STEP = args.n_step
    algo = args.algo
    episodes = args.episodes or getattr(C, f"{algo.upper()}_EPISODES")
    eval_every = args.eval_every or getattr(C, f"{algo.upper()}_EVAL_EVERY")
    tag = args.tag or f"{algo}_s{args.seed}"
    os.makedirs(os.path.join(C.RUNS_DIR, "ckpt"), exist_ok=True)

    print(C.summary())
    print(f"\nalgo={algo}  episodes={episodes}  seed={args.seed}  tag={tag}")
    print(f"alarm kuplaji={'ACIK' if args.alert else 'kapali'}  "
          f"risk-shaping={'kapali' if args.no_risk_shaping else 'acik'}\n")

    env = StrikeMissionEnv(max_steps=args.max_steps, seed=args.seed,
                           alert_enabled=args.alert,
                           risk_shaping=not args.no_risk_shaping)
    agent = build_agent(algo, args.seed, args.device)

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
    PROBE = _torch.as_tensor(np.asarray(_probe, dtype=np.float32))

    def q_stats():
        net = (agent[C.AGENT_1].online if algo == "iql" else agent.online[C.AGENT_1])
        with _torch.no_grad():
            q = net(PROBE)[:, :4]
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
    floor = max(1.0, episodes * C.EPS_FLOOR_FRAC)
    # eps_progress 0..1 arasi bir ILERLEME; 0 -> EPS_START, 1 -> EPS_END.
    # --eps-start verilirse o degere karsilik gelen ilerlemeden BASLANIR:
    #     eps = EPS_START + frac*(EPS_END - EPS_START)  ->  frac0'i cozeriz.
    frac0 = 0.0
    if args.eps_start is not None:
        span = C.EPS_START - C.EPS_END
        frac0 = min(1.0, max(0.0, (C.EPS_START - args.eps_start) / span))
        print(f"epsilon {args.eps_start:.2f}'den basliyor "
              f"(ilerleme {frac0:.2f}), {C.EPS_END} tabanina "
              f"ep {int(floor)} civarinda iner")
    t_start = time.perf_counter()
    best = (-1.0, -1.0)   # (route_reached, mission_prob)

    for ep in range(1, episodes + 1):
        set_eps(agent, algo, frac0 + (1.0 - frac0) * min(1.0, ep / floor))
        # Curriculum: radar sayisi 10 -> 40 rampalanir (bkz. sampler). Harita
        # tohumu VERILMEZ -> env kendi rng'sinden ceker, yani her episode taze
        # harita. Ezberlenecek sabit havuz yok; asiri ogrenmeye karsi asil
        # savunma bu (degerlendirme tohumlariyla da kesismiyor).
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
        ls = losses if isinstance(losses, list) else (losses[C.AGENT_1] + losses[C.AGENT_2])
        if ls:
            ma["loss"].append(float(np.mean(ls)))

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
            m = evaluate(env, agent, algo, args.eval_episodes)
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
            # mission_prob sismez: hedefe varmayan rota 0 alir.
            # BIRINCIL: route_reached (rotasi hedefe VARIYOR mu).
            # Burak: "amacimiz hedefe en cok varan rotayi checkpoint'lemek".
            # ESITLIK BOZUCU: mission_prob (varis x hayatta kalma) — ayni
            # varis oranini veren iki checkpoint'ten GUVENLI olani secilir.
            # Tuple karsilastirmasi tam olarak bunu yapar (leksikografik).
            #
            # team_success KULLANILMIYOR: zar sonucuna bagli, 30 haritada
            # 1-2 sayimdan ibaret, tohumdan tohuma yer degistiriyor.
            # analytic_surv_team ASLA kullanilmamali: SISER (gidilen yolu
            # olcer, hedefe varmayan politikada 1.0'a yaklasir). Bu projede
            # UC KEZ tuzak kurdu — en sonuncusu tam bu satirdi ve secici
            # sistematik olarak EN AZ HAREKET EDEN modeli seciyordu:
            #   IQL  gercek tepe ep1000 -> ep500 secildi (VARIS %0.0)
            #   QMIX gercek tepe ep750  -> ep500 secildi (VARIS %3.3)
            score = (m["route_reached"], m["mission_prob"])
            stem = os.path.join(C.RUNS_DIR, "ckpt", tag)
            if score >= best:
                best = score
                save(agent, algo, stem)
            save(agent, algo, stem + "_last")

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
