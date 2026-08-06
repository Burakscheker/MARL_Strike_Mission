"""Rastgele haritada degerlendirme — Strike_Mission.md §11.6.

ASIL METRIK surv_ratio. Gerekce: 40 radarli rastgele haritada oracle tavani
ortalama %32 ama medyan %7.2 (baselines/scan_random_maps). Yani ham basari
orani harita sansiyla 10 kat oynuyor ve tek basina ajanin iyi mi kotu mu
oldugunu SOYLEMIYOR. surv_ratio bunu normalize eder:

    surv_ratio = analytic_survival(ajanin yolu) / analytic_survival(oracle yolu)

1.0 = ajan oracle kadar iyi bir yol secti (zar ne getirirse getirsin).
Zar sonucundan bagimsiz, yolun DETERMINISTIK bir fonksiyonu — sansla iyi
gorunmek mumkun degil (odul hackleme duruşu, §11.8).

ORTAK HARITA SETI: tum politikalar env/sampler.eval_map_seeds() tohumlarindan
uretilen AYNI haritalarda olculur. Bu tohumlar egitim araligiyla KESISMEZ,
yani hicbiri ezberlenmis olamaz.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import config as C
from baselines.risk_oracle import (RISK_W, direction_costs, exposure,
                                   greedy_path, survival_prob)
from env.sampler import eval_map_seeds
from env.strike_env import StrikeMissionEnv

DIRS4 = C.DIRS


# --------------------------------------------------------------- referanslar

def oracle_path(env) -> list:
    """Ortamin KENDI risk-mesafe haritasindan oracle yolu.

    env.dist zaten `1 adim + RISK_W * p(giris)` maliyetiyle kurulmus (bkz.
    _build_map), yani bu yol odul fonksiyonunun optimize ettigi seyin ta
    kendisini optimize ediyor — saf risk minimizasyonu DEGIL. Dogru referans
    bu: "bizim odulumuz altinda en iyi yol".
    """
    cost = direction_costs(env.zone, RISK_W, env.hazard_mode)
    return greedy_path(C.START, C.GOAL, env.dist, cost)


def random_monotone_path(rng, n=C.GRID_N) -> list:
    """Sadece SAG/ASAGI — hedefe giden en kisa yollardan rastgele biri.

    Dogru taban cizgisi bu, "rastgele yuruyus" DEGIL: rastgele yuruyus
    1000x1000'de hedefe hicbir zaman varamaz ve karsilastirma anlamsizlasir.
    """
    r = c = 0
    path = [(0, 0)]
    while (r, c) != (n - 1, n - 1):
        if r == n - 1:
            c += 1
        elif c == n - 1:
            r += 1
        elif rng.random() < 0.5:
            c += 1
        else:
            r += 1
        path.append((r, c))
    return path


def staircase_path(n=C.GRID_N) -> list:
    """Naif "hedefe dogru git": sag/asagi donusumlu merdiven."""
    path = [(0, 0)]
    while path[-1] != (n - 1, n - 1):
        r, c = path[-1]
        if c < n - 1:
            path.append((r, c + 1))
        if path[-1] != (n - 1, n - 1) and path[-1][0] < n - 1:
            r, c = path[-1]
            path.append((r + 1, c))
    return path


# ------------------------------------------------------------------ politika

def rollout(env, agent, algo: str, map_seed: int) -> dict:
    """Egitilmis ajani TEK haritada eps=0 ile kosur."""
    from env.two_agent import play_episode, play_episode_qmix, play_episode_vdn
    runner = {"iql": play_episode, "vdn": play_episode_vdn,
              "qmix": play_episode_qmix}[algo]
    info, _ = runner(env, agent, train=False,
                     reset_kwargs={"map_seed": map_seed, "n_radar": C.N_RADAR})
    return info


# --------------------------------------------------------------------- ana

def evaluate_maps(n_maps: int, agent=None, algo: str | None = None,
                  seed: int = 12345) -> dict:
    env = StrikeMissionEnv(seed=seed, radar_random=True, n_radar=C.N_RADAR)
    # ROTA ORTAMI: zar kapali, ucak olmez. Ajanin NIYET ETTIGI tam rotayi
    # gormek icin (bkz. StrikeMissionEnv.death_enabled). Bunsuz surv_ratio
    # yarida olen politikalarda SISIYOR: kisa yol = "guvenli" yol.
    route_env = (StrikeMissionEnv(seed=seed, radar_random=True,
                                  n_radar=C.N_RADAR, death_enabled=False)
                 if agent is not None else None)
    rng = np.random.default_rng(seed)
    rows = []

    for i, ms in enumerate(eval_map_seeds(n_maps)):
        env.reset(map_seed=ms, n_radar=C.N_RADAR)
        z, mode = env.zone, env.hazard_mode

        orc = oracle_path(env)
        s_orc = survival_prob(orc, z, mode)
        row = {
            "map_seed": ms,
            "safe_frac": float((z == 0).mean()),
            "inner_frac": float((z == 2).mean()),
            "start_zone": int(z[C.START]),
            "goal_zone": int(z[C.GOAL]),
            "oracle_surv": s_orc,
            "oracle_len": len(orc) - 1,
            "oracle_fits": (len(orc) - 1) <= C.MAX_STEPS,
            "stair_surv": survival_prob(staircase_path(), z, mode),
            "rndmono_surv": float(np.mean([
                survival_prob(random_monotone_path(rng), z, mode)
                for _ in range(20)])),
        }

        if agent is not None:
            info = rollout(env, agent, algo, ms)
            # Zar KAPALI kosu: ajanin niyet ettigi tam rota.
            rinfo = rollout(route_env, agent, algo, ms)
            # mission_prob = "bu rotayla gorevi GERCEKTEN tamamlama olasiligi".
            # Hedefe varmayan rota 0 alir — hic hareket etmeyip surv=1.0
            # toplamak boylece imkansiz.
            m1 = rinfo["surv1"] if rinfo["reached1"] else 0.0
            m2 = rinfo["surv2"] if rinfo["reached2"] else 0.0
            mission = 1.0 - (1.0 - m1) * (1.0 - m2)      # >=1 ucak tamamlar
            row.update({
                "agent_surv_best": max(info["surv1"], info["surv2"]),
                "team_success": float(info["team_success"]),
                "n_dead": float(info["n_dead"]),
                "steps": float(info["steps"]),
                "timeout": float(info["timeout"]),
                "route_overlap": float(info["route_overlap"]),
                "route_reached": float(rinfo["reached1"] or rinfo["reached2"]),
                "route_steps": float(rinfo["steps"]),
                "mission_prob": mission,
                # ASIL METRIK: rotanin gorev tamamlama olasiligi / oracle'inki.
                # Oracle 0 ise (harita gercekten cozumsuz) tanimsiz — o
                # haritalar ortalamaya KATILMAZ.
                "surv_ratio": (max(m1, m2) / s_orc) if s_orc > 1e-12 else np.nan,
            })
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_maps} harita")
    return {"rows": rows}


def summarize(rows: list, has_agent: bool) -> str:
    def col(k):
        return np.array([r[k] for r in rows if k in r], dtype=float)

    out = []
    out.append(f"{len(rows)} held-out harita ({C.N_RADAR} radar, "
               f"tohum {C.EVAL_SEED_BASE}+)")
    out.append(f"guvenli hucre ort %{100*col('safe_frac').mean():.1f}   "
               f"ic halka %{100*col('inner_frac').mean():.1f}")
    out.append(f"B bir halkada: {int((col('start_zone')>0).sum())}/{len(rows)}   "
               f"H bir halkada: {int((col('goal_zone')>0).sum())}/{len(rows)}")
    out.append(f"oracle yolu MAX_STEPS'e siganlar: "
               f"{int(col('oracle_fits').sum())}/{len(rows)}")
    out.append("")
    out.append(f"{'politika':<26}{'hayatta (ort)':>15}{'medyan':>10}")
    for name, k in (("Dijkstra oracle (TAVAN)", "oracle_surv"),
                    ("merdiven (naif capraz)", "stair_surv"),
                    ("rastgele monoton", "rndmono_surv")):
        v = col(k)
        out.append(f"{name:<26}{v.mean():>15.4f}{np.median(v):>10.4f}")

    if has_agent:
        v = col("mission_prob")
        out.append(f"{'AJAN (rota, zar kapali)':<26}{v.mean():>15.4f}"
                   f"{np.median(v):>10.4f}")
        out.append("")
        out.append(f"rotasi hedefe VARIYOR: %{100*col('route_reached').mean():.1f} "
                   f"({int(col('route_reached').sum())}/{len(rows)} harita)   "
                   f"rota uzunlugu {col('route_steps').mean():.0f} adim")
        sr = col("surv_ratio")
        sr = sr[np.isfinite(sr)]
        out.append(f"*** surv_ratio (rota/oracle): ort {sr.mean():.4f}  "
                   f"medyan {np.median(sr):.4f}  (1.0 = mukemmel) ***")
        out.append("")
        out.append("-- zar ACIK kosu (gercek episode) --")
        out.append(f"takim basarisi (>=1 vardi): %{100*col('team_success').mean():.1f}"
                   f"   oracle tavani %{100*(1-(1-col('oracle_surv'))**2).mean():.1f}")
        out.append(f"olu ucak/episode {col('n_dead').mean():.2f}   "
                   f"timeout %{100*col('timeout').mean():.1f}   "
                   f"adim {col('steps').mean():.0f}   "
                   f"yol ortusme {col('route_overlap').mean():.3f}")
        out.append(f"NOT: 'agent_surv_best' ({col('agent_surv_best').mean():.4f}) "
                   f"GIDILEN yolu olcer ve yarida olen politikalarda SISER "
                   f"(kisa yol = guvenli gorunur) — karsilastirma icin "
                   f"mission_prob kullan.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default=None, choices=("iql", "vdn", "qmix"))
    ap.add_argument("--ckpt", default=None, help="egitilmis model (.pt)")
    ap.add_argument("--maps", type=int, default=C.EVAL_N_MAPS)
    ap.add_argument("--tag", default="eval")
    args = ap.parse_args()

    agent = None
    if args.ckpt:
        if not args.algo:
            raise SystemExit("--ckpt ile --algo da gerekli")
        from train import build_agent
        agent = build_agent(args.algo, 0, "cpu")
        (agent[C.AGENT_1].load(args.ckpt) if args.algo == "iql"
         else agent.load(args.ckpt))
        print(f"model yuklendi: {args.ckpt}")
    else:
        print("model YOK — sadece referans politikalar olculuyor")

    res = evaluate_maps(args.maps, agent, args.algo)
    text = summarize(res["rows"], agent is not None)
    print("\n" + text)

    os.makedirs(C.RUNS_DIR, exist_ok=True)
    out = os.path.join(C.RUNS_DIR, f"{args.tag}_maps.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res["rows"], f)
    with open(os.path.join(C.RUNS_DIR, f"{args.tag}_summary.txt"), "w",
              encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nyazildi: {out}")


if __name__ == "__main__":
    main()
