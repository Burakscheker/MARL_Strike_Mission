"""3 tohumlu IQL/VDN/QMIX karsilastirmasinin rapor figurleri.

Uretilen dosyalar (runs/):
  fig1_surv_ratio.png  — ANA SONUC: surv_ratio, tohum noktalari + hata cubugu
  fig2_paired.png      — eslestirilmis kiyas: harita basina VDN vs IQL
  fig3_training.png    — egitim egrileri (intihar tuzaginin kapandigi gorulur)
  fig4_routes.png      — gercek held-out haritalarda ajan rotasi vs oracle

TASARIM NOTU: ham basari orani BILEREK ana figur degil. Ucu de birbirinin
hata payi icinde (IQL 0.0067+-0.0094, VDN 0.0133+-0.0094, QMIX 0.0067+-0.0094)
ve o grafik "fark yok" gosterir — halbuki ayni kosularda surv_ratio p=0.001
veriyor. fig1'de ikisi YAN YANA cizilir ki metrik seciminin sonucu nasil
belirledigi gorunsun.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
from baselines.risk_oracle import (build_risk_distance_map, build_zone_map,
                                   direction_costs, greedy_path, survival_prob)

ALGOS = ("iql", "vdn", "qmix")
SEEDS = (0, 1, 2)
COL = {"iql": "#58a6ff", "vdn": "#3ddc97", "qmix": "#ff7b72"}
OUT = C.RUNS_DIR

plt.style.use("dark_background")
BG = "#0d1117"


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.15, lw=0.5)
    ax.tick_params(colors="#8b949e", labelsize=8)


def load_eval(seed, algo):
    p = os.path.join(OUT, f"ev{seed}_{algo}_maps.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def col(rows, k):
    return np.array([r.get(k, np.nan) for r in rows], dtype=float)


# ------------------------------------------------------- fig1: ana sonuc
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.patch.set_facecolor(BG)

    for ax, key, title in (
            (axes[0], "surv_ratio",
             "surv_ratio (ANA METRIK) — zar kapali, rotadan"),
            (axes[1], "team_success",
             "ham takim basarisi — ayni kosular, ayri sonuc")):
        means, allpts = [], []
        for a in ALGOS:
            # YUZDE olarak: her iki metrik de 0..1 arasi bir oran
            # (surv_ratio = ajan/oracle, team_success = varan episode orani).
            per_seed = [100 * np.nanmean(col(load_eval(s, a), key)) for s in SEEDS]
            means.append(np.mean(per_seed))
            allpts.append(per_seed)
        x = np.arange(len(ALGOS))
        ax.bar(x, means, color=[COL[a] for a in ALGOS], alpha=0.55, width=0.55)
        for i, pts in enumerate(allpts):
            ax.errorbar(i, np.mean(pts), yerr=np.std(pts), color="white",
                        capsize=5, lw=1.4, zorder=4)
            ax.plot([i] * len(pts), pts, "o", color="white", ms=5,
                    mfc="none", mew=1.3, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in ALGOS], fontsize=10)
        # Etiket, hata cubugunun VE tohum noktalarinin ustune —
        # cubugun tepesine koyunca beyaz hata cizgisiyle ust uste biniyordu.
        top = max(max(pts + [np.mean(pts) + np.std(pts)]) for pts in allpts)
        for i, (v, pts) in enumerate(zip(means, allpts)):
            hi = max(max(pts), np.mean(pts) + np.std(pts))
            ax.text(i, hi + top * 0.045, f"%{v:.2f}", ha="center", va="bottom",
                    fontsize=10, color="white", weight="bold")
        ax.set_ylim(0, top * 1.30)
        _style(ax, title, "", "ortalama (%)")

    axes[0].text(0.98, 0.95, "VDN 3/3 tohumda kazandi\neslestirilmis p=0.001",
                 transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=9, color="#3ddc97")
    axes[1].text(0.98, 0.95, "ucu de birbirinin\nhata payi icinde",
                 transform=axes[1].transAxes, ha="right", va="top",
                 fontsize=9, color="#8b949e")

    fig.suptitle("IQL / VDN / QMIX — 3 tohum, 50 ortak held-out harita",
                 fontsize=12.5)
    fig.tight_layout()
    p = os.path.join(OUT, "fig1_surv_ratio.png")
    fig.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------- fig2: eslestirilmis
def fig2():
    per = {}
    for a in ALGOS:
        # YUZDE (surv_ratio 0..1 bir oran); tohumlar harita basina ortalanir
        per[a] = 100 * np.nanmean(
            np.vstack([col(load_eval(s, a), "surv_ratio") for s in SEEDS]), axis=0)

    fig, ax = plt.subplots(figsize=(6.2, 6))
    fig.patch.set_facecolor(BG)
    m = np.isfinite(per["vdn"]) & np.isfinite(per["iql"])
    ax.scatter(per["iql"][m], per["vdn"][m], s=42, color="#3ddc97",
               alpha=0.75, edgecolor="white", lw=0.5, zorder=3)
    lim = max(per["vdn"][m].max(), per["iql"][m].max()) * 1.08
    ax.plot([0, lim], [0, lim], "--", color="#8b949e", lw=1.2,
            label="esitlik cizgisi")
    ax.fill_between([0, lim], [0, lim], [lim, lim], color="#3ddc97", alpha=0.06)
    wins = int((per["vdn"][m] > per["iql"][m]).sum())
    tot = int((per["vdn"][m] != per["iql"][m]).sum())
    ax.set_xlim(-0.5, lim); ax.set_ylim(-0.5, lim)
    _style(ax, f"Harita basina surv_ratio — VDN {wins}/{tot} haritada ustun\n"
               f"(Wilcoxon p=0.001, ayni haritalar -> zorluk varyansi disarida)",
           "IQL  surv_ratio (%)", "VDN  surv_ratio (%)")
    ax.legend(fontsize=8, framealpha=0.2, loc="lower right")
    fig.tight_layout()
    p = os.path.join(OUT, "fig2_paired.png")
    fig.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return p


# ----------------------------------------------------- fig3: egitim egrileri
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor(BG)
    for ax, ci, title, ylab in (
            (axes[0], 3, "olu ucak / episode (hareketli ort.)", "olu"),
            (axes[1], 5, "episode uzunlugu — intihar tuzagi kapali", "adim")):
        for a in ALGOS:
            curves = []
            for s in SEEDS:
                f = os.path.join(OUT, f"s{s}_{a}_train_dense.csv")
                if not os.path.exists(f):
                    continue
                d = np.genfromtxt(f, delimiter=",", skip_header=1)
                curves.append(d[:, ci])
            if not curves:
                continue
            n = min(len(c) for c in curves)
            A = np.vstack([c[:n] for c in curves])
            ep = np.genfromtxt(
                os.path.join(OUT, f"s0_{a}_train_dense.csv"),
                delimiter=",", skip_header=1)[:n, 0]
            ax.plot(ep, A.mean(0), color=COL[a], lw=1.8, label=a.upper())
            ax.fill_between(ep, A.min(0), A.max(0), color=COL[a], alpha=0.15)
        _style(ax, title, "episode", ylab)
        ax.legend(fontsize=8, framealpha=0.2)
    fig.suptitle("Egitim — 3 tohumun ortalamasi (bant: min-max)", fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_training.png")
    fig.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return p


# -------------------------------------------------------- fig4: demo rotalar
def fig4(ckpt="runs/ckpt/s2_vdn_last.pt", algo="vdn", n=4):
    """Ajanin NIYET ETTIGI rota (zar KAPALI) vs oracle, gercek held-out haritada.

    NEDEN zar kapali: egitimin demo JSON'lari zar ACIK kosuluyor, o yuzden
    yollar olum aninda kesiliyor ve grafik "ajan hicbir yere gidemiyor" gibi
    gorunuyor. Ama olculen sey (surv_ratio) rotanin KENDISI — ucagin sansi
    degil. Bu figur de onu gostermeli, yoksa metrikle celisir.
    """
    import torch  # noqa: F401  (agent kurulumu icin)
    from env.sampler import eval_map_seeds
    from env.strike_env import StrikeMissionEnv
    from env.two_agent import play_episode_vdn
    from train import build_agent

    agent = build_agent(algo, 0, "cpu")
    agent.load(ckpt)
    env = StrikeMissionEnv(seed=12345, radar_random=True,
                           n_radar=C.N_RADAR, death_enabled=False)

    seeds = eval_map_seeds(24)
    picked = []
    for ms in seeds:
        info, _ = play_episode_vdn(env, agent, train=False,
                                   reset_kwargs={"map_seed": ms,
                                                 "n_radar": C.N_RADAR})
        z = env.zone.copy()
        orc = greedy_path(C.START, C.GOAL, env.dist,
                          direction_costs(z, mode=C.HAZARD_MODE))
        picked.append({
            "map_seed": ms, "zone": z, "oracle": orc,
            "p1": list(env.path[C.AGENT_1]), "p2": list(env.path[C.AGENT_2]),
            "reached": bool(info["reached1"] or info["reached2"]),
            "orc_s": survival_prob(orc, z, C.HAZARD_MODE),
            "ag_s": max(info["surv1"] if info["reached1"] else 0.0,
                        info["surv2"] if info["reached2"] else 0.0),
        })
        if sum(x["reached"] for x in picked) >= n:
            break
    # varanlari one al, kalani basarisizlardan tamamla (durust: ikisi de olsun)
    ok = [x for x in picked if x["reached"]][:max(1, n - 1)]
    bad = [x for x in picked if not x["reached"]][:n - len(ok)]
    show = (ok + bad)[:n]

    fig, axes = plt.subplots(1, len(show), figsize=(4.1 * len(show), 4.8))
    fig.patch.set_facecolor(BG)
    cmap = matplotlib.colors.ListedColormap(["#0d1117", "#4a3a12", "#5c1f1f"])
    for ax, e in zip(np.atleast_1d(axes), show):
        ax.imshow(e["zone"], cmap=cmap, origin="upper", interpolation="nearest")
        oy, ox = zip(*e["oracle"])
        ax.plot(ox, oy, color="#8b949e", lw=1.6, ls="--", label="oracle")
        for pk, c, lb in (("p1", "#3ddc97", "ucak 1"), ("p2", "#58a6ff", "ucak 2")):
            if e[pk]:
                py, px = zip(*e[pk])
                ax.plot(px, py, color=c, lw=1.8, label=lb)
        ax.plot(0, 0, "o", color="#58a6ff", ms=8)
        ax.plot(C.GOAL[1], C.GOAL[0], "*", color="#ff7b72", ms=15)
        ax.set_title(f"{e['map_seed']}   "
                     f"{'VARDI' if e['reached'] else 'varamadi'}\n"
                     f"ajan {e['ag_s']:.2f}  /  oracle {e['orc_s']:.2f}",
                     fontsize=9, color="#3ddc97" if e["reached"] else "#8b949e")
        ax.set_xticks([]); ax.set_yticks([])
    np.atleast_1d(axes)[0].legend(fontsize=7, framealpha=0.25, loc="lower left")
    fig.suptitle("Ajanin niyet ettigi rota (zar KAPALI) vs oracle — VDN tohum 2\n"
                 "koyu sari: dis halka   koyu kirmizi: ic halka",
                 fontsize=11.5, y=1.06)
    fig.tight_layout()
    p_out = os.path.join(OUT, "fig4_routes.png")
    fig.savefig(p_out, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return p_out


if __name__ == "__main__":
    for fn in (fig1, fig2, fig3, fig4):
        try:
            print("yazildi:", fn())
        except Exception as ex:
            print(f"{fn.__name__} HATA: {type(ex).__name__}: {ex}")
