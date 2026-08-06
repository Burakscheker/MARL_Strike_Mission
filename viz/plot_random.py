"""Rastgele harita gorselleri — Strike_Mission.md §11.

Uc cikti:
  runs/viz/random_maps.png       held-out haritalar + oracle / naif yollar
  runs/viz/baseline_survival.png politikalarin harita basina hayatta kalmasi
  runs/viz/agent_paths.png       (demo JSON varsa) egitilmis ajanin yollari

DIKKAT — harita artik episode'a OZGU. Yol cizimi radar setini episode'un
kendi kaydindan okumali; sabit bir haritanin uzerine cizmek YANLIS resim
uretir (viz/plot_map.py bu yuzden olu kod: 51x51 + sabit 3 radar gomulu).
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import config as C
from baselines.risk_oracle import (RISK_W, build_risk_distance_map,
                                   build_zone_map, direction_costs,
                                   greedy_path, survival_prob)
from env.sampler import eval_map_seeds, sample_radars

OUT_DIR = os.path.join(C.RUNS_DIR, "viz")
BG = "#0d1117"
# 0 guvenli / 1 dis halka / 2 ic halka
ZONE_CMAP = ListedColormap(["#0d1117", "#5a3d0a", "#7d1f1f"])


def build(map_seed: int, n_radar: int = C.N_RADAR):
    rng = np.random.default_rng(map_seed)
    radars = sample_radars(n_radar, rng, C.GRID_N)
    z = build_zone_map(C.GRID_N, radars, C.OUTER_HALF, C.INNER_HALF)
    d = build_risk_distance_map(goal=C.GOAL, zone=z, mode=C.HAZARD_MODE,
                                max_iter=C.MAP_MAX_ITER)
    cost = direction_costs(z, RISK_W, C.HAZARD_MODE)
    return radars, z, d, cost


def staircase():
    p = [(0, 0)]
    n = C.GRID_N
    while p[-1] != (n - 1, n - 1):
        r, c = p[-1]
        if c < n - 1:
            p.append((r, c + 1))
        if p[-1] != (n - 1, n - 1) and p[-1][0] < n - 1:
            r, c = p[-1]
            p.append((r + 1, c))
    return p


def draw_map(ax, z, title):
    # imshow ham zone dizisini cizer — ustuste binen radarlarin nasil TEK bir
    # birlesik kitleye dondugu boyle gorunur (dikdortgen dikdortgen cizmek bunu
    # gizlerdi, bkz. §11.3 "halkalar birlesiyor").
    ax.imshow(z, cmap=ZONE_CMAP, vmin=0, vmax=2, interpolation="nearest",
              origin="upper")
    ax.plot(0, 0, "o", color="#58a6ff", ms=9, zorder=6)
    ax.plot(C.GRID_N - 1, C.GRID_N - 1, "*", color="#ff7b72", ms=14, zorder=6)
    ax.set_title(title, fontsize=9, color="#c9d1d9")
    ax.set_xticks([]); ax.set_yticks([])


def path_xy(path):
    a = np.asarray(path)
    return a[:, 1], a[:, 0]          # imshow: x=col, y=row


def fig_random_maps(n_show=4):
    seeds = eval_map_seeds(n_show)
    fig, axes = plt.subplots(1, n_show, figsize=(4.2 * n_show, 4.8),
                             facecolor=BG)
    for ax, ms in zip(np.atleast_1d(axes), seeds):
        radars, z, d, cost = build(ms)
        orc = greedy_path(C.START, C.GOAL, d, cost)
        st = staircase()
        s_o = survival_prob(orc, z, C.HAZARD_MODE)
        s_s = survival_prob(st, z, C.HAZARD_MODE)
        draw_map(ax, z, f"tohum {ms}\nguvenli %{100*(z==0).mean():.0f}  "
                        f"ic halka %{100*(z==2).mean():.0f}")
        x, y = path_xy(st)
        ax.plot(x, y, "-", color="#ff5555", lw=1.6, alpha=0.9,
                label=f"merdiven  {s_s:.3f}")
        x, y = path_xy(orc)
        ax.plot(x, y, "-", color="#58a6ff", lw=2.0,
                label=f"oracle  {s_o:.3f}")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.35,
                  labelcolor="#c9d1d9")
    fig.suptitle("Held-out rastgele haritalar (40 radar) — oracle yolu vs naif "
                 "merdiven\nkoyu kirmizi = ic halka (%90 olum/giris), "
                 "kahve = dis halka (%20)", color="#c9d1d9", fontsize=12, y=1.06)
    # rect: suptitle iki satirli, tight_layout'a yer birakmasini soylemezsek
    # alt grafik basliklariyla CAKISIYOR (ilk cikti boyle geldi).
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = os.path.join(OUT_DIR, "random_maps.png")
    fig.savefig(out, dpi=110, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_baseline_survival(json_path):
    with open(json_path, encoding="utf-8") as f:
        rows = json.load(f)
    orc = np.array([r["oracle_surv"] for r in rows])
    st = np.array([r["stair_surv"] for r in rows])
    rm = np.array([r["rndmono_surv"] for r in rows])
    idx = np.argsort(-orc)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=BG)
    for ax in (a1, a2):
        ax.set_facecolor(BG)
        ax.tick_params(colors="#8b949e")
        for s in ax.spines.values():
            s.set_color("#30363d")

    a1.plot(orc[idx], "-", color="#58a6ff", lw=2, label="Dijkstra oracle (TAVAN)")
    a1.plot(st[idx], ".", color="#ff5555", ms=4, label="merdiven (naif)")
    a1.plot(rm[idx], ".", color="#d29922", ms=4, label="rastgele monoton")
    a1.set_yscale("symlog", linthresh=1e-3)
    a1.set_xlabel("held-out harita (oracle'a gore sirali)", color="#c9d1d9")
    a1.set_ylabel("analitik hayatta kalma", color="#c9d1d9")
    a1.set_title(f"Harita basina tavan ve tabanlar (n={len(rows)})",
                 color="#c9d1d9")
    a1.legend(fontsize=8, framealpha=0.3, labelcolor="#c9d1d9")
    a1.grid(alpha=0.15)

    names = ["oracle\n(TAVAN)", "merdiven", "rastgele\nmonoton"]
    vals = [orc.mean(), st.mean(), rm.mean()]
    meds = [np.median(orc), np.median(st), np.median(rm)]
    xs = np.arange(3)
    a2.bar(xs - 0.18, vals, 0.36, color="#58a6ff", label="ortalama")
    a2.bar(xs + 0.18, meds, 0.36, color="#3ddc97", label="medyan")
    for i, (v, m) in enumerate(zip(vals, meds)):
        a2.text(i - 0.18, v + 0.012, f"{v:.4f}", ha="center", fontsize=8,
                color="#c9d1d9")
        a2.text(i + 0.18, m + 0.012, f"{m:.4f}", ha="center", fontsize=8,
                color="#c9d1d9")
    a2.set_xticks(xs); a2.set_xticklabels(names, color="#c9d1d9")
    a2.set_ylabel("analitik hayatta kalma", color="#c9d1d9")
    a2.set_title("AJAN BU IKISI ARASINDA BIR YERDE OLMALI\n"
                 "oracle'a yakinsa iyi, merdivene yakinsa ogrenmemis",
                 color="#c9d1d9", fontsize=10)
    a2.legend(fontsize=8, framealpha=0.3, labelcolor="#c9d1d9")
    a2.grid(alpha=0.15, axis="y")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "baseline_survival.png")
    fig.savefig(out, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_agent_paths(demo_json, n_show=4):
    """Egitilmis ajanin yollari — her episode KENDI haritasinin uzerine."""
    with open(demo_json, encoding="utf-8") as f:
        eps = json.load(f)
    eps = [e for e in eps if e.get("radars")][:n_show]
    if not eps:
        return None
    fig, axes = plt.subplots(1, len(eps), figsize=(4.2 * len(eps), 4.8),
                             facecolor=BG)
    for ax, e in zip(np.atleast_1d(axes), eps):
        radars = [tuple(r) for r in e["radars"]]
        z = build_zone_map(C.GRID_N, radars, C.OUTER_HALF, C.INNER_HALF)
        draw_map(ax, z, f"ep{e['episode']}  takim="
                        f"{'EVET' if e['team_success'] else 'hayir'}  "
                        f"adim={e['steps']}")
        for key, col, lab in (("path1", "#58a6ff", "A1"), ("path2", "#3ddc97", "A2")):
            p = e.get(key) or []
            if len(p) > 1:
                x, y = path_xy(p)
                ax.plot(x, y, "-", color=col, lw=1.6,
                        label=f"{lab} surv={e[f'surv{key[-1]}']:.3f}")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.35,
                  labelcolor="#c9d1d9")
    fig.suptitle("Egitilmis ajanin held-out haritalardaki yollari",
                 color="#c9d1d9", fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "agent_paths.png")
    fig.savefig(out, dpi=110, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines-json",
                    default=os.path.join(C.RUNS_DIR, "baselines_maps.json"))
    ap.add_argument("--demo-json", default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("yazildi:", fig_random_maps())
    if os.path.exists(args.baselines_json):
        print("yazildi:", fig_baseline_survival(args.baselines_json))
    else:
        print(f"atlandi: {args.baselines_json} yok "
              f"(once `python -m eval.evaluate --tag baselines`)")
    if args.demo_json and os.path.exists(args.demo_json):
        print("yazildi:", fig_agent_paths(args.demo_json))


if __name__ == "__main__":
    main()
