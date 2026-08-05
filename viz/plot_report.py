"""Egitim egrileri + gosterim episode'larinin harita cizimi.

Kosum:
    python -m viz.plot_report --tag vdn_fixed
    python -m viz.plot_report --tag vdn_fixed --compare vdn_resume

Uretilenler (runs/viz/ altina):
    {tag}_egitim_egrisi.png   episode basina takim basarisi + A1/A2 olme orani
    {tag}_yollar.png          10 deterministik episode'un ciziilmis yollari
    {a}_vs_{b}_karsilastirma.png   iki kosunun basari egrisi ust uste
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import config as C

VIZ_DIR = os.path.join(C.RUNS_DIR, "viz")

BG = "#0d1117"
FG = "#c9d1d9"
COL_A1 = "#58a6ff"
COL_A2 = "#ffa657"
COL_OK = "#3ddc97"
COL_BAD = "#ff5555"
COL_OUTER = "#ffa500"
COL_INNER = "#3ddc97"


def _style():
    plt.style.use("dark_background")
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG,
                         "savefig.facecolor": BG, "text.color": FG,
                         "axes.labelcolor": FG, "xtick.color": "#8b949e",
                         "ytick.color": "#8b949e", "axes.edgecolor": "#30363d",
                         "grid.color": "#21262d"})


def _rolling(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < 2:
        return x.astype(float)
    w = min(w, len(x))
    kernel = np.ones(w) / w
    # 'same' yerine kenarlari duzeltilmis ortalama: bastaki/sondaki pencere
    # eksik oldugu icin bolen de kucultuluyor, yoksa egri ucta 0'a dogru
    # yanli bicimde suruklenir.
    num = np.convolve(x.astype(float), kernel, mode="same")
    den = np.convolve(np.ones_like(x, dtype=float), kernel, mode="same")
    return num / den


def load_episodes(tag: str) -> dict[str, np.ndarray]:
    p = os.path.join(C.RUNS_DIR, f"{tag}_episodes.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    cols: dict[str, list] = {}
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return {k: np.asarray(v) for k, v in cols.items()}


# ------------------------------------------------------------- egitim egrisi

def plot_training_curves(tag: str, window: int | None = None) -> str:
    _style()
    d = load_episodes(tag)
    ep = d["episode"]
    w = window or max(5, len(ep) // 10)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    # REFERANS CIZGILERI — bunlar olmadan egri yorumlanamaz. Bu haritada
    # "hep sag, kenara dayaninca hep asagi" diyen SABIT politika zaten %100
    # takim basarisi ve sifir olum aliyor (bkz. baselines/policies.py).
    # Yani egrinin yukselmesi tek basina "ogrendi" demek DEGIL; tavan trivial
    # olarak ulasilabilir durumda. Alt referans rastgele monoton politika.
    series = [
        ("Takim basari orani (>=1 ucak hedefe vardi)", d["team_success"], COL_OK,
         1.00, 0.20),
        ("Ucak 1 olme orani", d["dead1"], COL_A1, 0.00, 0.90),
        ("Ucak 2 olme orani", d["dead2"], COL_A2, 0.00, 0.90),
    ]
    for ax, (title, y, col, ref_const, ref_rand) in zip(axes, series):
        ax.axhline(ref_const, color="#3ddc97", ls=":", lw=1.6,
                   label="SABIT politika (= oracle, trivial tavan)")
        ax.axhline(ref_rand, color="#ff5555", ls=":", lw=1.4,
                   label="rastgele monoton (taban)")
        ax.scatter(ep, y, s=6, color=col, alpha=0.18, linewidths=0,
                   label="episode basina (ham)")
        ax.plot(ep, _rolling(y, w), color=col, lw=2.2,
                label=f"hareketli ortalama (pencere {w})")
        ax.set_ylim(-0.05, 1.08)
        ax.set_ylabel("oran")
        ax.set_title(title, loc="left", fontsize=11, color=FG)
        ax.grid(alpha=0.35)
        ax.legend(loc="center right", fontsize=8, framealpha=0.25)

    # epsilon'u en uste ikinci eksen olarak koy — "ne zaman kesif bitti"
    # sorusu egrileri yorumlarken sart.
    ax2 = axes[0].twinx()
    ax2.plot(ep, d["eps"], color="#8b949e", ls="--", lw=1.2)
    ax2.set_ylabel("epsilon", color="#8b949e")
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(colors="#8b949e")

    axes[-1].set_xlabel("episode")
    n_ok = int(d["team_success"].sum())
    fig.suptitle(f"{tag} — {len(ep)} episode egitim  "
                 f"(takim basarisi {n_ok}/{len(ep)} = %{100*n_ok/len(ep):.1f})",
                 fontsize=13, color=FG)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(VIZ_DIR, exist_ok=True)
    out = os.path.join(VIZ_DIR, f"{tag}_egitim_egrisi.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------- yol cizimi

def _draw_map(ax):
    """Radar halkalarini ve B/H'yi ciz."""
    for r0, c0 in C.RADARS:
        for half, col in ((C.OUTER_HALF, COL_OUTER), (C.INNER_HALF, COL_INNER)):
            ax.add_patch(Rectangle((c0 - half, r0 - half), 2 * half, 2 * half,
                                   facecolor=col, alpha=0.10,
                                   edgecolor=col, lw=1.2))
    ax.plot(C.START[1], C.START[0], "o", color=COL_A1, ms=7, zorder=6)
    ax.plot(C.GOAL[1], C.GOAL[0], "*", color=COL_BAD, ms=13, zorder=6)
    ax.set_xlim(-20, C.GRID_N + 20)
    ax.set_ylim(C.GRID_N + 20, -20)          # row asagi dogru artar
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def plot_demo_paths(tag: str, cols: int = 5) -> str:
    _style()
    p = os.path.join(C.RUNS_DIR, f"{tag}_demo_episodes.json")
    with open(p, encoding="utf-8") as f:
        demos = json.load(f)

    rows = (len(demos) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, d in zip(axes, demos):
        _draw_map(ax)
        for key, col, reached, alive in (
            ("path1", COL_A1, d["reached1"], d["alive1"]),
            ("path2", COL_A2, d["reached2"], d["alive2"]),
        ):
            path = np.asarray(d[key])
            if len(path) < 2:
                continue
            ax.plot(path[:, 1], path[:, 0], "-", color=col, lw=1.6, alpha=0.95)
            if not alive:      # dusuruldugu nokta
                ax.plot(path[-1, 1], path[-1, 0], "x", color=COL_BAD,
                        ms=11, mew=2.5, zorder=7)
            elif not reached:  # timeout — nerede kaldiysa
                ax.plot(path[-1, 1], path[-1, 0], "s", color="#8b949e",
                        ms=6, zorder=7)
        st = "OK" if d["team_success"] else "BASARISIZ"
        stc = COL_OK if d["team_success"] else COL_BAD
        a1 = "vardi" if d["reached1"] else ("OLDU" if not d["alive1"] else "timeout")
        a2 = "vardi" if d["reached2"] else ("OLDU" if not d["alive2"] else "timeout")
        ax.set_title(f"ep{d['episode']}  {st}\nA1 {a1} / A2 {a2}\n"
                     f"{d['steps']} adim, maruziyet "
                     f"{d['outer1']+d['outer2']}/{d['inner1']+d['inner2']}",
                     fontsize=8, color=stc)

    for ax in axes[len(demos):]:
        ax.axis("off")

    handles = [
        Line2D([], [], color=COL_A1, lw=2, label="Ucak 1"),
        Line2D([], [], color=COL_A2, lw=2, label="Ucak 2"),
        Line2D([], [], color=COL_BAD, marker="x", ls="", ms=9, mew=2,
               label="dusuruldu"),
        Line2D([], [], color="#8b949e", marker="s", ls="", ms=6, label="timeout"),
        Line2D([], [], color=COL_OUTER, lw=2, label="dis halka (%20 giriste)"),
        Line2D([], [], color=COL_INNER, lw=2, label="ic halka (%90 giriste)"),
        Line2D([], [], color=COL_BAD, marker="*", ls="", ms=12, label="H hedef"),
    ]
    n_ok = sum(d["team_success"] for d in demos)
    fig.suptitle(f"{tag} — egitim sonrasi {len(demos)} deterministik episode "
                 f"(eps=0)   takim basarisi {n_ok}/{len(demos)}",
                 fontsize=13, color=FG)
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8,
               framealpha=0.2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    os.makedirs(VIZ_DIR, exist_ok=True)
    out = os.path.join(VIZ_DIR, f"{tag}_yollar.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------ karsilastirma

def plot_comparison(tags: list[str], window: int | None = None) -> str:
    _style()
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    palette = [COL_OK, COL_A2, COL_A1, "#d2a8ff"]
    for i, tag in enumerate(tags):
        d = load_episodes(tag)
        ep = d["episode"]
        w = window or max(5, len(ep) // 10)
        col = palette[i % len(palette)]
        for ax, key in zip(axes, ("team_success", "dead1", "dead2")):
            ax.plot(ep, _rolling(d[key], w), color=col, lw=2.2, label=tag)
    for ax, title in zip(axes, ("Takim basari orani", "Ucak 1 olme orani",
                                "Ucak 2 olme orani")):
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, loc="left", fontsize=11, color=FG)
        ax.grid(alpha=0.35)
        ax.legend(fontsize=9, framealpha=0.25)
    axes[-1].set_xlabel("episode")
    fig.suptitle(" vs ".join(tags), fontsize=13, color=FG)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(VIZ_DIR, exist_ok=True)
    out = os.path.join(VIZ_DIR, f"{'_vs_'.join(tags)}_karsilastirma.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--window", type=int, default=None)
    args = ap.parse_args()

    print(plot_training_curves(args.tag, args.window))
    try:
        print(plot_demo_paths(args.tag))
    except FileNotFoundError:
        print("  (demo JSON yok — egitim henuz bitmemis olabilir)")
    if args.compare:
        print(plot_comparison([args.tag, *args.compare], args.window))


if __name__ == "__main__":
    main()
