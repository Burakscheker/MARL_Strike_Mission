"""Cok tohumlu IQL/VDN/QMIX karsilastirmasi — Strike_Mission.md §0.0.

NEDEN BU DOSYA VAR: tek tohumla "VDN ayristi" sonucu uretildi ve TEKRARLANMADI
(tohum 0'da VDN 5 - IQL 0, tohum 1'de IQL 3 - VDN 1). Ham basari orani 30
episode'da 1-2 sayimdan ibaret oldugu icin tohumdan tohuma yer degistiriyor.
Bu betik, sonucu tohum gurultusunden ayirmak icin uc seyi birlikte yapar:

  1. TOHUMLAR ARASI dagilim (ort +- std), tek tohum yerine.
  2. ESLESTIRILMIS karsilastirma: tum politikalar AYNI 50 haritada olculdugu
     icin harita-basina fark alinabilir. Bu, harita zorlugu varyansini
     tamamen yok eder ve ayni veriden cok daha guclu bir kiyas cikarir.
  3. Wilcoxon isaretli sira testi (scipy'siz, normal yaklasiklikla): "fark
     sifir" hipotezinin bu veriyle ne kadar bagdastigini soyler.

ANA METRIK surv_ratio, ham basari DEGIL: surv_ratio zar atilmadan, ajanin
NIYET ETTIGI rotadan hesaplanir (bkz. evaluate.py), yani gurultusuzdur ve
ayni episode sayisiyla cok daha fazla bilgi tasir.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import config as C

ALGOS = ("iql", "vdn", "qmix")
SEEDS = (0, 1, 2)
METRICS = (("surv_ratio", "surv_ratio (ANA)"), ("mission_prob", "mission_prob"),
           ("route_reached", "rota hedefe variyor"),
           ("team_success", "takim basarisi (zar)"))


def load(seed: int, algo: str):
    p = os.path.join(C.RUNS_DIR, f"ev{seed}_{algo}_maps.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def col(rows, key):
    return np.array([r.get(key, np.nan) for r in rows], dtype=float)


def wilcoxon_p(d: np.ndarray) -> float:
    """Isaretli sira testi, normal yaklasiklik (n>=10 icin yeterli).

    scipy bagimliligi eklemiyoruz — bu projenin tek bagimliligi numpy/torch.
    Sifir farklar atilir (standart Wilcoxon kurali).
    """
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n < 6:
        return float("nan")
    rank = np.argsort(np.argsort(np.abs(d))) + 1.0
    w = rank[d > 0].sum()
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w - mu) / sd
    # iki yonlu p
    return math.erfc(abs(z) / math.sqrt(2))


def main():
    data = {a: {s: load(s, a) for s in SEEDS} for a in ALGOS}
    have = [(a, s) for a in ALGOS for s in SEEDS if data[a][s]]
    print(f"bulunan kosu: {len(have)}/9")
    missing = [(a, s) for a in ALGOS for s in SEEDS if not data[a][s]]
    if missing:
        print(f"EKSIK: {missing}")
    print()

    # --- referans (harita seti ortak oldugu icin hepsinde ayni)
    ref = next(data[a][s] for a, s in have)
    print(f"{len(ref)} held-out harita, {C.N_RADAR} radar, "
          f"halka {2*C.OUTER_HALF+1}/{2*C.INNER_HALF+1}")
    orc = col(ref, "oracle_surv")
    print(f"oracle TAVAN: ort {orc.mean():.4f}  medyan {np.median(orc):.4f}  "
          f"takim tavani %{100*(1-(1-orc)**2).mean():.1f}\n")

    # --- tohum tohum + ortalama
    for key, label in METRICS:
        print(f"--- {label} ---")
        print(f"{'algo':<6}" + "".join(f"{'tohum '+str(s):>12}" for s in SEEDS)
              + f"{'ORT':>12}{'std':>9}")
        for a in ALGOS:
            vals = []
            for s in SEEDS:
                rows = data[a][s]
                if rows is None:
                    vals.append(np.nan); continue
                v = col(rows, key)
                vals.append(np.nanmean(v))
            vals = np.array(vals, dtype=float)
            print(f"{a:<6}" + "".join(f"{v:>12.4f}" for v in vals)
                  + f"{np.nanmean(vals):>12.4f}{np.nanstd(vals):>9.4f}")
        print()

    # --- ESLESTIRILMIS karsilastirma (ayni haritalar!)
    print("=== ESLESTIRILMIS surv_ratio (harita basina, tohumlar ortalanmis) ===")
    per = {}
    for a in ALGOS:
        stack = [col(data[a][s], "surv_ratio") for s in SEEDS if data[a][s]]
        per[a] = np.nanmean(np.vstack(stack), axis=0) if stack else None

    for x, y in (("vdn", "iql"), ("qmix", "iql"), ("vdn", "qmix")):
        if per[x] is None or per[y] is None:
            continue
        d = per[x] - per[y]
        fin = np.isfinite(d)
        wins, losses = int((d[fin] > 0).sum()), int((d[fin] < 0).sum())
        p = wilcoxon_p(d[fin])
        verdict = ("FARK VAR" if p < 0.05 else
                   "fark gosterilemedi" if p == p else "veri az")
        print(f"{x.upper():>5} vs {y.upper():<5}  ort fark {np.nanmean(d):+.4f}   "
              f"{x} kazandi {wins}/{wins+losses} haritada   p={p:.3f}  -> {verdict}")

    print("\nNOT: p, Wilcoxon isaretli sira testi (iki yonlu). p<0.05 = 'fark")
    print("sifir' hipotezi bu veriyle bagdasmiyor. Eslestirilmis oldugu icin")
    print("harita zorlugu varyansi dislanmis durumda.")


if __name__ == "__main__":
    main()
