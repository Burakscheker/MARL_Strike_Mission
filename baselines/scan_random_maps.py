"""40 rastgele radarla harita fizibilitesi.

SORU: 40 radar, +-110 dis / +-70 ic halka ile harita hala gecilebilir mi?
Kaba hesap endise verici: 40 x 221^2 = 1.95M hucre, grid 1M hucre — yani
toplam radar alani gridin IKI KATI. Ortusme cok olacak.

PER-RADAR GIRIS MUHASEBESI (ustuste binme oldugu icin sart):
Her radar AYRI bir tespit sistemi. Bir radarin dis halkasina girmek o radar
icin %20'lik bir zar demek; baska bir radarin alanindayken UCUNCU bir radarin
alanina girmek YENI bir zardir. Yani "bolge seviyesi arttiginda zar at"
kurali 40 ustuste binen radarda yanlis olur — radar SAYISI onemli.

Kenar maliyeti bunun icin dogrudan hesaplanabilir: bir dikdortgene YUKARIDAN
girmek, tam olarak o dikdortgenin UST SATIRINA basmaktir. Yani her radar
icin 4 kenar (ust satir / alt satir / sol sutun / sag sutun) +1 alir.
Ustuste binme sorun degil, cunku her radar bagimsiz sayiliyor. O(radar) —
harita basina bedava.
"""
import math
import time

import numpy as np

N = 1000
OUT_H, IN_H = 110, 70
P_OUT, P_IN = 0.20, 0.90
LN_OUT = -math.log(1 - P_OUT)      # 0.2231
LN_IN = -math.log(1 - P_IN)        # 2.3026
START, GOAL = (0, 0), (N - 1, N - 1)


def zone_and_entry(radars, out_h=OUT_H, in_h=IN_H, n=N):
    """zone (0/1/2) + 4 yon icin 'bu hamleyle kac YENI radar alanina girdim'."""
    z = np.zeros((n, n), dtype=np.uint8)
    # enter[d][cell] = d yonunde hareket edip BU hucreye gelince tetiklenen
    # risk (log cinsinden). d: hucreye hangi yonden GIRILDIGI.
    ent = {k: np.zeros((n, n)) for k in ("from_up", "from_down", "from_left", "from_right")}

    for r0, c0 in radars:
        for half, val, lg in ((out_h, 1, LN_OUT), (in_h, 2, LN_IN)):
            r_lo, r_hi = r0 - half, r0 + half
            c_lo, c_hi = c0 - half, c0 + half
            rl, rh = max(0, r_lo), min(n - 1, r_hi)
            cl, ch = max(0, c_lo), min(n - 1, c_hi)
            if rl > rh or cl > ch:
                continue
            blk = z[rl:rh + 1, cl:ch + 1]
            np.maximum(blk, val, out=blk)
            # dikdortgene giris kenarlari
            if r_lo >= 0:
                ent["from_up"][r_lo, cl:ch + 1] += lg
            if r_hi < n:
                ent["from_down"][r_hi, cl:ch + 1] += lg
            if c_lo >= 0:
                ent["from_left"][rl:rh + 1, c_lo] += lg
            if c_hi < n:
                ent["from_right"][rl:rh + 1, c_hi] += lg
    return z, ent


def dijkstra_logrisk(ent, step_w=0.0, n=N, max_iter=60):
    """Hedefe giden en dusuk TOPLAM LOG-RISK yolu (fast sweeping).

    step_w=0 -> saf risk minimizasyonu (uzunluk umursanmaz).
    """
    # cost(u->v): v'ye hangi yonden girildigine bagli
    # dist[u] = min_v cost(u->v) + dist[v]
    c_up = step_w + ent["from_down"]      # (r,c)->(r-1,c): (r-1,c)'ye ASAGIDAN girilir
    c_down = step_w + ent["from_up"]
    c_left = step_w + ent["from_right"]
    c_right = step_w + ent["from_left"]

    d = np.full((n, n), np.inf)
    d[GOAL] = 0.0
    for _ in range(max_iter):
        before = d.copy()
        for r in range(1, n):
            np.minimum(d[r], d[r - 1] + c_up[r - 1], out=d[r])
        for r in range(n - 2, -1, -1):
            np.minimum(d[r], d[r + 1] + c_down[r + 1], out=d[r])
        for c in range(1, n):
            np.minimum(d[:, c], d[:, c - 1] + c_left[:, c - 1], out=d[:, c])
        for c in range(n - 2, -1, -1):
            np.minimum(d[:, c], d[:, c + 1] + c_right[:, c + 1], out=d[:, c])
        if not np.any(d < before - 1e-9):
            break
    return d


def scan(n_radar, samples=20, seed=0, out_h=OUT_H, in_h=IN_H):
    rng = np.random.default_rng(seed)
    surv, safe_frac, inner_frac, t0 = [], [], [], time.perf_counter()
    for _ in range(samples):
        radars = [(int(rng.integers(0, N)), int(rng.integers(0, N)))
                  for _ in range(n_radar)]
        z, ent = zone_and_entry(radars, out_h, in_h)
        safe_frac.append((z == 0).mean())
        inner_frac.append((z == 2).mean())
        d = dijkstra_logrisk(ent)
        surv.append(math.exp(-d[START]))
    dt = (time.perf_counter() - t0) / samples
    return (np.mean(surv), np.median(surv), np.mean(safe_frac),
            np.mean(inner_frac), dt)


print(f"{'radar':>6}{'halka':>12}{'guvenli %':>11}{'ic halka %':>12}"
      f"{'ORACLE hayatta':>16}{'medyan':>9}{'s/harita':>10}")
for nr in (5, 10, 20, 30, 40):
    m, md, sf, inf_, dt = scan(nr, samples=12)
    print(f"{nr:>6}{'110/70':>12}{100*sf:>10.1f}%{100*inf_:>11.1f}%"
          f"{m:>16.4f}{md:>9.4f}{dt:>10.2f}")

print()
for half in ((60, 38), (80, 50), (110, 70)):
    m, md, sf, inf_, dt = scan(40, samples=12, out_h=half[0], in_h=half[1])
    print(f"{40:>6}{f'{half[0]}/{half[1]}':>12}{100*sf:>10.1f}%{100*inf_:>11.1f}%"
          f"{m:>16.4f}{md:>9.4f}{dt:>10.2f}")

print("\nNOT: 'ORACLE hayatta' = riski MINIMIZE eden yolun hayatta kalma")
print("olasiligi. Ajan bundan daha iyisini yapamaz — tavan bu.")
