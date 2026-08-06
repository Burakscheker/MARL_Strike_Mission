"""Rastgele radarli haritalarda fizibilite: harita hala gecilebilir mi?

SORU: 40 radar, +-110 dis / +-70 ic halka ile B'den H'ye gidilebilir mi?
Kaba hesap endise verici: 40 x 221^2 = 1.95M hucre, grid 1M hucre — yani
toplam radar alani gridin IKI KATI, ortusme kacinilmaz.

RISK KURALI (Burak, 2026-08-06) — SEVIYE TABANLI, TOPLAMA YOK:
"isterse ayni anda 4 radarin detection zone'unda olsun yine olme ihtimali
yuzde 20, toplamiyoruz". Bir hucrenin tehlikesi SADECE hangi halkada
oldugu ile belirlenir:

    zone(hucre) = max(radarlar uzerinden)   0=guvenli, 1=dis, 2=ic

Bu, build_zone_map()'in zaten yaptigi sey (np.maximum) — yani ortam bastan
dogru kurulmus. Ustuste binen radarlari AYRI tespit sistemleri sayip zarlari
carpma fikri degerlendirildi ve REDDEDILDI.

Bu betik ortamin KENDI risk makinesini (baselines.risk_oracle) kullanir,
paralel bir kopya degil — yani buradaki tavan sayilari egitimde kullanilan
maliyet fonksiyonuyla BIREBIR ayni tanimdan geliyor.

ADIM MALIYETI NEDEN SART: saf risk minimizasyonunda guvenli hucrelerde
dolasmak bedava oldugu icin oracle keyfi uzunlukta yol bulur ("hayatta kalma
%95, ama 9000 adim" — MAX_STEPS=2800'e sigmaz). O yuzden risk_w suprusu
yapilip MAX_STEPS'e SIGAN en iyi yol raporlanir.
"""
import argparse
import math
import time

import numpy as np

import config as C
from baselines.risk_oracle import (build_risk_distance_map, build_zone_map,
                                   direction_costs, exposure, greedy_path,
                                   survival_prob)

# risk_w = "bir birim olum olasiligi kac adima bedel". Odul fonksiyonundan
# turetilen varsayilan |R_RISK_COEF/R_STEP| = 1500 ortada; iki yana acilip
# MAX_STEPS'e sigan en iyisi seciliyor.
RISK_WEIGHTS = (150.0, 1500.0, 15000.0)


def sample_radars(n_radar, rng, n=C.GRID_N):
    """Uniform merkez, cakisma SERBEST (Burak: merkezler 5 hucre yakin olabilir)."""
    return [(int(rng.integers(0, n)), int(rng.integers(0, n)))
            for _ in range(n_radar)]


def best_oracle(z, mode, max_steps=C.MAX_STEPS):
    """MAX_STEPS'e sigan en yuksek hayatta kalma olasiligi + yol istatistigi."""
    best = None
    for w in RISK_WEIGHTS:
        d = build_risk_distance_map(zone=z, risk_w=w, mode=mode, max_iter=120)
        # cost MUTLAKA gecilmeli: greedy_path kenar maliyetsiz calisirsa
        # varsayilan SABIT haritanin maliyetini kullanir (bkz. greedy_path notu).
        p = greedy_path(C.START, C.GOAL, d, direction_costs(z, w, mode))
        if p[-1] != tuple(C.GOAL):
            continue
        steps = len(p) - 1
        s = survival_prob(p, z, mode)
        if steps <= max_steps and (best is None or s > best[1]):
            best = (steps, s, *exposure(p, z))
    return best


def scan(n_radar, mode, samples=8, seed=0,
         out_h=C.OUTER_HALF, in_h=C.INNER_HALF):
    rng = np.random.default_rng(seed)
    rows, t0 = [], time.perf_counter()
    for _ in range(samples):
        z = build_zone_map(C.GRID_N, sample_radars(n_radar, rng), out_h, in_h)
        b = best_oracle(z, mode)
        if b is None:                     # MAX_STEPS'e sigan yol yok
            rows.append(((z == 0).mean(), (z == 2).mean(), math.nan, 0.0))
            continue
        rows.append(((z == 0).mean(), (z == 2).mean(), b[0], b[1]))
    a = np.array(rows, dtype=float)
    return (a[:, 0].mean(), a[:, 1].mean(), np.nanmean(a[:, 2]),
            a[:, 3].mean(), float(np.median(a[:, 3])),
            (time.perf_counter() - t0) / samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=C.HAZARD_MODE,
                    choices=("per_entry", "per_step"),
                    help="per_entry = ortamin varsayilani (girise tek zar)")
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()

    print(f"grid {C.GRID_N}x{C.GRID_N}, optimal yol {2*(C.GRID_N-1)} adim, "
          f"limit {C.MAX_STEPS}")
    print(f"risk modeli: {args.mode}"
          + ("  (bolgeye GIRISTE tek zar: dis %20, ic %90)"
             if args.mode == "per_entry" else
             f"  (ADIM basi: dis {C.P_DEATH[1]:.6f}, ic {C.P_DEATH[2]:.6f})"))
    print("zone = radarlar uzerinden MAX — ustuste binme riski ARTIRMAZ\n")

    hdr = (f"{'radar':>6}{'halka':>11}{'guvenli':>9}{'ic halka':>10}"
           f"{'yol adim':>10}{'ORACLE hayatta':>16}{'medyan':>9}{'s/harita':>10}")
    print(hdr)
    for nr in (5, 10, 20, 30, 40):
        sf, inf_, ln, m, md, dt = scan(nr, args.mode, args.samples)
        print(f"{nr:>6}{'220/140':>11}{100*sf:>8.1f}%{100*inf_:>9.1f}%"
              f"{ln:>10.0f}{m:>16.4f}{md:>9.4f}{dt:>10.2f}")

    print()
    for oh, ih in ((60, 38), (80, 50), (110, 70)):
        sf, inf_, ln, m, md, dt = scan(40, args.mode, args.samples,
                                       out_h=oh, in_h=ih)
        print(f"{40:>6}{f'{2*oh}/{2*ih}':>11}{100*sf:>8.1f}%{100*inf_:>9.1f}%"
              f"{ln:>10.0f}{m:>16.4f}{md:>9.4f}{dt:>10.2f}")

    print("\nNOT: 'ORACLE hayatta' = MAX_STEPS'e sigan, riski minimize eden")
    print("yolun hayatta kalma olasiligi. Ajan bundan iyisini yapamaz — tavan bu.")
    print("Takim tavani (>=1 ucak varir) = 1 - (1 - p)^2.")


if __name__ == "__main__":
    main()
