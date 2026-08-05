"""Harita zorluk analizi — Strike_Mission.md §0'in KANITI.

Kosum: python -m baselines.map_check

BULGU (bu dosyanin varolus sebebi): Burak'in verdigi SABIT haritada
"hep SAG, kenara dayaninca hep ASAGI" seklindeki TRIVIAL sabit politika
Dijkstra oracle'iyla BIREBIR ayni sonucu veriyor — 1998 adim (optimal),
sifir radar maruziyeti, %100 takim basarisi. Yani ogrenilecek hicbir sey yok.

KOK NEDEN: B sol-ust kose, H sag-alt kose ve uc radar da IC BOLGEDE. Gridin
KENARI bastan sona radarsiz bir otoyol, ve kose-kose gidildigi icin kenardan
dolasmanin uzunluk bedeli SIFIR. Guvenli yol = en kisa yol = trivial politika.

Bu modul o bulguyu olcer ve "haritayi nasil anlamli yapariz" sorusunu
sayilarla cevaplar: rastgele radar konumlarinda trivial cozumun ne siklikta
calistigini tarar.
"""
from __future__ import annotations

import numpy as np

import config as C
from baselines.risk_oracle import (build_zone_map, exposure, greedy_path,
                                   risk_distance_map, survival_prob, zone_map)


# ------------------------------------------------- monoton yol erisilebilirlik

def has_safe_monotone_path(safe: np.ndarray) -> bool:
    """(0,0) -> (n-1,n-1) arasi, SADECE sag/asagi adimlarla, hic tehlikeli
    hucreye girmeyen bir yol var mi?

    Boyle bir yol varsa optimal uzunluk (2(n-1)) ile SIFIR risk ayni anda
    saglanabiliyor demektir — yani problemde risk/uzunluk odunlesmesi YOK.

    Satir satir DP. Satir icindeki `reach[c] = safe[c] and (up[c] or
    reach[c-1])` bagimliligi vektorize ediliyor: bir hucreye soldan erisim,
    ARADA hic engel olmadan once gelen bir "yukaridan giris" varsa mumkundur.
    """
    n = safe.shape[0]
    idx = np.arange(n)
    reach = np.zeros(n, dtype=bool)
    up = np.zeros(n, dtype=bool)
    up[0] = safe[0, 0]                       # ilk satirin "yukaridan girisi"
    for r in range(n):
        row = safe[r]
        if r > 0:
            up = reach & safe[r]             # bir onceki satirdan asagi inis
        seed = np.where(up & row, idx, -1)
        last_seed = np.maximum.accumulate(seed)
        last_blocked = np.maximum.accumulate(np.where(~row, idx, -1))
        reach = row & (last_seed > last_blocked) & (last_seed >= 0)
        if not reach.any():
            return False
    return bool(reach[n - 1])


def constant_policy_path(safe: np.ndarray, order=("R", "D")) -> tuple[bool, int]:
    """'hep SAG sonra hep ASAGI' (veya tersi) yolunu izle; guvenli mi, kac adim."""
    n = safe.shape[0]
    if order[0] == "R":
        cells = [(0, c) for c in range(n)] + [(r, n - 1) for r in range(1, n)]
    else:
        cells = [(r, 0) for r in range(n)] + [(n - 1, c) for c in range(1, n)]
    ok = all(safe[r, c] for r, c in cells)
    return ok, len(cells) - 1


# --------------------------------------------------------------- sabit harita

def fixed_map_report():
    print("=" * 74)
    print("1) SABIT HARITA (Burak'in verdigi 3 radar)")
    print("=" * 74)
    z = zone_map()
    n = C.GRID_N
    n_out, n_in = int((z == 1).sum()), int((z == 2).sum())
    print(f"grid {n}x{n}  |  guvenli={z.size - n_out - n_in}  dis={n_out}  "
          f"ic={n_in}  (tehlikeli %{100*(n_out+n_in)/z.size:.1f})")
    print(f"radarlar (row,col): {C.RADARS}")
    print(f"p_death/adim: dis {C.P_DEATH[1]:.6f}  ic {C.P_DEATH[2]:.6f}")

    safe = (z == 0)
    d = risk_distance_map()
    orc = greedy_path(C.START, C.GOAL, d)
    opt = 2 * (n - 1)

    print(f"\n{'yol':<30}{'uzunluk':>9}{'optimal mi':>12}{'hayatta kalma':>15}"
          f"{'dis':>7}{'ic':>6}")
    rows = []
    for name, cells in (
        ("Dijkstra oracle", orc),
        ("SABIT politika (sag->asagi)",
         [(0, c) for c in range(n)] + [(r, n - 1) for r in range(1, n)]),
        ("SABIT politika (asagi->sag)",
         [(r, 0) for r in range(n)] + [(n - 1, c) for c in range(1, n)]),
    ):
        s = survival_prob(cells, z)
        o, i = exposure(cells, z)
        L = len(cells) - 1
        rows.append((name, L, L == opt, s, o, i))
        print(f"{name:<30}{L:>9}{'EVET' if L == opt else 'hayir':>12}"
              f"{s:>14.4f} {o:>7}{i:>6}")

    trivial = rows[1][2] and rows[1][3] > 0.999
    print(f"\n>> TRIVIAL Mi: {'EVET' if trivial else 'hayir'} — "
          f"sabit politika hem optimal uzunlukta hem sifir riskli."
          if trivial else "\n>> trivial degil")
    print(f">> guvenli monoton yol var mi: {has_safe_monotone_path(safe)}")


# ------------------------------------------------------- rastgele radar tarama

def random_radar_scan(samples=200, n_radar=C.N_RADAR, seed=0,
                      margin=0, outer_half=C.OUTER_HALF,
                      inner_half=C.INNER_HALF):
    """Rastgele radar konumlarinda haritanin ne kadar 'anlamli' oldugunu tara.

    Uc kategori:
      TRIVIAL : kenar (kose) yolu guvenli -> sabit politika kazanir, ogrenme yok
      KOLAY   : kenar yolu kapali AMA sifir-riskli optimal (monoton) yol var
                -> gercek yol bulma gerekiyor, ama risk/uzunluk odunlesmesi yok
      ZOR     : sifir-riskli optimal yol YOK -> ajan risk ile uzunluk arasinda
                gercek bir karar vermek zorunda   <<< ASIL ILGINC OLAN
    """
    rng = np.random.default_rng(seed)
    n = C.GRID_N
    lo, hi = margin, n - 1 - margin
    counts = {"trivial": 0, "kolay": 0, "zor": 0}
    for _ in range(samples):
        radars = [(int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1)))
                  for _ in range(n_radar)]
        z = build_zone_map(n, radars, outer_half, inner_half)
        # B ve H'nin kendisi tehlikeliyse o harita gecersiz sayilmaz —
        # ucak zaten oradan kalkiyor; sadece riskli baslamis olur.
        safe = (z == 0)
        edge_ok = (constant_policy_path(safe, ("R", "D"))[0]
                   or constant_policy_path(safe, ("D", "R"))[0])
        if edge_ok:
            counts["trivial"] += 1
        elif has_safe_monotone_path(safe):
            counts["kolay"] += 1
        else:
            counts["zor"] += 1
    return counts


def scan_report(samples=200):
    print()
    print("=" * 74)
    print(f"2) RASTGELE RADAR TARAMASI ({samples} harita)")
    print("=" * 74)
    print("kenar-marj = radar merkezinin gridin kenarina ne kadar")
    print("yaklasabildigi. margin=0 -> radarlar kenari da kapatabilir.\n")
    print(f"{'senaryo':<34}{'TRIVIAL':>10}{'kolay':>9}{'ZOR':>8}")
    for label, kw in (
        ("3 radar, margin=0", dict(n_radar=3, margin=0)),
        ("3 radar, margin=150 (ic bolge)", dict(n_radar=3, margin=150)),
        ("5 radar, margin=0", dict(n_radar=5, margin=0)),
        ("8 radar, margin=0", dict(n_radar=8, margin=0)),
        ("3 radar, 2x buyuk halka", dict(n_radar=3, margin=0,
                                         outer_half=220, inner_half=140)),
    ):
        c = random_radar_scan(samples=samples, **kw)
        tot = sum(c.values())
        print(f"{label:<34}{100*c['trivial']/tot:9.0f}%{100*c['kolay']/tot:8.0f}%"
              f"{100*c['zor']/tot:7.0f}%")
    print("\n>> 'ZOR' = ajanin risk ile yol uzunlugu arasinda GERCEK bir karar")
    print("   vermek zorunda oldugu haritalar. Egitim sinyalinin tamami orada.")


def main():
    fixed_map_report()
    scan_report(samples=150)


if __name__ == "__main__":
    main()
