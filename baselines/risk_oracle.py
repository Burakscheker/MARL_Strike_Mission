"""Risk haritasi + risk-farkinda mesafe haritasi (dogruluk zemini).

MARL-Pathfinding'in BFS oracle'inin karsiligi. Fark: orada maliyet HOP SAYISI
(BFS yeterdi), burada maliyet ADIM + RISK (agirlikli, yani Dijkstra gerekiyor).

Uc sey uretir:

 1. zone_map()          — her hucre icin 0/1/2 (guvenli / dis halka / ic halka)
 2. risk_distance_map() — hedefe "adim + risk" maliyetli en ucuz mesafe.
                          HEM reward shaping HEM gozlem skalarlari bunu kullanir
                          (MARL-Pathfinding'de bfs_distance_map ayni iki isi
                          yapiyordu — o tasarim aynen korundu).
 3. survival_prob(path) — verilen yolun ANALITIK hayatta kalma olasiligi.
                          Monte Carlo gurultusu YOK; bir politikanin gercekten
                          guvenli mi yoksa sansli mi oldugunu ancak bu ayirir
                          (Strike_Mission.md "altin kural").

HIZ NOTU: 1000x1000 = 1.000.000 dugum. heapq'lu klasik Dijkstra saf Python'da
~3M heap islemi demek (~30sn). Onun yerine FAST SWEEPING kullaniliyor: 4 yonlu
sirali tarama, her tarama satir/sutun ekseninde vektorize. Sabit noktaya kadar
tekrarlanir — bu Bellman-Ford'un iyi siralamali Gauss-Seidel hali, yani
sonucu KESIN (yaklasik degil). Olcum: ~1-2 saniye, ve sonuc runs/ altina
onbelleklenir (radarlar sabit oldugu icin bir kez hesaplanir).
"""
from __future__ import annotations

import os

import numpy as np

from config import (GOAL, GRID_N, INNER_HALF, OUTER_HALF, P_DEATH,
                    R_RISK_COEF, R_STEP, RADARS, RISK_CACHE, ZONE_CACHE)

# risk maliyetinin ADIM maliyeti cinsinden agirligi.
# cost(hucre) = 1 + RISK_W * p_death(hucre)  [adim-esdegeri]
# RISK_W = |R_RISK_COEF / R_STEP| — yani "bir birim beklenen olum maliyeti kac
# adima bedel" sorusunun odul fonksiyonundaki cevabiyla BIREBIR ayni. Boylece
# mesafe haritasi ile odul fonksiyonu ayni seyi optimize eder.
RISK_W = abs(R_RISK_COEF / R_STEP)


# --------------------------------------------------------------------- zone

def build_zone_map(n: int = GRID_N, radars=RADARS,
                   outer_half: int = OUTER_HALF,
                   inner_half: int = INNER_HALF) -> np.ndarray:
    """(n, n) uint8: 0 guvenli, 1 dis halka, 2 ic halka. Ic halka DIS'i ezer."""
    z = np.zeros((n, n), dtype=np.uint8)
    for r0, c0 in radars:
        for half, val in ((outer_half, 1), (inner_half, 2)):
            r_lo, r_hi = max(0, r0 - half), min(n - 1, r0 + half)
            c_lo, c_hi = max(0, c0 - half), min(n - 1, c0 + half)
            block = z[r_lo:r_hi + 1, c_lo:c_hi + 1]
            np.maximum(block, val, out=block)
    return z


def zone_map(cache: str | None = ZONE_CACHE) -> np.ndarray:
    if cache and os.path.exists(cache):
        return np.load(cache)
    z = build_zone_map()
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.save(cache, z)
    return z


def death_prob_map(zone: np.ndarray | None = None) -> np.ndarray:
    """(n,n) float32 — hucre basina adim basi olum olasiligi."""
    z = zone_map() if zone is None else zone
    return np.asarray(P_DEATH, dtype=np.float32)[z]


# ------------------------------------------------------- risk mesafe haritasi

def _sweep(dist: np.ndarray, w: np.ndarray) -> bool:
    """Tek tam tarama seti (4 yon). Degisiklik olduysa True doner.

    Her yon icin bir eksende SIRALI (Gauss-Seidel), diger eksende vektorize
    ilerlenir — yani 1000 satirlik dongu, her adimi 1000 elemanlik numpy
    islemi. Saf Python'da hucre hucre dolasmaya gore ~1000x hizli.
    """
    before = dist.copy()
    n = dist.shape[0]

    for r in range(1, n):                       # asagi
        np.minimum(dist[r], dist[r - 1] + w[r], out=dist[r])
    for r in range(n - 2, -1, -1):              # yukari
        np.minimum(dist[r], dist[r + 1] + w[r], out=dist[r])
    for c in range(1, n):                       # saga
        np.minimum(dist[:, c], dist[:, c - 1] + w[:, c], out=dist[:, c])
    for c in range(n - 2, -1, -1):              # sola
        np.minimum(dist[:, c], dist[:, c + 1] + w[:, c], out=dist[:, c])

    return bool(np.any(dist < before - 1e-6))


def build_risk_distance_map(goal=GOAL, zone: np.ndarray | None = None,
                            risk_w: float = RISK_W,
                            max_iter: int = 40,
                            verbose: bool = False) -> np.ndarray:
    """Hedefe "adim + risk" maliyetli en ucuz mesafe. (n,n) float32.

    dist[cell] = cell'den GOAL'a giderken odenecek toplam maliyet
    (hedefin kendi hucre maliyeti dahil DEGIL, cikis hucresininki dahil DEGIL —
    yani yalnizca GIRILEN hucrelerin maliyeti; standart node-weighted en kisa yol).

    Komsu FARKI (d_own - d_komsu) bu yuzden guvenli duz bir adimda tam +1.0
    cikar — MARL-Pathfinding'in BFS hop farkiyla AYNI olcek, transfer edilen
    agirliklarin bu skalarlari ayni olcekte gormesi icin onemli.
    """
    z = zone_map() if zone is None else zone
    w = (1.0 + risk_w * np.asarray(P_DEATH, dtype=np.float64)[z]).astype(np.float64)

    dist = np.full(z.shape, np.inf, dtype=np.float64)
    dist[goal] = 0.0

    for it in range(max_iter):
        changed = _sweep(dist, w)
        if verbose:
            print(f"  sweep {it + 1}: {'degisti' if changed else 'sabit nokta'}")
        if not changed:
            break
    else:
        raise RuntimeError("risk mesafe haritasi yakinsamadi — max_iter artir")
    return dist.astype(np.float32)


def risk_distance_map(cache: str | None = RISK_CACHE, verbose: bool = False) -> np.ndarray:
    if cache and os.path.exists(cache):
        return np.load(cache)
    d = build_risk_distance_map(verbose=verbose)
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.save(cache, d)
    return d


# ----------------------------------------------------------------- analitik

def survival_prob(path, zone: np.ndarray | None = None,
                  include_start: bool = False) -> float:
    """Verilen yolun ANALITIK hayatta kalma olasiligi: prod(1 - p_death).

    include_start=False: baslangic hucresi sayilmaz (ortamda da t=0'da zar
    atilmiyor, ilk zar ilk HAREKETTEN sonra). Monte Carlo yok — bu deger
    gurultusuz, tek bir episode'dan bile olculebilir.
    """
    z = zone_map() if zone is None else zone
    p = np.asarray(P_DEATH, dtype=np.float64)
    cells = path if include_start else path[1:]
    s = 1.0
    for r, c in cells:
        s *= (1.0 - p[z[r, c]])
    return s


def exposure(path, zone: np.ndarray | None = None) -> tuple[int, int]:
    """(dis_halkada_adim, ic_halkada_adim)."""
    z = zone_map() if zone is None else zone
    outer = sum(1 for r, c in path if z[r, c] == 1)
    inner = sum(1 for r, c in path if z[r, c] == 2)
    return outer, inner


def greedy_path(start=(0, 0), goal=GOAL, dmap: np.ndarray | None = None,
                max_steps: int = 10_000) -> list[tuple[int, int]]:
    """Risk-mesafe haritasinda tepe-inisi ile ORACLE yolunu cikar.

    dist haritasi zaten en ucuz maliyeti tasidigi icin, her adimda en dusuk
    dist'li komsuya gitmek optimal yolu verir (Dijkstra'nin geri-izlemesi).
    """
    d = risk_distance_map() if dmap is None else dmap
    n = d.shape[0]
    cur = tuple(start)
    path = [cur]
    for _ in range(max_steps):
        if cur == tuple(goal):
            return path
        best, best_d = None, d[cur]
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            rr, cc = cur[0] + dr, cur[1] + dc
            if 0 <= rr < n and 0 <= cc < n and d[rr, cc] < best_d:
                best, best_d = (rr, cc), d[rr, cc]
        if best is None:
            break
        cur = best
        path.append(cur)
    return path


# --------------------------------------------------------------------- CLI

def main():
    import time
    from config import MAX_STEPS, START

    t0 = time.perf_counter()
    z = zone_map()
    print(f"zone haritasi: {time.perf_counter() - t0:.2f}s")
    n_out = int((z == 1).sum())
    n_in = int((z == 2).sum())
    tot = z.size
    print(f"  guvenli={tot - n_out - n_in}  dis={n_out}  ic={n_in}  "
          f"(tehlikeli %{100 * (n_out + n_in) / tot:.1f})")

    t0 = time.perf_counter()
    d = risk_distance_map(verbose=True)
    print(f"risk mesafe haritasi: {time.perf_counter() - t0:.2f}s")

    n = GRID_N
    # DIKKAT: (i,i) listesi 4-YONLU BIR YOL DEGIL — diyagonal hamle aksiyon
    # uzayinda yok. Naif "hedefe dogru git" politikasinin gercek karsiligi
    # MERDIVEN yolu (sag/asagi donusumlu), ve bu radar bolgesinden GECEN hucre
    # sayisini ~2 katina cikardigi icin hayatta kalma olasiligi cok daha dusuk.
    stair = [(0, 0)]
    for i in range(n - 1):
        stair.append((stair[-1][0], stair[-1][1] + 1))
        stair.append((stair[-1][0] + 1, stair[-1][1]))
    Lpath = [(0, c) for c in range(n)] + [(r, n - 1) for r in range(1, n)]
    orc = greedy_path(START, GOAL, d)

    print(f"\n{'politika':<26}{'uzunluk':>9}{'hayatta kalma':>16}"
          f"{'dis':>8}{'ic':>7}")
    for name, p in (("merdiven (naif capraz)", stair),
                    ("L yolu (sag-ust kose)", Lpath),
                    ("Dijkstra oracle", orc)):
        s = survival_prob(p, z)
        o, i = exposure(p, z)
        print(f"{name:<26}{len(p) - 1:>9}{s:>15.4f} {o:>8}{i:>7}")

    # Rastgele monoton yol (asil baseline — bkz. Strike_Mission.md §5:
    # "random walk" degil "random monotone").
    rng = np.random.default_rng(0)
    trials, acc = 300, 0.0
    for _ in range(trials):
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
        acc += survival_prob(path, z)
    rnd = acc / trials
    print(f"{'rastgele monoton':<26}{2 * (n - 1):>9}{rnd:>15.4f}")

    print(f"\noptimal (manhattan)        {2 * (n - 1):>9}")
    print(f"MAX_STEPS                  {MAX_STEPS:>9}")
    o_s = survival_prob(orc, z)
    print(f"takim (>=1 varir): oracle {1 - (1 - o_s) ** 2:.4f}   "
          f"rastgele-monoton {1 - (1 - rnd) ** 2:.4f}")


if __name__ == "__main__":
    main()
