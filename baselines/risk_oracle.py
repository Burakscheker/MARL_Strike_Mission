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

from config import (DIRS, GOAL, GRID_N, HAZARD_MODE, INNER_HALF, OUTER_HALF,
                    P_DEATH, P_INNER_TOTAL, P_OUTER_TOTAL, R_RISK_COEF,
                    R_STEP, RADARS, RISK_CACHE, ZONE_CACHE)

# DIRS = ((-1,0),(0,1),(1,0),(0,-1)) -> UP, RIGHT, DOWN, LEFT
DIR_KEYS = ("up", "right", "down", "left")

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

def move_risk(z_from: np.ndarray, z_to: np.ndarray,
              mode: str = HAZARD_MODE) -> np.ndarray:
    """Bir hucreden komsusuna GECERKEN olum olasiligi.

    per_entry (varsayilan, patronun kurali): risk SADECE bolge seviyesi
    ARTARKEN odenir — 0->1 %20, 1->2 %90, 0->2 (kose durumu) %90. Ayni
    seviyede kalmak veya cikmak BEDAVA. Yani "bolgede ne kadar kaldigi"
    maliyeti hic etkilemez, "girdi mi girmedi mi" etkiler.

    per_step (ablation): varilan hucrenin adim basi hazard'i.
    """
    if mode == "per_step":
        return np.asarray(P_DEATH, dtype=np.float64)[z_to]
    lut = np.zeros((3, 3), dtype=np.float64)
    lut[0, 1] = P_OUTER_TOTAL
    lut[0, 2] = P_INNER_TOTAL
    lut[1, 2] = P_INNER_TOTAL
    return lut[z_from, z_to]


def direction_costs(z: np.ndarray, risk_w: float = RISK_W,
                    mode: str = HAZARD_MODE) -> dict:
    """4 yon icin "u'dan v'ye HAREKET" maliyeti: 1 adim + risk_w * p(u->v).

    Node-agirlikli (her hucreye sabit maliyet) surumden KENAR-agirlikliya
    gecis, per_entry modunun matematiksel geregi: orada maliyet hucrede
    DURMAKTAN degil, sinirdan GECMEKTEN doguyor.
    """
    n = z.shape[0]
    cost = {}
    # up[r] = (r,c) -> (r-1,c) hareketinin maliyeti (r >= 1)
    cost["up"] = np.full((n, n), np.inf)
    cost["up"][1:] = 1.0 + risk_w * move_risk(z[1:], z[:-1], mode)
    # down[r] = (r,c) -> (r+1,c)
    cost["down"] = np.full((n, n), np.inf)
    cost["down"][:-1] = 1.0 + risk_w * move_risk(z[:-1], z[1:], mode)
    # left[:,c] = (r,c) -> (r,c-1)
    cost["left"] = np.full((n, n), np.inf)
    cost["left"][:, 1:] = 1.0 + risk_w * move_risk(z[:, 1:], z[:, :-1], mode)
    # right[:,c] = (r,c) -> (r,c+1)
    cost["right"] = np.full((n, n), np.inf)
    cost["right"][:, :-1] = 1.0 + risk_w * move_risk(z[:, :-1], z[:, 1:], mode)
    return cost


def oracle_action(pos, dist: np.ndarray, cost: dict, n: int) -> int:
    """Bellman argmin: d[u] = min_v ( cost(u->v) + d[v] ).

    KENAR maliyeti sart — bir halkaya girmek per_entry'de 1 + 1500*0.9 = 1351
    adim-esdegeri. Sadece d[v]'ye bakmak (tepe inisi) yolu felakete surukler
    (bkz. greedy_path'teki ayni hata). BC on-egitiminde (train_bc.py) VE
    RL fine-tune sirasindaki oracle-capa (BC-anchored VDN, agents/vdn.py)
    kaybinda ortak kullanilir — ikisi de AYNI uzmani taklit etmeli.
    """
    r, c = pos
    best_a, best_v = 0, np.inf
    for a, (dr, dc) in enumerate(DIRS):
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            v = float(cost[DIR_KEYS[a]][r, c]) + float(dist[nr, nc])
            if v < best_v:
                best_v, best_a = v, a
    return best_a


def _sweep(dist: np.ndarray, cost: dict) -> bool:
    """Tek tam tarama seti (4 yon). Degisiklik olduysa True doner.

    dist[u] = u'dan hedefe en ucuz maliyet, yani
        dist[u] = min_v ( cost(u->v) + dist[v] ).
    Asagidaki her dongu bir eksende SIRALI (Gauss-Seidel), diger eksende
    vektorize ilerler — 1000 satirlik dongu, her adimi 1000 elemanlik numpy
    islemi. Sabit noktaya kadar tekrarlanan bu sema Bellman-Ford'un iyi
    siralamali hali, yani sonucu KESIN.
    """
    before = dist.copy()
    n = dist.shape[0]

    # dist[r]'yi dist[r-1] uzerinden guncelle -> hareket (r,c) -> (r-1,c) = "up"
    for r in range(1, n):
        np.minimum(dist[r], dist[r - 1] + cost["up"][r], out=dist[r])
    for r in range(n - 2, -1, -1):
        np.minimum(dist[r], dist[r + 1] + cost["down"][r], out=dist[r])
    for c in range(1, n):
        np.minimum(dist[:, c], dist[:, c - 1] + cost["left"][:, c], out=dist[:, c])
    for c in range(n - 2, -1, -1):
        np.minimum(dist[:, c], dist[:, c + 1] + cost["right"][:, c], out=dist[:, c])

    return bool(np.any(dist < before - 1e-6))


def build_risk_distance_map(goal=GOAL, zone: np.ndarray | None = None,
                            risk_w: float = RISK_W,
                            mode: str = HAZARD_MODE,
                            max_iter: int = 40,
                            verbose: bool = False) -> np.ndarray:
    """Hedefe "adim + risk" maliyetli en ucuz mesafe. (n,n) float32.

    Komsu FARKI (d_own - d_komsu) guvenli duz bir adimda tam +1.0 cikar —
    MARL-Pathfinding'in BFS hop farkiyla AYNI olcek, transfer edilen
    agirliklarin bu skalarlari tanidik bir araliktan gormesi icin onemli.
    """
    z = zone_map() if zone is None else zone
    cost = direction_costs(z, risk_w, mode)

    dist = np.full(z.shape, np.inf, dtype=np.float64)
    dist[goal] = 0.0

    for it in range(max_iter):
        changed = _sweep(dist, cost)
        if verbose:
            print(f"  sweep {it + 1}: {'degisti' if changed else 'sabit nokta'}")
        if not changed:
            break
    else:
        raise RuntimeError("risk mesafe haritasi yakinsamadi — max_iter artir")
    return dist.astype(np.float32)


def risk_distance_map(cache: str | None = RISK_CACHE, verbose: bool = False,
                      mode: str = HAZARD_MODE,
                      risk_w: float = RISK_W) -> np.ndarray:
    # Onbellek adi MOD ve RISK_W'yi icerir: ikisi de haritayi TAMAMEN
    # degistiriyor, ayni dosyaya yazilirsa parametre degisince sessizce yanlis
    # harita kullanilir (ve hicbir yerde patlamaz — en tehlikeli hata turu).
    # risk_w eklendi cunku R_RISK_COEF 15.0 -> 7.5 olunca RISK_W de
    # 1500 -> 750 dustu ve eski onbellek gecerliymis gibi okunuyordu.
    if cache:
        root, ext = os.path.splitext(cache)
        cache = f"{root}_{mode}_w{risk_w:g}{ext}"
        if os.path.exists(cache):
            return np.load(cache)
    d = build_risk_distance_map(verbose=verbose, mode=mode, risk_w=risk_w)
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.save(cache, d)
    return d


# ----------------------------------------------------------------- analitik

def survival_prob(path, zone: np.ndarray | None = None,
                  mode: str = HAZARD_MODE, prev0: int = 0) -> float:
    """Verilen yolun ANALITIK hayatta kalma olasiligi.

    per_entry: SADECE bolge seviyesinin ARTTIGI gecislerde zar atilir
               (patronun kurali: sure onemsiz, giris onemli).
    per_step  : her hucrede zar (ablation).

    Monte Carlo yok — bu deger gurultusuz, tek bir episode'dan bile olculebilir.
    Basari oranini ASLA tek basina raporlama; bunu yanina koy.
    """
    z = zone_map() if zone is None else zone
    s = 1.0
    if mode == "per_step":
        p = np.asarray(P_DEATH, dtype=np.float64)
        for r, c in path[1:]:
            s *= (1.0 - p[z[r, c]])
        return s
    # prev0=0 (varsayilan): B bir halkanin icindeyse bu bir GIRIS sayilir ve
    # zar atilir — Burak'in kurali, "atmamazlik yapma". env.reset() de
    # _prev_zone'u 0'dan baslatir, ikisi ayni.
    # prev0 parametresi SADECE yol parcasi olcerken lazim: bir yolun ortasindan
    # baslayan bir dilimi degerlendirirken ucagin oraya hangi bolgeden geldigi
    # verilmelidir (bkz. tests/_run_scripted).
    prev = prev0
    for r, c in path:
        cur = int(z[r, c])
        if cur > prev:
            s *= (1.0 - (P_INNER_TOTAL if cur == 2 else P_OUTER_TOTAL))
        prev = cur
    return s


def exposure(path, zone: np.ndarray | None = None) -> tuple[int, int]:
    """(dis_halkada_adim, ic_halkada_adim)."""
    z = zone_map() if zone is None else zone
    outer = sum(1 for r, c in path if z[r, c] == 1)
    inner = sum(1 for r, c in path if z[r, c] == 2)
    return outer, inner


_STEP_DIRS = (("up", -1, 0), ("right", 0, 1), ("down", 1, 0), ("left", 0, -1))


def entries(path, zone: np.ndarray | None = None, prev0: int = 0) -> tuple[int, int]:
    """(dis_halka_girisi, ic_halka_girisi) — per_entry'de ZAR SAYISI.

    "Kac adim bolgede kaldi" (exposure) DEGIL, "kac kez ayri bir bolgeye
    girdi". per_entry modunda hayatta kalma tam olarak
    0.8^(dis giris) * 0.1^(ic giris) oldugu icin ogrenilmesi gereken beceri
    budur; reward shaping tartismasinin dogru olcusu de bu.
    """
    z = zone_map() if zone is None else zone
    prev, out, inn = prev0, 0, 0
    for r, c in path:
        cur = int(z[r, c])
        if cur > prev:
            if cur == 2:
                inn += 1
            else:
                out += 1
        prev = cur
    return out, inn


def greedy_path(start=(0, 0), goal=GOAL, dmap: np.ndarray | None = None,
                cost: dict | None = None,
                max_steps: int = 20_000) -> list[tuple[int, int]]:
    """Risk-mesafe haritasindan ORACLE yolunu cikar (Dijkstra geri-izlemesi).

    BUG (bulundu ve duzeltildi): eskiden her adimda SADECE `d` degeri en kucuk
    komsuya gidiliyordu (tepe-inisi). Bu, maliyet DUGUM-agirlikli oldugunda
    yaklasik dogru; ama bizim maliyetimiz KENAR-agirlikli — bir hucreye
    girmenin bedeli hangi bolgeden geldigine bagli (`move_risk`). per_entry'de
    bir halkaya girmek 1 + 1500*0.9 = 1351 adim-esdegeri; `d`'si biraz daha
    kucuk diye o komsuya atlamak yolu felakete surukluyordu.

    Dogru geri izleme, Bellman denkleminin kendisi:
        d[u] = min_v ( cost(u->v) + d[v] )
    yani argmin ALINIRKEN kenar maliyeti de toplanmali.

    OLCULDU (40 rastgele radar, per_entry): eski surum hayatta kalma 0.0008,
    duzeltilmis surum AYNI haritada 0.0800 — yani oracle tavani 100 KAT
    dusuk raporlaniyordu. per_step'te de 0.0063 -> 0.5028. Tavan yanlis
    olunca "ajan ne kadar iyi" sorusu da yanlis cevaplaniyordu.

    cost=None verilirse varsayilan harita/moddan uretilir (sabit-harita
    cagiranlar icin geriye donuk uyumlu). Rastgele haritada MUTLAKA o
    haritanin direction_costs()'u gecilmeli.
    """
    d = risk_distance_map() if dmap is None else dmap
    if cost is None:
        cost = direction_costs(zone_map())
    n = d.shape[0]
    cur = tuple(start)
    path = [cur]
    for _ in range(max_steps):
        if cur == tuple(goal):
            return path
        best, best_v = None, np.inf
        for name, dr, dc in _STEP_DIRS:
            rr, cc = cur[0] + dr, cur[1] + dc
            if 0 <= rr < n and 0 <= cc < n:
                v = cost[name][cur] + d[rr, cc]
                if v < best_v:
                    best_v, best = v, (rr, cc)
        # Maliyetler >= 1.0 oldugu icin d yol boyunca KESIN azalir; sonsuz
        # dongu imkansiz. best None ise (hepsi inf) cikilir.
        if best is None or not np.isfinite(best_v):
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
