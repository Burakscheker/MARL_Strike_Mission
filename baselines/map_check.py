"""Strike Mission tasarim dogrulamasi — Strike_Mission.md §0'in KANITI.

Bu dosya plandaki her sayiyi yeniden uretir. Plandaki bir sayidan supheye
dusersen once bunu kosur, sonra tartisirsin. Asama 2'de yerini duzgun bir
`baselines/risk_oracle.py` alacak (env/config ile paylasilan sabitler
uzerinden); bu dosya plani yazarken kullanilan BAGIMSIZ hesap olarak kaliyor —
kasitli olarak config.py'ye bagimli DEGIL, yani ortam kodundaki bir hata bu
kontrolu de sessizce bozamaz.

Kosum: python -m baselines.map_check

Sorular:
 1. STEP=20 ile 1000x1000 grid kac latis noktasi, radar kareleri tam simetrik mi?
 2. Duz capraz (B->H) hangi radarlardan geciyor?
 3. SIFIR-riskli ve AYNI ZAMANDA optimal (100 adim) yol var mi? Kac tane?
 4. Per-step olum olasiligi kalibrasyonu ne cikiyor?
 5. Dijkstra ile en yuksek hayatta-kalma olasilikli yol ne veriyor?
"""
import heapq
import math
from collections import deque

STEP = 20
N = 51                       # 0..50 latis noktasi (1000/20 + 1)

# (x,y) -> (row, col):  row = (500-y)/20,  col = (x+500)/20
def to_cell(x, y):
    return ((500 - y) // STEP, (x + 500) // STEP)

RADARS_XY = [("R1", -280, 220), ("R2", 200, 100), ("R3", -100, -280)]
OUTER_HALF_U = 110           # birim
INNER_HALF_U = 70
OUTER_HALF = OUTER_HALF_U // STEP    # 5 (110/20 = 5.5 -> |d|<=5)
INNER_HALF = INNER_HALF_U // STEP    # 3 (70/20 = 3.5  -> |d|<=3)

START = to_cell(-500, 500)   # B  (sol ust)
GOAL = to_cell(500, -500)    # H  (sag alt)

print("=== 1. Geometri ===")
print(f"latis: {N}x{N} nokta, adim {STEP} birim")
print(f"B={START}  H={GOAL}  manhattan={abs(GOAL[0]-START[0])+abs(GOAL[1]-START[1])}")
print(f"dis halka yaricap={OUTER_HALF} hucre ({2*OUTER_HALF+1} nokta genis)"
      f"  -> {(2*OUTER_HALF+1)*STEP} birim (hedef 220)")
print(f"ic  halka yaricap={INNER_HALF} hucre ({2*INNER_HALF+1} nokta genis)"
      f"  -> {(2*INNER_HALF+1)*STEP} birim (hedef 140)")

RADARS = []
for name, x, y in RADARS_XY:
    r, c = to_cell(x, y)
    exact = ((500 - y) % STEP == 0) and ((x + 500) % STEP == 0)
    RADARS.append((name, r, c))
    print(f"  {name} (x={x},y={y}) -> hucre (row={r},col={c})  latis-uzerinde={exact}")


def zone(r, c):
    """0=guvenli, 1=dis halka, 2=ic halka (en tehlikelisi kazanir)."""
    z = 0
    for _, rr, cc in RADARS:
        dr, dc = abs(r - rr), abs(c - cc)
        if dr <= INNER_HALF and dc <= INNER_HALF:
            return 2
        if dr <= OUTER_HALF and dc <= OUTER_HALF:
            z = 1
    return z


print("\n=== 2. Duz capraz (row==col) hangi bolgelerden geciyor ===")
diag = [(i, i) for i in range(N)]
inner_hits = [p for p in diag if zone(*p) == 2]
outer_hits = [p for p in diag if zone(*p) == 1]
print(f"ic halkada {len(inner_hits)} nokta: {inner_hits}")
print(f"dis halkada {len(outer_hits)} nokta: {outer_hits}")

print("\n=== 4. Per-step olum kalibrasyonu ===")
# "Bolgeyi boydan boya gecersen toplam olum riski %20 / %90" olacak sekilde
CROSS_OUTER = 2 * OUTER_HALF + 1     # 11 adim
CROSS_INNER = 2 * INNER_HALF + 1     # 7 adim
P_OUTER_TOTAL, P_INNER_TOTAL = 0.20, 0.90
p_out = 1 - (1 - P_OUTER_TOTAL) ** (1 / CROSS_OUTER)
p_in = 1 - (1 - P_INNER_TOTAL) ** (1 / CROSS_INNER)
print(f"dis: {CROSS_OUTER} adimda toplam %20 -> adim basi p={p_out:.5f} (%{p_out*100:.2f})")
print(f"ic : {CROSS_INNER} adimda toplam %90 -> adim basi p={p_in:.5f} (%{p_in*100:.2f})")
P_DEATH = (0.0, p_out, p_in)

print("\n=== 3. Sifir-riskli optimal (monoton, 100 adim) yol var mi? ===")
# monoton yol: sadece sag (col+1) ve asagi (row+1). DP ile say.
safe = [[zone(r, c) == 0 for c in range(N)] for r in range(N)]
dp = [[0] * N for _ in range(N)]
dp[0][0] = 1 if safe[0][0] else 0
for r in range(N):
    for c in range(N):
        if r == 0 and c == 0:
            continue
        if not safe[r][c]:
            continue
        dp[r][c] = (dp[r - 1][c] if r else 0) + (dp[r][c - 1] if c else 0)
n_safe_monotone = dp[GOAL[0]][GOAL[1]]
print(f"sifir-riskli monoton yol sayisi: {n_safe_monotone}")
total_monotone = math.comb(100, 50)
print(f"toplam monoton yol: {total_monotone:.3e}  -> guvenli oran %{100*n_safe_monotone/total_monotone:.4f}")

print("\n=== 5. Dijkstra: en yuksek hayatta-kalma olasilikli yol ===")
# maliyet = -ln(1 - p_death(hucre)); hedefe varana kadar toplanir.
# 4 yonlu hareket (geri adim serbest) — gercek ortamla ayni aksiyon kumesi.
INF = float("inf")
dist = [[INF] * N for _ in range(N)]
prev = {}
sr, sc = START
dist[sr][sc] = -math.log(1 - P_DEATH[zone(sr, sc)])
pq = [(dist[sr][sc], sr, sc)]
while pq:
    d, r, c = heapq.heappop(pq)
    if d > dist[r][c]:
        continue
    for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < N and 0 <= nc < N):
            continue
        nd = d - math.log(1 - P_DEATH[zone(nr, nc)])
        if nd < dist[nr][nc] - 1e-12:
            dist[nr][nc] = nd
            prev[(nr, nc)] = (r, c)
            heapq.heappush(pq, (nd, nr, nc))
best = dist[GOAL[0]][GOAL[1]]
print(f"en iyi hayatta kalma olasiligi: {math.exp(-best):.6f}")

# duz caprazin hayatta kalma olasiligi (kiyas)
surv_diag = 1.0
for p in diag:
    surv_diag *= (1 - P_DEATH[zone(*p)])
print(f"duz capraz yolun hayatta kalma olasiligi: {surv_diag:.6f}")

# L yolu (once saga sonra asagi)
Lpath = [(0, c) for c in range(N)] + [(r, N - 1) for r in range(1, N)]
surv_L = 1.0
for p in Lpath:
    surv_L *= (1 - P_DEATH[zone(*p)])
print(f"L yolu (sag-ust kose) hayatta kalma: {surv_L:.6f}, uzunluk={len(Lpath)-1}")

print("\n=== 6. Rastgele monoton yolun beklenen hayatta kalmasi (baseline) ===")
import random
random.seed(0)
tot = 0.0
TRIALS = 20000
for _ in range(TRIALS):
    r = c = 0
    s = 1 - P_DEATH[zone(0, 0)]
    for _ in range(100):
        if r == N - 1:
            c += 1
        elif c == N - 1:
            r += 1
        elif random.random() < 0.5:
            c += 1
        else:
            r += 1
        s *= (1 - P_DEATH[zone(r, c)])
    tot += s
print(f"random-monoton baseline hayatta kalma: {tot/TRIALS:.4f}")
print(f"  -> takim (2 bagimsiz ucak) en az biri varir: {1-(1-tot/TRIALS)**2:.4f}")

print("\n=== 7. Tehlike yogunlugu ===")
cells = [(r, c) for r in range(N) for c in range(N)]
n1 = sum(1 for p in cells if zone(*p) == 1)
n2 = sum(1 for p in cells if zone(*p) == 2)
print(f"guvenli={len(cells)-n1-n2}  dis={n1}  ic={n2}  "
      f"(tehlikeli oran %{100*(n1+n2)/len(cells):.1f})")
