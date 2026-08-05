"""51x51 latisin Burak'in orijinal haritasiyla cakistigini GORSEL olarak dogrula."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

STEP, N = 20, 51
RADARS = [("R1", -280, 220), ("R2", 200, 100), ("R3", -100, -280)]
OH, IH = 5, 3          # hucre yaricapi (110/20 -> 5, 70/20 -> 3)


def cell_to_xy(r, c):
    return (-500 + STEP * c, 500 - STEP * r)


def to_cell(x, y):
    return ((500 - y) // STEP, (x + 500) // STEP)


RC = [(n,) + to_cell(x, y) for n, x, y in RADARS]


def zone(r, c):
    z = 0
    for _, rr, cc in RC:
        if abs(r - rr) <= IH and abs(c - cc) <= IH:
            return 2
        if abs(r - rr) <= OH and abs(c - cc) <= OH:
            z = 1
    return z


plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(11, 11))
ax.set_facecolor("#0d1117")

# latis noktalari (soluk)
for i in range(0, N, 5):
    ax.axhline(500 - STEP * i, color="#1f6f4a", lw=0.4, alpha=0.5)
    ax.axvline(-500 + STEP * i, color="#1f6f4a", lw=0.4, alpha=0.5)

# radar kareleri â€” latis hucrelerinden URETILDI (elle cizilmedi)
for name, rr, cc in RC:
    for half, col, lab in ((OH, "#ffa500", "dis 220x220"), (IH, "#3ddc97", "ic 140x140")):
        x0, y0 = cell_to_xy(rr + half, cc - half)      # sol alt
        w = h = (2 * half + 1 - 1) * STEP + STEP       # kenar uzunlugu birim
        ax.add_patch(Rectangle((x0 - STEP / 2, y0 - STEP / 2), w, h,
                               fill=True, facecolor=col, alpha=0.10,
                               edgecolor=col, lw=2))
    cx, cy = cell_to_xy(rr, cc)
    ax.plot(cx, cy, "+", color="white", ms=12, mew=2)
    ax.text(cx + 12, cy - 26, f"{name}  hucre({rr},{cc})", color="white", fontsize=9)

# --- yollar
diag = [(i, i) for i in range(N)]
Lsafe = [(0, c) for c in range(N)] + [(r, N - 1) for r in range(1, N)]

for path, col, lab in ((diag, "#ff5555", "duz capraz â€” hayatta kalma %24.7"),
                       (Lsafe, "#58a6ff", "guvenli L yolu â€” %100, ayni 100 adim")):
    xs, ys = zip(*[cell_to_xy(*p) for p in path])
    ax.plot(xs, ys, "-", color=col, lw=2.5, label=lab, zorder=5)
    hot = [p for p in path if zone(*p) > 0]
    if hot:
        hx, hy = zip(*[cell_to_xy(*p) for p in hot])
        ax.plot(hx, hy, "x", color=col, ms=9, mew=2, zorder=6)

bx, by = cell_to_xy(0, 0)
hx2, hy2 = cell_to_xy(50, 50)
ax.plot(bx, by, "o", color="#58a6ff", ms=14, zorder=7)
ax.text(bx + 15, by - 35, "B (0,0)", color="#58a6ff", fontsize=13, weight="bold")
ax.plot(hx2, hy2, "*", color="#ff7b72", ms=20, zorder=7)
ax.text(hx2 - 150, hy2 + 25, "H (50,50)", color="#ff7b72", fontsize=13, weight="bold")

ax.set_xlim(-520, 520)
ax.set_ylim(-520, 520)
ax.set_aspect("equal")
ax.set_title("51x51 latis (STEP=20) â€” radar kareleri hucrelerden URETILDI\n"
             "220/20=11 hucre, 140/20=7 hucre, uc merkez de tam latis uzerinde",
             fontsize=12)
ax.legend(loc="lower left", fontsize=10, framealpha=0.3)
ax.tick_params(colors="#8b949e")
import os
os.makedirs("runs", exist_ok=True)
out = os.path.join("runs", "lattice_check.png")
fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="#0d1117")
print("yazildi:", out)

