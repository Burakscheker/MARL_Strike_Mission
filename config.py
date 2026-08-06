"""Tum hiperparametreler ve sabitler burada. Koda deger gomme.

Strike_Mission.md ile senkron tut: ozellikle odul degerleri (§4) ve MAX_STEPS.

OLCEK KARARI: grid TAM COZUNURLUK, 1000x1000 (hucre = 1 birim). Plandaki
STEP_SIZE=20 indirgemesi Burak'in karariyla IPTAL edildi. Bunun bedeli ve
neyin degismek zorunda kaldigi asagida GAMMA ve PATCH_STRIDE notlarinda.
"""
import math

# ---------------------------------------------------------------- grid / ortam
GRID_N = 1000                  # hucre (0..999), 1 hucre = 1 birim
N_AGENTS = 2

AGENT_1 = 0
AGENT_2 = 1

# Aksiyonlar
UP, RIGHT, DOWN, LEFT, NOOP = range(5)
N_ACTIONS = 5
ACTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT", "NOOP")
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))

# Dunya koordinati (x,y) -> hucre (row, col):
#   col = x + 500,  row = 500 - y      (x: -500..500, y: -500..500)
START = (0, 0)                 # B  (-500, +500) sol ust
GOAL = (GRID_N - 1, GRID_N - 1)  # H  (+500, -500) sag alt

# Radar merkezleri (row, col) — Burak'in verdigi (x,y)'lerden turetildi.
# R1 (-280,220) R2 (200,100) R3 (-100,-280)
RADARS = ((280, 220), (400, 700), (780, 400))
N_RADAR = len(RADARS)

# Halka yariciplari (hucre). 220x220 -> +-110, 140x140 -> +-70.
# |d| <= HALF kullaniliyor: kenar 2*HALF+1 = 221 / 141 hucre. Nominal 220/140
# yerine 221/141 olmasi (%0.45 fark) bilincli — merkez etrafinda TAM SIMETRIK
# kare veriyor, tek hucrelik kayma ogrenmeyi hicbir sekilde etkilemiyor.
OUTER_HALF = 110
INNER_HALF = 70

# Adim limiti. Optimal yol = manhattan(B,H) = 1998 adim. 1.4x tampon.
MAX_STEPS = 2800

# ---------------------------------------------------------------- risk modeli
# Strike_Mission.md §0.2: ADIM BASI hazard. Burak'in verdigi %20/%90 "bolgeyi
# BOYDAN BOYA gecersen toplam olum riski" olarak okunup adima cevriliyor:
#     p_step = 1 - (1 - p_toplam)^(1 / caprazlama_adimi)
# Bu formul OLCEKTEN BAGIMSIZ: 51x51'de de 1000x1000'de de duz-capraz yolun
# hayatta kalma olasiligi %24.7 cikiyor (baselines/map_check.py dogruladi).
P_OUTER_TOTAL = 0.20           # dis halkayi boydan boya gecersen olum
P_INNER_TOTAL = 0.90           # ic halkayi boydan boya gecersen olum

_CROSS_OUTER = 2 * OUTER_HALF + 1     # 221 adim
_CROSS_INNER = 2 * INNER_HALF + 1     # 141 adim
P_DEATH = (
    0.0,
    1.0 - (1.0 - P_OUTER_TOTAL) ** (1.0 / _CROSS_OUTER),   # ~0.001010 / adim
    1.0 - (1.0 - P_INNER_TOTAL) ** (1.0 / _CROSS_INNER),   # ~0.016184 / adim
)

# RISK MODELI — Burak'in patronunun netlestirdigi kural (2026-08-05):
#   "detection zone'da ister 20 adim at ister 2 adim gir-cik yap, olum
#    ihtimali ESIT. 0.2*0.02*0.2 gibi bir birikme YOK."
# Yani risk SUREYE BAGLI DEGIL: bolgeye GIRIS basina TEK zar.
#   dis halkaya giris -> %20 olum
#   ic halkaya giris  -> %90 olum   (dis'ten ic'e gecerken ayrica zar)
# Cikip tekrar girmek YENI bir zardir (yeni tespit/angajman).
#
# "per_step" (adim basi hazard, P_DEATH ile) ABLATION olarak duruyor:
# --hazard per_step ile kosulur. Ikisinin karsilastirmasi rapora girer,
# cunku ikisi NITELIKSEL OLARAK FARKLI problemler uretiyor: per_step'te
# "bolgeyi tegetten sıyır" ogrenilecek bir beceridir, per_entry'de degildir
# (girdiysen girmissindir) — orada tek beceri "HIC girme".
HAZARD_MODE = "per_entry"

# Radar alarm kuplaji — Strike_Mission.md §0.4 / Asama 6.
# Bir ucak dis halkaya girince o radar alarma gecer, olum olasiligi carpilir.
# VARSAYILAN KAPALI: once sade ortam dogrulanacak.
ALERT_ENABLED = False
ALERT_MULT = 2.0
ALERT_DECAY = 300              # adim (1000x1000 olcegine gore; 51x51'de ~15)

# ---------------------------------------------------------------- odul (§4)
# 1000x1000 YENIDEN KALIBRASYONU: optimal yol 1998 adim (51x51'de 100 idi).
# R_STEP'i -0.05'te birakirsak toplam adim maliyeti -100 olur ve +50'lik hedef
# odulunu 2x ezer — ajan "hic hareket etme" ogrenir. Adim maliyeti yol
# uzunluguyla ORANTILI kalmali: toplam adim maliyeti / hedef odulu ~= 0.4
# (MARL-Pathfinding'de 5/10 = 0.5 idi, kanitlanmis oran).
R_STEP = -0.01                 # 1998 adim -> toplam -20
R_DEATH = -15.0                # bir ucak dusuruldu
R_FIRST_GOAL = +50.0           # ILK ucak hedefe vardi -> takim odulu FULL
R_SECOND_GOAL = +12.0          # ikinci ucak da vardi ("ikide olsa" bonusu)
R_ALL_DEAD = -10.0             # ikisi de dusuruldu (R_DEATH'lerin USTUNE)
R_TIMEOUT = -10.0              # hicbiri varmadan sure doldu

# Adim basi risk maliyeti = R_RISK_COEF * p_death(hucre). Seyrek/gurultulu
# olum sinyalini yogun/deterministik hale getirir. 0.0 -> kapali.
#
# 15.0 -> 7.5 DUZELTMESI: eski deger |R_DEATH| ile ayniydi ve yorum bunu
# "varyans azaltma" diye gerekcelendiriyordu. Ama varyans azaltma normalde
# stokastik terimin YERINE gecer; burada USTUNE ekleniyordu:
#     tehlikeli hucre  ->  R_RISK_COEF * p   (pesin, deterministik)
#                      ->  ayrica zar at, olursen R_DEATH   (stokastik)
# Ikisinin beklenen degeri de |R_DEATH| * p oldugu icin risk EFEKTIF OLARAK
# IKI KEZ cezalandiriliyordu: ajanin risk kacinmasi spec'in 2 kati, yani
# tasarlanandan daha urkek bir politika ogreniyordu.
# 7.5 = |R_DEATH| / 2 ile iki terimin TOPLAMI spec'teki tek cezaya esitlenir.
# Ablation olarak 0.0 (sadece stokastik) ve 15.0 (eski) de kosulmali.
R_RISK_COEF = 7.5

# ---------------------------------------------------------------- egitim (ortak)
SEED = 0

# GAMMA — 1000x1000'in EN KRITIK sonucu, 0.99'da BIRAKILAMAZ.
# Iki bagimsiz nedenden:
#  (1) Episode 2000+ adim. gamma=0.99 ile hedef odulu 0.99^2000 = 2e-9'a
#      iner — ajan hedefi FIZIKSEL OLARAK goremez.
#  (2) Potential-based shaping'in "drag" terimi: r' = COEF*(gamma*Phi' - Phi)
#      = COEF*(DPhi - (1-gamma)*Phi). Hedefe dogru bir adimda DPhi = 1/1998 =
#      0.0005. gamma=0.99'da drag = 0.01*Phi, yani ilerleme sinyalinin 10-20
#      KATI ve TERS isaretli — shaping ajani ilerlemekten CAYDIRIRDI.
# gamma=0.9998'de drag = 0.0002*Phi <= 0.0002, DPhi=0.0005'in altinda kalir
# (Phi=1'de bile 2.5x marj). Etkin ufuk 1/(1-gamma) = 5000 adim > 2800 episode.
GAMMA = 0.9998

GRAD_CLIP = 10.0
HIDDEN = 128
LEARN_EVERY = 8

# SHAPING_COEF: hedefe dogru bir adimin shaping sinyali |R_STEP|'in ~5 katı
# olsun (MARL-Pathfinding'de olculmus dogru oran: en zayif sinyal "hedefe git"
# olmamali). Phi=0.5'te net sinyal = COEF*(0.0005 - 0.0001) = COEF*0.0004.
# 0.05 / 0.0004 = 125 -> 120 secildi.
SHAPING_COEF = 120.0

EPS_START, EPS_END = 1.0, 0.05
EPS_FLOOR_FRAC = 0.5           # epsilon egitimin bu KESRINDE tabana iner

# ---------------------------------------------------------------- DQN (Asama 3)
DQN_EPISODES = 3_000
DQN_BUFFER = 200_000
DQN_BATCH = 32
DQN_EPS_DECAY_STEPS = 200_000
DQN_LEARN_START = 2_000
DQN_LR = 1e-4
DQN_TARGET_UPDATE = 2_000
DQN_EVAL_EVERY = 250

# ---------------------------------------------------------------- IQL (Asama 4)
IQL_EPISODES = 2_000
# BELLEK: buffer satiri = OBS_DIM(898) x 4 byte x 2 dizi (obs + next_obs)
# = 7.2 KB. 150k satir = 1.08 GB, ajan basina -> IQL toplam ~2.2 GB.
# (MARL-Pathfinding'de ayni hesapla 100k kullaniliyordu; episode'lar orada
# ~180 adimdi, burada ~2000 — ayni sayida episode cok daha fazla transition
# uretiyor, o yuzden buffer'in kapsadigi episode penceresi kacinilmaz dar.)
IQL_BUFFER = 150_000
IQL_BATCH = 32
IQL_EPS_DECAY_STEPS = 1_000_000
IQL_LEARN_START = 2_000
IQL_LR = 1e-4
IQL_TARGET_UPDATE = 2_000
IQL_EVAL_EVERY = 200

TRAIN_HARM_WINDOW = 100
TRAIN_HARM_LOG_EVERY = 25
DEMO_EPISODES = 10
DEMO_SEED = 777

# ---------------------------------------------------------------- VDN (Asama 5)
# DIKKAT: LR ve TARGET_UPDATE degerleri MARL-Pathfinding'de 3 ayri tam-olcekli
# kosuyu cokerttikten sonra bulundu (1e-4 + 2000 -> ep~1750'de tepe yapip
# cokme; 3e-5 + 4000 -> 12000 episode monoton yukselis, hic cokme yok).
# Degistirmeden egitme.
VDN_EPISODES = 2_000
# BELLEK: joint satir = 4 x OBS_DIM x 4 byte = 14.4 KB (obs1,obs2 + next'leri).
# 120k satir = 1.72 GB.
VDN_BUFFER = 120_000           # ortak (joint) transition — her satir BIR global timestep
VDN_BATCH = 32
VDN_EPS_DECAY_STEPS = 2_000_000
VDN_LEARN_START = 2_000
VDN_LR = 3e-5
VDN_TARGET_UPDATE = 4_000
VDN_EVAL_EVERY = 200

# ---------------------------------------------------------------- QMIX (Asama 7)
QMIX_EPISODES = 2_000
# VDN satirina ek olarak state + next_state (2 x 890 x 4 byte = 7.1 KB)
# -> satir 21.5 KB. 80k satir = 1.72 GB (VDN ile ayni bellek butcesi).
QMIX_BUFFER = 80_000
QMIX_BATCH = 32
QMIX_EPS_DECAY_STEPS = 2_000_000
QMIX_LEARN_START = 2_000
QMIX_LR = 3e-5
QMIX_TARGET_UPDATE = 4_000
QMIX_EVAL_EVERY = 200
QMIX_MIXER_EMBED = 32

# ---------------------------------------------------------------- gozlem
# TRANSFER KISITI (Burak'in istegi: MARL-Pathfinding'in egitilmis modellerini
# resume et). O projede OBS_DIM = 2*21*21 + 16 = 898, STATE_DIM = 2*21*21 + 8
# = 890. CNNQNet'in parametre sayisi AdaptiveAvgPool2d sayesinde PATCH_SIZE'dan
# bagimsiz, ama scalar_enc'in ilk Linear'i N_SCALARS'a, head'in ilki
# (flat + SCALAR_EMBED + N_SCALARS)'a bagli. Yani ceckpoint'in BIREBIR
# yuklenmesi icin OBS_CHANNELS=2 ve N_SCALARS=16 AYNEN korunmali. Korundu.
PATCH_RADIUS = 10              # 21x21 ornek
PATCH_SIZE = 2 * PATCH_RADIUS + 1

# PATCH_STRIDE — 1000x1000'in ikinci kritik sonucu.
# 51x51'de 21x21 pencere bir radarin (11 hucre) tamamini goruyordu. 1000x1000'de
# dis halka 221 HUCRE genisliginde; stride=1'de ajan penceresinin tamami tek bir
# halkanin icinde kalir ve SINIRI HIC GOREMEZ (kanal sabit 1.0 -> bilgisiz).
# Cozum: pencere SEYREK ornekleniyor — 21 ornek x 16 hucre arayla +-160 hucre
# kapsar, yani bir dis halkayi (+-110) sinirlariyla birlikte gorur.
# Tensor SEKLI degismedigi icin checkpoint uyumu bozulmuyor.
PATCH_STRIDE = 16

OBS_CHANNELS = 2
# kanal 0 = TEHLIKE HARITASI (0 guvenli, 0.5 dis halka, 1.0 ic halka)
#           MARL-Pathfinding'deki "yasak bolge" kanaliyla AYNI ROL: "buradan
#           kacin, buyuk deger daha kotu". Ikili degil dereceli olmasi transferi
#           bozmuyor (ag zaten surekli girdi aliyor).
# kanal 1 = kendi izi (ziyaret edilmis hucreler) — MARL-Pathfinding ile BIREBIR
#           ayni anlam.

N_SCALARS = 16                 # slot slot MARL-Pathfinding ile hizalandi:
# 0 agent_id | 1 other_terminal (orada: faz) | 2 t/max | 3 own_row | 4 own_col
# 5 dy_goal  | 6 dx_goal | 7 dist_goal
# 8 dy_other | 9 dx_other | 10 dist_other
# 11 risk_dist_own (orada: bfs_own)
# 12-15 risk_dist komsu FARKI: yukari/sag/asagi/sol (orada: bfs komsu farki)
OBS_DIM = OBS_CHANNELS * PATCH_SIZE * PATCH_SIZE + N_SCALARS   # 898

STATE_CHANNELS = 2             # [A1 cevresi tehlike, A2 cevresi tehlike]
STATE_SCALARS = 8              # A1_row,A1_col,A2_row,A2_col,goal_row,goal_col,
                               # alive_bits, t/max
STATE_DIM = STATE_CHANNELS * PATCH_SIZE * PATCH_SIZE + STATE_SCALARS   # 890

# ---------------------------------------------------------------- ag mimarisi
CNN_CHANNELS = (16, 32)
CNN_POOL_SIZE = 4
SCALAR_EMBED = 128

# ---------------------------------------------------------------- yollar
RUNS_DIR = "runs"
RISK_CACHE = "runs/risk_dist.npy"      # Dijkstra risk-mesafe haritasi onbellegi
ZONE_CACHE = "runs/zone_map.npy"

# MARL-Pathfinding checkpoint'lerinin bulundugu klasor (--resume-from icin).
PATHFINDING_CKPT_DIR = r"..\MARL-Pathfinding\runs\ckpt"


def summary() -> str:
    """Kalibrasyonun tek bakista dogrulanmasi icin — train.py basinda basilir."""
    opt = (GRID_N - 1) * 2
    return "\n".join([
        f"grid            : {GRID_N}x{GRID_N} (1 hucre = 1 birim)",
        f"B -> H          : {START} -> {GOAL}, optimal {opt} adim, limit {MAX_STEPS}",
        f"radarlar        : {RADARS}  dis +-{OUTER_HALF}, ic +-{INNER_HALF}",
        (f"risk modeli     : per_entry — GIRIS basina tek zar, "
         f"dis %{P_OUTER_TOTAL*100:.0f}  ic %{P_INNER_TOTAL*100:.0f} "
         f"(surede birikme YOK)"
         if HAZARD_MODE == "per_entry" else
         f"risk modeli     : per_step (ABLATION) — adim basi "
         f"dis {P_DEATH[1]:.6f}  ic {P_DEATH[2]:.6f}"),
        f"gamma           : {GAMMA}  (etkin ufuk {1/(1-GAMMA):.0f} adim)",
        f"shaping         : COEF={SHAPING_COEF}, adim basi net "
        f"~{SHAPING_COEF*(1/opt - (1-GAMMA)*0.5):.4f} (|R_STEP|={abs(R_STEP)})",
        f"odul            : adim {R_STEP}, olum {R_DEATH}, "
        f"ilk varis +{R_FIRST_GOAL}, ikinci +{R_SECOND_GOAL}",
        f"gozlem          : {OBS_DIM} = 2x{PATCH_SIZE}x{PATCH_SIZE}(stride "
        f"{PATCH_STRIDE}, +-{PATCH_RADIUS*PATCH_STRIDE} hucre) + {N_SCALARS} skalar",
        f"alarm kuplaji   : {'ACIK' if ALERT_ENABLED else 'kapali'}",
    ])


if __name__ == "__main__":
    print(summary())
