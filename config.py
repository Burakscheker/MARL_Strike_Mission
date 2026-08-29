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

# SABIT harita (Asama 1-10 regresyonu icin duruyor). Merkezler (row, col),
# Burak'in verdigi (x,y)'lerden turetildi: R1 (-280,220) R2 (200,100) R3 (-100,-280)
RADARS = ((280, 220), (400, 700), (780, 400))

# RASTGELE HARITA (Asama 11, Burak 2026-08-06): "her episode farkli bi haritada
# olsun trainde de testde de", merkezler uniform, CAKISMA SERBEST
# ("merkezleri arasinda 5 fark olabilir, ustuste biner alanlari").
#
# 40 -> 30 RADAR (Burak, 2026-08-08). Olculdu (30 held-out harita, 160/100):
#   40 radar -> guvenli %39.0  ic halka %32.0  oracle ort 0.5556  takim ~%65
#   30 radar -> guvenli %48.8  ic halka %25.1  oracle ort 0.6977  takim ~%91
# H'nin bir halkanin icinde kalma orani da %38 -> %20'ye dustu (o haritalarda
# hedefe varmak icin ic halkaya girmek ZORUNLU oldugu icin tavani cakiyordu).
# Problem hala gercek planlama gerektiriyor: naif merdiven 0.0005, yani
# taban ile tavan arasinda ~1400 kat var.
#
# DIKKAT — KIYASLANABILIRLIK: 40 radarda alinan 3 tohumlu sonuc
# (VDN %5.59 > IQL %2.16 > QMIX %0.47) bu degisiklikten SONRAKI kosularla
# KARSILASTIRILAMAZ. Harita zorlugu ve eval seti degisti; o tablo "40 radar"
# etiketiyle arsivde kalir.
RADAR_RANDOM = True
# 30 -> 25 (2026-08-19, Burak): VDN'in 8000-adim/tohum1 sonucu sonrasi radar
# sayisi geri dusuruldu. ms3ks1_vdn8k_r25 checkpoint'i BU deger (25) ile
# egitildi/degerlendirildi — 30'a geri donersek checkpoint uyumsuz haritada
# olculur (daha zor, daha kotu sonuc cikar, YANLIS kiyas).
N_RADAR = 25

# Curriculum: erken egitimde seyrek harita (bol pozitif ornek), sonra yogun.
# Olculdu (baselines/scan_random_maps): 10 radarda oracle tavani %92.5 ve
# medyan %100. Yogun uctan SIFIRDAN baslamak neredeyse hic basarili episode
# gormemek demek. END degeri N_RADAR ile birlikte guncellenmeli.
CURRICULUM_RADAR_START = 10
CURRICULUM_RADAR_END = 25
CURRICULUM_FRAC = 0.6          # egitimin bu kesrinde END'e ulasir

# DEGERLENDIRME her zaman N_RADAR'da ve SABIT tohumlu ORTAK harita setinde.
# Egitim haritalari bu tohumlardan uretilmez (bkz. asagidaki ayrim kurali) —
# ezberlenecek bir sey olmamasi icin sart.
EVAL_N_MAPS = 100
EVAL_SEED_BASE = 900_000_000   # egitim tohumlari 0..1e8 araliginda kalir
TRAIN_SEED_MAX = 100_000_000

# Risk-mesafe haritasi fast-sweeping tur ustu siniri. Yogun/dolambacli
# haritalar 40'a sigmiyor (olculdu), 120 bol tampon.
MAP_MAX_ITER = 120

# Halka yariciplari (hucre). 160x160 -> +-80, 100x100 -> +-50.
# |d| <= HALF kullaniliyor: kenar 2*HALF+1 = 161 / 101 hucre.
#
# 220/140 -> 160/100 KUCULTMESI (Burak, 2026-08-06: "160'a 100 yap, mecbur").
# GEREKCE (olculdu, baselines/scan_random_maps): 40 radar x 221^2 = 1.95M hucre,
# grid 1M hucre — yani radar alani gridin IKI KATI. Sonuc: haritanin sadece
# %17.7'si guvenli, %52.2'si IC halka ve oracle tavani %17.6 (medyan %9.3).
# Yani MUKEMMEL politika bile episode'larin %70'inde basarisiz gorunuyordu;
# uc algoritmayi ayirt etmek icin gereken sinyal gurultunun altinda kaliyordu.
#   40 radar, 220/140 -> guvenli %17.7   oracle %17.6
#   40 radar, 160/100 -> guvenli %38.6   oracle %63.4
#   40 radar, 120/76  -> guvenli %57.8   oracle %87.3
# 160/100 orta yol: harita hala yogun ve gercek yol planlamasi gerektiriyor
# (guvenli hucre %38.6, serbest alan perkolasyon esigi ~%59'un ALTINDA degil,
# yani koridorlar var ama bulunmasi gerekiyor), ama tavan olculebilir bir
# bolgeye cikiyor. Radar SAYISI (40) ve rastgelelik AYNEN korunuyor.
#
# Otomatik uyum: P_DEATH asagida 2*HALF+1 caprazlama adimindan turedigi icin
# adim basi hazard kendini yeniden kalibre eder; PATCH_STRIDE=16 ile gozlem
# penceresi +-160 hucre gordugu icin 161 genisligindeki dis halkanin TAMAMI
# hala tek karede goruluyor.
OUTER_HALF = 80
INNER_HALF = 50

# Adim limiti. Optimal yol = manhattan(B,H) = 1998 adim.
#
# 2800 -> 4000 -> 3000 (2026-08-08). Once 2800'un ogrenilmis (optimal
# olmayan) rotalar icin DAR oldugu olculdu (r30s0_vdn, 50 held-out harita,
# AYNI checkpoint, yeniden egitim YOK):
#   MAX_STEPS 2800 -> surv_ratio %18.3  VARIS %48  timeout %38  (rota 2697 adim)
#             4000 -> surv_ratio %29.3  VARIS %64  timeout %36
#             6000 -> surv_ratio %32.0  VARIS %72  timeout %28
# 4000 tatli nokta olarak secilmisti. Sonra Burak: "Tolga'nin projesinde
# MAX_STEPS 3000, bizde de oyle olsun" -- iki proje arasi kiyaslanabilirlik
# icin 3000'e cekildi.
#
# 3000 -> 8000 (2026-08-08..24). Sonraki MAX_STEPS taramasinda (2800/3000/
# 4000/6000/8000/10000/12000) 8000 yerel optimum bulundu ve BUTUN GPU/
# paralel-rollout kosulari (bu oturumun tamami) `--max-steps 8000` CLI
# bayragiyla calisti — ama bu SABIT (config.MAX_STEPS) hep 3000'de KALDI.
#
# BULUNAN KRITIK BUG (2026-08-25, dis inceleme + dogrudan dogrulandi):
# tests/test_env.py'deki intihar-kapisi testi C.MAX_STEPS (3000) uzerinden
# kontrol ediyordu, GERCEKTE kullanilan 8000 degil. Hesap:
#     MAX_STEPS=3000: intihar=-100  timeout=-80   KAPALI (guvenli)
#     MAX_STEPS=8000: intihar=-100  timeout=-130  ACIK — intihar 30 puan
#                     DAHA KARLI CIKIYORDU
# Yani bu oturumdaki TUM egitimler (transfer/scratch/batch128 dahil) bu
# acik kapiyla kosmus olabilir — gozlenen surekli yuksek olum orani
# (olu(ma) 1.2-2.0) ve kaotik osilasyonun bir kismi BUNDAN kaynaklanmis
# olabilir. MAX_STEPS artik GERCEKTEN kullanilan degere (8000) cekildi ki
# testler CLI'dan BAGIMSIZ olarak gercek konfigurasyonu dogrulasin;
# R_ALL_DEAD asagida buna gore yeniden kalibre edildi.
#
# 8000 -> 4000 (2026-08-28, Burak'in istegi): 4-algoritma (VDN/QMIX/MAPPO/
# HAPPO) kiyasini HIZLANDIRMAK icin — kotu bir politika bile en fazla 4000
# adimda timeout yer, MARL-pathtfinding referansiyla ayni deger. Intihar
# kapisi (Kapi 2) YENIDEN KONTROL EDILDI: 2*R_DEATH+R_ALL_DEAD=-140 <=
# R_TIMEOUT+MAX_STEPS*R_STEP=-50+4000*(-0.01)=-90 — HALA saglaniyor (marj
# 10'dan 50'ye CIKTI, daha guvenli), R_ALL_DEAD'e dokunmaya GEREK YOK.
MAX_STEPS = 4000

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

# SIKISMA CEZASI (2026-08-20, Burak): "cok kolay" (risksiz) haritalarda bile
# ajan BASLANGICIN hemen yaninda TIKANIP KALIYORDU — fiziksel engel yokken
# bile 8000 adimin tamamini ilerlemeden harciyordu (bkz. viz/plot_easy_fail.py
# ciktisi). R_STEP zaten var olan (kucuk) bir "burada durmanin bedeli var"
# sinyaliydi ama yeterince keskin degildi. Bu, START'in STUCK_BOX x STUCK_BOX
# kutusu icinde STUCK_GRACE_STEPS'ten fazla kalinirsa HER ADIM ek bir ceza —
# temkinliligi (loitering) doğrudan cezalandirir, kacinmayi degil.
STUCK_GRACE_STEPS = 30
STUCK_BOX = 10                 # START'tan itibaren satir/sutun genisligi
# STUCK_WINDOW: ceza SINIRSIZ birikmez — grace suresinden sonra en fazla bu
# kadar adim boyunca uygulanir (sonra durur). SINIRSIZ olsaydi 8000 adimlik
# bir episode'da toplam -0.5*7970*2 = ~-8000'e kadar cikar, R_TIMEOUT (-50)
# / R_ALL_DEAD (-70) / R_DEATH (-15) yaninda anlamsizca buyurdu ve odul
# olcegini bozardi (bu projede zaten olcek/sapma sorunlari yasadik).
STUCK_WINDOW = 200
R_STUCK = -0.5                 # R_STEP'in 50 kati, pencere icinde HER ADIM

# --- ODUL HACKLEME KAPILARI (Strike_Mission.md §11.8) ---------------------
# Asagidaki iki deger BIRLIKTE ayarlanir; tek basina degistirmek bir acik
# acar. Rastgele haritada tavan dustugu icin (medyan oracle %7.2) ajanin
# "hic denememek" ve "hemen olmek" gibi dejenere cikislari kârli hale
# gelebiliyor. Kapatilmasi gereken iki kapi:
#
# KAPI 1 — OYALANMA. R_TIMEOUT -10 iken guvenli bir kosede dolanip sureyi
# doldurmak, ucup olmekten UCUZDU (olculdu: oyalanmak -38, ucmak -65).
# Ajan "hic deneme" ogrenirdi. -50'ye cekildi.
R_TIMEOUT = -50.0              # hicbiri varmadan sure doldu
#
# KAPI 2 — INTIHAR. Kapi 1'i kapatmak yenisini aciyor: umudu kesen ajan icin
# artik EN UCUZ cikis kasten bir ic halkaya ucup episode'u erken bitirmek olur.
#
# ILK DENEMEM YANLISTI ve ajan bunu 600 episode'da BULDU. "2*R_DEATH +
# R_ALL_DEAD <= R_TIMEOUT" yazmistim (-55 <= -50) ve kapali sandim; ADIM
# MALIYETINI hesaba katmamistim. Erken olen ajan kalan adimlarin maliyetinden
# de KURTULUYOR — 2800 adimlik bir episode'da bu -28 puan.
# OLCULDU (ayni harita, scripted politikalar):
#     INTIHAR  getiri -53.85 (267 adim, 2 olu)
#     OYALAN   getiri -77.92 (2800 adim, timeout)
#   -> intihar 24 puan KARLI. Egitimde de tam bu gorundu: adim ortalamasi
#      2529'dan 446'ya dustu, olum 2.00'a cikti, basari %0'da kaldi.
#
# DOGRU esitsizlik adim maliyetini icerir (en kotu durum: t=0'da olmek):
#     2*R_DEATH + R_ALL_DEAD  <=  R_TIMEOUT + MAX_STEPS * R_STEP
#     2*(-15) + (-50) = -80   <=  -50 + (-28) = -78        -> KAPALI
#
# Ilerlerken olmek yine serbest: shaping yol boyunca toplandigi icin
# ("deneyip yolda olen" ~ +120*ilerleme) haritayi kat eden ajan hic
# denemeyenden cok daha yuksek puan alir. Cezalandirilan sey KASTEN erken
# bitirmek.
# -70 -> -110 (2026-08-25): MAX_STEPS 3000->8000 olunca (yukarida) bu kapi
# SESSIZCE ACILMISTI — -70 ile 2*(-15)+(-70)=-100, oyalanma ise
# -50+8000*(-0.01)=-130; -100 > -130 yani INTIHAR 30 PUAN KARLIYDI. Bu
# oturumdaki TUM 8000-adim kosulari (transfer/scratch/batch128) bu acik
# kapiyla calisti. -110 ile: 2*(-15)+(-110)=-140 <= -130 -> KAPALI, 10
# puan marj (MAX_STEPS 4000'deki stilin ayni orani). MAX_STEPS bir daha
# degisirse bu esitsizlik tekrar kontrol edilmeli (tests/test_env.py
# otomatik dogruluyor — ama SADECE calisirken kullanilan C.MAX_STEPS icin;
# CLI --max-steps ile FARKLI bir deger geciyorsan bu testi KORUMAZ, bkz.
# yukaridaki MAX_STEPS notu).
R_ALL_DEAD = -110.0            # ikisi de dusuruldu (R_DEATH'lerin USTUNE)

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
#
# 7.5 -> 15.0 GERI ALINDI (2026-08-06, olcume dayali — §11.11).
# "Iki kez sayiliyor" argumani, IKI TERIMIN DE ogreniciye ulastigini
# varsayiyordu. Olculdu: ULASMIYOR.
#   - Stokastik R_DEATH terminal/seyrek gecislerde: tamponun ~%0.2'si, ve o
#     satirlarda TD hatasi buyuk oldugu icin Huber gradyani +-1'e kirpiyor.
#     R_ALL_DEAD'i 200 KATINA cikarmak trajektoriyi degistirmedi — yani bu
#     kanal pratikte SESSIZ.
#   - Yogun R_RISK_COEF terimi ise HER tehlikeli adimda odeniyor ve TD
#     hatalari kucuk (medyan 0.02, p99 0.70) => karesel bolgede, buyuklugu
#     gradyana TAM giriyor.
# Yani risk bilgisini tasiyan tek gercek kanal YOGUN olan. Onu yariya
# indirmek, ajanin gordugu riski yariya indirmekti.
# 15.0 = |R_DEATH| ile yogun kanal beklenen olum maliyetinin TAMAMINI tasir.
# Ablation olarak 0.0 (sadece stokastik) ve 7.5 de kosulmali.
# 15.0 -> 65.0 (2026-08-08). OLCULDU: ajan rota basina 2.8 IC HALKA girisi
# yapiyor, oracle 0.2. Her giris %90 olum zari -> hayatta kalma 0.026 vs 0.63.
# Sebep FIYATLAMA: ic halkaya girmenin ajana maliyeti R_RISK_COEF*0.9 = 13.5,
# hedefe varmanin odulu +50. Yani uc halkayi delip gecmek MATEMATIKSEL OLARAK
# KARLI (-40.5 vs +50). Ajan dogru hesapliyor, biz yanlis fiyatlamisiz.
# Gercek bedel sadece olum cezasi degil, KAYBEDILEN GOREV ODULU de:
#     p * (|R_DEATH| + R_FIRST_GOAL) = 0.9 * (15 + 50) = 58.5
# yani 13.5 degil 58.5 olmaliydi -> 4.3 kat dusuk fiyatlamisiz.
# 65 = |R_DEATH| + R_FIRST_GOAL.
#
# DIKKAT: RISK_W = |R_RISK_COEF/R_STEP| oldugu icin bu deger risk-mesafe
# haritasini (shaping potansiyeli) VE oracle referansini birlikte degistirir:
# 1500 -> 6500. Oracle daha cok dolasir, yani tavan da degisir; eski
# surv_ratio sayilariyla dogrudan kiyaslanamaz.
R_RISK_COEF = 65.0

# GEREKSIZ RISK CEZASI (2026-08-21, Burak'in gozlemi): basarisiz haritalari
# tek tek izlerken ucaklarin bos/guvenli alanda "durup dururken" bir halkaya
# girdigi goruldu. viz/analyze_unforced_deaths.py ile OLCULDU: %40
# checkpoint'te (ms3ks1_vdn8k_r25) 100 haritada 88 olumun 71'i (%81) GIRIS
# anindaki eski pozisyondan zonu ARTIRMAYAN GERCEK bir alternatif yon
# varken oldu (46'si dogrudan tamamen guvenli bolgeden). Yani cogu olum
# haritanin zorlamasi degil, politikanin GEREKSIZ tercihi.
#
# R_RISK_COEF'i buyutmek (120 denendi) BLANKET bir mudahale oldugu icin
# kolay haritalarda bile cökuse yol acti (§ yukarida). Bu ceza ONA GORE
# FARKLI: SADECE "kacinilabilir giris" anini hedefler (bkz.
# StrikeMissionEnv._unnecessary_entry) — haritanin geregi olan zorunlu
# geciscleri etkilemez, cunku alternatif yoksa hic tetiklenmez.
#
# Buyukluk: mevcut R_DEATH(-15)/R_TIMEOUT(-50) ile ayni mertebede secildi,
# dis halka girisini (13 -> 33) hem ic halka girisini (58.5 -> 78.5)
# belirgin sekilde daha maliyetli yapar. HENUZ EGITILMEDI/OLCULMEDI —
# ODUL GORUNURLUGU notundaki (yukarida) Huber-doygunlugu riski gecerli:
# bu da nispeten SEYREK bir olay (ep basi ~0.7-0.9), R_ALL_DEAD'in 200x
# buyutulup TRAJEKTORIYI HIC degistirmedigi durumla ayni kaderi paylasabilir.
# Egitim sonrasi ayni script ile (once/sonra) kacinilabilir-giris orani
# olculup DOGRULANMALI.
#
# 20.0 -> 0.0 (2026-08-29, gece throughput/skor dongusu, it3): fast-eps
# (--eps-start 0.1) kurulumuyla ajanin sorunu ARTIK "gereksiz risk alma"
# DEGIL, TERSI — asiri temkin (olu 0.39/ep ama timeout %35, rota meandering,
# team %69'da takili). Bu ek +20 ceza ring girisini daha da caydiriyor,
# yani su anki basarisizlik moduyla TERS yonde itiyor. Hafiza notu zaten
# "net iyilesme SAGLAMADI ... istenirse kaldirilabilir" diyordu. 0.0'a
# cekildi; it3 testinde peak eval %69'u geciyor mu diye olculecek.
# GERI ALINABILIR: sadece bu satiri 20.0'a dondur.
R_UNNECESSARY_RISK = 0.0

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

# n-ADIM GETIRI DENENDI VE KALDIRILDI (2026-08-17): olculdu, uzun egitim
# politikayi BOZUYORDU (VDN surv_ratio ep1000 %4.23 -> ep3000 %2.80, QMIX
# %0.00 -> %0.00) ve bozulma epsilon DUSERKEN artiyordu (kesiften degil
# ogrenmeden). Kullanilmayan (varsayilan kapali) ve zararli oldugu olculen
# kod tutulmadi — bkz. git gecmisi (agents/nstep.py).
GRAD_CLIP = 10.0
HIDDEN = 128
# LEARN_EVERY=16 + batch=64 DENENDI VE GERI ALINDI (2026-08-08).
# Hipotez: cagri sayisini yariya indirip batch'i ikiye katlarsak ayni ornek
# akisini korur, Python/optimizer YUKUNDEN kazaniriz.
# OLCULDU (ayni ortam, 4 episode, tam egitim dongusu):
#   LEARN_EVERY=8,  batch=32  ->  1.447 ms/adim
#   LEARN_EVERY=16, batch=64  ->  1.486 ms/adim
# Kazanc YOK, hafif kotu. Cunku learn() maliyetini cagri yuku degil BATCH
# HESABI belirliyor; 2x batch, 1/2 cagri = sabit toplam hesap.
# Karsiliksiz risk (buyuk batch = az gradyan gurultusu, ogrenme dinamigi
# degisir), o yuzden eski degerlere donuldu.
LEARN_EVERY = 8

# --- ODUL GORUNURLUGU (Strike_Mission.md §11.11) --------------------------
# OLCULDU: R_ALL_DEAD'i -25'ten -5000'e cikarmak (200 kat!) egitim
# trajektorisini HIC degistirmedi. Yani odul BUYUKLUGU gradyana girmiyordu.
# Iki etki ust uste biniyor:
#   1) smooth_l1_loss |TD hatasi| > beta'da gradyani +-1'e sabitler —
#      buyuklugu degil sadece ISARETI tasir. Terminal odullerimiz -50..-5000,
#      yani daima doygun bolgede.
#   2) LR 3e-5 x gradyan tavani 1 -> Q tek guncellemede en fazla 3e-5 hareket
#      eder. Q hedefe (~+-100) hic yaklasamiyor, iki hedef arasindaki fark
#      materyalize olmuyor.
#
# REWARD_SCALE: ogrenicinin gordugu TUM odulleri carpar (ortam ciktisinda,
# tek yerde). Amac Q hedeflerini O(1-5) araligina indirmek — hem Huber'in
# karesel bolgesine sokar hem de LR'in Q'yu makul surede oraya tasimasini
# saglar. config'deki odul degerleri INSAN OLCEGINDE kalir (kapi aritmetigi
# §11.8'de oldugu gibi okunabilir olsun diye).
#
# HUBER_BETA: karesel bolgenin genisligi. beta buyudukce buyukluk daha genis
# bir aralikta tasinir (gradyan = hata/beta), ama gradyan TAVANI yine 1 —
# yani beta tek basina LR sorununu COZMEZ. Ikisi ayri ayri ve birlikte
# olculdu; secim tests/test_reward_visible.py'nin sonucuna gore yapildi.
# SECIM (olcume dayali, 2026-08-06): IKISI DE NO-OP birakildi.
# Gerekce: TD hatasi dagilimi olculdu -> medyan 0.02, p90 0.05, p99 0.70,
# yani gecislerin %99'u ZATEN Huber'in karesel bolgesinde. Doygunluk sadece
# kuyrukta (terminal gecisler, max 21.7). Iki knob da bu kuyruğu duzeltirken
# YOGUN sinyali zayiflatiyor:
#   REWARD_SCALE=0.05 -> shaping gradyani 20 kat kuculur
#   HUBER_BETA=50     -> gradyan = hata/beta, tipik hatada 50 kat kuculur
# Yani "nadir terminal odulu gorunur kilmak" icin "her adimda ogrenilen
# sinyali" feda etmek olurdu. Dogru cozum bu degil (bkz. R_RISK_COEF notu):
# riski SEYREK/stokastik kanaldan degil YOGUN kanaldan tasimak.
# 1.0/1.0 -> 0.05/50.0 DENENDI VE GERI ALINDI (2026-08-21/22). Dogrudan
# olculdu (tests/test_reward_visible.py, beta=2,5,10,25 eklendi): SADECE
# HUBER_BETA'yi buyutmek odulu gorunmez birakiyordu; SADECE REWARD_SCALE=
# 0.05+HUBER_BETA=50 BIRLIKTE odul buyuklugunu gorunur kildi. Ama bunu telafi
# etmek icin gereken VDN_LR artisi (asagida ayrintili) UC AYRI tam 500-episode
# kosuda da basarisiz oldu: LR=1e-2 pervasiz olum (olu(ma) 2.00'a kilitlendi,
# gorev en iyi 0.10), LR=1e-3 donuk/timeout (gorev en iyi 0.045, 10 evaldan
# 8'i 0.0000). Iki UC LR'de iki TAMAMEN ZIT basarisizlik modu — dar aralikta
# "dogru" bir LR olmadiginin guclu isareti. Eski %40 checkpoint'in tarifine
# (1.0/1.0) GERI DONULDU.
REWARD_SCALE = 1.0             # 1.0 = kapali
HUBER_BETA = 1.0               # torch varsayilani

# SHAPING_COEF: hedefe dogru bir adimin shaping sinyali |R_STEP|'in ~5 katı
# olsun (MARL-Pathfinding'de olculmus dogru oran: en zayif sinyal "hedefe git"
# olmamali). Phi=0.5'te net sinyal = COEF*(0.0005 - 0.0001) = COEF*0.0004.
# 0.05 / 0.0004 = 125 -> 120 secildi.
SHAPING_COEF = 120.0

EPS_START, EPS_END = 1.0, 0.05
EPS_FLOOR_FRAC = 0.5           # epsilon egitimin bu KESRINDE tabana iner

TRAIN_HARM_WINDOW = 100
TRAIN_HARM_LOG_EVERY = 25
DEMO_EPISODES = 10
DEMO_SEED = 777

# ---------------------------------------------------------------- VDN (Asama 5)
# DIKKAT: LR ve TARGET_UPDATE degerleri MARL-Pathfinding'de 3 ayri tam-olcekli
# kosuyu cokerttikten sonra bulundu (1e-4 + 2000 -> ep~1750'de tepe yapip
# cokme; 3e-5 + 4000 -> 12000 episode monoton yukselis, hic cokme yok).
# Degistirmeden egitme — ISTISNA asagida (REWARD_SCALE/HUBER_BETA degisimi
# icin BILEREK degistirildi).
VDN_EPISODES = 2_000
# BELLEK: joint satir = 4 x OBS_DIM x 4 byte = 14.4 KB (obs1,obs2 + next'leri).
# 250k satir = 3.6 GB.
#
# 120k -> 250k, 32 -> 128 (2026-08-24, Burak: "batch/buffer buyuklugunu
# artir"). GEREKCE: ince-tarama (eval_every=25) ile GORULEN sey (bkz.
# yukaridaki VDN_EVAL_EVERY notu) "erken zirve, yavas cokme" degil, HER 25
# episode'da 0.00-0.40 arasi KAOTIK ZIPLAMA + her tohumda ep500'de cokusdu.
# Bu, gradyan tahmininin YUKSEK VARYANSLI oldugunun klasik belirtisi —
# batch=32 kucuk bir orneklemden TD hedefi tahmin ediyor, her learn()
# cagrisi farkli (gurultulu) bir yone iteliyor olabilir. Buyuk batch bu
# varyansi azaltir (istatistiksel olarak ~sqrt(4)=2x daha az gurultu).
# VDN_BUFFER da orantili buyutuldu — n_envs=32 paralel toplama buffer'i
# eskisinden HIZLI dolduruyor, daha genis bir pencere daha CESITLI (daha az
# birbirine bagimli) ornekleme saglar.
VDN_BUFFER = 250_000           # ortak (joint) transition — her satir BIR global timestep
VDN_BATCH = 128
VDN_EPS_DECAY_STEPS = 2_000_000
VDN_LEARN_START = 2_000
# 3e-5 -> 1e-2 DENENDI VE GERI ALINDI (2026-08-22): 25-episode kisa prob
# temiz gorunmustu (loss max 1.14, q_max sinirli) ama TAM 500 episode'da
# KATASTROFIK cikti — olu(ma) ep300'den itibaren 2.00'a KILITLENDI (HER
# episode'da iki ucak da olduruyor), en iyi mission_prob sadece 0.10 (eski
# %40 checkpoint'in 0.34'unun cok altinda). VARIS (zar kapali rota) %78-98
# gibi YUKSEK gorunuyordu ama YANILTICIYDI — ajan rotayi "biliyor" ama zar
# acikken pervasizca riske giriyordu. Kisa prob'un "aninda patlamiyor"
# sonucu "uzun vadede stabil" ANLAMINA GELMIYORMUS — tam da yukarida
# uyarilan risk gerceklesti.
#
# 1e-2 -> 1e-3 DE DENENDI VE GERI ALINDI (2026-08-22): 25-episode prob temiz
# gorunmustu (loss max 0.058) ama tam 500 episode'da TAM TERSI yonde
# basarisiz oldu — bu sefer politika DONDU/timeout yedi (10 evaldan 8'i
# gorev=0.0000, demo'da 10 episode'un 6'si 8000 adimin tamamini
# kullanip hicbir sey yapmadan bitiyordu). Iki uc LR de basarisiz, ikisi de
# ZIT yonlerde (pervasizlik vs donukluk) — REWARD_SCALE/HUBER_BETA degisimi
# TAMAMEN TERK EDILDI, eski (1.0/1.0) degerlere donuldu (yukarida).
# 3e-5'e GERI DONULDU — eski %40 checkpoint'in (ms3ks1_vdn8k_r25) tarifi.
VDN_LR = 3e-5
VDN_TARGET_UPDATE = 4_000

# OGRETMEN-CAPASI (teacher-anchored VDN, 2026-08-26, dis inceleme onerisi):
# vanilla BC->TD fine-tune (train_bc.py cikisindan --resume-from ile devam)
# COKTU — 25 episode'da mission_prob 0.0001'e dustu (BC-oncesi: %42 takim,
# %98 VARIS). Sebep: BC cross-entropy'yle egitildigi icin cikis olcegi
# gercek Q-degerleriyle (ort. 17-37, aksiyon-farki 0.05-0.1) uyumsuz;
# normal TD guncellemesi bu olcegi ilk birkac yuz adimda yeniden kalibre
# ederken BC'nin ogrendigi siralamayi siliyordu. Cozum: TD kaybina SUREKLI
# oracle-capraz-entropi eklemek (agents/vdn.py VDNAgent.learn), lambda
# baslangicta baskin, egitim ilerledikce azalir ama SIFIRA INMEZ (taban
# > 0 -- BC'nin ogrettigi guvenli rota bilgisi RL guncellemeleri tarafindan
# TAMAMEN silinmesin diye).
VDN_BC_LAMBDA_START = 1.0
VDN_BC_LAMBDA_END = 0.1
VDN_BC_LAMBDA_DECAY_FRAC = 0.5   # EPS_FLOOR_FRAC ile ayni ritimde taban deger

# DUELING + ONCELIKLI DENEYIM TEKRARI (PER, Schaul ve ark. 2016 / Wang ve
# ark. 2016) — 2026-08-26, SAF RL denemesi (BC/oracle YOK, uc ayri BC->RL
# fine-tune denemesi basarisiz oldu). Motivasyon: cokme deseninde HER
# SEFERINDE Q'nun MUTLAK OLCEGI (V) degisirken aksiyonlar ARASI SIRALAMA
# (A, action-gap 0.05-0.1) siliniyordu (bkz. §11.14 q_gap notu) — Dueling
# ikisini mimari olarak ayirir (agents/networks.py). Ayrica 250k uniform
# buffer'da NADIR ama KRITIK gecisler (olum, varis, riskli giris) binlerce
# siradan adima seyreliyor — PER bunlari TD-hatasi buyuklugune gore daha
# sik ornekler (agents/vdn.py SumTree).
PER_ALPHA = 0.6            # oncelik ussu (0=uniform, 1=tam-orantili)
PER_BETA_START = 0.4       # importance-sampling agirligi baslangici
PER_BETA_END = 1.0         # egitim sonunda ONYARGISIZ (tam duzeltilmis)
PER_EPS = 1e-3             # sifir-hatali gecisler icin taban oncelik (asla 0 olmasin)
# DENEY (2026-08-16) DENENDI VE GERI ALINDI: q_mean hicbir egitimde
# duzlemiyor, surekli artiyor (3.6->19.4 ilk 1000ep, devaminda 24.5'e) ve
# loss da AYNI YONDE artiyor. Hipotez: VDN_TARGET_UPDATE=4000 HAM ADIM,
# ortalama episode ~1700-2800 adim surdugu icin target ~1.5-2 EPISODE'DA
# BIR tam senkron oluyor — bu asiri sik, "sabit hedef" gecikmesini
# neredeyse sifirliyor sanildi. Sert+seyrek yerine yumusak (Polyak,
# tau=0.005, her learn() adiminda) denendi — SONUC DAHA KOTU: q_mean DAHA
# HIZLI firladi (ep625'te 35 vs sertte ep825'te 19), VARIS ep250=90->32,
# ep500=72->0. NEDEN: tau=0.005'in yari-omru ~139 ogrenme adimi
# (ln(2)/tau), sert senkronun ORTALAMA bayatligi (~250 adim, senkronlar
# arasi donuk kalarak) daha bile UZUNDU. Yani "target'e daha cok gecikme
# ekliyorum" sanip DAHA AZ gecikme eklemis olduk — target hic durmadan
# online'i kovaliyor, TD regresyonunun ihtiyac duydugu SABIT referans
# ortadan kalkiyor. Sert senkrona geri donuldu; ıraksama kabul edilip
# mission_prob-birincil checkpoint secimiyle en iyi nokta yakalaniyor.
# 250 -> 50 -> 100 -> 50 (2026-08-21, Burak): VDN erken zirve yapip sonra
# bozuluyor, 250'lik aralikta sadece 2 eval noktasi gercek zirveyi
# KACIRABILIR. Once 50'ye cekildi, ama evaluate() TEK-ortam/batch=1
# yolunu kullandigi icin PAHALI cikti (eval basina ~16 dk GPU'da, n_envs=32
# egitimle ayni surecte) — 100'e geri cekilmisti. Sonra evaluate() de
# vdn_vec_evaluate() ile PARALEL/batch'li hale getirildi (bkz. train.py) —
# OLCULDU: eval ~16 dk -> ~3.9 dk (4.1x). Bu maliyeti dusurdugu icin 50'ye
# GERI DONULDU: 10 eval noktasi x ~4dk ≈ 40 dk ek yuk, toplam kosu ~1.8
# saat — 250'in 5 kati cozunurluk, makul maliyetle.
#
# 50 -> 25 (2026-08-22, Burak): ms5_vdn8k_gpu32 kosusunda egitim-ici
# takim(ma) ep275'te %59'a cikmisti ama en yakin GERCEK (held-out) eval
# noktalari ep250 (%28) ve ep300 (%38) idi — arada ne oldugunu bilmiyoruz.
# eval artik ucuz oldugu icin (bkz. yukarida) o bandi daha ince taramak
# icin 25'e cekildi: 20 eval noktasi x ~4dk ≈ 80dk ek yuk, toplam ~2.5 saat.
#
# GUVENLIK NOTU (eval_every DEGERININ KENDISI artik egitim yorungesini
# ETKILEMIYOR): daha once "eval_every 250->50 degistirince sonuclar tamamen
# degisti" diye BULUNMUSTU ama sebep eval_every DEGILDI — o kosu YANLISLIKLA
# farkli thread sayisiyla da calismisti (MKL/OpenMP thread-sayisina bagli
# farkli yuvarlama -> farkli egitim yorungesi) VE evaluate() egitim
# env'inin RNG'sini SIZDIRIYORDU (bulunup duzeltildi, bkz. train.py
# evaluate() saved_rng notu). Simdi eval eps=0 kullandigi icin agent.rng'yi
# HIC tuketmiyor (eps>0.0 guard'i sayesinde) ve vdn_vec_evaluate() ayrı env
# nesneleri kullaniyor — evaluate() egitim yorungesine artik hicbir
# sekilde DOKUNMUYOR, sadece SURESI degisiyor.
VDN_EVAL_EVERY = 25

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
QMIX_EVAL_EVERY = 250
QMIX_MIXER_EMBED = 32

# ---------------------------------------------------------------- MAPPO / HAPPO
# 2026-08-28: euzxx/MARL-pathtfinding (mappo_happo dali) agents/ppo.py'den
# PORTLANDI. IQL bu degisiklikle KALDIRILDI — VDN/QMIX (off-policy, TD) yaninda
# artik MAPPO/HAPPO (on-policy, PPO) var. Mimari fark, VDN/QMIX'ten:
#   - Aktorler (agents/mappo_happo.py Actor'leri) build_qnet() ile AYNI CNN
#     govdesini kullanir — cikisi Q-degeri DEGIL, politika logit'i.
#   - TEK MERKEZI KRITIK (CentralCritic), env.state()'i (QMIX'in mixer'inin
#     kullandigi AYNI global gozlem) girdi alir — CTDE (centralized training,
#     decentralized execution).
#   - Off-policy replay YOK: ROLLOUT_EPISODES tam episode toplanir, GAE
#     hesaplanir, PPO_EPOCHS kez minibatch SGD yapilir, batch ATILIR.
# MAPPO/HAPPO farkı SADECE actor guncelleme sirasinda: HAPPO ajanlari
# RASTGELE sirayla, ONCEKI ajanin politika-orani carpanini (importance
# sampling factor) SIRADAKI ajanin advantage'ina carparak guncelliyor
# (sequential/monotonic-improvement garantisi, Kuba ve ark. 2021 HAPPO
# makalesi) — MAPPO'da bu carpan hep 1 (bagimsiz guncelleme).
GAE_LAMBDA = 0.95
PPO_ACTOR_LR = 1e-4
PPO_CRITIC_LR = 1e-4
PPO_CLIP_COEF = 0.2
PPO_EPOCHS = 5
PPO_MINIBATCH_SIZE = 256
PPO_ROLLOUT_EPISODES = 32      # bu kadar TAM episode -> 1 PPO guncellemesi
PPO_ENTROPY_COEF = 0.01
PPO_VALUE_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5

MAPPO_EPISODES = 2_000
MAPPO_EVAL_EVERY = 25
HAPPO_EPISODES = 2_000
HAPPO_EVAL_EVERY = 25

# ---------------------------------------------------------------- gozlem
# TRANSFER KISITI (Burak'in istegi: MARL-Pathfinding'in egitilmis modellerini
# resume et). O projede OBS_DIM = 2*21*21 + 16 = 898, STATE_DIM = 2*21*21 + 8
# = 890. CNNQNet'in parametre sayisi AdaptiveAvgPool2d sayesinde PATCH_SIZE'dan
# bagimsiz, ama scalar_enc'in ilk Linear'i N_SCALARS'a, head'in ilki
# (flat + SCALAR_EMBED + N_SCALARS)'a bagli. N_SCALARS=16 iken bu ikisi de
# BIREBIR transfer oluyordu. N_SCALARS=18'e cikinca (asagida, "son hareket"
# ozelligi) bu iki katman ARTIK TRANSFER OLMUYOR — konvolusyon katmanlari
# hala olur (bkz. asagidaki N_SCALARS notu).
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

# GLOBAL_PATCH_STRIDE (2026-08-25, dis inceleme onerisi): yerel kanal (stride
# 16) sadece +-160 hucre goruyor, oracle ise TUM haritayi. Ayni PATCH_SIZE
# (21x21) sekliyle ama COK DAHA GENIS aralikla ORNEKLENEN ikinci bir tehlike
# kanali — "coklu-olcek" gorus: 50 hucre arayla 21 ornek = +-500 hucre, yani
# 1000x1000 gridin YARISINI her yonde kapsiyor (harita ortasindaysa TAMAMINI).
# Tensor SEKLI ayni kaldigi icin agents/networks.py'ye HICBIR degisiklik
# GEREKMIYOR — sadece OBS_CHANNELS 2->3 oldu, CNNQNet zaten bu parametreden
# genel. Sinir disi (harita disi) HUCRELER icin ayni "0.0=guvenli" kurali
# (kanal 0 ile TUTARLI, mevcut haritanin kenarlarindaki davranisla ayni).
GLOBAL_PATCH_STRIDE = 50
PATCH_STRIDE = 16

OBS_CHANNELS = 3
# kanal 0 = YEREL TEHLIKE (stride 16, +-160 hucre) (0 guvenli, 0.5 dis halka,
#           1.0 ic halka). MARL-Pathfinding'deki "yasak bolge" kanaliyla AYNI
#           ROL: "buradan kacin, buyuk deger daha kotu". Ikili degil dereceli
#           olmasi transferi bozmuyor (ag zaten surekli girdi aliyor).
# kanal 1 = kendi izi (ziyaret edilmis hucreler) — MARL-Pathfinding ile BIREBIR
#           ayni anlam.
# kanal 2 = KURESEL TEHLIKE (stride 50, +-500 hucre) — YENI (2026-08-25).
#           Ayni tehlike haritasi, cok daha genis/kaba ornekleme. Amac:
#           ajanin "yerel olarak temiz ama ileride yogun bir radar kumesi
#           var" durumunu ONCEDEN gormesi — oracle'in sahip oldugu kuresel
#           bilginin sadelestirilmis bir yaklasimi. MARL-Pathfinding'de
#           KARSILIGI YOK, transfer edilirken bu kanal HER ZAMAN sifirdan
#           baslar (agirlik sekli zaten degisti, ayrica sorun degil).

# 16 -> 18 DENEY (2026-08-20, Burak): "cok kolay" (risksiz) haritalarda ajan
# baslangicin hemen yaninda TIKANIP KALIYORDU — 8000 adimin tamamini
# kullanip ilerlemeden (bkz. viz/plot_easy_fail.py, runs/fig_easy_fail.png).
# Acik alanda fiziksel engel YOK, yani bu bir SALINIM/kararsizlik belirtisi.
# PATCH_STRIDE ile "kendi izini daha net gor" denendi, BASARISIZ oldu (menzil
# kaybi zarari asti). Bu sefer DAHA DOGRUDAN bir sinyal: ajanin SON ATTIGI
# ADIM (16-17: son hareketin dr,dc'si, DIRS olceginde -1/0/1). Salinim
# onleme (ayni yonu tersine cevirmeme) icin standart bir teknik.
#
# UYARI (checkpoint uyumu): N_SCALARS degistigi icin scalar_enc'in ilk
# katmani VE head'in ilk katmani (skip-baglantisi n_scalars alir) MARL-
# Pathfinding'in / eski ms3ks1_vdn8k_r25 checkpoint'inin agirliklariyla
# SEKIL UYUŞMUYOR — o iki katman artik transfer OLMAYACAK (agents/transfer.py
# bunu rapor eder, sessizce atlamaz). Konvolüsyon katmanlari (asil
# gorsel/navigasyon bilgisi) etkilenmez, hala transfer olur.
# EYLEM-OZGU ANLIK RISK (2026-08-27, dis inceleme onerisi — BC/PER/Dueling
# hattinin UCU DE elendikten sonra): mevcut 12-15 slotlari "bu yone gitmek
# HEDEFE mesafeyi ne kadar kisaltir" der (risk-mesafe haritasindan, yani
# GELECEKTEKI beklenen risk dahil, TEK bir bilesik sayi) ama "bu yone
# girersem BU ADIMDA ne kadar tespit/olum riski aliyorum" sorusunu AYRI
# bir sinyal olarak vermez — ag bunu 3x21x21'lik seyrek/kuresel goruntuden
# CIKARMAYA calisiyordu. Ölum oraninin 1.3/episode'da takilı kalmasinin
# (rotasi %93 hedefe ULASIYOR ama ajan sik sik yanlis halkaya giriyor)
# sebebinin bu "eksik yari" oldugu hipotez edildi. Cozum: 4 yon icin
# self._hazard()'in (env/strike_env.py) AYNI olasilik modelini (per_entry,
# alert carpani dahil) "su an bulundugum bolgeden bu komsuya gecersem"
# sorusuyla tekrar kullanan 4 YENI skalar. Oracle'in SECTIGI aksiyon
# DEGIL — sadece o eylemin ANLIK fiziksel risk bilgisi; aksiyonu YINE RL
# ajani secer. `--dueling`/`--prioritized`/`--bc-lambda-start` YOK, saf
# VDN + mask-fix mimarisi (bkz. strike-mission-vdn-egitim-bulgulari
# hafizasi) KORUNUYOR.
N_SCALARS = 22                  # slot slot MARL-Pathfinding ile hizalandi (0-15):
# 0 agent_id | 1 other_terminal (orada: faz) | 2 t/max | 3 own_row | 4 own_col
# 5 dy_goal  | 6 dx_goal | 7 dist_goal
# 8 dy_other | 9 dx_other | 10 dist_other
# 11 risk_dist_own (orada: bfs_own)
# 12-15 risk_dist komsu FARKI: yukari/sag/asagi/sol (orada: bfs komsu farki)
# 16-17 SON HAREKET (dr, dc) — MARL-Pathfinding'de YOK, YENI slot
# 18-21 EYLEM-OZGU ANLIK RISK: yukari/sag/asagi/sol yone girmenin BU ADIMDAKI
#       olum olasiligi (self._hazard() ile AYNI model) — MARL-Pathfinding'de
#       YOK, YENI slot (2026-08-27)
OBS_DIM = OBS_CHANNELS * PATCH_SIZE * PATCH_SIZE + N_SCALARS   # 1345 (3x21x21+22)

STATE_CHANNELS = 2             # [A1 cevresi tehlike, A2 cevresi tehlike]
# 16 skalar: A1_row,A1_col,A2_row,A2_col,goal_row,goal_col,alive_bits,t/max
# (8) + A1'in 4 yon eylem-ozgu riski + A2'ninki (8) — 2026-08-28 dis inceleme:
# merkezi kritik (QMIX mixer + MAPPO/HAPPO) ONCEDEN bu bilgiyi HIC gormuyordu,
# aktorler observe() ile goruyordu ama kritik gormuyordu (bkz. env.state()).
STATE_SCALARS = 16
STATE_DIM = STATE_CHANNELS * PATCH_SIZE * PATCH_SIZE + STATE_SCALARS   # 898

# ---------------------------------------------------------------- ag mimarisi
CNN_CHANNELS = (16, 32)
CNN_POOL_SIZE = 4
SCALAR_EMBED = 128

# ---------------------------------------------------------------- yollar
RUNS_DIR = "runs"
# ONBELLEK RASTGELE HARITADA KAPALI OLMAK ZORUNDA. Harita her episode
# degistigi icin onbellek "gecerliymis gibi" eski haritayi dondurur ve
# HICBIR YERDE PATLAMAZ — ajan A haritasinda ucarken B haritasinin risk
# haritasiyla odullendirilir. Sessiz yanlislik, en tehlikeli hata turu.
# (Ayni tuzaga R_RISK_COEF degisiminde de dusulmustu; oradaki cozum dosya
# adina parametreleri gomekti, burada tek dogru cozum onbellegi KAPATMAK.)
RISK_CACHE = None if RADAR_RANDOM else "runs/risk_dist.npy"
ZONE_CACHE = None if RADAR_RANDOM else "runs/zone_map.npy"

# MARL-Pathfinding checkpoint'lerinin bulundugu klasor (--resume-from icin).
PATHFINDING_CKPT_DIR = r"..\MARL-Pathfinding\runs\ckpt"


def summary() -> str:
    """Kalibrasyonun tek bakista dogrulanmasi icin — train.py basinda basilir."""
    opt = (GRID_N - 1) * 2
    return "\n".join([
        f"grid            : {GRID_N}x{GRID_N} (1 hucre = 1 birim)",
        f"B -> H          : {START} -> {GOAL}, optimal {opt} adim, limit {MAX_STEPS}",
        (f"radarlar        : RASTGELE {N_RADAR} adet/episode (cakisma serbest), "
         f"dis +-{OUTER_HALF}, ic +-{INNER_HALF}"
         if RADAR_RANDOM else
         f"radarlar        : SABIT {RADARS}  dis +-{OUTER_HALF}, ic +-{INNER_HALF}"),
        (f"harita tohumu   : egitim 0..{TRAIN_SEED_MAX:.0e}, test "
         f"{EVAL_SEED_BASE:.0e}+ ({EVAL_N_MAPS} harita) — KESISMEZ; "
         f"curriculum {CURRICULUM_RADAR_START}->{CURRICULUM_RADAR_END} radar"
         if RADAR_RANDOM else "harita tohumu   : -"),
        f"odul hackleme   : timeout {R_TIMEOUT} < 2xolum {2*R_DEATH}; "
        f"ikisi de olur {2*R_DEATH + R_ALL_DEAD} <= timeout+adim "
        f"{R_TIMEOUT + MAX_STEPS * R_STEP} (MAX_STEPS={MAX_STEPS})",
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
        f"gozlem          : {OBS_DIM} = {OBS_CHANNELS}x{PATCH_SIZE}x{PATCH_SIZE}"
        f"(yerel stride {PATCH_STRIDE} +-{PATCH_RADIUS*PATCH_STRIDE} hucre, "
        f"kuresel stride {GLOBAL_PATCH_STRIDE} +-{PATCH_RADIUS*GLOBAL_PATCH_STRIDE} "
        f"hucre) + {N_SCALARS} skalar",
        f"alarm kuplaji   : {'ACIK' if ALERT_ENABLED else 'kapali'}",
    ])


if __name__ == "__main__":
    print(summary())
