# Strike Mission — MARL ile Radar Kaçınmalı İki Uçak Görevi

> **1000x1000 grid, 2 uçak, eşzamanlı kalkış, 3 sabit radar, ortak hedef.**
> Uçaklar aynı anda B'den kalkar, aynı yoldan gidebilirler (çarpışma yok).
> Radar halkalarında **stokastik düşürülme** riski var. **En az bir uçak
> hedefe varırsa takım ödülü fullenir.**
> IQL / VDN / QMIX ile eğitip karşılaştıracağız.

Durum: 📋 Planlama · Branch: `iql_vdn_qmix` · Güncelleme: 2026-08-05

Kardeş proje: [`MARL-Pathfinding`](../MARL-Pathfinding/PLAN.md) — ortam farklı,
**ajan/eğitim/eval altyapısı birebir taşınacak.** Orada 5x5→50x50 yolculuğunda
öğrenilen tuzaklar §9'daki tabloda hazır duruyor; aynı duvara ikinci kez
toslamayacağız.

---

## 0. Ölçülmüş gerçekler (tahmin değil, hesapladım)

Aşağıdaki sayıların **hepsi ölçüldü**, `python -m baselines.map_check` ve
`python -m baselines.risk_oracle` ile yeniden üretilebilir.

### 0.1 Ölçek: **tam çözünürlük, 1000x1000** (hücre = 1 birim)

Planın ilk sürümünde `STEP_SIZE=20` ile 51x51 latise indirgemiştim; Burak
bunu iptal etti, tam çözünürlükte koşuyoruz. Ölçekten bağımsız kalan ve
kalmayan şeyler:

| | Değer | Not |
|---|---:|---|
| Grid | 1000x1000 hücre | B = **(0,0)**, H = **(999,999)** |
| B→H Manhattan | **1998 adım** | `MAX_STEPS = 2800` (1.4x tampon) |
| Dış halka | ±110 hücre (221 kenar) | |
| İç halka | ±70 hücre (141 kenar) | |
| Tehlikeli hücre oranı | **%14.7** | 86.880 dış + 59.643 iç |

Dönüşüm: `row = 500 − y`, `col = x + 500`.

| Radar | (x, y) | Hücre (row, col) |
|---|---|---|
| R1 | (−280, 220) | **(280, 220)** |
| R2 | (200, 100) | **(400, 700)** |
| R3 | (−100, −280) | **(780, 400)** |

**Tam çözünürlüğün zorladığı iki değişiklik** (ikisi de ölçüldü, tahmin değil):

1. **`GAMMA = 0.99` → `0.9998`.** Episode 2000+ adım. `0.99^2000 = 2e-9` —
   ajan hedefi fiziksel olarak göremez. Dahası potential-based shaping'in
   "drag" terimi `(1−γ)·Φ`, ilerleme terimi `ΔΦ = 1/1998 = 0.0005`; γ=0.99'da
   drag 0.01 çıkıyor, yani shaping ajanı ilerlemekten **caydırırdı**.
   γ=0.9998'de drag ≤ 0.0002, ilerleme 2.5x baskın.
2. **`PATCH_STRIDE = 16`.** 21x21 pencere stride=1'de sadece ±10 hücre görür;
   dış halka 221 hücre genişliğinde, yani ajanın penceresi tamamen halkanın
   içinde kalır ve **sınırı hiç göremez** (kanal sabit 1.0, bilgisiz). 16
   hücre arayla örneklenince pencere ±160 hücre kapsıyor, bir radarı
   sınırlarıyla görüyor. Tensör şekli değişmediği için checkpoint uyumu bozulmuyor.

### 0.1b Ölçülen hız 🔑

VDN, 1000x1000, CPU: **5.0 saniye / episode** (2800 adım).

| Episode | Süre |
|---:|---|
| 500 | ~42 dk |
| 1.000 | ~1.4 saat |
| 2.000 | ~2.8 saat |
| 10.000 | ~14 saat |

QMIX ~2x daha yavaş olacak (mixer'ın state kodlayıcısı). Bu, tam çözünürlüğün
gerçek maliyeti: 51x51'de aynı iş ~20x hızlıydı.

### 0.2 Risk modeli: "girişte tek zar" değil, **adım başına hazard** 🔑

> **GÜNCELLEME 2026-08-05 — patron netleştirdi:** *"detection zone'da ister 20
> adım at ister 2 adım gir-çık yap, ölüm ihtimali eşit; `0.2*0.02*0.2` gibi bir
> birikme yok."* Yani **kural per-entry**: bölgeye **giriş başına tek zar**
> (dış %20, iç %90), süre önemsiz. `HAZARD_MODE = "per_entry"` varsayılan
> oldu; aşağıdaki "adım başına hazard" tartışması artık **ablation** olarak
> duruyor (`--hazard per_step`).
>
> Bunun tasarımsal sonucu: per_entry'de "bölgeyi teğetten sıyır" diye
> öğrenilecek bir beceri **yok** — girdiysen girmişsindir. Tek beceri
> **hiç girmemek**. Bu problemi belirgin şekilde basitleştiriyor ve
> §0.3'teki trivial'lik sorununu daha da keskinleştiriyor.

Senin tarifin ("turuncu alana girdiğinde %80 yaşama, iç alanda %10 yaşama")
iki şekilde okunabilir ve **ikisi tamamen farklı problemler üretiyor**:

| Okuma | Sonuç |
|---|---|
| **Girişte tek zar** | Bölgede 1 adım kalmak ile 11 adım kalmak **aynı** risk. Ajan "nasılsa tek zar" deyip tam ortadan geçer; teğet geçme / kenardan sıyırma gibi öğrenilecek incelik yok olur. |
| **Adım başına hazard** 🔑 | Risk, bölgede **geçirilen süreyle** artar. Ajan gerçekten "mümkün olduğunca uzak dur" öğrenir — senin asıl istediğin davranış bu. |

**Adım başına modeli seçtim**, ama senin sayılarını (%20 / %90) referans olarak
koruyacak şekilde kalibre ettim: *"bölgeyi boydan boya geçersen toplam ölüm
riskin %20 (dış) / %90 (iç) olsun."*

```
p_step = 1 − (1 − p_toplam)^(1 / çaprazlama_adımı)

dış:  1 − 0.80^(1/221) = 0.001009   → %0.101 / adım
iç :  1 − 0.10^(1/141) = 0.016198   → %1.620 / adım
```

Bu formül **ölçekten bağımsız**: 51x51'de de 1000x1000'de de "halkayı boydan
boya geç" toplam riski aynı %20 / %90 çıkıyor. `tests/test_env.py` ortamın
gerçek zarını analitik değerle karşılaştırıyor — ölçüldü: **0.0975 vs 0.1016**
(3σ içinde). Yani ortamın rastgeleliği oracle'ın matematiğiyle uyuşuyor.

İç halka içinde dış halka **saymaz** (iç kazanır, çift ceza yok). Zar her
timestep'te, her uçak için **bağımsız** atılır.

> `config.HAZARD_MODE = "per_step" | "per_entry"` bayrağı ile ikisi de
> koşulabilir — ablation olarak rapora girer. Varsayılan `per_step`.

### 0.3 🚨 SABİT HARİTA TRIVIAL — ölçüldü, en önemli bulgu

Dijkstra ile (maliyet = `−ln(1 − p_ölüm)` + adım) hayatta kalma olasılığını
maksimize eden yolu kesin olarak hesapladım, sonra da **hiç eğitilmemiş bir
ağla ve elle yazılmış sabit bir politikayla** karşılaştırdım
(`python -m baselines.policies`):

| Politika | Takım başarısı | Uzunluk | Dış/İç maruziyet | Analitik hayatta kalma |
|---|---:|---:|---:|---:|
| Rastgele monoton (baseline) | **20%** | 730† | 164 / 36 | 0.080 |
| Naif çapraz (merdiven) | — | 1998 | 160 / 161 | **0.080** = 0.8×0.1 |
| **SABİT politika (hep sağ → hep aşağı)** | **100%** | **1998** | **0 / 0** | **1.000** |
| **SABİT politika (hep aşağı → hep sağ)** | **100%** | **1998** | **0 / 0** | **1.000** |
| **Dijkstra oracle** | **100%** | **1998** | **0 / 0** | **1.000** |

† ölümle kesildiği için kısa. Sayılar per-entry risk modeliyle
(`python -m baselines.policies`).

🚨 **Sabit politika oracle'ın kendisi.** Yani bu haritada öğrenilecek hiçbir
şey yok — hiç eğitilmemiş bir ağ bile, argmax'ı tesadüfen "sağ"a düşerse,
mükemmel skoru alır. İlk duman testinde 4 episode'luk VDN koşusu eval'de
%100 / 1998 adım / sıfır maruziyet verdi ve **bu öğrenme değildi**, bu artefakttı.

**Kök neden:** B sol-üst köşe, H sağ-alt köşe, üç radar da iç bölgede.
Gridin **kenarı baştan sona radarsız bir otoyol**, ve köşeden köşeye
gidildiği için kenardan dolaşmanın uzunluk bedeli **sıfır**.
Güvenli yol = en kısa yol = trivial politika. Risk ile uzunluk arasında
**hiçbir ödünleşme yok**, dolayısıyla optimize edilecek bir şey de yok.

### 0.3b Haritayı nasıl anlamlı yaparız — taradım

150 rastgele radar konfigürasyonu tarayıp üç kategoriye ayırdım
(`python -m baselines.map_check`):

- **TRIVIAL** — kenar yolu güvenli → sabit politika kazanır, öğrenme yok
- **KOLAY** — kenar kapalı ama sıfır-riskli optimal yol hâlâ var → gerçek yol
  bulma gerekiyor, risk/uzunluk ödünleşmesi yok
- **ZOR** — sıfır-riskli optimal yol YOK → ajan risk ile uzunluk arasında
  **gerçek bir karar** vermek zorunda 🔑

| Senaryo | TRIVIAL | KOLAY | **ZOR** |
|---|---:|---:|---:|
| 3 radar, iç bölgede (**senin haritan gibi**) | **100%** | 0% | **0%** |
| 3 radar, kenara da yaklaşabiliyor | 75% | 19% | 7% |
| 5 radar, kenara da yaklaşabiliyor | 56% | 34% | 10% |
| 8 radar, kenara da yaklaşabiliyor | 32% | 51% | 17% |
| **3 radar, 2x büyük halka (440/280)** | 43% | 24% | **33%** |

👉 **İki net sonuç:**

1. **Radarları iç bölgeye koymak haritayı her seferinde trivial yapıyor**
   (%100). Random radar'a geçmek tek başına yetmez — radarlar **kenarı da
   kapatabilmeli**.
2. **En güçlü kaldıraç halka boyutu.** Halkaları 2x büyütmek "zor" oranını
   %0'dan %33'e çıkarıyor — radar sayısını 8'e çıkarmaktan (%17) daha etkili.

Bu, MARL-Pathfinding'in §0.3'teki "zor alt küme" metodolojisinin birebir
analoğu ve aynı sonucu veriyor: **metrikleri zor alt kümede raporla, genel
ortalamada her şey %100 çıkar ve hiçbir şey öğrenmemiş olursun.**

### 0.4 Peki VDN/QMIX niye gerekli? — kuplaj sorunu ve çözümü

VDN/QMIX'in IQL'e üstünlüğü **ancak bir ajanın eylemi diğerinin ödülünü
etkilediğinde** ortaya çıkar. Şu anki tanımda böyle bir kanal yok: ölüm zarları
bağımsız, çarpışma yok, kaynak paylaşımı yok.

**Çözüm — radar alarm (uyarılmış radar) mekanizması:**

> Bir uçak bir radarın dış halkasına girdiği anda o radar **alarma geçer**.
> Alarmdaki radarın ölüm olasılığı `ALERT_MULT` katına çıkar (varsayılan 2.0).
> Alarm `ALERT_DECAY` adım sonra söner.

Gerçek hava harbi doktrinine uygun (radar kilitlenir, batarya uyanır) ve tam
olarak istediğimiz kuplajı yaratır: **"aynı radarın menziline ikimiz birden
girmeyelim, farklı koridorlardan gidelim."**

Bu, MARL-Pathfinding'deki "A1, A2'nin yolunu kilitler" hikâyesinin birebir
analoğu — ve oradaki tez burada da geçerli:

- **IQL**: A1'in radarı uyandırması A1'in *kendi* ödülünde görünmez → A1 bunu
  asla öğrenemez.
- **VDN**: tek takım TD hatası A1'in ağına geri akar → A1 "bu hamle A2'yi
  öldürüyor" sinyalini alır.
- **QMIX**: A2'nin değeri A1'in seçimine **koşullu** (A1 radarı uyandırdıysa A2
  ne yaparsa yapsın riskli) — yani ödül **toplamsal değil**. VDN'in additivity
  varsayımı ihlal ediliyor, QMIX'in state-koşullu mixer'i bunu temsil edebilir.

👉 **Raporun tezi:** *"Bağımsız-risk kurulumunda IQL ≈ VDN ≈ QMIX; radar-alarm
kuplajı devreye girince VDN IQL'i, QMIX de VDN'i geçer."* İki koldan ölçülebilir,
temiz bir deney tasarımı. Alarm **varsayılan olarak KAPALI** — önce senin
tarif ettiğin sade ortamı kurup doğruluyoruz (§Aşama 1-5), sonra açıyoruz (§Aşama 6).

---

## 1. Formal tanım

### Ortam

- Latis 51x51, hücre `(row, col) ∈ 0..50`. Fiziksel `STEP_SIZE = 20` birim.
- Aksiyonlar: `0=YUKARI, 1=SAĞ, 2=AŞAĞI, 3=SOL, 4=NOOP` (5 aksiyon).
- **Eşzamanlı hareket** — MARL-Pathfinding'in sıralı/turn-based akışının aksine.
  İki uçak her timestep'te birlikte hamle yapar. Bunun iki bedava faydası var:
  gölge NOOP hilesine gerek yok (VDN/QMIX doğal haliyle bağlanıyor) ve episode
  uzunluğu yarıya iniyor.
- Grid dışına hamle **maskelenir** (ceza değil, seçilemez).
- `NOOP` maskesi: yaşayan/varmamış uçakta **kapalı** (beklemek asla optimal
  değil), terminal olmuş uçakta **tek açık aksiyon**.
- Çarpışma yok, yasak bölge yok — aynı hücrede durabilirler, aynı yoldan
  gidebilirler.

### Uçak durumu

Her uçak üç durumdan birinde: `ALIVE` → (`REACHED` | `DEAD`).
Terminal olan uçak haritada kalır ama hareket etmez (NOOP), Q'su VDN toplamına
girmeye **devam eder** (MARL-Pathfinding'in gölge NOOP tasarımıyla aynı
mekanik, farklı gerekçe).

Episode biter: iki uçak da terminal **veya** `t = MAX_STEPS`.

### Risk uygulaması (her timestep, her yaşayan uçak için)

```python
z = zone(pos)                      # 0 guvenli, 1 dis halka, 2 ic halka
p = P_DEATH[z] * (ALERT_MULT if radar_alarmda else 1.0)
if rng.random() < p:  ucak.durum = DEAD
```

Zar **hareketten sonra**, varılan hücreye göre atılır. Başlangıç hücresi (B)
için t=0'da bir kez atılır (B güvenli bölgede, pratikte etkisiz).

---

## 2. MARL algoritma yığını

| Algoritma | Fikir | Bu problemde beklentim |
|---|---|---|
| **IQL** | Her uçak kendi DQN'i, diğerini ortamın parçası sanar. | Baseline. Sade ortamda **VDN kadar iyi olmalı** (§0.3). Alarm açılınca geride kalmalı. |
| **VDN** | `Q_tot = Q_1 + Q_2`, tek TD hatası ikisine yayılır (CTDE). | Alarm açıkken credit assignment'ı çözer: "radarı uyandıran A1'di". |
| **QMIX** | `Q_tot = f(Q_1, Q_2 \| s)`, monotonik mixing + hypernetwork. | Alarm açıkken VDN'i geçmeli — ödül toplamsal değil (§0.4). |

**Ağırlık paylaşımı YOK** — iki ayrı Q ağı. MARL-Pathfinding'de tek paylaşılan
ağ 3 ayrı tam-ölçekli koşuda çöktü (`agents/vdn.py` dosya-stringi), ayrı ağlarla
istikrarlı kaldı. Burada roller daha simetrik olduğu için risk daha düşük ama
**kanıtlanmış konfigürasyondan sapmıyoruz.**

**Kütüphane kullanma, elle yaz.** PyMARL/EPyMARL Windows'ta kurulum kâbusu; VDN
~40 satır, QMIX ~90 satır. MARL-Pathfinding'in `agents/` klasörü zaten
%80 hazır — ortam farkı sadece `play_episode` döngüsünde.

---

## 3. Gözlem ve ağ tasarımı

Gözlem tasarımı **bugünden random radar'a hazır** (§Aşama 10) — radar konumları
gözlemin içinde, sabit değil. Böylece random'a geçerken ağ mimarisi hiç
değişmeyecek.

### Uçak gözlemi

**Yerel patch** (`PATCH_RADIUS = 6` → 13x13), 2 kanal:

| Kanal | İçerik |
|---|---|
| 0 | dış halka maskesi (uçağın çevresindeki 13x13 pencerede) |
| 1 | iç halka maskesi |

`PATCH_RADIUS=6` seçimi: dış halkanın yarıçapı 5 hücre, yani **bir radar
tamamen pencereye sığıyor** + 1 hücre pay. Daha büyük pencere bedava bilgi
vermiyor, replay buffer'ı şişiriyor.

**Skalarlar (25 adet):**

```
agent_id, t/MAX_STEPS,
own_row, own_col,                                  (normalize)
dx_goal, dy_goal, dist_goal,
dx_other, dy_other, dist_other, other_alive, other_reached,
self_in_outer, self_in_inner,
en_yakin_3_radar × (dx, dy, dist)                  (9 skalar, mesafeye göre SIRALI)
cum_log_survival                                   (birikmiş risk)
n_alerted / N_RADAR                                (Aşama 6)
```

> Radarlar **mesafeye göre sıralı** veriliyor — permütasyon-değişmezlik.
> R1/R2/R3 sırasıyla verilirse ağ radar kimliğini ezberler ve random radar'a
> geçince çöker.

`OBS_DIM = 2 × 13 × 13 + 25 = 363`

### Global state (QMIX mixer)

4 kanal patch (her uçağın çevresi × 2 halka) + 14 skalar
(iki konum, hedef, t, alive/reached bayrakları, 3 alarm biti)
→ `STATE_DIM = 4 × 169 + 14 = 690`

### Ağ

`agents/networks.py` MARL-Pathfinding'den **değiştirilmeden** taşınıyor:
`CNNQNet` (stride-2 conv → `AdaptiveAvgPool2d` → head) + ayrı skalar-kodlayıcı
dalı + ham skalar skip bağlantısı.

> Skalar dalı neden şart: MARL-Pathfinding'de CNN dalı 512 boyut üretirken
> skalarlar 11'di; ajana "hedef şu yönde" diyen 2 boyut, 523 girdinin %0.4'ü
> olup kayboluyordu. `SCALAR_EMBED=128` ile oran %20'ye çıktı ve öğrenme
> düzeldi. Aynı sorun burada da olurdu — hazır çözümle başlıyoruz.

---

## 4. Ödül tasarımı

VDN/QMIX **tek takım ödülü** ister — ayrıştırmayı algoritma yapar.
IQL için ayrıca `info["r_ind"]` üretilir (sadece kendi step-cost + kendi
varış/ölüm terimleri).

| Olay | `r_team` | Ne zaman |
|---|---:|---|
| Her global timestep | −0.05 | her t |
| Dış halkada bulunma (uçak başına) | −0.05 | her t | 
| İç halkada bulunma (uçak başına) | −0.25 | her t |
| **Bir uçak düşürüldü** | **−15.0** | anında |
| **İLK uçak hedefe vardı** | **+50.0** | 🔑 takım ödülü full |
| İkinci uçak da vardı | +12.0 | "ikide olsa" bonusu |
| İkisi de düşürüldü | −10.0 ek | tam başarısızlık |
| Timeout (hiçbiri varmadı) | −10.0 | `t = MAX_STEPS` |
| Shaping | `+20.0 × (γΦ' − Φ)` | her t, yaşayan uçak başına |

`Φ(s) = 1 − manhattan(pos, H) / max_manhattan` (0 = en uzak, 1 = hedefte).

### Kalibrasyon mantığı

- **Güvenli optimal yol:** 100 × (−0.05) + 50 + 12 = **+57**.
- **Düz çaprazdan iki uçak:** E[ödül] ≈ 0.43×50 + 0.06×12 − 2×0.75×15 ≈ **−0.7**.
  Fark ~58 puan — ajan güvenli yolu bulmak zorunda.
- **Ölüm cezası (−15) vs varış ödülü (+50):** oran 1:3.3. Daha sert bir ölüm
  cezası (−50 gibi) uçağı aşırı ihtiyatlı yapıp timeout'a iter; daha yumuşağı
  riski önemsizleştirir.
- Risk-shaping terimleri (−0.05 / −0.25) **kasıtlı olarak küçük**: ölüm
  cezasının yerini almıyorlar, sadece erken eğitimde seyrek ölüm sinyalini
  yoğunlaştırıyorlar. `--no-risk-shaping` bayrağıyla kapatılıp etkisi ölçülecek.

### Üç kural (MARL-Pathfinding'den taşınan)

1. **Shaping potential-based olsun** (Ng ve ark. 1999) — optimal politikayı
   değiştirmediği kanıtlı. Naif shaping ajanı ödül farmına sokar.
2. **Terminal'de `Φ = 0` kabul et**, yoksa shaping terminal geçişte yanlış
   bootstrap üretir.
3. **Ölçek büyütme.** `R_AGENT_GOAL`'ü 50→300 yapmak MARL-Pathfinding'de QMIX'i
   çökertti (LR=3e-5 ile ~350 büyüklüğünde Q temsil etmek kırılgan).
   50/12/−15 aralığı o kanıtlanmış ölçeğe yakın kalıyor.

---

## 5. Baseline'lar ve doğruluk zemini

MARL-Pathfinding'in BFS oracle'ının analoğu: burada **Dijkstra**, çünkü
maliyet toplamsal değil çarpımsal — logaritma ile toplamsala çeviriyoruz.

`baselines/risk_oracle.py`:

| Baseline | Tanım | Ölçülen |
|---|---|---|
| **Random-monotone** | Hedefe doğru rastgele monoton adım, radar körü | %21.2 hayatta kalma / %37.9 takım |
| **Greedy diagonal** | Düz çapraz | %24.7 / %43.4 |
| **Dijkstra oracle** | `−ln(1−p)` maliyetli en güvenli yol | **%100 / %100**, 100 adım |
| **Length-constrained oracle** | En güvenli yol, uzunluk ≤ L kısıtıyla | Pareto eğrisi (risk vs uzunluk) |

> Baseline **"random walk" değil, "random monotone"** olmalı. Rastgele yürüyüş
> 51x51'de hedefe neredeyse hiç varmaz ve RL'in kazancını yapay olarak şişirir
> (MARL-Pathfinding §Aşama 1 uyarısının aynısı).

Oracle **analitik** bir sayı da veriyor: verilen bir yolun tam hayatta kalma
olasılığı `Π(1 − p_zone)`. Yani ajanın ürettiği her yolu "oracle kaç puandı"
ile kıyaslayabiliyoruz — Monte Carlo gürültüsü olmadan. Bu, gap metriğini
inanılmaz keskinleştiriyor.

---

## 6. Aşamalar

Her aşamanın kabul kriteri var. **Kriter geçmeden sonrakine geçme.**

### Aşama 0 — Kurulum
- [x] `git init` + branch **`iql_vdn_qmix`** + remote `Burakscheker/MARL_Strike_Mission`
- [x] Klasör iskeleti, `.gitignore` (`.venv/`, `__pycache__/`, `runs/`)
- [x] Bu doküman
- [ ] `.venv` + `pip install -r requirements.txt` (numpy, matplotlib, torch CPU)
- [ ] `config.py` — tüm hiperparametreler tek dosyada, koda değer gömme
- [ ] İlk commit + push

### Aşama 1 — Ortam (`env/strike_env.py`)

```python
class StrikeMissionEnv:
    def reset(self, seed=None) -> dict[agent_id, obs]
    def step(self, actions: dict) -> (obs, rewards, dones, infos)
    def state(self) -> np.ndarray          # QMIX mixer icin
    def render(self) -> str                # ASCII
    def action_mask(self, agent) -> np.ndarray[5]
```

ASCII render (debug'ın %80'i buradan gelir):
```
B . . . . . . .        B = baslangic   H = hedef
. o o o . . . .        o = dis halka   x = ic halka
. o x x o . . .        1 = ucak1       2 = ucak2   * = ikisi ayni hucrede
. o x x o . . .        + = dusurulmus ucak
. . . . . . . H
```

**Kabul kriterleri (istatistiksel, `tests/test_env.py`):**
- [ ] `zone()` fonksiyonu §0.1'deki üç radar için tam 11x11 / 7x7 alan veriyor
- [ ] 20.000 episode `random_monotone` politika → hayatta kalma **%21.2 ± 1**
      (analitik değerle uyuşuyor)
- [ ] Düz çapraz → **%24.7 ± 1**
- [ ] Dijkstra oracle yolu → **%100** (hiç ölüm yok)
- [ ] Aksiyon maskesi: kenarda grid-dışı kapalı, terminal uçakta sadece NOOP
- [ ] Terminal uçağın gözlemi güncellenmeye devam ediyor (VDN kredi kanalı)

### Aşama 2 — Risk oracle + tarama (`baselines/risk_oracle.py`)
- [ ] `zone_map()`, `survival_prob(path)`, `dijkstra_safest()`, `path_length()`
- [ ] §0.3 tablosunu **kendi kodunla yeniden üret** (%21.2 / %24.7 / %100)
- [ ] Sıfır-riskli monoton yol sayısını DP ile doğrula (**%1.07**)
- [ ] `runs/map_stats.csv` — harita istatistikleri

### Aşama 3 — Tek uçak DQN (`agents/dqn.py`)
Sanity check: tek uçak, radar riski var, hedefe git. MARL'a geçmeden önce
DQN makinesinin ve gözlem borusunun doğruluğunu kanıtlar.
- [ ] **Kabul:** ε=0 deterministik değerlendirmede **hayatta kalma ≥ %98**,
      ortalama yol uzunluğu ≤ 104 (oracle 100)

### Aşama 4 — IQL baseline (`agents/iql.py`)
İki bağımsız DQN, ortak ödül **yok** (`info["r_ind"]`).
- [ ] **Kabul:** takım başarısı ≥ %98 (baseline %37.9)
- [ ] Ölçüm: iki uçağın yolları ne kadar örtüşüyor (`route_overlap`)?
      Sade ortamda örtüşme yüksek olmalı — koordinasyon yok, ikisi de aynı
      güvenli yolu bulur. **Bu beklenen sonuç, hata değil.**

### Aşama 5 — VDN (`agents/vdn.py`)
```
Q_tot = Q_1(o_1, a_1) + Q_2(o_2, a_2)
loss  = (r_team + γ·max Q_tot_target − Q_tot)²
```
- [ ] Joint replay buffer (her satır BİR global timestep, iki uçağın bilgisi)
- [ ] **Sağlık kontrolü:** bir uçak terminal olduktan sonra `Q(obs, NOOP)`
      değişmeye devam ediyor mu? Sabit kalıyorsa gözlem güncellenmiyor demektir
      — MARL-Pathfinding'de bu **en sinsi bug**'dı.
- [ ] **Kabul:** takım başarısı ≥ %99, IQL ile **fark yok** (§0.3 tahmini).
      Fark çıkarsa neden çıktığını araştır, sevinme.

### Aşama 6 — Radar alarm mekanizması 🔑 **kritik**
§0.4'ün uygulaması. Ortama tek bir kuplaj ekleniyor, algoritmalar değişmiyor.
- [ ] `config.ALERT_ENABLED`, `ALERT_MULT = 2.0`, `ALERT_DECAY = 15` adım
- [ ] Alarm bitleri gözleme ve global state'e ekleniyor
- [ ] IQL / VDN / QMIX **aynı bütçeyle yeniden koşuluyor**
- [ ] **Kabul:** VDN'in takım başarısı IQL'i **anlamlı** şekilde geçiyor
      **VEYA** geçmiyorsa neden geçmediği ölçülerek raporlanıyor.
      Negatif sonuç da sonuçtur — abartma, gizleme.
- [ ] Ölçüm: `route_overlap` VDN'de IQL'e göre **düşmeli** (uçaklar ayrışıyor)

### Aşama 7 — QMIX (`agents/qmix.py`)
- [ ] Hypernetwork + monotonik mixer (`abs(W)`), `embed_dim=32`
- [ ] Global state kodlayıcı (küçük CNN — düz `nn.Linear(690, ...)` değil)
- [ ] **Kabul:** §0.4 hipotezi test edilecek — QMIX, alarm açıkken VDN'i
      geçiyor mu?

### Aşama 8 — Değerlendirme (`eval/evaluate.py`)

| Metrik | Tanım | Hedef |
|---|---|---|
| **Takım başarısı** | ≥1 uçak hedefe vardı | 🔑 **ANA METRİK** — baseline %37.9 |
| Both-reached | İki uçak da vardı | ikincil |
| Uçak başına hayatta kalma | | baseline %21.2 |
| **Radar maruziyeti** | dış/iç halkada geçirilen adım sayısı | 🔑 → 0 |
| Yol uzunluğu / optimal gap | Dijkstra oracle'a göre | → 0 |
| Analitik hayatta kalma | Üretilen yolun `Π(1−p)` değeri | 🔑 gürültüsüz |
| `route_overlap` | İki yolun ortak hücre oranı | koordinasyon göstergesi |
| Timeout oranı | | → 0 |
| Sample efficiency | %95 takım başarısına kaç episode | algoritma kıyası |

- [ ] Tek ortak `run_episode()` döngüsü — scripted baseline'lar ile öğrenen
      ajanlar arasında kod-yolu sapması riski yok (MARL-Pathfinding dersi)
- [ ] `runs/eval_report.md`
- [ ] 3 seed × 3 algoritma × 2 mod (alarm kapalı/açık), ortalama ± std

### Aşama 9 — Görselleştirme (`viz/plot_report.py`)
- [ ] Harita çizimi: radar kareleri (turuncu/yeşil, senin görselindeki gibi),
      iki uçağın yolu, ölüm noktaları ✕ ile
- [ ] Öğrenme eğrileri: takım başarısı + radar maruziyeti (yoğun, eğitim-içi)
- [ ] Isı haritası: 1000 episode üzerinden hangi hücrelerden kaç kez geçildi
      → politikanın "tercih ettiği koridor" görsel olarak çıkar
- [ ] `plot_algorithm_comparison()`: IQL vs VDN vs QMIX, alarm açık/kapalı

### Aşama 10 — Random radar (genelleme) — senin "sonradan" dediğin yer
- [ ] `env/sampler.py` — radar sayısı/konumu her episode rastgele
- [ ] **Zorluk sınıflandırması:** Dijkstra ile "sıfır-riskli optimal yol var mı"
      → `kolay` / `zor` kovaları. MARL-Pathfinding'in §0.3 metodolojisinin
      birebir analoğu — ve **asıl fark zor alt kümede görünür.**
- [ ] Curriculum: `p_hard = min(0.8, 0.2 + 0.6·(ep/total))`
- [ ] Transfer testi: sabit haritada eğit → random haritada test et
      (gözlem tasarımı buna hazır, §3)

---

## 7. Klasör yapısı

```
MARL_Strike_Mission/
├─ Strike_Mission.md         # bu dosya
├─ README.md
├─ requirements.txt
├─ config.py                 # TÜM hiperparametreler
├─ env/
│  ├─ strike_env.py          # StrikeMissionEnv (esZamanli)
│  ├─ two_agent.py           # play_episode / play_episode_vdn / _qmix
│  ├─ single_agent.py        # Asama 3 sarmalayicisi
│  └─ sampler.py             # Asama 10: random radar + curriculum
├─ agents/
│  ├─ networks.py            # CNNQNet, masked_q       (MARL-Pathfinding'den)
│  ├─ buffer.py              # ReplayBuffer            (MARL-Pathfinding'den)
│  ├─ dqn.py                 # tek ucak                (MARL-Pathfinding'den)
│  ├─ iql.py                 # iki bagimsiz DQN
│  ├─ vdn.py                 # Q_tot = sum Q_i + JointReplayBuffer
│  └─ qmix.py                # QMixer + MixerReplayBuffer
├─ baselines/
│  ├─ risk_oracle.py         # Dijkstra + survival_prob + tarama
│  └─ policies.py            # random_monotone, greedy_diagonal, oracle_follow
├─ train.py                  # python train.py --algo vdn --seed 0
├─ eval/evaluate.py
├─ viz/plot_report.py
├─ tests/
│  ├─ test_env.py            # zone(), maske, terminal davranisi
│  └─ test_oracle.py         # Dijkstra, analitik survival, %1.07 DP
└─ runs/                     # git'e girmez
```

---

## 8. Hiperparametreler (başlangıç seti — MARL-Pathfinding'de kanıtlanmış)

| | Değer | Not |
|---|---|---|
| `GRID_N` / `STEP_SIZE` | 51 / 20 | §0.1 |
| `MAX_STEPS` | 140 | optimal 100 × 1.4 tampon |
| γ | 0.99 | |
| ε | 1.0 → 0.05, eğitimin ilk %50'sinde | **episode-tabanlı**, adım-tabanlı değil |
| `LEARN_EVERY` | 8 | CNN `learn()` CPU'da pahalı, ~%38 hızlanma |
| Batch | 32 transition | 128 değil — CNN'de learn() ~10ms |
| Grad clip | 10.0 | mixer'da patlamayı önler |
| DQN/IQL LR | 1e-4 | 5e-4'te Q-divergence görüldü |
| **VDN/QMIX LR** | **3e-5** | 1e-4'te VDN ep~1750'de tepe yapıp çöktü |
| **VDN/QMIX target update** | **4000 adım** | 2000'de moving-target çökmesi |
| Double DQN | açık | overestimation'ı kırar |
| Ağırlık paylaşımı | **kapalı** | §2, kanıtlanmış |
| Toplam eğitim | 20-30k episode | tahmini VDN ~1 saat, QMIX ~2 saat |

> ⚠️ **VDN/QMIX'in LR ve target_update değerlerini değiştirmeden eğitme.**
> MARL-Pathfinding'de bu iki sayı 3 ayrı tam-ölçekli koşuyu çökertti; düzeltilmiş
> değerlerle 12.000 episode boyunca monoton yükseldi, hiç çökme görülmedi.

---

## 9. Tuzaklar tablosu (belirti → kök neden → çözüm)

Üstteki blok MARL-Pathfinding'de **fiilen yaşanmış** ve çözülmüş vakalar —
bedavaya devralıyoruz. Alttaki blok bu projeye özgü riskler.

| Belirti | Kök neden | Çözüm |
|---|---|---|
| Eğitim ilerledikçe **çöküyor** (tepe yapıp düşüyor) | VDN/QMIX'te LR yüksek + target update sık; ortak TD hatası ayrı-ağlı DQN'den kırılgan | `VDN_LR=3e-5`, `TARGET_UPDATE=4000` |
| Q değerleri patlıyor / loss üstel büyüyor | Moving-target pozitif geri besleme | Target update aralığını artır, LR düşür |
| Timeout'ta ajan "değeri 0" sanıyor | Time-limit bootstrapping | `info["truncated"]` ayrımı, buffer'a `done=False` |
| VDN, IQL ile **birebir aynı** | Terminal ajanın gözlemi güncellenmiyor → `Q_2` sabit → gradyan akmıyor | `Q(obs,NOOP)`'un t ile değiştiğini logla (§Aşama 5) |
| Ajan hedefi hiç bulamıyor, hep timeout | Seyrek ödül; büyük gridde rastgele yürüyüşle varış karesel zor | Potential-based shaping, `SHAPING_COEF=20` |
| "Hedefe git" sinyali en zayıf sinyal | CNN dalı 512 boyut, skalarlar 11 → %0.4 | `SCALAR_EMBED=128` + ham skalar skip |
| Eval sonucu eğitimden kötü | Eval'de ε > 0 kalmış | ε=0, deterministik greedy |
| Türkçe karakterde `UnicodeEncodeError` | Windows'ta stdout dosyaya yönlenince cp1252 | `sys.stdout.reconfigure(encoding="utf-8")` |
| `.bat` dosyası satırların ilk harfini yutuyor | LF satır sonu | CRLF'ye normalize et |
| **Ajan radar bölgesine hiç girmiyor ama hedefe de varmıyor** | Risk-shaping cezası ölüm cezasına göre çok büyük, ajan "hiç hareket etme" öğreniyor | `R_RADAR_*`'ı küçült veya kapat (`--no-risk-shaping`) |
| **Ajan doğrudan çaprazdan gidiyor, ölümü umursamıyor** | Ölüm cezası step-cost'a göre küçük | `R_DEATH` ≥ 10× toplam step-cost olmalı (şu an 15 vs 5) |
| **Başarı oranı yüksek ama radar maruziyeti de yüksek** | Şans eseri sağ kalmış episode'lar; success tek başına yanıltıcı | **Her zaman `analytic_survival`'ı birlikte raporla** (gürültüsüz) |
| **IQL = VDN = QMIX** | Kuplaj yok, problem ayrıştırılabilir | Beklenen (§0.3). Alarmı aç (§Aşama 6) |
| Ağ radar kimliğini ezberliyor, random'da çöküyor | Radarlar gözleme sabit sırayla veriliyor | Mesafeye göre **sırala** (§3) |
| İki uçak birbirini hiç "görmüyor" | `dx_other`/`dy_other` skalarları eksik veya normalize edilmemiş | Gözlem testinde iki uçağı ayrı yerlere koyup skalarları kontrol et |

---

## 10. Zaman tahmini

| Aşama | Süre | Kümülatif |
|---|---:|---:|
| 0 Kurulum + config | 1 s | 1 |
| 1 Ortam + testler | 4 s | 5 |
| 2 Risk oracle + tarama | 3 s | 8 |
| 3 Tek uçak DQN | 3 s | 11 |
| 4 IQL | 2 s | 13 |
| 5 VDN | 3 s | 16 |
| **6 Alarm mekanizması** | **3 s** | 19 |
| 7 QMIX | 3 s | 22 |
| 8 Değerlendirme | 3 s | 25 |
| 9 Görselleştirme | 3 s | 28 |
| 10 Random radar (opsiyonel) | 5 s | **~33 saat** |

Odaklanmış **5-6 günlük** iş (Aşama 10 hariç). Riskli iki yer: **Aşama 6**
(kuplaj gerçekten fark yaratıyor mu) ve **Aşama 1** (risk modelinin analitik
değerlerle uyuşması). Tamponu oraya bırak.

Zaman daralırsa kesme sırası: 10 → 9 → 7(QMIX) → 3(DQN).
**Aşama 2 ve 6 asla kesilmez** — biri doğruluk zemini, diğeri projenin
"neden MARL" sorusuna cevabı.

---

## 11. Tek paragraf özet

1000x1000 gridi `STEP_SIZE=20` ile 51x51 latise indir (radar kareleri tam
bölünüyor, B→H = 100 adım, eğitim bütçesi kanıtlanmış) → riski **adım başına
hazard** olarak modelle (dış %2.01, iç %28.03 — senin %20/%90'ını "boydan boya
geçiş" referansıyla kalibre ederek) → Dijkstra oracle'ı yaz ve §0.3'teki
%21.2 / %24.7 / %100 sayılarını kendi kodunla yeniden üret → tek uçak DQN'i
doğrula → IQL/VDN'i koş ve **sade haritada üçünün de aynı çıktığını dürüstçe
göster** → **radar alarm kuplajını aç** ki VDN'in IQL'i geçmesi için fiziksel
bir kanal oluşsun → QMIX ile additivity hipotezini test et → çiz, yaz →
zaman kalırsa radarları random'a çevirip zor alt kümede ölç.

**Altın kural:** Başarı oranını **asla tek başına** raporlama. Yanına her zaman
`analytic_survival` (üretilen yolun matematiksel hayatta kalma olasılığı) ve
`radar_exposure` koy — şans eseri sağ kalan bir politika ile gerçekten güvenli
bir politika ancak böyle ayrılır.
