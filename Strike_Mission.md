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

## 11. Aşama 11 — RANDOM HARİTA: 40 radar, her episode yeni 🔑

> **Karar (2026-08-05, Burak):** Harita artık sabit değil. **Her episode'da
> (hem eğitimde hem testte) 40 radar rastgele yerlere konur.** B ve H sabit
> kalır (0,0) → (999,999). Radar merkezleri farklı olur ama **çakışabilir** —
> merkezleri 5 hücre farklı iki radar olabilir, alanları üst üste biner.

Bu, §0.3'teki trivial'lik sorununu **tamamen çözüyor** ve projeyi gerçek bir
genelleme problemine çeviriyor: ajan artık bir yolu ezberleyemez, her episode
yeni bir haritada gerçek yol planlaması yapmak zorunda.

### 11.1 Ölçülen fizibilite — önce buna bak

`python -m baselines.scan_random_maps --mode per_entry` (satır başına 8 harita,
`greedy_path` düzeltmesinden **sonra** — bkz. §11.3b):

| Radar | Güvenli hücre | İç halkada | **Oracle hayatta kalma** | medyan |
|---:|---:|---:|---:|---:|
| 5 | 80.5% | 8.7% | **97.5%** | 100% |
| 10 | 64.2% | 17.1% | **92.5%** | 100% |
| 20 | 41.9% | 30.7% | **65.0%** | 80% |
| 30 | 28.6% | 41.8% | **34.6%** | 8.0% |
| **40** | **18.0%** | **52.2%** | **32.1%** | **7.2%** |

Halka boyutu, radar sayısından daha güçlü bir kaldıraç:

| Radar | Halka (dış/iç, dünya birimi) | Güvenli | Oracle hayatta kalma |
|---:|---|---:|---:|
| 40 | 120 / 76 | 57.8% | **81.0%** |
| 40 | 160 / 100 | 38.5% | **58.3%** |
| 40 | 220 / 140 | 18.0% | **32.1%** |

> "Oracle hayatta kalma" = `MAX_STEPS`'e sığan, riski **minimize eden** yolun
> hayatta kalma olasılığı. Ajan bundan iyisini yapamaz — **tavan bu.**

👉 **40 radar × 220/140 çalışır, ama dağılım çok çarpık.** Ortalama %32.1
iken **medyan sadece %7.2**. Sebep `per_entry`'nin ayrık yapısı: hayatta
kalma `0.8^(dış giriş) × 0.1^(iç giriş)` biçiminde **kuantize**. Gözlenen
tipik değerler:

| Oracle'ın mecbur kaldığı geçiş | Hayatta kalma | Sıklık |
|---|---:|---|
| 1 dış halka | 0.80 | şanslı haritalar |
| 2 dış halka | 0.64 | |
| 1 dış + 1 **iç** | 0.08 | **tipik (medyan)** |
| 2 dış + 1 iç | 0.064 | |

Yani **haritaların yarısından fazlasında en iyi yol bile bir iç halkayı
delmek zorunda** — ve iç halka %90 ölüm. Takım tavanı: ortalama haritada
`1−(1−0.321)² = %53.9`, medyan haritada `1−(1−0.08)² = %15.4`.

Bu, `surv_ratio` metriğini (§11.6) **zorunlu** kılıyor: ham başarı oranı
harita şansıyla 10 kat oynuyor, ajanın iyi mi kötü mü olduğunu söylemiyor.

### 11.2 🚨 Ödül kalibrasyonu — iki gerçek sorun

> ⚠️ Bu bölümün ilk hali hatalıydı: shaping terimini hesaba katmıyor ve
> düzeltilmemiş oracle sayılarını kullanıyordu. Aşağısı düzeltilmiş hali.

**Sorun A — Φ potansiyeli başlangıçta SATÜRE oluyor.** ✅ *düzeltildi*

`_phi()` şöyleydi: `Φ = 1 − min(1, dist/max_man)`, `max_man = 2(n−1) = 1998`.
Ama `dist` **risk-mesafesi**: `1 adım + RISK_W × p(giriş)`. Bir iç halkaya
girmek `750 × 0.9 = 675` adım-eşdeğeri ekliyor, yani `dist` `max_man`'i
aşabiliyor ve `min(1, …)` kırpması Φ'yi 0'a kilitliyor.

**Ölçüldü** (15 rastgele 40-radar haritası, `RISK_W = 750`):

| | değer |
|---|---:|
| `dist(B) / max_man` ortalama | **1.29** |
| aynı oran, en kötü harita | **1.75** |
| `max_man`'i aşan harita | **14/15** |
| oracle yolunun Φ=0 olan kısmı, ortalama | **%16.5** |
| aynı, en kötü harita | **%38.3** |

> ⚠️ Bu bölümün ilk hali "haritanın büyük kısmı" diyordu — **abartıydı.**
> Gerçek: yolun ortalama **%16.5**'i sinyalsiz. Ama bu ölü bölge tam olarak
> **başlangıçta**, yani ajanın her episode'a başladığı ve yönlendirmeye en
> çok ihtiyaç duyduğu yerde. Düzeltmeye değer, ama "shaping tamamen ölü"
> değil.

**Düzeltme (uygulandı):** normalizasyon episode'un KENDİ haritasına göre:
```python
self.dist_scale = max(dist[s1], dist[s2], 1.0)    # reset()'te, harita başına
Φ = 1.0 - min(1.0, dist[pos] / self.dist_scale)   # Φ(B)≈0, Φ(H)=1 her haritada
```
Aynı kırpma gözlem skaları #11'i (`min(1, d_own/max_man)`) de sabit 1.0'a
sabitliyordu — ajan "hedefe ne kadar kaldı"yı hiç göremiyordu; o da
`dist_scale` ile normalize edildi.

Potential-based shaping **herhangi bir Φ** için politika-değişmez (Ng ve
ark. 1999), yani bu düzeltme teorik garantiyi bozmaz — sadece sinyali geri
getirir.

**Yan bulgu — `d` ile `survival_prob` tutarsız.** Ölçümde bir harita
`dist(B) = 1998` (yani maliyet modeline göre tamamen güvenli yol var) ama
`survival_prob = 0.100` verdi. Sebep: B **bir iç halkanın içinde** ve
`survival_prob` `prev = 0`'dan başladığı için ilk hücrede %90'lık zar
atıyor; `direction_costs` ise sadece hücreler arası GEÇİŞLERİ ücretlendirdiği
için bunu hiç görmüyor. §11.3'teki "kalkış istisnası" maddesi bu — ölçümle
doğrulandı, `prev = z[START]` düzeltmesi yapılınca ikisi tutarlı olacak.

**Sorun B — oyalanma teşviki.** Φ düzeltildikten sonra bile, medyan haritada
(oracle %8) beklenen değerler:

| | uçmak | oyalanmak |
|---|---:|---:|
| | uçmak | oyalanmak |
|---|---:|---:|
| ilk/ikinci varış | +7.7 | 0 |
| ölüm cezaları | −36.1 | 0 |
| adım maliyeti | −20.2 | −28.0 |
| risk_shaping peşin ödemesi (`R_RISK_COEF=7.5` ile) | −16.5 | 0 |
| `R_TIMEOUT` | 0 | −10.0 |
| **toplam** | **≈ −65.1** | **≈ −38.0** |

Oyalanmak hâlâ **27 puan daha kârlı**. İki kaldıraç:

1. **`risk_shaping` riski İKİ KEZ sayıyordu** ✅ *düzeltildi*.
   `R_RISK_COEF × p` peşin ödeniyor *ve* ayrıca stokastik `R_DEATH` zarı
   atılıyor; ikisinin de beklenen değeri `|R_DEATH| × p`. `R_RISK_COEF = 15
   = |R_DEATH|` olduğu için ajanın efektif risk kaçınması **spec'in 2 katıydı**
   — tasarladığımızdan ürkek bir politika. `R_RISK_COEF = 7.5` yapıldı; iki
   terimin toplamı artık spec'teki tek cezaya eşit. Ablation olarak 0.0
   (sadece stokastik) ve 15.0 (eski) da koşulmalı.
2. **`R_TIMEOUT = −50`** ⏳ *bekliyor* — denememek, deneyip ölmekten pahalı
   olmalı. Bu, rastgele harita geçişiyle birlikte yapılacak (§11.7 adım 2);
   sabit haritada oyalanma teşviki yok çünkü orada tavan %100.

`R_DEATH = −15` ve `R_FIRST_GOAL = +50` aynı kalır (ölüm cezasını büyütmek
ajanı daha ürkek yapar, ters etki).

### 11.3 Risk muhasebesi: **seviye tabanlı, çakışma TOPLANMAZ** ✅

> **Karar (Burak, 2026-08-06):** *"İsterse aynı anda 4 radarın detection
> zone'unda olsun yine ölme ihtimali yüzde 20, toplamıyoruz."*

Bir hücrenin tehlikesi **sadece hangi halkada olduğuyla** belirlenir, kaç
radarın kapsadığıyla değil:

```
zone(hücre) = max(tüm radarlar üzerinden)    0 = güvenli, 1 = dış, 2 = iç
```

**Bu zaten kodda böyle.** `risk_oracle.build_zone_map()` radarları
`np.maximum` ile bindiriyor, `move_risk()` zarı yalnızca **seviye arttığında**
atıyor (`0→1` %20, `0→2` ve `1→2` %90). Yani Aşama 11 için risk motorunda
**değişiklik gerekmiyor** — tek değişen, radar setinin her episode yeniden
örneklenmesi.

> Alternatif "her radar ayrı bir tespit sistemi, üst üste binen 4 radar → 4
> ayrı zar" muhasebesi önerilmişti; Burak tarafından **reddedildi.** Kayıt
> için: o model tavanı belirgin şekilde düşürüyordu.

**Kuralın en önemli sonucu — halkalar BİRLEŞİYOR.** 40 radarın dış halkaları
üst üste bindiği için harita "40 ayrı engel" değil **birkaç büyük birleşik
bölge**. `per_entry` ile bir bölgeye girmek %20'ye mal olur ve **içeride ne
kadar dolaşırsan bedava**. Optimal davranış bu yüzden "girdiğin bölgeden
gereksiz çıkma, aynı sınırı tekrar tekrar geçme".

Öğrenilecek beceri **topolojik**: kaç ayrı bölge sınırı geçtiğin önemli,
o bölgelerde kaç adım kaldığın değil. Bu, `per_step` ablasyonundan niteliksel
olarak farklı bir problem (orada beceri "bölgeyi teğetten sıyırmak" olurdu) —
ikisinin karşılaştırması rapora girmeli.

**Kalkış istisnası — düzeltilecek.** 40 radarla B'nin bir halkanın içinde
kalma olasılığı yüksek. Hem ortam (`_prev_zone`) hem `survival_prob` şu anda
`prev = 0`'dan başlıyor; B iç halkadaysa uçak daha ilk adımda %90'lık zar
yiyor — halbuki kendi üssünden kalkıyor. İkisi de `prev = zone[START]` ile
başlamalı.

### 11.3b 🐛 `greedy_path` kenar-maliyeti hatası — BULUNDU ve DÜZELTİLDİ

Fizibilite taramasını yazarken tutarsızlık çıktı: 160/100 halka, **daha
büyük** olan 220/140'tan kötü sonuç veriyordu. Küçük radar daha güvenli
olmak zorunda, yani ölçüm hatalıydı.

**Kök neden:** `greedy_path` her adımda `d` değeri en küçük komşuya iniyordu
(tepe-inişi). Maliyet **düğüm**-ağırlıklı olsa yaklaşık doğruydu; ama bizim
maliyetimiz **kenar**-ağırlıklı — bir hücreye girmenin bedeli hangi bölgeden
geldiğine bağlı. `per_entry`'de bir halkaya girmek `1 + 1500×0.9 = 1351`
adım-eşdeğeri; `d`'si azıcık küçük diye o komşuya atlamak yolu mahvediyordu.
Doğrusu Bellman denkleminin kendisi:

```
d[u] = min_v ( cost(u→v) + d[v] )      # argmin alınırken kenar maliyeti de toplanmalı
```

**Ölçüldü** (40 rastgele radar, aynı haritalar):

| mod | eski `greedy_path` | düzeltilmiş |
|---|---:|---:|
| per_entry | 0.0008 | **0.0800** |
| per_step | 0.0063 | **0.5028** |

Oracle tavanı **100 kata kadar düşük** raporlanıyordu. Tavan yanlışsa
`surv_ratio` da yanlış — yani bu hata bulunmasaydı ajanın "oracle'ın kaçta
kaçı" sorusu baştan sona yanlış cevaplanacaktı. Sabit haritada da geçerli:
`map_check.py`, `policies.py` ve `tests/test_env.py` aynı fonksiyonu
kullanıyor, hepsinin sayıları yeniden üretilmeli.

### 11.4 Curriculum — kolay haritadan zora

Medyan haritada %8'lik tavanla **sıfırdan** başlamak, erken eğitimde
neredeyse hiç pozitif örnek görmemek demek. MARL-Pathfinding'in dersine
sadık kalıp radar sayısını rampalıyoruz:

```python
n_radar = round(10 + 30 * min(1.0, episode / (0.6 * total_episodes)))   # 10 -> 40
```

10 radarda oracle %92.5 ve **medyan %100** (bol pozitif örnek), 40'ta ortalama
%32 / medyan %7. Ajan önce "hedefe git ve halkalardan kaç" temel davranışını
öğrenir, sonra yoğun haritaya taşınır. **Değerlendirme her zaman 40 radarda**
yapılır — curriculum sadece eğitim örneklemesi, metrik değil.

### 11.5 Episode başına maliyet

Her episode yeni harita → `zone_map` + `risk_distance_map` yeniden hesaplanır.
Ölçüldü: **~0.33 s/harita** (40 radar, 1000×1000 fast sweeping dahil; tarama
betiği harita başına 3 farklı `risk_w` çözdüğü için orada ~1.0 s görünüyor).

Bu **ihmal edilebilir değil**: 20 000 episode × 0.33 s ≈ **1.8 saat** sadece
harita üretimi. İki hafifletme:
- Radar setini örnekle, ama risk-mesafe haritasını **ajan başına değil harita
  başına bir kez** çıkar (zaten öyle).
- `max_iter`'ı erken çıkışla sınırla; sweeping tipik olarak 6–10 turda sabit
  noktaya oturuyor, 120 tavanı sadece patolojik haritalar için.

`RISK_CACHE`/`ZONE_CACHE` artık işe yaramaz — harita her episode değişiyor,
onbellek **devre dışı bırakılmalı** (yoksa sessizce YANLIŞ haritayı okur;
bu hata türü hiçbir yerde patlamaz, en tehlikelisi).

### 11.6 Metrikler — hangisi anlamlı kalır

%17 tavanla ham `team_success` çok gürültülü. **Asıl metrik `analytic_surv`
olmalı**: ajanın SEÇTİĞİ yolun matematiksel hayatta kalma olasılığı. Zar
sonucundan bağımsız, tek episode'dan bile ölçülebilir, ve doğrudan
oracle'la kıyaslanabilir:

| Metrik | Tanım | Hedef |
|---|---|---|
| **`surv_ratio`** | `analytic_surv(ajan yolu) / analytic_surv(oracle yolu)` | 🔑 **YENİ ANA METRİK**, 1.0 = mükemmel |
| `team_success` | ≥1 uçak vardı | gürültülü, yine de raporlanır |
| Tetiklenen radar sayısı | kaç ayrı sistem angaje etti | oracle'ınkiyle kıyasla |
| `route_overlap` | iki uçağın yol örtüşmesi | koordinasyon göstergesi |

`surv_ratio` bu aşamanın en önemli eklentisi — onsuz "%20 başarı" iyi mi kötü
mü söylenemez (oracle da %17.6 ise mükemmeldir).

### 11.7 Yapılacaklar listesi (akşam için)

**Sırayla yapılmalı** — 1 ve 2 düzeltilmeden eğitim koşmanın anlamı yok.

**1. Öğrenme sinyalini geri getir (§11.2 Sorun A)** — ✅ **YAPILDI**
- [x] `env/strike_env.py` `reset()`: `self.dist_scale = max(dist[s1], dist[s2], 1.0)`
- [x] `_phi()` ve gözlem skaları #11: `max_man` yerine `dist_scale`
- [x] Doğrulandı: `Φ(B) = 0.0000`, `Φ(H) = 1.0000`, oracle yolu boyunca monoton
      artıyor; ölçüm 15 rastgele haritada tekrarlandı (§11.2 tablosu)

**2. Ödül kalibrasyonu (§11.2 Sorun B)** — ✅ **YAPILDI**
- [x] `config.py`: `R_RISK_COEF = 15.0 → 7.5` (risk iki kez sayılıyordu)
- [x] `risk_oracle.risk_distance_map` önbellek adına `risk_w` eklendi —
      `RISK_W` 1500→750 düşünce eski önbellek sessizce okunuyordu
- [x] `config.py`: `R_TIMEOUT = -50`, `R_ALL_DEAD = -25` (§11.8 kapıları)
- [ ] `baselines/policies.py` ile doğrula: scripted oracle'ın episode getirisi
      oyalanan politikadan **yüksek** çıkmalı (rastgele harita baseline'ları
      `eval/evaluate.py` işi — ortak harita seti gerekiyor)

**3. Rastgele harita altyapısı** — ✅ **YAPILDI**
- [x] `config.py`: `N_RADAR = 40`, `RADAR_RANDOM = True`, curriculum sabitleri,
      `RISK_CACHE = None`, `ZONE_CACHE = None`, `EVAL_SEED_BASE`
- [x] `env/sampler.py`: `sample_radars` (üniform, çakışma serbest),
      `train_map_seed` / `eval_map_seeds` (ayrık aralıklar), `curriculum_n_radar`
- [x] `env/strike_env.py`: `_build_map()` + `reset(map_seed=…, n_radar=…)`,
      her episode yeni radar seti, önbelleksiz
- [x] `_prev_zone = zone[B]` (kalkış istisnası) — `survival_prob` da aynı
- [x] `_radar_at` O(1)'e indirildi: seviye zone haritasından okunuyor, 40 radar
      taraması sadece alarm kuplajı açıkken (yoksa episode başına ~9M işlem)
- [x] `two_agent.py` çalıştırıcıları `reset_kwargs` alıyor; `train.py` eğitimde
      curriculum, değerlendirme ve demo'da held-out `map_seed` kullanıyor

**Ölçüldü** (smoke test): 5 reset → 5 farklı harita; harita kurma
**0.251 s/episode** (20 000 episode için ~1.4 saat ek yük); aynı `map_seed`
aynı haritayı veriyor; eğitim tohumu max 9.99e7 < test tohumu min 9e8.

**4. Ölçüm** — sırada
- [ ] `eval/evaluate.py`: `surv_ratio` + `eval_map_seeds()` ortak test seti
      (her algoritma AYNI 100 haritada ölçülmeli, yoksa kıyas anlamsız)
- [ ] `viz`: yol çizimi radar setini JSON'dan okumalı (`run_demo` artık
      `radars` + `map_seed` yazıyor, çizim tarafı henüz güncellenmedi)

**5. Regresyon** — ✅ **YAPILDI**
- [x] `test_random_maps` — her reset farklı harita, `map_seed` determinizmi,
      `n_radar` ezmesi, **eğitim/test tohum aralıkları kesişmiyor**, curriculum
- [x] `test_takeoff_exception` — 30 haritanın **12'sinde B bir halkanın içinde**
      (%40! nadir köşe durumu değil), `_prev_zone` doğru kuruluyor
- [x] `test_reward_hacking_gates` — §11.8'deki üç kapı
- [x] `test_phi_not_saturated` — yoğun haritada `dist(B) > max_man` (tuzak
      gerçek) ve `dist_scale` ile Φ monoton
- [ ] `map_check.py` / `policies.py` yeniden koşulup §0.3 sayıları güncellenmeli
      (`greedy_path` düzeltmesi sonrası)

### 11.8 🛡️ Ödül hackleme duruşu

> **Burak (2026-08-06):** *"aynı zamanda reward hacking'e de karşı bi
> duruşumuz olsun"*

Rastgele haritada tavan düştüğü için (medyan oracle %7.2) ajanın **görevi
yapmadan puan toplama** yolları kârlı hale gelebiliyor. Üç kapı tespit edildi
ve üçü de `tests/test_reward_hacking_gates` ile **kilitlendi** — ödül
değerleri elle değiştirilirse test patlar.

| # | Açık | Nasıl sömürülürdü | Kapatan kural |
|---|---|---|---|
| 1 | **Oyalanma** | Güvenli köşede dolan, süreyi doldur | `R_TIMEOUT < 2·R_DEATH` → `−50 < −30` ✅ |
| 2 | **İntihar** | Umudu kes, iç halkaya dal, episode'u erken bitir | `2·R_DEATH + R_ALL_DEAD ≤ R_TIMEOUT` → `−55 ≤ −50` ✅ |
| 3 | **Shaping farmlama** | İleri-geri gidip shaping ödülü biriktir | Potential-based shaping teleskopik (Ng ve ark. 1999): kapalı döngüde net katkı 0 ✅ |

**Kapı 2, Kapı 1'i kapatmanın YAN ÜRÜNÜ.** `R_TIMEOUT`'u −10'dan −50'ye
çekince, umudunu kesen ajan için en ucuz çıkış kasten ölmek oluyordu (eski
değerlerle `2×(−15) + (−10) = −40`, timeout −50 → intihar 10 puan kârlı).
`R_ALL_DEAD = −25` bunu kapatıyor. **İkisi birlikte ayarlanır**, tek başına
değiştirmek açık yaratır.

> Not: *ilerlerken* ölmek serbest ve olmalı — shaping yol boyunca zaten
> toplandığı için "deneyip yolda ölen" ajan "hiç denemeyen"den çok daha
> yüksek puan alır. Cezalandırdığımız şey **kasten erken bitirmek**.

**Aşırı öğrenmeye karşı duruş** (aynı madalyonun diğer yüzü — ezberlenmiş bir
politika da "hack"tir):

1. **Her episode taze harita.** Eğitimde harita tohumu env'in kendi
   rng'sinden çekilir, sabit bir havuz yok.
2. **Eğitim / test tohum aralıkları ayrık.** Eğitim `0 … 1e8`, test
   `9e8 …`. Ajanın testte gördüğü bir haritayı eğitimde görme olasılığı
   **sıfır** — testte iyi olmak genellemeden başka bir şeyle açıklanamaz.
3. **Demo'lar da held-out.** Rapora giren yol çizimleri eğitimde
   görülmemiş haritalardan.
4. **Curriculum sadece eğitimde.** Değerlendirme her zaman tam 40 radarda.
5. **Metrik hile-dayanıklı.** `surv_ratio` = ajanın seçtiği yolun analitik
   hayatta kalma olasılığı / oracle'ınki. Zar sonucundan bağımsız, yolun
   deterministik bir fonksiyonu — şansla iyi görünmek mümkün değil.

---

## 12. Tek paragraf özet

> **Not (2026-08-06):** Bu özetin ilk hali `STEP_SIZE=20` ile 51x51 latise
> inmeyi anlatıyordu. O karar **iptal edildi** — grid tam çözünürlükte
> (1000x1000) kalıyor (§1) ve harita her episode rastgele üretiliyor (§11).
> Aşağısı güncel hali.

Gridi **1000x1000 tam çözünürlükte** tut (bunun bedeli: episode 2000+ adım,
bu yüzden `GAMMA = 0.9998` — 0.99'da hem hedef ödülü hem shaping sinyali
matematiksel olarak yok oluyor; ve gözlem penceresi `PATCH_STRIDE=16` ile
seyrek örnekleniyor ki ajan 221 hücrelik bir halkanın *sınırını* görebilsin)
→ riski **adım başına hazard** olarak modelle (senin %20/%90'ını "boydan boya
geçiş" referansıyla kalibre ederek; formül ölçekten bağımsız) → Dijkstra
oracle'ı yaz ve §0.3'teki hayatta kalma sayılarını kendi kodunla yeniden üret
→ tek uçak DQN'i doğrula → IQL/VDN'i koş ve **sabit haritada üçünün de aynı
çıktığını dürüstçe göster** → **§11'e geç: her episode 40 rastgele radar**
(trivial'lik sorununu kökten çözer, projeyi gerçek bir genelleme problemine
çevirir; `R_TIMEOUT = −50` düzeltmesi ve **per-radar giriş** muhasebesi bu
aşamanın ön koşulu) → **radar alarm kuplajını aç** ki VDN'in IQL'i geçmesi
için fiziksel bir kanal oluşsun → QMIX ile additivity hipotezini test et →
çiz, yaz.

**Altın kural:** Başarı oranını **asla tek başına** raporlama. Yanına her zaman
`analytic_survival` (üretilen yolun matematiksel hayatta kalma olasılığı) ve
`radar_exposure` koy — şans eseri sağ kalan bir politika ile gerçekten güvenli
bir politika ancak böyle ayrılır. Random haritada (§11) bu kural **zorunluluğa**
dönüşür: oracle tavanı %17.6 olduğu için "%20 başarı" tek başına iyi mi kötü mü
söylemez — `surv_ratio` (ajan yolu / oracle yolu) olmadan hiçbir sayı yorumlanamaz.
