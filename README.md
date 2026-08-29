# MARL Strike Mission

İki uçak, 25 rastgele radar, bir hedef. **VDN / QMIX / MAPPO / HAPPO**
karşılaştırmalı kooperatif multi-agent reinforcement learning deneyi.

> **Bu bir ürün değil, algoritma seçme deneyi.** Asıl iş `../rs1/`'de:
> Rocksoft staj görevi — Elektronik Harp simülasyonunda single-agent RL'i
> multi-agent'a çevirmek. Bu proje "iki ajanı koordine etmek için hangi
> algoritma" sorusunun test tezgâhı. Detay: [Strike_Mission.md §0](Strike_Mission.md)

Durum: 🔬 **VDN olgun — takım başarısı %74.4** (deployment: stall-escape) /
**%69.2** (saf greedy). QMIX %33; MAPPO/HAPPO eşit bütçeli kıyas henüz yapılmadı.

---

## Sonuçlar

### VDN — en iyi sonuç

100 held-out haritada (25 radar, tohum 9e8+, eğitimle kesişmiyor),
`--eps-start 0.1` "fast-eps" tarifi + 5 zar-tohumu ortalaması:

| politika | takım başarısı | timeout | ölü/ep | not |
|---|---:|---:|---:|---|
| **stall-escape deployment** | **%74.4** *(72–76)* | %9 | 0.63 | `eval/deploy_eval.py` |
| saf greedy (`argmax Q`) | %69.2 *(68–70)* | %34 | 0.39 | `eval/evaluate.py` (kanonik) |
| — oracle tavanı | %81.5 | | | Dijkstra referansı |

Ek metrikler (greedy, tek koşu): `surv_ratio` **0.7635** (medyan **1.0000** —
haritaların yarısından fazlasında ajan Dijkstra oracle'ı kadar güvenli bir rota
çiziyor), rota hedefe varıyor %80, adım 2632 *(optimal 1998)*.
Checkpoint: `runs/ckpt/it2_vdn_epsfast.pt`

**stall-escape nedir:** greedy `argmax(Q)` ~25 haritada takılıyor — risk-mesafesi
azalmayı bırakıyor, ajan timeout yiyor. Sebep: action-gap ~0.03, Q-ağı "ileri git"
ile "bekle" farkını argmax'ın kararlı seçebileceği kadar büyük kodlayamıyor
(it8'de ölçüldü). Düzeltme: ajan **150 adımdır ilerlemiyorsa** `argmax` yerine
`softmax(Q/0.03)` örnekliyor (yalnız o ajan, yalnız stall süresince), ilerleyince
greedy'ye dönüyor. **Sadece öğrenilen Q** — elle hedef-arama, kural, oracle YOK;
stall sinyali de ajanın zaten gözlemde gördüğü risk-mesafesi. Eğitim değişmiyor,
sadece çıkış politikası — gerçek RL sistemlerinde standart bir teknik.

```bash
python -m eval.deploy_eval --ckpt runs/ckpt/it2_vdn_epsfast.pt
```

Buraya varmak için mimari tarafında 19 iterasyon denendi (aşağıda "Elenen
yaklaşımlar"); hiçbiri greedy'yi %69'un üstüne çıkarmadı. Sıçrama deployment
politikasından geldi.

### Dört algoritma — ön sonuçlar

⚠️ **Bu tablo eşit bütçeli bir kıyas DEĞİL.** Tüm gayret VDN'e gitti (19
iterasyon, 100 harita, çok sayıda tarif). QMIX bir kez 100 haritada ölçüldü;
MAPPO/HAPPO küçük bütçe + 20 harita gördü. Aradaki fark büyük ölçüde bunu
yansıtıyor, algoritma gücünü değil.

| | VDN | QMIX | MAPPO | HAPPO |
|---|---:|---:|---:|---:|
| eğitim (episode) | ~250 | ~150 | ~100–160 | ~50 |
| eval harita | 100 | 100 | 20 (+40 eğitim-içi) | 20 |
| `surv_ratio` | **0.7635** | 0.4669 | 0.0586 | 0.0543 |
| rotası hedefe varıyor | 80% | **94%** | 40% | 30% |
| **takım başarısı** | **%74.4** / %69.2 greedy | %33.0 | ~%5–10 | ~%5 |
| ölü / episode | **0.39** | 1.29 | 1.25 | 1.15 |

QMIX (`runs/ckpt/it1_qmix.pt`, standart tarif): rotası 100 haritanın 94'ünde
hedefe **varıyor** — navigasyonu öğrenmiş — ama `surv_ratio` 0.47, yani rota
oracle'ınkinin yarısı kadar güvenli, episode başına 1.29 ölüm. QMIX'e fast-eps
tarifi düzgün bütçeyle hiç denenmedi (bir kez `--episodes 120` confound'uyla
denendi, ep24'te %45 tepe gördü ama o sayı eps-takvimi artefaktı).

MAPPO entropi curriculum (0.03→0.005) ile eğitim-içi takım %5 → %10'a çıktı ama
orada takıldı; HAPPO ep32'de kesildi. İkisi de on-policy ve `--n-envs 32` chunk
toplayıcısıyla (`ppo_parallel_rollout`, episode ~27 s → ~5.6 s) hızlandırıldı
ama eşit bütçeli tam koşu henüz yapılmadı.

**Neden eşit değil:** VDN'in paralel rollout altyapısı aylar önce olgunlaştı.
QMIX'in paralel yolu vardı ama CLI'a **hiç bağlanmamıştı** (2026-08-28'de bulundu
ve düzeltildi). MAPPO/HAPPO için ayrı chunk-tabanlı toplayıcı yazılması gerekti
(PPO'nun GAE'si episode sınırlarını net bilmek zorunda). **Dört algoritmanın
eşit bütçeli koşusu sıradaki iş.**

### Neden ham başarı oranı değil `surv_ratio`

Ham başarı oranı harita şansıyla oynuyor: aynı 100 haritada oracle'ın kendi
hayatta kalma olasılığı ortalama 0.7487, ama harita başına 0'a yakınla 1'e
yakın arasında geziniyor. `surv_ratio` ajanın *niyet ettiği* rotanın analitik
hayatta kalma olasılığını oracle'ınkine bölüyor: zar atılmıyor, gürültü yok, ve
hedefe varmayan rota 0 alıyor — yani "hiç hareket etme, güvende kal" ile
şişirilemiyor.

---

## Ortam

```
1000x1000 grid, TAM COZUNURLUK (1 hucre = 1 birim)
B (0,0)  ->  H (999,999)      optimal 1998 adim, limit 4000

25 RASTGELE radar / episode  (egitimde de testte de, cakisma serbest)
  dis halka 160x160  (+-80 hucre)   ->  girise %20 olum
  ic  halka 100x100  (+-50 hucre)   ->  girise %90 olum
```

**Risk kuralı `per_entry`:** bölgeye **girişte tek zar**. Sürede birikme yok —
iç halkada 2 adım da atsan 200 adım da atsan risk aynı. Çıkıp tekrar girmek
yeni bir zar. Üst üste binen radarlar riski **artırmaz** (`zone = max`): aynı
anda 4 radarın alanındaysan da dış halka %20.

Kalkış istisnası **yok** — B bir halkanın içindeyse zar atılır. (Haritaların
~%15'inde B, ~%20'sinde H bir halkanın içinde.)

İki uçak aynı anda kalkar, aynı yoldan gidebilir (çarpışma yok). **En az bir
uçak varırsa takım ödülü fullenir.**

### Ödül

| | değer | not |
|---|---:|---|
| adım | −0.01 | 1998 adım → toplam −20 |
| ölüm | −15 | uçak başına |
| ilk varış | **+50** | takım ödülü full |
| ikinci varış | +12 | "ikide olsa" bonusu |
| timeout | −50 | oyalanma kapısı |
| ikisi de öldü | −110 | intihar kapısı (ölüm cezalarının üstüne) |
| risk önizleme | `65 × p(giriş)` | yoğun, deterministik |

`GAMMA = 0.9998` — episode 2000+ adım olduğu için 0.99'da hem hedef ödülü hem
shaping sinyali matematiksel olarak yok oluyor. `SHAPING_COEF = 120`,
risk-farkında Dijkstra mesafesinden türetilmiş potansiyel tabanlı shaping.

**İntihar kapısı doğrulaması:** `2×(−15) + (−110) = −140 ≤ −50 + 4000×(−0.01) = −90`
— yani "ikisi de ölsün, bitsin" oyalanmaktan hep daha kötü. `MAX_STEPS`
değiştiğinde bu eşitsizlik testle yeniden kontrol ediliyor.

### Gözlem

```
OBS_DIM   = 1345 = 3x21x21 (yerel + iz + kuresel tehlike) + 22 skalar
STATE_DIM =  898 = 2x21x21 (iki ajan cevresi)             + 16 skalar
```

22 skaların son 4'ü **eylem-özgü anlık risk**: "bu yöne girersem BU ADIMDA ne
kadar ölüm riski alıyorum" (UP/RIGHT/DOWN/LEFT). Ortamın gerçek `_hazard()`
modeliyle **aynı fonksiyondan** üretiliyor, yani gözlem ile gerçek davranış
çelişemiyor. Global state'e de iki ajanın 4'er riski eklendi — merkezi kritik
(QMIX mixer + MAPPO/HAPPO) önceden bu bilgiyi hiç görmüyordu.

---

## Kurulum

```bash
python3 -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

## Kullanım

```bash
python -m pytest tests/ -q
```

VDN'in en iyi tarifi (fast-eps) — `--eps-start 0.1` şart, 1.0'dan başlamak
öğrenmeyi çökertiyor. Tepe ep20–25 civarında; sonra "aşırı-temkin drift"
başlıyor, `mission_prob`-birincil checkpoint seçici erken noktayı yakalıyor:

```bash
python train.py --algo vdn --episodes 250 --eval-every 15 --eval-episodes 50 --n-envs 32 --device cuda --eps-start 0.1 --tag vdn_r1
```

MAPPO / HAPPO (on-policy, paralel toplayıcı ile):

```bash
python train.py --algo mappo --episodes 300 --eval-every 32 --n-envs 32 --device cuda --tag mappo_r1
```

100 held-out haritada değerlendirme:

```bash
python -m eval.evaluate --algo vdn --ckpt runs/ckpt/vdn_r1.pt --maps 100 --tag vdn_r1
```

Referans politikaları (oracle / merdiven / rastgele monoton) tek başına ölçmek
için `--ckpt` vermeden çalıştır.

**Önemli bayraklar:**
`--n-envs 32` paralel rollout (dördü de destekliyor; GPU'da şart) ·
`--eps-start 0.1` VDN için kritik (QMIX'te düzgün bütçeyle denenmedi) ·
`--n-radar N` curriculum'u kapatıp radar sayısını sabitler ·
`--hazard per_step` risk modelini ablation olarak değiştirir

---

## Öğrenilenler

### Ölçümle bulunan hatalar

1. **Shaping'in terminal sızıntısı.** Episode'u bitiren adımda shaping
   *atlanıyordu*; atlamak Φ'yi sıfırlamak değil, o yüzden ajan öldüğü yerdeki Φ
   kadar ödülü cebinde tutuyordu. Haritanın ortasında ölmek ~+13 puan bedava
   kârdı ve **tüm algoritmalar** bu tuzağa düşüp %0'da buluşuyordu — fark
   algoritmadan değil ortamdan geliyordu. Düzeltme: terminal adımda `Φ' = 0`.

2. **Metrik şişmesi.** `survival_prob` gidilen yolu ölçtüğü için yarıda ölen
   politika "güvenli" görünüyordu (hiç hareket etmeyen ajan 1.000 alırdı).
   Çözüm: ölüm zarı kapalı koşup *niyet edilen* rotayı ölçmek, hedefe varmayan
   rotaya 0 vermek.

3. **Paralel rollout'ta timeout bootstrap'i.** Vec-env auto-reset, `next_obs`'u
   **reset sonrası** gözlemle değiştiriyordu. Ölüm için zararsız (`done`
   çarpanı sıfırlıyor) ama timeout'ta `push_done=False` olduğu için bootstrap
   *tam olarak* o gözlemi kullanıyor — ve o, rastgele yeni bir haritanın ilk
   karesiydi. Bu oturumun tüm GPU koşuları etkilenmişti. Aynı hata `next_mask`
   ve QMIX'in `state`'i için de vardı.

4. **CUDA non-determinizmi.** Aynı `--seed` ile GPU'da iki koşu **tamamen
   farklı** sonuç veriyordu (ep200'de takım %50 vs %22) — tek-tohum kıyasları
   güvenilmez hale getiriyordu. Kök neden: `nn.AdaptiveAvgPool2d`'nin CUDA
   backward'ının deterministik karşılığı **yok**;
   `torch.use_deterministic_algorithms(True)` açıkken bile sessizce
   non-deterministik kernele düşüyor. Çözüm: PyTorch'un kendi bölme formülünü
   manuel dilimleme+ortalama ile uygulayan `DeterministicAdaptiveAvgPool2d`
   (sayısal olarak eşdeğer, max fark 1.2e-7 — eski checkpoint'ler geçerli
   kalıyor). Doğrulama: iki bağımsız koşunun logları `diff` ile **birebir aynı**.

5. **QMIX'in paralel yolu bağlanmamıştı.** `qmix_parallel_rollout` yazılmıştı
   ama CLI dispatch'i sadece VDN'i kabul ediyordu — `--n-envs 32` verilse bile
   QMIX sessizce tek-env'e düşüyor, GPU'da hiç kazanç sağlamadan dispatch
   overhead'i yüzünden **yavaşlıyordu**.

### Elenen yaklaşımlar

Çalışmayan şeyi belgelemek de sonuçtur. `--eps-start 0.1` atılımından (%33 → %69)
sonra greedy tavanı %75'e taşımak için 19 mimari/eğitim iterasyonu denendi —
**hiçbiri greedy'yi %69'un üstüne çıkarmadı** (asıl sıçrama sonradan deployment
tarafından, stall-escape ile geldi):

| yaklaşım | sonuç |
|---|---|
| BC ön-eğitim → RL fine-tune (3 varyant) | ❌ BC'nin sınırsız Q-ölçeği TD ile yeniden kalibre olunca sıralama siliniyor |
| Dueling (± Prioritized Replay) | ❌ rastgele init edilen A-head 250 episode'da yakınsamıyor, pervasız |
| Advantage Learning (Bellemare 2016) | ❌ action-gap'i 15× açtı (0.03→0.36) ama greedy "aşırı temkinli"ye kaydı, %45 |
| Munchausen RL (Vieillard 2020) | ❌ entropi Q-manzarasını düzleştirdi, argmax bozuldu, %40 |
| QR-DQN distributional (16 kuantil ± iyimserlik) | ❌ 80-çıkışlı başlık fast-eps bütçesinde güvenli rota öğrenemedi, pervasız |
| LayerNorm Q-ağı (BroNet/CrossQ) | ❌ hedef-yön magnitude sinyalini normalize edip sildi, VARIS %2.5 |
| ham Manhattan shaping | ❌ regresyon (%69 → %20), geri alındı |
| batch 64 / 256 override | ❌ iki yönde de ters (128 sweet spot); büyük batch → karamsar sabit-noktaya daha hızlı yakınsıyor |
| hızlı curriculum (25 radara ep25'te) | ❌ hard-maps-erken = pozitif örnek yok = "hiç hareket etme" |
| checkpoint ensemble | ❌ snapshot'lar korele, zayıf olanlar Q-ortalamayı çekiyor |
| deployment uniform-eps / düz Boltzmann | ❌ timeout'u ölüme takas ediyor, net negatif |
| çoklu net-tohum taraması | ❌ yüksek varyans; seed 0 = %70, seed 1 = %27, seed 2 = %48 (şanslı outlier) |

**Kök neden:** `it2_vdn_epsfast.pt` %69 = kırılgan dengeli bir yerel optimum.
Değer fonksiyonu radar ölüm cezasını aşırı ağırlıklandırıp karamsar/aşırı-temkinli
bir sabit-noktaya oturuyor; fast-eps + erken checkpoint bunu *oturmadan önce*
yakalıyor. Dengenin herhangi bir bileşeni (batch, mimari, curriculum, target,
seed) değişince çöküyor. Temiz/hızlı/keskin öğrenme oraya *daha hızlı* götürüyor
— gürültü (eps, orta batch) oturmayı geciktiriyor. **Tek işe yarayan ek:**
deployment'ta stall-tetikli Boltzmann (yukarıda, %74.4) — ama o da öğrenmeyi
değil çıkış politikasını değiştiriyor.

Bu 19 iterasyonun kodu (AL/Munchausen/QR-DQN/LayerNorm/batch-override/…) tam
teşhisiyle `NOTES.md`'de; repodan `82e7f76`'da çıkarıldı (−392 satır),
`it2_vdn_epsfast` tarifi birebir korundu. Tek kalan ek: `eval/deploy_eval.py`
(yukarıdaki stuck-escape ölçümü).

İki atılım da mimariden değil **politikadan** geldi:
1. **Keşif takvimi** — `--eps-start 0.1` (epsilon'u 1.0 yerine 0.1'den
   başlatmak) takım başarısını %33 → %69'a çıkardı. Eğitilmiş bir başlangıç
   noktası varken yüksek epsilon, öğrenilmiş politikayı yüzlerce episode
   boyunca rastgele aksiyonlarla bozuyor.
2. **Çıkış politikası** — stall-tetikli Boltzmann greedy tavanını (%69)
   deployment'ta %74.4'e taşıdı. Q-ağı zaten doğru sıralamayı biliyor, argmax
   sadece action-gap çok küçük olduğu için kararlı seçemiyor.

`train_bc.py` (oracle davranış klonlama) bir **teşhis aracı** olarak yazıldı ve
%99.3 uzman eşleşmesiyle şunu kanıtladı: doğru cevap gözlemin içinde, yani
sorun temsilde değil öğrenmede.

---

## Kardeş projeler

[`MARL-Pathfinding`](../MARL-Pathfinding) — ajan/eğitim/eval altyapısı buradan
taşındı, `--resume-from pathfinding` ile eğitilmiş modeller yüklenebiliyor.

MAPPO/HAPPO implementasyonu [`euzxx/MARL-pathtfinding`](https://github.com/euzxx/MARL-pathtfinding)
(`mappo_happo` dalı) referans alınarak bu projenin CNN gövdesine ve ortamına
uyarlandı.
