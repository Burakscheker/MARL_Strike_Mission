# MARL Strike Mission

İki uçak, 25 rastgele radar, bir hedef. **VDN / QMIX / MAPPO / HAPPO**
karşılaştırmalı kooperatif multi-agent reinforcement learning deneyi.

> **Bu bir ürün değil, algoritma seçme deneyi.** Asıl iş `../rs1/`'de:
> Rocksoft staj görevi — Elektronik Harp simülasyonunda single-agent RL'i
> multi-agent'a çevirmek. Bu proje "iki ajanı koordine etmek için hangi
> algoritma" sorusunun test tezgâhı. Detay: [Strike_Mission.md §0](Strike_Mission.md)

Durum: 🔬 **VDN olgunlaştı; MAPPO/HAPPO yeni portlandı, kıyas henüz eşit bütçeli değil**

---

## Sonuçlar

### VDN — doğrulanmış en iyi sonuç

100 held-out haritada (25 radar, tohum 9e8+, eğitimle kesişmiyor),
`--eps-start 0.1` "fast-eps" tarifi:

| | değer |
|---|---:|
| **`surv_ratio`** (ana metrik) | **0.7635** (medyan 1.0000) |
| rotası hedefe varıyor | 80% (80/100 harita) |
| takım başarısı (zar açık) | **69.0%** *(oracle tavanı 81.5%)* |
| ölü uçak / episode | 0.39 |
| timeout | 35% |
| adım | 2632 *(optimal 1998)* |

`surv_ratio` medyanı **1.0000** — yani haritaların yarısından fazlasında ajan
Dijkstra oracle'ı kadar güvenli bir rota çiziyor. 5 tohumda tekrarlandı,
ortalama **%69.2**. Checkpoint: `runs/ckpt/it2_vdn_epsfast.pt`

### Dört algoritma — ön sonuçlar

⚠️ **Bu tablo eşit bütçeli bir kıyas DEĞİL.** VDN 500 episode + 100 harita ile
ölçüldü; diğerleri çok daha az eğitim ve 20 harita gördü. Aradaki fark büyük
ölçüde bunu yansıtıyor, algoritma gücünü değil.

| | VDN | QMIX | MAPPO | HAPPO |
|---|---:|---:|---:|---:|
| eğitim (episode) | ~500 | 150 | ~50 | ~50 |
| eval harita | 100 | 20 | 20 | 20 |
| `surv_ratio` | **0.7635** | 0.1437 | 0.0586 | 0.0543 |
| rotası hedefe varıyor | **80%** | 50% | 40% | 30% |
| takım başarısı | **69.0%** | 25.0% | 5.0% | 5.0% |
| ölü / episode | **0.39** | 1.00 | 1.25 | 1.15 |

**Neden eşit değil:** VDN'in `--n-envs 32` ile paralel rollout altyapısı aylar
önce olgunlaştı. QMIX'in paralel yolu vardı ama CLI'a **hiç bağlanmamıştı**
(2026-08-28'de bulundu ve düzeltildi); MAPPO/HAPPO ise seri episode topluyordu
— PPO'nun GAE'si episode sınırlarını net bilmek zorunda olduğu için VDN'in
auto-reset deseni doğrudan kullanılamıyor. Bunun için ayrı bir "chunk" tabanlı
paralel toplayıcı yazıldı (`ppo_parallel_rollout`), episode süresi
**~27 s → ~5.6 s**'ye düştü. Dört algoritmanın eşit bütçeli koşusu sıradaki iş.

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
öğrenmeyi çökertiyor:

```bash
python train.py --algo vdn --episodes 500 --eval-every 25 --n-envs 32 --device cuda --eps-start 0.1 --tag vdn_r1
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
`--eps-start 0.1` VDN/QMIX için kritik ·
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

Çalışmayan şeyi belgelemek de sonuçtur:

| yaklaşım | sonuç |
|---|---|
| BC ön-eğitim → RL fine-tune (3 varyant) | ❌ BC'nin sınırsız Q-ölçeği TD ile yeniden kalibre olunca sıralama siliniyor |
| Dueling + Prioritized Replay | ❌ öğreniyor (0→0.23) ama geç bozuluyor, eşiği geçmiyor |
| QR-DQN (distributional) | ❌ elendi |
| Munchausen RL / Advantage Learning | ❌ elendi |
| LayerNorm Q-ağı (BroNet/CrossQ) | ❌ elendi |
| ham Manhattan shaping | ❌ regresyon (%69 → %20), geri alındı |
| batch 64 / 256 override | ❌ ters etki |
| checkpoint ensemble | ❌ etkisiz |

Asıl atılım mimariden değil **keşif takviminden** geldi: `--eps-start 0.1`
(epsilon'u 1.0 yerine 0.1'den başlatmak) takım başarısını %33'ten %69'a
çıkardı. Eğitilmiş bir başlangıç noktası varken yüksek epsilon, öğrenilmiş
politikayı yüzlerce episode boyunca rastgele aksiyonlarla bozuyor.

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
