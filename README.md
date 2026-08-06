# MARL Strike Mission

İki uçak, 40 rastgele radar, bir hedef. **IQL / VDN / QMIX** karşılaştırmalı
kooperatif multi-agent reinforcement learning deneyi.

> **Bu bir ürün değil, algoritma seçme deneyi.** Asıl iş `../rs1/`'de:
> Rocksoft staj görevi — Elektronik Harp simülasyonunda single-agent RL'i
> multi-agent'a çevirmek. Bu proje "iki ajanı koordine etmek için hangi
> algoritma" sorusunun test tezgâhı. Detay: [Strike_Mission.md §0](Strike_Mission.md)

Durum: 🔬 **Deney koştu, ilk sonuçlar alındı** · Branch `iql_vdn_qmix`

---

## Sonuçlar

Aynı 50 held-out haritada, aynı tohum, aynı transfer başlangıcı, 1000 episode:

| | IQL | **VDN** | QMIX |
|---|---:|---:|---:|
| rotası hedefe varıyor | 8% | **30%** | **30%** |
| **`surv_ratio`** (ana metrik) | 0.0001 | **0.0423** | 0.0384 |
| `mission_prob` | 0.0000 | **0.0249** | 0.0174 |
| takım başarısı | 0.0% | 2.0% | 0.0% |
| ölü / episode | 1.38 | **1.18** | 1.56 |
| yol örtüşme | 0.369 | 0.600 | **0.072** |

*(oracle tavanı: takım %65.4)*

**Bulgu: değer ayrıştırması (VDN/QMIX) bağımsız öğrenmeyi (IQL) açık ara
geçiyor.** IQL'in rotası haritaların %8'inde hedefe varıyor, diğerlerinde %30 —
bu fark zar şansı değil, deterministik rota kalitesi. **VDN ile QMIX bu veriyle
birbirinden ayrılamıyor.**

⚠️ **Tek tohum.** QMIX 850, diğerleri 1000 episode. Mutlak seviye düşük (tavan
%65.4, en iyi 0.042) — sonuç *"hangisi daha iyi"* için geçerli, *"yeterince iyi
mi"* için değil. Rapora girmeden önce ≥3 tohumla tekrarlanmalı.

### Neden ham başarı oranı değil `surv_ratio`

Ham başarı üçünü **ayıramıyor** (2% / 0% / 0% — hepsi 0-1 episode). `surv_ratio`
ajanın *niyet ettiği* rotanın analitik hayatta kalma olasılığını oracle'ınkine
bölüyor: zar atılmıyor, gürültü yok, ve hedefe varmayan rota 0 alıyor (yani
"hiç hareket etme, güvende kal" ile şişirilemiyor).

---

## Ortam

```
1000x1000 grid, TAM COZUNURLUK (1 hucre = 1 birim)
B (-500,500) = (0,0)  ->  H (500,-500) = (999,999)   optimal 1998 adim, limit 2800

40 RASTGELE radar / episode  (egitimde de testte de, cakisma serbest)
  dis halka 160x160  (+-80 hucre)   ->  girise %20 olum
  ic  halka 100x100  (+-50 hucre)   ->  girise %90 olum
```

**Risk kuralı `per_entry`:** bölgeye **girişte tek zar**. Sürede birikme yok —
iç halkada 2 adım da atsan 200 adım da atsan risk aynı. Çıkıp tekrar girmek
yeni bir zar. Üst üste binen radarlar riski **artırmaz** (`zone = max`): aynı
anda 4 radarın alanındaysan da dış halka %20.

Kalkış istisnası **yok** — B bir halkanın içindeyse zar atılır.

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
| ikisi de öldü | −50 | intihar kapısı |
| risk önizleme | `15 × p(giriş)` | yoğun, deterministik |

`GAMMA = 0.9998` — episode 2000+ adım olduğu için 0.99'da hem hedef ödülü hem
shaping sinyali matematiksel olarak yok oluyor.

---

## Kurulum

```bash
python3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Kullanım

```bash
python -m tests.test_env
```

```bash
python -m baselines.scan_random_maps
```

```bash
python train.py --algo vdn --episodes 1000 --eval-every 250 --eval-episodes 30 --resume-from pathfinding --tag vdn_r1
```

```bash
python -m eval.evaluate --algo vdn --ckpt runs/ckpt/vdn_r1_last.pt --maps 50 --tag vdn_r1
```

Referans politikaları tek başına ölçmek için `--ckpt` vermeden çalıştır.

**Önemli bayraklar:** `--n-radar N` curriculum'u kapatıp radar sayısını sabitler ·
`--eps-start 0.2` eğitilmiş bir checkpoint'ten devam ederken şart (1.0'dan
başlamak öğrenilmiş politikayı yüzlerce episode boyunca rastgele aksiyonlarla
bozar) · `--hazard per_step` risk modelini ablation olarak değiştirir.

---

## Öğrenilenler

Bu projede üç ödül/metrik açığı **ölçümle** bulundu — üçü de "makul görünen ama
doğrulanmamış" varsayımlardan doğdu:

1. **Shaping'in terminal sızıntısı.** Episode'u bitiren adımda shaping
   *atlanıyordu*; atlamak Φ'yi sıfırlamak değil, o yüzden ajan öldüğü yerdeki
   Φ kadar ödülü cebinde tutuyordu. Haritanın ortasında ölmek ~+13 puan bedava
   kârdı ve **üç algoritma da** bu tuzağa düşüp %0'da buluşuyordu — yani fark
   algoritmadan değil ortamdan geliyordu. Düzeltme: terminal adımda `Φ' = 0`
   ile shaping **uygulanır**.
2. **Metrik şişmesi.** `survival_prob` gidilen yolu ölçtüğü için yarıda ölen
   politika "güvenli" görünüyordu (hiç hareket etmeyen ajan 1.000 alırdı).
   Çözüm: ölüm zarı kapalı koşup *niyet edilen* rotayı ölçmek, hedefe varmayan
   rotaya 0 vermek.
3. **IQL'in yarısı yüklenmiyordu.** `evaluate.py` IQL için sadece `agent1`'i
   yüklüyor, `agent2` rastgele kalıyordu — karşılaştırmayı sistematik olarak
   IQL aleyhine bozuyordu.

Ayrıca `train_bc.py` (oracle davranış klonlama) bir **teşhis aracı** olarak
yazıldı ve %99.3 uzman eşleşmesiyle şunu kanıtladı: doğru cevap gözlemin
içinde (komşuların risk-mesafe farkları), yani sorun temsilde değil öğrenmede.

## Kardeş proje

[`MARL-Pathfinding`](../MARL-Pathfinding) — ajan/eğitim/eval altyapısı buradan
taşındı. `--resume-from pathfinding` ile eğitilmiş modelleri doğrudan yükleniyor
(14/14 tensör, %100 uyum: gözlem 898 boyut ve 16 skaların sırası birebir aynı
tutuldu).
