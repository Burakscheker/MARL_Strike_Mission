# Strike Mission — MAPPO ve HAPPO Tasarımı

> 1000×1000 grid üzerinde iki uçağın, üç sabit radarın risk bölgelerinden
> kaçınarak ortak hedefe ulaşmayı öğrendiği kooperatif MARL ortamı.

**Durum:** Tasarım onaylandı

**Branch:** `mappo_happo`

**Güncelleme:** 2026-08-05

## 1. Amaç ve ilk sürümün kapsamı

İki uçak aynı başlangıç noktasından aynı anda kalkar. Her zaman adımında iki
uçak da eşzamanlı olarak bir hücre hareket eder. Aynı hücrede bulunabilir ve
aynı rotayı kullanabilirler; çarpışma mekaniği yoktur.

Görevin ana başarı koşulu, uçaklardan **en az birinin hedefe ulaşmasıdır**.
İlk uçak hedefe ulaştığı anda episode başarıyla biter ve tam takım başarı
ödülü verilir. Bir uçak düşürülürse diğeri göreve devam eder. İki uçak da
düşürülürse episode başarısız biter.

İlk sürüm yalnızca şu iki algoritmayı karşılaştırır:

- MAPPO (Multi-Agent Proximal Policy Optimization)
- HAPPO (Heterogeneous-Agent Proximal Policy Optimization)

IQL, VDN, QMIX, radar alarmı, rastgele radarlar, uçuş dinamiği, iletişim
kesintisi ve gözlem gürültüsü ilk sürümün kapsamı dışındadır.

## 2. Harita ve koordinatlar

Grid, tam sayı koordinatlarla tanımlanır:

```text
x ∈ [-500, 499]
y ∈ [-500, 499]
```

Bu aralık her eksende tam 1000 hücre üretir.

| Öğe | Koordinat |
|---|---:|
| İki uçağın başlangıcı B | `(-500, 495)` |
| Ortak hedef H | `(499, -494)` |
| Radar R1 | `(-280, 220)` |
| Radar R2 | `(200, 100)` |
| Radar R3 | `(-100, -280)` |

Her radarın aynı merkezli iki kare bölgesi vardır:

- Dış kare: 220×220 birim.
- İç kare: 140×140 birim.
- Dış risk halkası, dış karenin içinde fakat iç karenin dışında kalan alandır.
- İç risk bölgesi, iç karenin tamamıdır.

Kareler sürekli koordinat uzayında merkezlerinin çevresinde tanımlanır. Grid
hücreleri yarı-açık sınırlarla eşlenerek her eksende tam 220 ve 140 hücre elde
edilir; böylece çift sayılı kenar uzunluklarında 221 hücrelik kapsama hatası
oluşmaz.

Radar merkezi `(cx, cy)` için kesin üyelik formülleri:

```text
dış kare: cx - 110 <= x < cx + 110 ve cy - 110 <= y < cy + 110
iç kare:  cx -  70 <= x < cx +  70 ve cy -  70 <= y < cy +  70
dış halka = dış kare - iç kare
```

### Boyut değerlendirmesi

Tek bir dış kare harita alanının yaklaşık `%4,84`'ünü, tek bir iç kare
`%1,96`'sını kaplar. Üç radar örtüşmediğinde dış karelerin toplam kapsaması
yaklaşık `%14,52`, iç karelerin toplam kapsaması `%5,88` olur.

Bu oranlar ilk sabit harita için uygundur: radarlar görünür ve anlamlı bir
tehdit oluşturur, ancak haritayı kapatmaz. Haritada risksiz ve en kısa
uzunlukta sınır rotaları bulunduğu için bu kurulum önce ortamın doğruluğunu ve
eğitim hattını kanıtlamak için kullanılacaktır. Daha zor koordinasyon deneyleri
radar randomizasyonu aşamasında ele alınacaktır.

## 3. Aksiyonlar ve episode akışı

Her canlı uçak dört hareket aksiyonundan birini seçer. Ortamın ayrıca yalnızca
ölü uçaklar için açtığı dahili bir `NOOP` aksiyonu vardır:

```text
0 = yukarı    (x, y + 1)
1 = sağ       (x + 1, y)
2 = aşağı     (x, y - 1)
3 = sol       (x - 1, y)
4 = NOOP      (yalnızca ölü uçakta geçerli)
```

- Hareket çözünürlüğü her adımda 1 grid birimidir.
- Grid dışına çıkan aksiyonlar maskelenir ve seçilemez.
- İki aksiyon aynı state üzerinden hesaplanıp birlikte uygulanır.
- Ölü uçak hareket etmez; algoritma tarafında yalnızca zorunlu `NOOP` üretir.
- Başlangıç-hedef Manhattan mesafesi 1.988 adımdır.
- `MAX_STEPS = 3_000` olarak belirlenir.

Bir step sırası:

1. İki actor mevcut gözlemlerden aksiyon seçer.
2. Geçerli hareketler eşzamanlı uygulanır.
3. Yeni risk bölgesi girişleri belirlenir ve bağımsız ölüm zarları atılır.
4. Mesafe shaping'i, ölüm cezaları ve varsa başarı ödülü hesaplanır.
5. İlk hedef varışı, iki uçağın ölmesi veya 3.000 adımlık sınır terminal
   koşulunu üretir.

## 4. Radar risk modeli

Risk, bölgede geçirilen süreye değil **bölgeye girişe** bağlıdır.

| Geçiş | Ölüm | Hayatta kalma |
|---|---:|---:|
| Güvenli alan → dış halka | `%20` | `%80` |
| Dış halka → iç bölge | `%90` | `%10` |

Uçak aynı bölgede 2 adım da 20 adım da kalsa yeniden zar atılmaz. Bölgeden
çıkıp daha sonra yeniden girerse yeni bir giriş oluşur ve yeni zar atılır.
İç bölgeden dış halkaya geri çıkmak da dış halkaya yeni giriş sayılır.

Dışarıdan iç bölgeye ulaşan uçak önce dış halka, sonra iç bölge riskini alır:

```text
P(hayatta kalma) = 0,80 × 0,10 = 0,08
P(ölüm)          = 0,92
```

Her uçak, her radar ve her giriş için rastgelelik bağımsızdır. Aynı yolu aynı
anda kullanan iki uçak aynı zar sonucunu paylaşmaz. Ortam, son bölgeyi radar ve
uçak bazında saklayarak aynı bölgede tekrar zar atılmasını engeller.

## 5. Gözlem ve iletişim

İlk sürümde 1000×1000 görüntü tensörü veya CNN kullanılmaz. Her actor küçük,
normalize edilmiş bir özellik vektörü alır. Bu tercih eğitimi hızlandırır ve
haritadaki gerçek karar değişkenlerini doğrudan temsil eder.

Her uçak şunları görür:

- Kendi `(x, y)` konumu ve hayatta olma durumu.
- Takım arkadaşının `(x, y)` konumu ve hayatta olma durumu.
- Hedefin koordinatı ve hedefe göre bağıl uzaklık.
- Üç radarın merkezleri ve kendi konumuna göre bağıl uzaklıkları.
- Kendisinin ve takım arkadaşının mevcut radar bölgesi: güvenli, dış veya iç.
- Normalize edilmiş zaman adımı.

Bu, öğrenilmiş mesajlaşma protokolü değildir. Uçakların paylaştığı durumun bir
veri bağlantısıyla iletildiği varsayılır. İletişim kaybı ve gecikme daha sonraki
gerçekçilik aşamasına bırakılır.

Radar koordinatları sabit olmasına rağmen gözlemin içinde tutulur. Böylece
radarlar ileride random yapıldığında actor giriş boyutunu ve ağ mimarisini
değiştirmek gerekmez.

## 6. MAPPO/HAPPO mimarisi

Merkezi eğitim, dağıtık çalıştırma (CTDE) kullanılacaktır:

- Her uçak için ayrı bir actor MLP bulunur.
- Actor yalnızca kendi kompakt gözlemini ve aksiyon maskesini kullanır.
- Eğitim sırasında ortak merkezi critic iki uçağın ve haritanın global
  state'ini görür.
- Çalıştırma ve değerlendirme sırasında critic kullanılmaz.
- MAPPO ve HAPPO aynı actor/critic kapasitesini ve aynı eğitim bütçesini
  kullanır.

Başlangıç ağları:

```text
Actor:
observation → Linear(128) → Tanh → Linear(128) → Tanh → Linear(5)

Central critic:
global state → Linear(256) → Tanh → Linear(256) → Tanh → Linear(1)
```

MAPPO, iki actor'ı aynı donmuş rollout üzerinden PPO clipped objective ile
günceller. HAPPO her güncellemede seed'li bir ajan sırası seçer; actor'ları
sırayla günceller ve önce güncellenen actor'ların yeni/eski politika olasılık
oranlarını sonraki actor'ın avantajına kümülatif importance factor olarak
uygular.

İki uçak simetrik göreve sahip olsa da ayrı actor parametreleri korunur. Böylece
MAPPO ve HAPPO karşılaştırmasında parametre paylaşımı ek bir değişken olmaz.

## 7. Ödül tasarımı

Her step tek bir ortak takım ödülü üretir:

| Olay | Takım ödülü |
|---|---:|
| İlk uçak hedefe ulaştı | `+100` |
| Bir uçak düşürüldü | `-25` |
| Canlı bir uçak hedefe 1 birim yaklaştı | `+0,01` |
| Canlı bir uçak hedeften 1 birim uzaklaştı | `-0,01` |
| Her global zaman adımı | `-0,001` |

İki uçak da aynı step'te hedefe ulaşırsa başarı ödülü yine `+100` olur; ikinci
bir terminal bonus verilmez. `both_reached` ayrıca metrik olarak tutulur.
Bir uçak hedefe ulaşırken diğeri aynı step'te düşürülürse episode başarılıdır;
o step hem `+100` başarı ödülünü hem de `-25` ölüm cezasını içerir.

Yaklaşma/uzaklaşma shaping'i imzalı Manhattan mesafe farkıdır. İleri-geri bir
döngünün mesafe ödülü sıfırlanır; zaman cezası döngüyü ayrıca zararlı yapar.
Başarı ödülü, daha önce bir uçak kaybedilmiş olsa bile tam `+100` olarak
verilir. Önceki `-25` ölüm cezası toplam getiride korunarak güvenli rota
tercihini öğretir.

## 8. Rollout ve eğitim akışı

Rollout tamponu her ortak timestep için şunları saklar:

- İki actor gözlemi ve aksiyon maskesi.
- Global state.
- İki aksiyon ve eski log-olasılıkları.
- Merkezi critic değeri.
- Ortak takım ödülü.
- `terminated` ve `truncated` işaretleri.
- İki uçağın alive/dead/reached durumları.

GAE ortak zaman çizgisi üzerinde hesaplanır. Gerçek terminalde bootstrap değeri
sıfırdır; 3.000 adımlık truncation'da son global state'in critic değeriyle
bootstrap yapılır.

İlk PPO ayarları iki algoritma için aynıdır:

| Parametre | Değer |
|---|---:|
| `gamma` | `0.99` |
| `gae_lambda` | `0.95` |
| `clip_coef` | `0.2` |
| `actor_lr` | `3e-4` |
| `critic_lr` | `3e-4` |
| `ppo_epochs` | `5` |
| `minibatch_size` | `256` |
| `rollout_episodes` | `32` |
| `entropy_coef` | `0.01` |
| `value_coef` | `0.5` |
| `max_grad_norm` | `0.5` |

Advantage normalization ve value clipping açık olacaktır. Tam eğitim bütçesi,
önce kısa smoke koşuları ve seed 0 pilotuyla ölçülecek; iki algoritma için aynı
episode/step bütçesi kullanılacaktır.

## 9. Değerlendirme

Ana metrik, en az bir uçağın hedefe ulaşma oranıdır. Ayrıca şunlar raporlanır:

- İki uçağın da ulaştığı episode oranı.
- Uçak başına hedefe varma ve ölüm oranı.
- Dış ve iç bölge giriş sayıları.
- Radar kaynaklı ölüm sayısı.
- Ortalama episode adımı ve timeout oranı.
- Ortalama takım getirisi.
- İki rotanın hücre örtüşme oranı.
- Öğrenme eğrisi ve duvar saati.

Nihai karşılaştırma en az üç seed (`0`, `1`, `2`) ile yapılır. MAPPO ve HAPPO
aynı başlangıç haritasını, aynı seed listesini, aynı ağ kapasitesini ve aynı
eğitim bütçesini kullanır. Sonuçlar ortalama ve örnek standart sapma olarak
raporlanır.

## 10. Test stratejisi

Uygulama test güdümlü ilerleyecektir. Asgari testler:

1. Grid tam olarak 1000×1000 hücre ve sınır aksiyon maskeleri doğrudur.
2. Her radarın dış ve iç karesi tam 220×220 ve 140×140 hücre kapsar.
3. Aynı bölgede kalırken yalnızca bir ölüm zarı atılır.
4. Bölgeden çıkıp yeniden girildiğinde yeni zar atılır.
5. Dış ve ardından iç bölgeyi geçen uçakta iki bağımsız risk uygulanır.
6. İki uçağın risk zarları birbirinden bağımsızdır.
7. Hareketler eşzamanlıdır ve aynı hücreye giriş serbesttir.
8. Bir uçak öldüğünde diğeri devam eder; ilk hedef varışında episode biter.
9. Maskelenmiş categorical dağılım geçersiz aksiyona olasılık vermez.
10. GAE terminalde bootstrap yapmaz, truncation'da yapar.
11. MAPPO actor ve critic parametrelerini sonlu kayıpla günceller.
12. HAPPO seed'li sıralı güncelleme ve importance factor uygular.
13. Checkpoint kaydet-yükle aynı deterministik aksiyonu üretir.

## 11. Uygulama yol haritası

### Aşama 1 — Ortam ve geometri

- Tek config kaynağı oluştur.
- Radar karelerini ve bölge geçişlerini uygula.
- Eşzamanlı iki-uçak step akışını ve terminal koşullarını uygula.
- Deterministik RNG ve ASCII/Matplotlib render ekle.

### Aşama 2 — Oracle ve scripted baseline

- Risksiz en kısa yol için BFS/Dijkstra oracle oluştur.
- Sınırdan giden güvenli rota, doğrudan rota ve random-monotone baseline'ları
  ölç.
- Ortam risk frekanslarını kontrollü giriş testleriyle doğrula.

### Aşama 3 — Ortak PPO temelleri

- Actor, merkezi critic, aksiyon maskeleme, rollout ve GAE'yi uygula.
- Ortak checkpoint, seed ve metrik altyapısını kur.

### Aşama 4 — MAPPO

- Donmuş rollout üzerinden clipped PPO actor güncellemelerini uygula.
- Kısa smoke eğitimle parametre değişimi ve sonlu loss doğrula.

### Aşama 5 — HAPPO

- Seed'li ajan sırası ve ardışık actor güncellemelerini uygula.
- Kümülatif importance factor hesabını birim testle doğrula.

### Aşama 6 — Eğitim ve değerlendirme

- Seed 0 pilotuyla ortak eğitim bütçesini belirle.
- Seed 0/1/2 nihai MAPPO ve HAPPO koşularını yap.
- CSV/JSON ham sonuçlarını, Markdown karşılaştırmasını ve rota görsellerini
  üret.

### Sonraki sürümler

İlk sabit harita doğrulandıktan sonra sırasıyla radar konumu randomizasyonu,
zorluk curriculum'u, gözlem gürültüsü, iletişim gecikmesi/kaybı ve daha gerçekçi
uçuş dinamiği değerlendirilecektir. Bu maddeler için şimdiden kod veya soyutlama
eklenmeyecektir.

## 12. Kabul kriterleri

- Tüm ortam, risk, ödül, MAPPO ve HAPPO testleri geçer.
- Aynı seed ile ortam geçişleri ve ölüm zarları yeniden üretilebilir.
- MAPPO ve HAPPO smoke eğitimleri NaN/sonsuz değer olmadan tamamlanır.
- En az bir uçak hedefe ulaşınca `+100` başarı ödülü verilir ve episode biter.
- Bir uçağın ölümü diğer uçağın episode'unu sonlandırmaz.
- Radar bölgesinde kalma süresi ölüm ihtimalini artırmaz.
- Üç seed için final checkpoint'leri ve değerlendirme çıktıları oluşur.
- Sonuç raporu MAPPO/HAPPO'yu eşit bütçede karşılaştırır ve yalnızca ölçülmüş
  verilere dayanır.
