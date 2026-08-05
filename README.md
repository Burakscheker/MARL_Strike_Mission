# MARL Strike Mission

İki uçak, üç sabit radar ve bir ortak hedef içeren kooperatif multi-agent
reinforcement learning projesi. MAPPO ve HAPPO aynı ortam ve eğitim bütçesi
altında karşılaştırılacaktır.

```text
Grid:       1000×1000, x/y ∈ [-500, 499]
Hareket:    4 yön, adım başına 1 birim, eşzamanlı
Başlangıç:  B = (-500, 495), iki uçak aynı noktada
Hedef:      H = (499, -494)
Radarlar:   R1=(-280,220), R2=(200,100), R3=(-100,-280)
Risk:       dış 220×220 alana girişte %20 ölüm
            iç 140×140 alana girişte ayrıca %90 ölüm
Başarı:     en az bir uçak hedefe ulaştığında +100 ve episode sonu
```

Radar riski bölge içinde geçirilen her adımda değil, yalnızca bölgeye girişte
bir kez uygulanır. Dış ve ardından iç bölgeye giren bir uçağın toplam hayatta
kalma olasılığı `%8`'dir. Aynı hücrede kalmak ya da aynı rotayı kullanmak
serbesttir.

## Durum

İlk çalışır sürüm hazırdır: ortam, MAPPO/HAPPO trainer'ları, rollout toplama,
checkpoint, değerlendirme ve harita renderer'ı bulunur. Eğitilmiş model
dosyaları repoya dahil değildir.

Detaylı ve güncel tasarım: [Strike_Mission.md](Strike_Mission.md)

## Algoritmik yaklaşım

- İki ayrı actor MLP.
- Eğitimde ortak merkezi critic.
- Çalıştırmada kompakt koordinat gözlemleriyle dağıtık actor'lar.
- Takım arkadaşının konumu ve durumu actor gözleminde paylaşılır.
- MAPPO ve HAPPO için aynı ağ kapasitesi, seed'ler ve eğitim bütçesi.

## Kurulum

Python 3.10 veya üstü gerekir.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell aktivasyonu:

```powershell
.venv\Scripts\Activate.ps1
```

## Test ve harita

```bash
python -m unittest discover -s tests -v
python -m viz.plot_map --output runs/map.png
```

## Eğitim

MAPPO ve HAPPO aynı episode bütçesiyle ayrı çıktı klasörlerinde çalıştırılır:

```bash
python train.py --algo mappo --episodes 10000 --rollout-episodes 32 --seed 0 --output runs/mappo_seed0
python train.py --algo happo --episodes 10000 --rollout-episodes 32 --seed 0 --output runs/happo_seed0
```

Bir episode en fazla 3.000 adet 1-birim hareket içerir. Bu nedenle tam eğitim
koşuları uzundur; önce testlerdeki kısa-horizon smoke koşularını doğrulayın.

## Değerlendirme

```bash
python -m eval.evaluate --checkpoint runs/mappo_seed0/checkpoint.pt --episodes 1000 --output runs/mappo_seed0/eval
```

Değerlendirme `episodes.csv`, `summary.json` ve `report.md` üretir.

## Temel dosyalar

- `Strike_Mission.md`: onaylanan tasarım ve aşamalı yol haritası.
- `config.py`: geometri, ödül ve PPO sabitleri.
- `env/strike_env.py`: eşzamanlı iki-uçak ortamı ve giriş-bazlı risk.
- `agents/ppo.py`: ortak PPO altyapısı, MAPPO ve HAPPO.
- `train.py`: eğitim ve checkpoint üretimi.
- `eval/evaluate.py`: deterministik değerlendirme ve raporlama.
- `viz/plot_map.py`: gerçek 1000×1000 harita çizimi.
