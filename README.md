# MARL Strike Mission

İki uçak, üç radar, bir hedef. **IQL / VDN / QMIX** ile eğitilen kooperatif
multi-agent reinforcement learning projesi.

```
1000x1000 grid  ·  STEP_SIZE=20  ->  51x51 latis
B (-500, 500) = (0,0)  ->  H (500, -500) = (50,50)   |  optimal 100 adim

R1 (-280, 220)  R2 (200, 100)  R3 (-100, -280)
  dis halka 220x220 (11 hucre)  ->  %2.01 olum / adim
  ic  halka 140x140 ( 7 hucre)  ->  %28.03 olum / adim
```

İki uçak **aynı anda** kalkar, aynı yoldan gidebilir (çarpışma yok).
**En az bir uçak hedefe varırsa takım ödülü fullenir.** Düşürülen uçak ceza
getirir. Amaç: hedefe ulaşırken radarlardan mümkün olduğunca uzak durmak.

## Durum

📋 Planlama aşaması — bkz. **[Strike_Mission.md](Strike_Mission.md)**

| Aşama | Durum |
|---|---|
| 0 Kurulum | 🟡 devam ediyor |
| 1 Ortam | ⬜ |
| 2 Risk oracle | ⬜ |
| 3 Tek uçak DQN | ⬜ |
| 4 IQL | ⬜ |
| 5 VDN | ⬜ |
| 6 Radar alarm kuplajı | ⬜ |
| 7 QMIX | ⬜ |
| 8 Değerlendirme | ⬜ |
| 9 Görselleştirme | ⬜ |
| 10 Random radar | ⬜ |

## Ölçülmüş baseline'lar

`python -m baselines.map_check` ile yeniden üretilebilir:

| Politika | Tek uçak hayatta kalma | Takım (≥1 varır) |
|---|---:|---:|
| Rastgele monoton yol | 21.2% | 37.9% |
| Düz çapraz | 24.7% | 43.4% |
| **Dijkstra oracle** | **100.0%** | **100.0%** |

Haritanın %14.0'ü tehlikeli; monoton yolların %1.07'si sıfır riskli.

## Kurulum

```bash
python3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Kullanım

```bash
python -m baselines.map_check        # harita/risk dogrulama
python train.py --algo vdn --seed 0  # (Asama 5'ten itibaren)
python -m eval.evaluate              # (Asama 8'den itibaren)
```

## Kardeş proje

[`MARL-Pathfinding`](../MARL-Pathfinding) — 5x5'ten 50x50'ye giden sıralı-akışlı
MARL projesi. Ajan/eğitim/eval altyapısı ve `PLAN.md §8`'deki tuzaklar tablosu
buraya taşındı.
