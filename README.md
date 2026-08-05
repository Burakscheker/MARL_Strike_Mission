# MARL Strike Mission

İki uçak, üç radar, bir hedef. **IQL / VDN / QMIX** ile eğitilen kooperatif
multi-agent reinforcement learning projesi.

```
1000x1000 grid, TAM COZUNURLUK (1 hucre = 1 birim)
B (-500, 500) = (0,0)  ->  H (500, -500) = (999,999)  |  optimal 1998 adim

R1 (280,220)  R2 (400,700)  R3 (780,400)     [hucre koordinati]
  dis halka +-110 hucre  ->  %0.101 olum / adim
  ic  halka +- 70 hucre  ->  %1.620 olum / adim
  (kalibrasyon: halkayi BOYDAN BOYA gecersen toplam %20 / %90)
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

## 🚨 Ölçülmüş baseline'lar — sabit harita TRIVIAL

`python -m baselines.policies` ile yeniden üretilebilir:

| Politika | Takım başarısı | Uzunluk | Dış/İç maruziyet |
|---|---:|---:|---:|
| Rastgele monoton yol | 10% | 657 (ölümle kesildi) | 148 / 122 |
| **SABİT politika (hep sağ → hep aşağı)** | **100%** | **1998** | **0 / 0** |
| **Dijkstra oracle** | **100%** | **1998** | **0 / 0** |

Sabit politika oracle'ın kendisi: B ve H karşılıklı köşelerde, üç radar da iç
bölgede, dolayısıyla gridin kenarı **radarsız ve aynı zamanda en kısa** yol.
Öğrenilecek bir ödünleşme yok. Detay ve çözüm: [Strike_Mission.md §0.3](Strike_Mission.md).

Haritanın %14.7'si tehlikeli. `python -m baselines.map_check` rastgele radar
konfigürasyonlarını tarayıp trivial/kolay/zor oranlarını raporlar.

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
