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

Tasarım tamamlandı; uygulama planı inceleme sonrasında hazırlanacaktır.

Detaylı ve güncel tasarım: [Strike_Mission.md](Strike_Mission.md)

## Algoritmik yaklaşım

- İki ayrı actor MLP.
- Eğitimde ortak merkezi critic.
- Çalıştırmada kompakt koordinat gözlemleriyle dağıtık actor'lar.
- Takım arkadaşının konumu ve durumu actor gözleminde paylaşılır.
- MAPPO ve HAPPO için aynı ağ kapasitesi, seed'ler ve eğitim bütçesi.

## Mevcut dosyalar

- `Strike_Mission.md`: onaylanan tasarım ve aşamalı yol haritası.
- `viz/plot_map.py`: önceki tasarımdan kalan harita çizimi; yeni 1-birim
  geometri uygulanırken güncellenecek.
- `baselines/map_check.py`: önceki 20-birim/adım varsayımına dayanır; ortam
  aşamasında yeni giriş-bazlı risk oracle'ıyla değiştirilecek.
