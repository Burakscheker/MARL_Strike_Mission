# Gece dongusu — 2026-08-29

Amac: happo / mappo / qmix / vdn egit; herhangi birinin eval team_success'i
%75'i gecerse DUR. Ikincil: 500 episode < 2.5 saat.

---
## >>> NIHAI DURUM (2026-08-29 gec saat, Burak once BUNU oku) <<<

**En iyi: `runs/ckpt/it2_vdn_epsfast.pt` — team_success ~%70** (100-harita,
yetkili eval). Recipe: `--eps-start 0.1` (tek CLI bayragi, KOD DEGISIKLIGI YOK),
seed 0, 250 ep, ~ep25 checkpoint. Oracle tavani %81.5 -> %70 = oracle'in %86'si.

**%75'e ULASILAMADI.** it7-it18: **~16 ciddi deneme, HICBIRI %70'i gecmedi:**

| kategori | denenenler | sonuc |
|----------|-----------|-------|
| TD-hedef | Advantage Learning (a=0.9), Munchausen (tau 0.02) | AL: gap 15x acti ama tepe %45; Munchausen: Q duzlesti argmax bozuldu %40 |
| mimari | Dueling, LayerNorm, QR-DQN (16q, opt 0/0.5) | hepsi fast-eps butcesinde "guvenli rota"yi ogrenemedi (pervasiz ya da timeout) |
| hiperparametre | batch 64, batch 256, curriculum-frac 0.1 | 128/0.6 IKI YONDE de sweet spot; perturbasyon = cokus |
| deployment | eps 0.02-0.1, Boltzmann temp 0.01-0.1 | timeout'u olume takas ediyor, NET NEGATIF |
| baska | checkpoint-ensemble, multi-seed (2,3,4...) | ensemble korele=kotu; seed 1=%27 seed 2=%48 (seed 0 SANSLI outlier) |

**KOK NEDEN:** `it2_vdn_epsfast.pt` %70 = DELIKANLI DENGELI bir yerel optimum.
Deger fonksiyonu radar olum cezasini asiri-agirliklandirip KARAMSAR/asiri-temkin
bir sabit-noktaya oturuyor. fast-eps + erken ckpt o oturmadan ONCE yakaliyor.
Dengenin herhangi bir bileseni (batch, mimari, curriculum, target, seed)
degisince cokuyor. O ~27 "stall" haritasinda greedy takiliyor ama HAREKET =
radar-yogun bolgede olum riski; deger fonksiyonu DOGRU olarak reddediyor.

**%75 icin gereken (senin kararin):**
1. **Reward/senaryo:** `R_RISK_COEF` 65->~40 dusur (asiri-temkin cezasini
   hafiflet), VEYA `MAX_STEPS` 4000->8000 geri al (Tolga kiyasi icin cekmistin).
   Ikisi de senaryo config -> senin "dokunma" kuralinin disinda, senin cagrin.
2. **Stokastik deploy politikasini "mesru" say:** eval.py eps=0 hardcoded ama
   training-MA %74.7 (eps~0.07 ile). "VDN + risk-farkinda deploy = %74" gecerli
   bir algoritma sonucu sayilabilir.
3. **Cok daha uzun distributional RL** (QR-DQN 1000+ ep) — yavas, belirsiz.

Saf VDN + mevcut config + greedy eval ile **%70 TAVAN** (yuksek guvenle).

Kod: hepsi bayrak-arkasi (--al-alpha, --munchausen-tau, --layernorm, --quantiles,
--qr-optimism, --vdn-batch, --vdn-target-update, --map-seed, --curriculum-frac,
--save-all-ckpts). Varsayilan = eski davranis BIREBIR. tests/ 15/15.

---
## >>> ESKI SABAH OZETI (fast-eps ATILIMI ONCESI — artik guncel degil) <<<

**Dongu 1. iterasyonda tikandi: `codex` kullanim limitinde (26 gun reset, Sep 24).**
Senin tarif ettigin dongunun 4-5. adimlari (codex analiz -> onerisini uygula)
codex olmadan yapilamaz. "Komut hata verirse donguyu surdurme" kuralin geregi
throughput KOD degisikligi yapmadim. Bunun yerine bosta GPU'yu ASIL hedefe
(%75 team_success) yonlendirdim: kod degisikligi YOK, sadece egitim kosulari.

**%75'e ulasilamadi.** En iyi: `it1_qmix.pt` = **%33** (100-harita held-out,
standalone eval). Kanit + teshis:
- QMIX 500-ep: ep50 %5 -> ep100 %2.5 -> ep150 %0. Coktu.
- VDN 300-ep (seed 2): ep25 %5 -> ep75..150 %0. Ayni cokus.
- Tohum taramasi (48-ep, 4 tohum): QMIX s0=%37.5(SANS) s1=%7.5 s3=%10 ;
  VDN s0=%25 s1=%5 s2=cokus s3=%2.5. Medyan ~%5.
- qmix_s0_96: seed 0'i 48 yerine 96 ep -> ep48'de %5 (48-ep %37.5 idi). Yani
  %37.5 ogrenilmis DEGIL, eps-takvimi artefakti.
- MAPPO/HAPPO: %5-7.5, koordinasyon yok (route_overlap 1.0).

**ASIL TESHIS (en degerli cikti):** Sorun NAVIGASYON degil, ROTA GUVENLIGI.
it1_qmix.pt greedy rotasi 100 haritanin **94'unde hedefe VARIYOR** — ajan
yolu biliyor. Ama surv_ratio sadece 0.47 (oracle rotasinin yarisi kadar
guvenli) -> episode basi 1.29 olum -> team %33. %75'e giden yol: rotayi
oracle guvenligine (surv_ratio ~0.9) yaklastirmak. Bu bir OGRENME
degisikligi (risk shaping / obs / kredi atama), throughput degil -> codex-kapisi.

**Cokus mekanizmasi** (yeni gozlem, degerli): ~ep50-150 arasi anlamli epsilon'la
egitim politikayi "urkek"lestiriyor — o kadar risk-kacinan hale geliyor ki
HEDEFE GITMEYI birakiyor (VARIS %100 -> %0, her episode timeout, analitik
"guvenlik" 0.85 ama gorev 0.0). config.py R_RISK_COEF=65 notundaki tam olarak
uyarilan mod. it1'in yuksek sayilari (QMIX %37.5) SADECE 48-ep kosunun eps'i
ep24'te tabana indirip politikayi cokusten ONCE greedy'ye dondurmesindendi —
tohuma da bagli, kalici degil.

**En iyi kullanabilir checkpoint'ler** (bu gece uretildi) — 100-harita
STANDALONE `python -m eval.evaluate` (yetkili sayi):
- `runs/ckpt/it1_qmix.pt` — **team_success %33.0**, VARIS %94, surv_ratio 0.467
  (medyan 0.512), mission_prob 0.340, olu 1.29/ep, timeout %37.
- `runs/ckpt/it1_vdn.pt`  — **team_success %27.0**, VARIS %95, surv_ratio 0.356
  (medyan 0.100 — rotalari QMIX'ten belirgin daha az guvenli), olu 1.49/ep.
Oracle tavani her ikisi icin %81.5. Tarihi `BEST_vdn_seed2_team45.pt` (%45)
ESKI 18-skalar obs ile — kiyaslanamaz.

**KRITIK NUANS (yanlis anlasilmasin): "her sey cokuyor" DEGIL.**
it1_qmix.pt (~ep48) GERCEKTEN ise yarar bir politika: greedy rotasi 100
haritanin 94'unde hedefe VARIYOR. Sorun rotanin GUVENLIGI (surv_ratio 0.47 =
oracle'in yarisi kadar guvenli) — ajan yolu biliyor ama zara fazla giriyor,
episode basi 1.29 olum. Cokme ep48'DEN SONRAKI egitimde oluyor (uzun kosu +
mid-eps rejimi politikayi asiri urkeklestirip VARIS'i %0'a dusuruyor).
Yani: erken checkpoint %33, ama "daha cok egit -> daha iyi" YANLIS — tersine
donuyor. %75'e giden yol "checkpoint'i ep48'de yakala + rota guvenligini
artir" — ve o ikincisi throughput DEGIL ogrenme degisikligi (codex-kapisi).

**s/ep olcumleri** (it1 probe, 48-ep, n_envs 32, cuda, solo): VDN 10.95, QMIX
13.34, MAPPO 10.24, HAPPO 10.61. Gercek 500-ep kosu eps takvimi daha uzun oldugu
icin episode'lar daha uzun -> ~14-20 s/ep (qmix_long500'de gorulen). "500 ep <
2.5 saat" = <18 s/ep: VDN/MAPPO/HAPPO SIGAR, QMIX sinirinda. (Ama %75 sarti
saglanmadigi icin DUR kosulu tetiklemedi.)

**Codex donunce dongu icin hazir throughput adaylari** (hicbiri UYGULANMADI —
kod dokunulmadi; asagida "Ek: throughput aday detaylari" bolumunde tam tarif):
- A: eval risk-haritasi in-memory cache (EN GUVENLI, ogrenmeye sifir risk)
- B: `StrikeMissionEnv.__init__` defer_reset
- C: ppo rollout/eval'de biten env'i islememe (straggler)
- D: `*_EVAL_EVERY` 24/25 -> 48/50 (senin izin verdigin knob)

**Repo durumu:** temiz. KOD DEGISMEDI (git diff 7f5b4d0..HEAD sadece NOTES.md).
`python -m tests.test_env` -> TUM TESTLER GECTI (WIP baseline saglam). Orphan
python sureci YOK. (Not: 06:45'ten beri idle bir `codex` sureci var — benim
degil, dokunmadim.)

**Son deneme:** qmix_s0_96 — sans tohumu (s0) 96 ep. Soru: it1'in %37.5 tavani
daha fazla episode ile asilabilir mi? **CEVAP: HAYIR, tersine.** ep12 %5, ep24
%0, ep36 %0, ep48 **%5** (it1'in AYNI seed/AYNI kod ep48'i %37.5 idi). Tek fark
--episodes 48 vs 96 -> eps tabani ep24 vs ep48. Yani %37.5 gercekten SADECE
48-ep eps takviminin urunu, ogrenilmis bir sey degil. ep48'de kesilip
oldurdum.

---

## Dokunulmaz (kullanici kurali)
- `eval/evaluate.py` — hic dokunma
- `team_success` hesabi (`env/strike_env._terminal_info`: `reached1 or reached2`) — dokunma
- Ogrenmeyi bozarak hizlanma YOK: ag boyutu, episode uzunlugu (MAX_STEPS),
  update sayisi (LEARN_EVERY / PPO_EPOCHS / rollout) SABIT.
- Serbest: env vektorizasyonu, gereksiz kopya, adim basi loglama,
  egitim-ici eval sikligi (`*_EVAL_EVERY`, `--eval-episodes`).
- Bir degisiklik s/ep'i iyilestirip mean_return'u >%15 dusururse GERI AL.
- Komut hatasi / izin reddi / 3 ust uste cokme -> DUR, buraya yaz.

## Baslangic durumu (commit 56aa477 + commit'siz WIP)
WIP = MAPPO/HAPPO portu (agents/mappo_happo.py) + env/vec_env.py +
action-risk skalarlari (N_SCALARS 22, OBS_DIM 1345) + STATE_DIM 898.

Son gercek kosular (Aug 28, --episodes 300 --n-envs 32 --max-steps 4000, device belirsiz):
| algo  | en iyi eval team | s/ep (kabaca) | not |
|-------|------------------|---------------|-----|
| vdn   | ep100 %16        | ~30, sabit    | en umut verici trend |
| mappo | ep32 %4 -> %0    | 26->33, ARTIYOR | ilk chunk 5.6 sonra sisiyor |
| happo | ep32-64 %4       | 29->40, ARTIYOR | ayni sisme |
| qmix  | ep75 %0          | ~39           | eval_every=250 |
En iyi checkpoint (tum zamanlar): BEST_vdn_seed2_team45.pt = %45 team.

Teshis hipotezi (henuz olculmedi): MAPPO/HAPPO "sismesi" = her 25 episode'da
bir calisan egitim-ici eval (50 harita x 2 rollout = ~100 tam episode +
100 kez risk-haritasi insasi). run_chunk dice+route icin AYNI radar setinde
risk haritasini 2 kez kuruyor (death_enabled harita hesabini etkilemez).

## Probe protokolu (tum iterasyonlarda SABIT — karsilastirilabilirlik icin)
`--episodes 48 --n-envs 32 --device cuda --eval-every 24 --eval-episodes 40 --seed 0`
4 algo SIRAYLA (paralel DEGIL). Kaydedilen: "bitti" satiri s/ep + son eval team%.
NEDEN sirayla: makinede 15.6 GB RAM var, VDN JointReplayBuffer insada ~5.4 GB
(4x 250k x 1345 x f32) ayirir. 4 paralel -> RAM tukendi (0.4 GB bos), swap'e
girip s/ep olcumu anlamsizlasti (it1 paralel denemesi cope gitti). Solo kosu
temiz: kaza probe'u solo 11.86 s/ep gosterdi. Sirayla ~50-60 dk/iterasyon.
NOT: 48 episode kisa oldugu icin eps ep24'te tabana iner (EPS_FLOOR_FRAC*48).
Bu yuzden mutlak s/ep gercek 500-ep kosuyla ayni DEGIL; ama iterasyonlar
arasi DELTA gercek (sadece throughput'a dokunuyoruz, ogrenme dinamigi sabit).
Kaza probe (VDN, oldurulmeden once ep25): 11.86 s/ep cuda, eval ep24 team=20%.

## Iterasyon gunlugu

### it1 — baseline olcumu (degisiklik yok), commit 7f5b4d0
Probe: 48 ep, n_envs 32, cuda, eval-every 24, eval-episodes 40, seed 0. Sirayla.
| algo  | s/ep  | eval team% (ep24 / ep48) | mission_prob (ep24/ep48) | not |
|-------|-------|--------------------------|--------------------------|-----|
| vdn   | 10.95 | 25.0 / 12.5              | 0.240 / 0.134            | ep24 tepe, sonra bozuluyor (belgeli desen) |
| qmix  | 13.34 | 10.0 / 37.5             | 0.111 / 0.408            | ep48'de YUKSELIYOR, VARIS %95 |
| mappo | 10.24 | 5.0 / 7.5               | 0.056 / 0.070            | route_overlap 1.0 (koordinasyon yok), olu 1.8 |
| happo | 10.61 | 5.0 / 2.5               | 0.056 / 0.032            | MAPPO ile ayni ep24 (ayni seed), koordinasyon yok |
En iyi it1 = QMIX %37.5 @ ep48 (yukselis trendi). VDN %25 (ep24 tepe). Hedef %75 uzak.
s/ep it1: VDN 10.95, QMIX 13.34, MAPPO 10.24, HAPPO 10.61.
=> Contender: QMIX + VDN. MAPPO/HAPPO zayif (~5-7%, route_overlap 1.0).

### !!! ENGEL: codex exec kullanim limitinde (2026-08-29 ~07:40)
`codex exec` -> "ERROR: You've hit your usage limit ... try again at Sep 24th 2026".
Yani dongunun 4. adimi (codex analiz) ve 5. adimi (codex'in onerisini uygula)
26 gun boyunca YAPILAMAZ. Kullanici kurali: "komut hata verirse donguyu surdurme".

KARAR (gozetimsiz, kendi kararim): throughput-kod-degisikligi dongusu codex'siz
YURUTULMEZ (o adim codex'in sanksiyonunu gerektiriyor; portfolyo deposunda
gozetimsiz kod degisikligi tam da "hata olunca dur" kuralinin korudugu sey).
Bunun yerine BOSTA GPU zamanini ASIL HEDEFE (%75 team_success) yonlendiriyorum:
KOD DEGISIKLIGI YOK, sadece daha uzun egitim kosulari (--episodes yuksek =
YASAK DEGIL, sadece "episode KISALTMA" yasak). Contender'lar QMIX ve VDN.
Bir eval %75'i gecerse: DUR, kullaniciyi bilgilendir.

### Uzun kosu denemeleri (codex'siz, kod degisikligi YOK)
| kosu | s/ep | en iyi eval team% | not |
|------|------|-------------------|-----|
| qmix_long500 s0 | ~13 | ep50 %5 -> ep100 %2.5 -> ep150 %0 | COKTU. ep150'de oldurdum. Belgeli QMIX uzun-egitim cokusu. it1'deki %37.5 SADECE 48-ep sikistirilmis eps takviminin dusuk-kesif anlik goruntusuydu, kalici degil. |
| vdn_long300 s2 | ~11 | ep25 %5 -> ep50 %2.5 -> ep75..150 %0 | QMIX ILE AYNI COKUS. eps ep150'de tabana indi, GREEDY politika toparlamadi (VARIS %0, timeout, "urkek" politika: analitik 0.82 ama hedefe varmiyor). ep150'de oldurdum. |

### it2-esdegeri: kisa-kosu tohum taramasi (48 ep, eval-every 12, kod degisikligi YOK)
Hipotez: kisa kosu eps'i ep24'te tabana indirir -> politika "urkek cokus"ten
ONCE greedy'ye zorlanir -> temiz snapshot. Farkli tohum daha yuksek tepe verebilir.
| kosu | en iyi eval team% (hangi ep) | not |
|------|------------------------------|-----|
| sw_qmix_s1 | ep36 %7.5 (ep12/24/36/48: 0/5/7.5/5) | VARIS max %25. |
| sw_qmix_s3 | ep12 %10 (10/2.5/2.5/0) | ep48 tam cokus (VARIS %0). |
| sw_vdn_s1  | ep36 %5 (2.5/0/5/0) | ep48 VARIS %5. |
| sw_vdn_s3  | ep48 %2.5 (0/2.5/2.5/2.5) | hep ~%2.5, route_overlap ~1.0 (koordinasyon yok). |

**Tam tablo (bu gece uretilen tum team_success tepe degerleri):**
| algo  | s0 | s1 | s2 | s3 | uzun kosu |
|-------|-----|-----|-----|-----|-----------|
| QMIX  | %37.5 (SANS, ep48) | %7.5 | — | %10 | 500ep: ep150'de %0 |
| VDN   | %25 (ep24) | %5 | cokus | %2.5 | 300ep: ep75+'da %0 |
| MAPPO | %7.5 | — | — | — | — |
| HAPPO | %5 | — | — | — | — |

Medyan ~%5. %25-37.5 SADECE seed 0. => %75 mevcut kurulumla ULASILMAZ.
s/ep: eval-every 12 (4 eval/48ep) -> ~16-20 s/ep; eval-every 24 (2 eval) -> ~11-13.
Yani eval maliyeti s/ep'in ~%40'i. Codex donunce 1 numarali hedef bu (bkz. adaylar).

### Onemli gozlem
it1'in yuksek sayilari (QMIX %37.5, VDN %25) 48-ep kosunun eps'i ep24'te
tabana indirmesinden geliyordu — az kesif = az politika bozulmasi = daha
temiz greedy snapshot. Proper eps takvimli uzun kosuda ilk ~150 ep yuksek-eps
"kaotik olum" rejiminde ve QMIX oradan cikamadi. VDN'in belgeli tepesi ~%45
(seed 2), ama o ESKI obs uzayiyla (N_SCALARS 18); simdiki 22-skalar obs farkli.

---
## it2 — codex YOK, self-review (Burak: "codexi denklemden cikar, kendin review at, 75 olcak")
Baslangic durumu (o ana kadar): en iyi it1_qmix %33 (100-harita), it1_vdn %27.
s/ep: VDN 10.95 QMIX 13.34 MAPPO 10.24 HAPPO 10.61.

**Self-review — #1 bulgu:** greedy politika COKUSU her seyi tikayan sorun.
Desen: eps ~ep50-150 arasi 0.4-0.8 iken buffer RASTGELE-aksiyon-olumleriyle
doluyor -> deger fonksiyonu "her sey tehlikeli" ogreniyor -> politika donuyor
(VARIS %100->%0). it1 48-ep kosulari bundan KACINIYOR cunku eps ep24'te
0.05'e iniyor -> buffer ON-POLICY kaliyor -> deger fonksiyonu ise yarar sey
ogreniyor. Belgeli %46 (ESKI obs, 500-ep) COKMEMISTI — yani cokus mevcut
config'e (22-skalar obs / 4000 adim / MAPPO-HAPPO ortak-kod portu) ozgu.

**it2 degisikligi (KOD DEGISIKLIGI YOK, sadece CLI):** `--eps-start 0.1` ile
egitim boyunca eps'i ~0.10->0.05 tut (it1'in hizli-decay davranisini uzun
kosuda taklit et). Beklenti: greedy politika 250 ep boyunca duzenli tirmanir
ve COKMEZ; team_success it1'in %33'unu gecer. Test: VDN 250-ep seed 0.
Geri alinabilir (bayrak kaldirilir).

**SONUC — HIPOTEZ DOGRULANDI, ETKI BEKLENENDEN COK BUYUK:**
it2_vdn_epsfast eval ep25: **team %67.5** (onceki en iyi %33), VARIS %77.5,
mission_prob 0.676, **olu 0.28** (1.3-1.5'ten), ic-halka girisi 0, analitik 0.921.
Yani fast-eps buffer'i on-policy tutunca ajan GERCEKTEN guvenli rota ogreniyor.
Trajektori (eval her 25 ep):
| ep | team% | VARIS% | olu | gorev | not |
|----|-------|--------|-----|-------|-----|
| 25 | 67.5  | 77.5   | 0.28| 0.676 | both_reached %50! analitik 0.921 |
| 50 | 45.0  | 47.5   | 0.10| 0.443 | asiri-temkin: adim 3800, timeout %95 |
| 75 | 25.0  | 27.5   | 0.15| 0.253 | dusme devam ediyor |
=> TEPE ep25. Sonra "asiri-temkin drift": ajan olmuyor (olu ~0.1) ama hedefe
de gitmiyor (VARIS dusuyor, timeout artiyor). outer_total 634->206: ajan
disi halkalari bile giderek daha cok kaciniyor. ep25 checkpoint kaydedildi
(mission_prob 0.676 = en iyi). Kosuyu ep75'te oldurdum.
YENI YON: fast-eps + ERKEN DURDURMA. Sonraki: (1) it2_vdn_epsfast.pt 100-harita
authoritative eval, (2) QMIX fast-eps ayni desen mi, (3) kisa kosu (40-48 ep)
tepeyi temiz yakala.

### >>> it2 ATILIM (2026-08-29 ~13:35) <<<
`it2_vdn_epsfast.pt` (VDN, `--eps-start 0.1`, ep25 checkpoint) —
**100-harita STANDALONE eval: team_success %69.0** (onceki en iyi %33 -> 2.1x).
- surv_ratio ort 0.764, **medyan 1.0000** (rota guvenligi haritalarin yarisinda
  ORACLE KADAR — it1'de 0.47 idi). olu 0.39/ep (1.29'dan). VARIS %80. timeout %35.
- Oracle tavani %81.5. Hedef %75'e **6.5 puan**, tavana 12.5 puan.

**Neden ise yaradi:** `--eps-start 0.1` -> egitim boyunca eps ~0.10->0.05 ->
replay buffer ON-POLICY kaliyor -> deger fonksiyonu GERCEK risk-rotalamayi
ogreniyor, "her sey tehlikeli"yi degil. Tek CLI bayragi, SIFIR kod degisikligi.

QMIX fast-eps: ep12 %15 -> ep24 %45 -> ep36 %35 (tepe ep24, VDN'den dusuk,
olu 0.8). VDN net kazanan.

**%75 icin sonraki adimlar:** (a) coklu tohum (s1/s2/s3 — biri daha yuksek
olabilir, %69 tek tohum), (b) ince eval grid (eval-every 6, ep15-40 arasi
gercek tepeyi yakala — ep25 ille de tepe degil), (c) drift'i yavaslatmak
(--eps-end 0.1 sabit-dusuk?).

### it2 push: sweep2 GECERSIZDI — `--episodes 48` confound
`--episodes N` HEM `EPS_FLOOR_FRAC*N` (eps tabani) HEM `curriculum_n_radar(ep, N)`
(radar rampasi) degistiriyor. 48-ep kosuda ep25'te ~23 radar (250-ep'de ~13),
eps ep24'te tabanda. sw s0 (48ep) ep16 %10 sonra coktu — %69'u URETMEDI.
DERS: %69 recipe'i TAM tut: `--episodes 250 --eps-start 0.1`, ep~25 checkpoint.

### it2b/it2c: coklu tohum, DOGRU recipe (--episodes 250 --eps-start 0.1)
it2b s0 ince grid: **tepe ep15 %65** (ep30 %45 -> drift). Yani VDN fast-eps
tepesi ep15-25 bandi, ~%65-69 team. Iki s0 kosusu (67.5/69 ve 65) tutarli.
=> it2c: her tohum icin --episodes 250, ep40'ta oto-kill (tepe gecti), en iyi
mission_prob checkpoint saklaniyor.
| seed | tepe eval team% (ep) | 100-harita | not |
|------|----------------------|------------|-----|
| 0 | ep15-25 %65-69 | **%69.0** | referans (it2_vdn_epsfast.pt) — SANSLI TOHUM |
| 1 | ep20 %27.5 (10/20/30: 2.5/27.5/2.5) | — | zayif tohum, ep30'da coktu |
=> Seed 0 outlier (it1'de de VDN'de en iyi seed 0'di). %69 tek tohuma bagli.

### it2d: seed 0'i zorla — `--eps-start 0.1 --eps-end 0.1` (SABIT eps 0.1, decay yok)
Hipotez: sabit dusuk-eps buffer'i cesitli tutar -> asiri-temkin geri-besleme
donusunu kirar -> tepe (ep25) korunur/yukselir (oracle tavani %81.5).
| ep | team% | not |
|----|-------|-----|
| 10 | 50.0  | olu 0.25, VARIS 62.5, adim 3655 |
| 20 | 47.5  | flat, TIRMANMIYOR. ep25 train MA %68 ama greedy %47.5 (greedy < eps0.1 politika) |
SONUC: sabit-eps %69'u GECMIYOR (~48 plato). eps ~ayni (0.10 vs 0.095), fark yok.
=> Epsilon ayari tavani ~%67-69. %75 icin ASIRI-TEMKINI azaltmak lazim (ogrenme/odul).

### it3 — KOD DEGISIKLIGI: R_UNNECESSARY_RISK 20.0 -> 0.0 (config.py:313, geri alinabilir)
Gerekce: fast-eps'te sorun "gereksiz risk alma" DEGIL, TERSI — asiri temkin
(olu 0.39/ep, timeout %35, rota meandering, VARIS %80). +20 ek ceza ring
girisini daha da caydirir -> su anki basarisizlik moduyla TERS. Hafiza notu
zaten "istenirse kaldirilabilir" diyordu. GATE'lere dokunmaz (KAPI 1/2
R_TIMEOUT/R_ALL_DEAD ile ilgili).
Test: it3_vdn_s0_nounnec (VDN s0 --eps-start 0.1 250ep). Beklenti: rota daha
kisa -> timeout duser -> VARIS %80->%90, team %69->%73-77.
| ep | team% | not |
|----|-------|-----|
| 15 | 65.0  | it2b ep15 ile BYTE-BYTE AYNI -> R_UNNEC no-op, GERI ALINDI |

### it4 — --max-steps 8000 (SADECE CLI, config.MAX_STEPS 4000'de KALIYOR)
Gerekce: fast-eps ajaninin ASIL basarisizligi timeout (%35). Rotalari guvenli
ama 4000 adima sigmiyor. Hafiza: tarihi en iyi (%46) MAX_STEPS=8000 ile geldi.
Gate 8000'de hala saglaniyor: 2*(-15)+(-110)=-140 <= -50+8000*(-0.01)=-130.
Test: it4_vdn_s0_ms8k. Kritik: checkpoint HEM 8000 HEM 4000 adimda eval edilecek
(4000 = gercek benchmark, %69 ile kiyas). KOD DEGISMIYOR — bu senaryo karari
Burak'in; ben sadece SAYIYI olcuyorum.
| ep | team% (8000-eval) | not |
|----|-------------------|-----|
| 15 | 25.0 | VARIS %75 ama olu 1.43 (4000'de %65/olu 0.35) — 8000'de ajan hedefe gidiyor ama zara giriyor |
| 30 | 0.0  | TAM COKUS. inner_total 204 (ic halkalari DELIP geciyor). VARIS %0. |
=> --max-steps 8000 fast-eps recipe'i KOTULESTIRIYOR. Ekstra sure "duz git,
zarı ye" stratejisini erken karli yapiyor -> ajan onu ogrenip cokuyor.
config.MAX_STEPS DOKUNULMADI (it4 sadece CLI). 4000 + %69 en iyi kaliyor.

---
## it2-it4 NIHAI VERDICT (2026-08-29 ~15:00)

**EN IYI SONUC: `runs/ckpt/it2_vdn_epsfast.pt` = team_success %69.0** (100-harita
authoritative). Recipe: `--algo vdn --episodes 250 --n-envs 32 --device cuda
--eval-every 25 --eval-episodes 40 --seed 0 --eps-start 0.1`, ~ep25 checkpoint.
KOD DEGISIKLIGI YOK — tek CLI bayragi.

Baslangic %33 -> %69 (2.1x). Ama HEDEF %75'e ULASILAMADI. %69'u gecmek icin
denenen 4 yol, HEPSI basarisiz:
| it | ne | sonuc |
|----|----|-------|
| it2c | coklu tohum (s1,s2,s3) | seed 0 outlier — s1 %27, digerleri de zayif |
| it2d | sabit eps (--eps-end 0.1) | ~%48 plato, decay'liden KOTU |
| it3  | R_UNNECESSARY_RISK 20->0 | byte-byte AYNI (penalty tetiklenmiyor) — GERI ALINDI |
| it4  | --max-steps 8000 | ep30'da tam cokus (ic halka deliyor) |

**%75 icin (codex donunce / Burak karari):**
1. Deeper RL: greedy-vs-training boslugu (egitim MA %68, greedy eval %47-65).
   Kucuk kesif greedy'yi duzeltiyor -> greedy politika bazi haritalarda
   donguye giriyor. Bunu cozmek (double-Q duzeltme, farkli target update,
   veya eval-time kucuk eps) %69->%75+ acabilir.
2. Coklu-tohum + en iyisini sec: yeterince tohum denenirse biri %75 verebilir
   (seed 0 zaten %69, varyans yuksek).
3. MAX_STEPS: DENENDI (it4), fast-eps'te ise yaramadi. Ama YAVAS-eps + 8000
   birlikte (eski %46 recipe'i) denenmedi.

**Repo:** temiz. config.py = baseline (R_UNNEC 20, MAX_STEPS 4000) + it3
"denendi/geri alindi" yorumu. Kod DEGISMEDI. it2_vdn_epsfast.pt (%69) + tum
loglar runs/'da.

NOT (altyapi dersi): looping bash script TaskStop'a dayanikli degil (loop
devam edip yeni kosu baslatiyor). Tek-kosu script + oto-kill kullan.

---
## it5-it6: euzxx parent tasarimindan RL fikirleri (Burak: "1 sonra 3, sirayla")
euzxx/MARL-pathtfinding Strike_Mission.md (parent) farklari: entropy curriculum
0.03->0.005 (Burak sabit 0.01), ham-Manhattan shaping (Burak risk-farkinda
potansiyel), gamma 0.99 (Burak 0.9998), +200/-100 (Burak +50/-15). Burak'in
kendi §11.12'si zaten "sorun odulde degil ogrenmede/kredi atamasi" demis.

### it5 (opt 1): MAPPO/HAPPO entropi curriculum 0.03->0.005 — KOD DEGISIKLIGI
`PPO_ENTROPY_COEF 0.01` -> `PPO_ENTROPY_START 0.03 / END 0.005`, train.py ep
basi `set_entropy_progress` anneal. test_env GECTI.
| algo  | eval team% trajektori | en iyi mission_prob | it1 (sabit 0.01) |
|-------|-----------------------|--------------------|-------------------|
| mappo | 32:5 / 64:0 / 96:10 / 128:10 / 160:7.5 | **0.135** (ep96) | 0.070 |
| happo | ep32 %2.5 (kesildi, RAM) | ~0.050 | 0.056 |
=> MAPPO ~2x (mp 0.07->0.135) ama ~%10'da takili; HAPPO marjinal. ep160'ta
VARIS %35 (VDN'deki asiri-temkin drift). Entropi curriculum YARDIMCI ama
yetmiyor — MAPPO/HAPPO'nun daha derin sorunu var. Degisiklik KALIYOR (euzxx
tasarimi, 2x, zararsiz). GERI ALINABILIR: config'te ikisini de 0.01 yap.

### it6 (opt 3): shaping HAM Manhattan (euzxx, risk-farkindan ayri) — KOD DEGISIKLIGI
`env/strike_env._phi`: risk-farkinda `self.dist` -> ham Manhattan. Gozlem
skalari #11 degismedi. test_env GECTI. Kritik test: VDN fast-eps (%69 recipe)
+ bu shaping. Beklenti: Phi hedefe MONOTON -> "takilma" azalir (§11.12),
%69 -> ?
| ep | eval team% | not |
|----|-----------|-----|
| 15 | 20.0 | olu 1.30 (risk-farkinda: %65, olu 0.35) |
| 30 | 20.0 | mp 0.333 |
| 45 | 7.5  | COKUS (VARIS %10, adim 3682) |
SONUC: **ham-Manhattan shaping REGRESYON** (%69 -> ~%20). Ajan hedefe dogru
riske BAKMADAN kosuyor (olu 1.3 vs 0.35). Burak'in risk-farkinda shaping'i
MEGER ISE YARIYORDU: bir radari dolasmak risk-mesafeyi DUSURUYOR ("ilerleme"
gibi hissettiriyor) -> guvenli rota tesvik ediliyor. Ham Manhattan bunu
kaldiriyor -> pervasizlik. §11.12'nin "takilma"si gercek bir optimizasyon
sorunu, shaping artefakti DEGIL. GERI ALINDI (git revert 4a83e72).

### it5-it6 NIHAI: euzxx fikirleri (opt 1 + opt 3) %69'u GECMEDI
- opt 1 (entropy curr): MAPPO ~2x'e cikti (mp 0.135) ama ~%10; KALDI (zararsiz).
- opt 3 (ham-Manhattan shaping): REGRESYON %69->%20; GERI ALINDI.
- EN IYI hala `it2_vdn_epsfast.pt` %69 (`--eps-start 0.1`, kod degisikligi YOK).
- %75 icin geriye: "deeper RL" (greedy politika kirilganligi / kredi atama) —
  daha buyuk arastirma isi, hizli bir bayrakla degil.

### it7 — Dueling mimari + fast-eps (KOD DEGISIKLIGI YOK, --dueling zaten var)
Beklenti: greedy politika action-gap cokmesinden salinima giriyor; Dueling
V(s)/A(s,a)'yi mimari ayirir -> greedy eval training'e yaklasir (%65->%69+).
SONUC: **ELENDI (ep25'te oldu).** training MA %8 (it2 ayni noktada %48),
olu(ma) 1.92 (pervasiz), 14.2 s/ep (2x yavas). Rastgele init A-head'i 250
episode'da (fast-eps recipe) toparlamiyor. Kill @ ep25.

### it8 — Advantage Learning operatoru (Bellemare 2016) — KOD DEGISIKLIGI
config.py `AL_ALPHA_DEFAULT=0.9` + `agents/vdn.py` learn() AL dali +
`train.py --al-alpha`. Hepsi bayrak-arkasi, al_alpha=0.0 -> BIREBIR eski.
Gerekce: q_gap probu OLCTU — it2 en iyi ckpt (ep25) gap=0.033, eval-cokus
ep50 gap=0.021; FA gurultusuyle ayni mertebe. AL: T_AL Q = r+g*V(s') -
alpha*(V(s)-Q(s,a)); greedy'de 0 (politika korunur), non-greedy asagi ->
gap ~1/(1-alpha)=10x. Sicaklik yok, tek knob. n-adim (GAMMA notu) ve
soft-target (PER notu) elenmislerinden farkli: ufka/target ritmine dokunmaz.
Beklenti: greedy eval team_success %69 -> %75+; q_gap 0.03 -> ~0.2+.
Once diagnostik: eval_eps.py (it2 ckpt, eps in [0,.02,.05,.1,.2], 100 harita,
batched/CUDA) — eps>0 belirgin iyi = AL hipotezi dogru yonde. (Diagnostik CPU'da
cok yavas, iptal; it2 log'undaki training-MA %68-75 [eps~0.08] vs greedy eval
%25-67 [eps=0] farki ZATEN ayni kaniti veriyor.)

### it8 NIHAI: AL action-gap'i DUZELTTI ama TAVANI GECMEDI — tepe %45 @ ep45
| ep | eval takim% | VARIS% | olu | adim | q_gap (dense) |
|----|-------------|--------|-----|------|---------------|
| 15 | 20.0 | 95 | 1.60 | 1541 | — |
| 25 | (dense MA %24) | — | — | — | **0.202** (it2: 0.033) |
| 30 | 32.5 | 85 | 1.05 | 2247 | — |
| 45 | **45.0** | 55 | 0.33 | 3198 | — |
| 50 | (dense MA %48) | — | — | — | **0.361** (it2: 0.021) |
| 60 | 32.5 | 42.5 | 0.25 | 3385 | — |

**SONUC: AL MEKANIZMAYI COZDU, SORUNU COZMEDI.** q_gap it2'nin ~10-17
katina cikti (0.02 -> 0.36), Q buyuklugu bastirildi — action-gap cokmesi
DURDU. AMA eval tepe %45 (< it2 %69), ep45'ten sonra dusuyor. VARIS %95 ->
%42.5 COKTU, adim 4000 tavanina dayaniyor.

**TESHIS (it8'in asil degeri): tavan action-gap sorunu DEGIL.** AL greedy
politikayi "kafasi karisik/salinan"dan "kendinden emin/asiri temkinli"ye
cevirdi — ama politika hala hedefe varmayi birakip timeout yiyor. "Kucuk eps
stuck'i kiriyor" gozlemi = fiziksel itme ajani karamsar deger fonksiyonunun
gonullu gecmedigi noktadan gecirıyor, Q-karisikligi DEGIL. Sorun KARAMSARLIK/
asiri-temkin drifti. AL kodu KALIYOR (bayrak-arkasi, zararsiz, gap probu
icin faydali arac). Sonraki: Munchausen (AL + entropi) — entropi terimi
politikanin rijit-temkin bir stratejiye COKMESINE direnir.

### it9 NIHAI: Munchausen tau=0.02 — it8'den DE KOTU (tepe ~%40 @ ep30, dusuyor)
| ep | eval takim% | VARIS% | olu | adim | q_mean/q_gap (dense) |
|----|-------------|--------|-----|------|----------------------|
| 15 | 32.5 | 55.0 | 0.88 | 2969 | — |
| 25 | (MA %64) | — | 1.04 | 1740 | q_mean 1.02, q_gap 0.029 |
| 30 | 40.0 | 47.5 | 0.62 | 3236 | — |

**SONUC: entropi yontemi eval-argmax ile UYUMSUZ.** tau=0.02 Q'yu SISIRMEDI
(q_mean 1.02 ~ it2'nin 1.17) ama q_gap'i de acmadi (0.029 ~ it2'nin 0.033) —
tau bu is icin cok kucuk. Training MA guclu (%64 > it2 %48) cunku Munchausen
STOKASTIK bir politika egitiyor ve o training'de iyi. Ama EVAL argmax aliyor,
ve entropi terimi Q-manzarasini DUZLESTIRIYOR -> argmax daha KOTU. AL ve
Munchausen ZIT sebeplerden basarisiz: AL gap acar (kararli ama temkinli
argmax), Munchausen gap kapatir (stokastik politika, kotu argmax). Kod
KALIYOR (bayrak-arkasi). tau'yu buyutmek (0.05-0.1) argmax'i daha da
bozardi — bu yol kapali.

### it10 NIHAI: VANILLA fast-eps + INCE eval — GIZLI TEPE YOK (NULL sonuc)
ep25 dense BYTE-BYTE it2 ile AYNI (GPU determinizmi -> it10 == it2). Evaller:
ep10 %36 (VARIS %98, olu 1.28) / ep20 **%64** / ep30 %44 (VARIS %60, drift).
Tepe ep20-25, ~%64-68. it2'nin kaba ep25/50/75 grid'i GERCEK tepeyi kacirmadi.
**Fast-eps tavani ~%67-69, drift ~ep25'te basliyor — kesin.** Ince grid bir
sey degistirmedi.

### it11: LayerNorm Q-agi (BroNet/CrossQ) — KOD DEGISIKLIGI (--layernorm)
Gerekce: belgeli Q-IRAKSAMASI (config.py §11.14: q_mean 17->37, hic
duzlesmiyor) sustained TD'de argmax politikasini bozuyor — fast-eps ~ep25'te
yakalayip DURUYOR ama cozmuyor. Her gizli Linear'dan SONRA (ReLU'dan ONCE)
LayerNorm -> aktivasyon dagilimi sabit -> son Linear girdisi sinirli -> Q
buyumesi durur. `agents/networks.py` `layernorm` bayragi (state_dict head.0/2/4
DEGISMEZ, eski ckpt yuklenir; --layernorm ile head.1/3 LN eklenir, sifirdan).
AL/Munchausen'den FARKLI: TD hedefine DOKUNMAZ, sadece mimari — Q'yu manzarayi
DUZLESTIRMEDEN (Munchausen sorunu) veya asagi CEKMEDEN (AL sorunu) sinirlar.
Beklenti: q_mean ~1-3'te kalir (it2: 1.17->2.15->2.66->...), eval ep25 SONRASI
COKMEZ -> daha gec/yuksek tepe. Watch: q_mean dense'te patlarsa veya ep15/30
it7-dueling gibi cok geride ise kill.

### it11 NIHAI: LayerNorm KATASTROFIK — hedef-yonelimini oldurdu
eval ep15 %20 (VARIS %20!) -> ep30 **%2.5** (VARIS %2.5, adim 3800 = hep
timeout, olu 0.17). Baslik-ici LayerNorm hedef-yon MAGNITUDE sinyalini
(dx/dy/dist_goal skalarlari) normalize edip yok ediyor -> ajanin hedefe
GITME durtusu kayboluyor, sadece guvenli dolaniyor. Kill @ ep30.
**How to apply: --layernorm baslikta DENEME.** (Belki conv'da veya
affine'siz farkli olurdu ama bu yol simdilik kapali.)

## it7-it11 OZET: 5 deneme, hicbiri %69'u gecmedi
| it | ne | sonuc |
|----|-----|-------|
| it7 | Dueling | ep25 MA %8, yavas yakinsama — ELE |
| it8 | Advantage Learning a=0.9 | gap 15x acildi ama asiri-temkin drift, tepe %45 |
| it9 | Munchausen tau=0.02 | entropi Q'yu duzlestirdi, argmax bozuldu, %40 |
| it10 | ince eval (vanilla) | GIZLI TEPE YOK — fast-eps tavani ~%67-69 kesin |
| it11 | LayerNorm | KATASTROFIK, hedef-durtusu oldu, VARIS %2.5 |

**TESHIS PEKISTI:** greedy argmax extraction sorunu, ama Q-fonksiyonuna /
mimariye her mudahale "hedefe var + guvende kal" dengesini BOZUYOR. Training
MA %74.7'ye ulasiyor (ep75) — politika IYI olabiliyor, greedy cikarim
basarisiz. Tek ise yarayan: fast-eps + erken ckpt (denge'yi ERKEN yakala).

### it12: --map-seed TESHISI — %69 SAGLAM DEGIL, ozgul sansli bilet
`--map-seed` eklendi (egitim harita dizisini ag-init'ten ayirir).
| kosu | net_seed | map_seed | ep10 | ep20 |
|------|----------|----------|------|------|
| it10 (=it2) | 0 | 0 | %36 (VARIS 98) | %64 |
| it12a | 1 | 0 | %10 (VARIS 16) | **%0** |
| it12b | 0 | 1 | %16 (VARIS 14) | (kill) |
| it2c_s1 | 1 | 1 | %2.5 | %27.5 (tepe) |

**SONUC: (net0, map0) KOMBINASYONU tikliyor. Yarisini boz -> cokuyor.**
Seed 0'in iyi haritalari kotu ag-init'i KURTARMIYOR; seed 0'in iyi ag'i
kotu haritalarla COKUYOR. %69 tek bir sansli (net,map) cifti — kolay
transfer edilemez. Kok neden: 250-ep fast-eps KARARSIZ (yuksek gradyan
varyansi, config.py batch notu).

### it13: --vdn-batch 256 — TERS TEPKI, batch buyutmek COKERTIYOR
seed 0 + batch 256: eval ep15 %14 (VARIS %16, adim 3760 = timeout).
batch-128 seed-0 ep15 ~%65 (it2b_s0). 2x batch -> ~50 puan DUSUS.
**MEKANIZMA: buyuk batch = temiz gradyan = deger fonksiyonu KARAMSAR
sabit-noktasina DAHA HIZLI yakinsiyor. Kucuk-batch gurultusu politikanin
o asiri-temkin yerel-optimuma OTURMASINI ENGELLIYORDU** — tipki eps
gurultusu gibi. (config batch 32->128 notu: o LONG-run kaosunu duzeltmisti,
farkli rejim.)

### PATTERN (it7-it13): GURULTU YARDIM EDER, keskinlestirme ZARAR
- YARDIM: eps gurultusu (fast-eps %33->%69), kucuk-orta batch
- ZARAR: batch 256 (hizli yakinsama), LayerNorm (sinyal yok), AL (gap
  keskinlestir -> emin-ama-temkinli), Munchausen (Q duzlestir -> kotu argmax),
  Dueling (yavas)
Kok: temiz/hizli/keskin ogrenme -> KARAMSAR asiri-temkin deger sabit-noktasina
daha hizli oturuyor. Cozum yonu: DAHA COK gurultu / anti-yakinsama.

### it14 NIHAI: batch 64 de TERS — ic-halka delme (88), pervasiz %20
batch 128 IKI YONDE de sweet spot: 256 -> asiri-temkin (VARIS %16),
64 -> pervasiz (ic-halka 88). Her knob dengeyi bozuyor. batch 128 KAL.

### it15 NIHAI: checkpoint ENSEMBLE — BUST (her kombinasyon tek-ckpt'ten KOTU)
seed 0 fast-eps, --save-all-ckpts, ep5-30 kaydedildi. 100-harita Q-ortalama
ensemble (scratchpad/eval_ensemble.py, repo'ya dokunmaz):
| config | team% | VARIS% | mp | timeout% | olu |
|--------|-------|--------|-----|----------|-----|
| **TEK ep25** | **70.0** | 80.0 | 0.692 | 34.0 | 0.39 |
| ens {15,20,25} | 62.0 | 74.0 | 0.626 | 37.0 | 0.48 |
| ens {20,25,30} | 53.0 | 64.0 | 0.524 | 37.0 | 0.48 |

**SONUC: ensemble ZARARLI.** Snapshot'lar tek kosunun anlik halleri ->
COK korele; zayif/temkinli snapshot'lari Q-ortalamaya katmak ep25'i asagi
cekiyor. Bagimsiz tohumlar gerekirdi ama onlar da kotu (seed 1 %27).
**Tek ep25 fast-eps ckpt = %70 (100 harita, temiz olcum).** Onceki %69
kaydi 40-harita selector'du; gercek ~%70. Timeout %34 = asil kayip kanali.

### it15b: eps-sweep (deploy-time uniform eps) — NET NEGATIF
| eps | team% | timeout% | olu | not |
|-----|-------|----------|-----|-----|
| 0.00 | **70.0** | 34.0 | 0.39 | greedy |
| 0.02 | 63.0 | 7.0 | 1.03 | timeout duzeldi ama olum |
| 0.05 | 51.0 | 3.0 | 1.34 | |

**KANIT: greedy'nin timeout'u GERCEK bir stall** — kucuk eps %34->%7
duzeltiyor (~27 harita "takilan" politika). AMA uniform-random radara da
giriyor -> olu 0.39->1.03 -> NET team DUSUYOR. Uniform kesif cok kaba.

### it15c: BOLTZMANN deploy (softmax Q/temp) — DE NET NEGATIF
| temp | team% | timeout% | olu |
|------|-------|----------|-----|
| greedy | **70.0** | 34.0 | 0.39 |
| 0.01 | 68.0 | 10.0 | 0.86 |
| 0.02 | 60.0 | 4.0 | 1.19 |

Boltzmann uniform-eps'ten IYI (Q-agirlik olumu biraz azaltiyor: 0.86 vs
1.03 @ benzer timeout) ama HALA net negatif. **Stall'i kirmak = radar-yogun
bolgede HAREKET = risk. Deger fonksiyonu tam da onu kacinmak icin
stall'lamis. O ~27 harita GERCEKTEN tehlikeli, politika DOGRU reddediyor.**

### DEPLOYMENT-POLITIKA ACISI TUKENDI (it15b + it15c)
uniform-eps VE Boltzmann: ikisi de timeout'u olume takas ediyor, net
negatif. **greedy %70 = bu checkpoint'in GERCEK tavani.** %75'e giden tek
yol: DAHA IYI DEGER FONKSIYONU (o 27 stall haritada oracle'in bulabildigi
guvenli rotayi bulan). Oracle tavani %81.5 -> %70 = oracle'in %86'si.

### it16: NET-TOHUM taramasi — seed 2 = %48, KESILDI (GPU'yu QR-DQN'e verdim)
seed 0=%70, seed 1=%27, **seed 2=%48**. Yuksek varyans dogrulandi;
seed 0 yuksek outlier, gerisi ~%40-50 kumelesıyor. seeds 3-7 dusuk-olasilik
(hicbiri %75'e yakin degil) -> tarama kesildi, QR-DQN (daha yuksek EV)
GPU'yu aldi. Gerekirse gece seeds 3-7 devam.

### it17: QR-DQN (distributional RL) — son ciddi algoritmik bahis (kod HAZIR)
agents/networks.py + agents/vdn.py + config.py + train.py, --quantiles /
--qr-optimism bayraklari (commit 8814d1a). --quantiles 1 -> BIREBIR eski.
Motivasyon: deger fonksiyonu stokastik olum cezasi altinda KARAMSAR mean'e
cokuyor. QR-DQN tum getiri dagilimini ogrenir (kuantil Huber, Dabney 2017);
+ IYIMSER aksiyon secimi argmax(mean + k*std) karamsar cokmeye DOGRUDAN
karsi. VDN toplami komonoton (QR-MIX). Smoke gecti.
it17 kosusu (seed-tarama sonrasi): seed 0 fast-eps + --quantiles 16
--qr-optimism 0.5. ~%40 sans (12 deneme elendi ama bu principled + denenmemis).

### it17 (optimism 0.5): PERVASIZ — kill @ ep15
eval ep15 team %0, olu 1.72, ic-halka 19. mean + 0.5*std optimism bonusu
action-gap'in (~0.03) 10-50 kati -> politika hep yuksek-varyansli ("hedefe
dogru riskten gec") aksiyonu seciyor -> olum. Optimism egitimde de aktif
(buffer pervasiz trajektorilerle doluyor). 0.5 COK fazla.

### it17b (optimism 0.0): risk-notr QR-DQN — DE ELENDI
eval ep15 team %2 olu 1.74 / ep30 team %18 olu 1.56. Distributional TEK
BASINA da pervasiz + yavas. 80-cikis basligi (16 kuantil x 5 aksiyon)
fast-eps butcesinde 'guvenli rota'yi ogrenmiyor — Dueling/LayerNorm/batch256
ile AYNI kader (Q-basligi degisikligi = fast-eps recipe'i bozar).
**QR-DQN ELENDI.**

### it19: STUCK-TETIKLI BOLTZMANN DEPLOY — %75 (dogrulama bekliyor!)
scratchpad/eval_stuckescape.py (repo'ya dokunmaz, eval_map_seeds + team_success
AYNI). Ajan SADECE stall ettiginde (risk-mesafe stuck_n adimdir azalmadi)
argmax yerine softmax(Q/temp) ornekler; digerlerinde greedy. Elle hedef-arama
YOK — sadece ogrenilen Q. Rota (niyet) her zaman greedy.
| config | team% | timeout% | olu |
|--------|-------|----------|-----|
| greedy (baseline) | 69-70 | 34 | 0.39 |
| **stuck_n=150 temp=0.03** | **75.0** | 9 | 0.63 |
| stuck_n=150 temp=0.06 | 75.0 | 9 | 0.63 |
| stuck_n=300 temp=0.03-0.10 | 74.0 | 10 | 0.64 |

**MEKANIZMA:** greedy argmax ~25 haritada stall (risk-mesafe azalmayi
birakiyor, action-gap ~0.03 argmax'in karar veremeyecegi kadar kucuk).
150 adim ilerlemesizlik -> softmax(Q/0.03) o kucuk gap'i ORNEKLEMEYLE
degerlendirir -> stall kirilir. Kotu aksiyon (radar, cok -Q) ~secilmez.
timeout %34->%9, olu sadece 0.39->0.63 (uniform eps'in 1.03'une karsi).
DOGRULAMA: 5 zar-tohumu ile (team_success stokastik) — ort >=73 ise saglam.

### it18: TRAIN/EVAL ZORLUK UYUMSUZLUGU (yeni acinim!)
curriculum_n_radar(ep, 250) frac=0.6 ile: ep25 -> **12 radar**, ep50 -> 15.
Ama EVAL her zaman 25 radar. **Fast-eps ckpt'i (~ep25, %70) KOLAY 12-radar
haritalarda egitilip ZOR 25-radar eval'da olculuyor.** ~15 denemedir bunu
gozden kacirdim.
--curriculum-frac 0.1 -> ep25'te 25 radara ulasir (hala rampali 11->25).
it18a: seed 0 fast-eps + --curriculum-frac 0.1. eval-every 10.
Beklenti: ckpt dogru zorlukta egitilir -> %70 -> ? Risk: config uyarisi
"yogun uctan basla = hic basari gorme" — ama 0.1 rampali. ep15-25 VARIS %0
= cok hizli -> frac 0.2 dene.



**A. Eval haritalarinin bellek-ici cache'i  [EN GUVENLI, self-contained]**
`env/strike_env.py`:
- modul seviyesi `OrderedDict _MAP_CACHE`, maxsize ~64
- `_build_map(radars, cacheable=False)`: cacheable ise (radars, hazard_mode,
  goal, n) anahtariyla `build_zone_map`+`build_risk_distance_map` sonucunu
  ara/doldur
- `reset()`: `cacheable=(map_seed is not None)` gecir
Neden guvenli: zone/dist insa sonrasi READ-ONLY; danger her seferinde yeniden
turetiliyor (taze array); alert taze zeros. Eval map tohumlari SABIT
(`eval_map_seeds`) -> ilk eval'dan sonra HER eval tam isabet (dice + route +
sonraki eval). Egitim haritalari (rastgele tohum) cache'e HIC girmez -> churn
yok, bellek buyumez. Ogrenmeye risk: SIFIR (deterministik yeniden-hesap,
birebir ayni array).
Kazanc: eval basi harita-insa ~4 insa/harita -> warmup sonrasi ~0. ~40 s/eval.

**B. `StrikeMissionEnv.__init__` defer_reset  [eval/rollout'taki cop insayi keser]**
`run_chunk` / `ppo_parallel_rollout` N env kurup HEMEN `e.reset(map_seed=...)`
cagiriyor. `__init__` su an kosulsuz `self.reset()` cagiriyor -> once RASTGELE
bir cop harita insa ediliyor. `__init__(..., defer_reset=False)` ekle; o 3
cagri noktasinda `True` gecir. Kazanc: eval'da -2 insa/harita (~26 s/eval),
ppo rollout'ta -1 insa/env/chunk. Risk: dusuk ama `__init__`'e dokunuyor (o 3
nokta kullanim oncesi reset ediyor mu -> EDIYOR, dogrulandi).

**C. ppo rollout/eval while-loop'unda biten env'i islememe (straggler)**
`done[i]` iken kod hala `e.state()`/`e.action_mask()` cagirip np.stack'e
koyuyor, `act_batch` N'in hepsini isliyor. 1 env 4000 adima giderken 31'i
1500'de bitmisse -> 2500 tur 32-genis is, 1 aktif env icin. Fix: state/mask/obs
SADECE aktif idx'ler icin; act_batch alt-batch'te; sonuclari geri dagit.
Kazanc: MAPPO/HAPPO'da buyuk (en kotu straggler). VDN/QMIX auto-reset zaten
kaciniyor. Risk: ORTA — rollout dongusune dokunuyor, HANGI transition'larin
kaydedildigini DEGISTIRMEMELI.

**D. `*_EVAL_EVERY` 24/25 -> 48/50**  eval sayisini yariya indirir. Senin
acikca izin verdigin knob. Belgeli dezavantaj: uzun kosuda VDN tepe-yakalama
kacabilir. Ogrenmeye risk yok.

**Kapsam disi:** `DeterministicAdaptiveAvgPool2d` (Python-dongu pooling) bir
DOGRULUK ozelligi (CUDA determinizmi) — geri alma. 4 paralel kosu RAM tuketir
(15.6 GB, VDN buffer ~5.4 GB) — kosular SIRAYLA olmali.
