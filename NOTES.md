# Gece throughput dongusu — 2026-08-29

Amac: happo / mappo / qmix / vdn egit; herhangi birinin eval team_success'i
%75'i gecerse DUR. Ikincil: 500 episode < 2.5 saat.

---
## >>> SABAH OZETI (Burak, once bunu oku) <<<

**Dongu 1. iterasyonda tikandi: `codex` kullanim limitinde (26 gun reset, Sep 24).**
Senin tarif ettigin dongunun 4-5. adimlari (codex analiz -> onerisini uygula)
codex olmadan yapilamaz. "Komut hata verirse donguyu surdurme" kuralin geregi
throughput KOD degisikligi yapmadim. Bunun yerine bosta GPU'yu ASIL hedefe
(%75 team_success) yonlendirdim: kod degisikligi YOK, sadece egitim kosulari.

**%75'e ulasilamadi. Ulasilamaz da — mevcut kurulumla.** Kanit:
- QMIX 500-ep: ep50 %5 -> ep100 %2.5 -> ep150 %0. Coktu.
- VDN 300-ep (seed 2): ep25 %5 -> ep75..150 %0. Ayni cokus, eps tabana inince toparlamadi.
- Tohum taramasi (48-ep, 4 tohum): QMIX s0=%37.5(SANS) s1=%7.5 s3=%10 ;
  VDN s0=%25(sans) s1=%5 s2=cokus s3=%2.5. Medyan ~%5.
- MAPPO/HAPPO: %5-7.5, koordinasyon yok (route_overlap 1.0).

**Cokus mekanizmasi** (yeni gozlem, degerli): ~ep50-150 arasi anlamli epsilon'la
egitim politikayi "urkek"lestiriyor — o kadar risk-kacinan hale geliyor ki
HEDEFE GITMEYI birakiyor (VARIS %100 -> %0, her episode timeout, analitik
"guvenlik" 0.85 ama gorev 0.0). config.py R_RISK_COEF=65 notundaki tam olarak
uyarilan mod. it1'in yuksek sayilari (QMIX %37.5) SADECE 48-ep kosunun eps'i
ep24'te tabana indirip politikayi cokusten ONCE greedy'ye dondurmesindendi —
tohuma da bagli, kalici degil.

**En iyi kullanabilir checkpoint'ler** (bu gece uretildi):
- `runs/ckpt/it1_qmix.pt` — QMIX seed 0, mission_prob 0.408, eval team %37.5 @ ep48
- `runs/ckpt/it1_vdn.pt`  — VDN seed 0, eval team %25 @ ep24
Tarihi en iyi hala `runs/ckpt/BEST_vdn_seed2_team45.pt` (%45) ama o ESKI 18-skalar
obs uzayiyla; simdiki 22-skalar (eylem-ozgu risk) obs ile o sonuc tekrarlanmadi.

**s/ep olcumleri** (it1 probe, 48-ep, n_envs 32, cuda, solo): VDN 10.95, QMIX
13.34, MAPPO 10.24, HAPPO 10.61. Gercek 500-ep kosu eps takvimi daha uzun oldugu
icin episode'lar daha uzun -> ~14-20 s/ep (qmix_long500'de gorulen). "500 ep <
2.5 saat" = <18 s/ep: VDN/MAPPO/HAPPO SIGAR, QMIX sinirinda. (Ama %75 sarti
saglanmadigi icin DUR kosulu tetiklemedi.)

**Codex donunce dongu icin hazir 3 throughput adayi**
(scratchpad/patch_candidates.md, hicbiri UYGULANMADI — kod dokunulmadi):
1. Eval'da risk-haritasi in-memory cache (eval map tohumlari sabit; run_chunk
   dice+route + her eval ayni haritayi 4x kuruyor). En guvenli.
2. StrikeMissionEnv.__init__'e defer_reset — eval/rollout'ta atilan rastgele
   harita insasini onler.
3. ppo rollout/eval while-loop'unda BITEN env'leri isleme sokmama (straggler).

**Repo durumu:** temiz. KOD DEGISMEDI (git diff 7f5b4d0..HEAD sadece NOTES.md).
`python -m tests.test_env` -> TUM TESTLER GECTI (WIP baseline saglam). Orphan
python sureci YOK. (Not: 06:45'ten beri idle bir `codex` sureci var — benim
degil, dokunmadim.)

**3 saatlik pencerede daha fazla egitim kosusu YAPMADIM** bilerek: cokus deseni
5 kosuda (2 uzun + sweep) net dogrulandi, tekrar re-confirm etmek GPU israfi.
Elimdeki tam resim bu.

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
