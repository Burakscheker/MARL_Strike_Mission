# Gece throughput dongusu — 2026-08-29

Amac: happo / mappo / qmix / vdn egit; herhangi birinin eval team_success'i
%75'i gecerse DUR. Ikincil: 500 episode < 2.5 saat.

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

## Iterasyon gunlugu
| it | tarih/saat | s/ep h/m/q/v | eval team% h/m/q/v | degisiklik | beklenti |
|----|-----------|--------------|--------------------|-----------|----------|
