"""Zar-ACIK kosuda olen ucaklarin OLUM ANI kacinabilir miydi? — teshis.

Kullanici gozlemi: basarisiz haritalarin 10-15'inde ucak bos alanda "durup
dururken" bir halkaya giriyor ve olebiliyor. Bu script her olum icin, olumu
tetikleyen GIRIS adimini bulur (zone artisi olan adim) ve o andaki pozisyonda
baska bir yon (4 komsu + NOOP) zonu ARTIRMADAN mevcut muydu diye bakar.

  KACINILABILIR : ayni adimda zonu artirmayan baska bir hareket vardi
                   (NOOP haric -- o zaten HER ZAMAN guvenlidir, onu saymiyoruz;
                   burada ozellikle ILERLEME saglayan/farkli bir komsu araniyor)
  ZORUNLU       : o pozisyonda TUM komsu haneler de ayni ya da daha yuksek
                   zonda -- gercekten sikismis, secenek yok

Ayrica GIRIS anindaki eski zon 0 (tamamen guvenli bolgeden) miydi diye de
ayirir -- kullanicinin "hicbir sey yokken" dedigi tam olarak bu.
"""
from __future__ import annotations

import numpy as np

import config as C
from env.sampler import eval_map_seeds
from env.strike_env import StrikeMissionEnv
from env.two_agent import play_episode_qmix, play_episode_vdn
from train import build_agent

RUNNER = {"vdn": play_episode_vdn, "qmix": play_episode_qmix}
DIRS4 = ((-1, 0), (0, 1), (1, 0), (0, -1))


def in_bounds(p):
    return 0 <= p[0] < C.GRID_N and 0 <= p[1] < C.GRID_N


def analyze(agent, algo, ckpt, n_scan, max_steps, seed=12345):
    if algo == "vdn":
        agent_obj = build_agent("vdn", 0, "cpu")
        agent_obj.load(ckpt)
    else:
        agent_obj = build_agent("qmix", 0, "cpu")
        agent_obj.load(ckpt)

    env = StrikeMissionEnv(seed=seed, radar_random=True, n_radar=C.N_RADAR,
                           max_steps=max_steps, death_enabled=True)
    env.rng = np.random.default_rng(seed)
    runner = RUNNER[algo]

    total_deaths = 0
    avoidable = 0
    forced = 0
    from_zone0 = 0
    examples = []

    for ms in eval_map_seeds(n_scan):
        info, _ = runner(env, agent_obj, train=False,
                         reset_kwargs={"map_seed": ms, "n_radar": C.N_RADAR})
        zone = env.zone
        for a in (C.AGENT_1, C.AGENT_2):
            if env.alive[a]:
                continue                       # bu ucak olmedi
            path = env.path[a]
            # zone artisi olan ADIMI bul (prev_zone < cur_zone)
            prev_zone_seen = 0
            entry_step = None
            for i in range(1, len(path)):
                prev_pos, cur_pos = path[i - 1], path[i]
                cz = int(zone[cur_pos[0], cur_pos[1]])
                if cz > prev_zone_seen:
                    entry_step = i
                    entry_prev_pos, entry_cur_pos = prev_pos, cur_pos
                    entry_prev_zone, entry_cur_zone = prev_zone_seen, cz
                    # bu SON giris olabilir (ic halkaya art arda giris de
                    # olabilir) -- olum ANINDAKI son girisi istiyoruz, devam et
                prev_zone_seen = cz
            if entry_step is None:
                continue                       # zone hic artmadi ama oldu? (olmamali)

            total_deaths += 1
            if entry_prev_zone == 0:
                from_zone0 += 1

            # o pozisyonda alternatif var miydi (zonu artirmayan baska komsu)?
            alt_found = False
            for dr, dc in DIRS4:
                cand = (entry_prev_pos[0] + dr, entry_prev_pos[1] + dc)
                if cand == entry_cur_pos:
                    continue                   # secilen hareketin kendisi
                if not in_bounds(cand):
                    continue
                cand_z = int(zone[cand[0], cand[1]])
                if cand_z <= entry_prev_zone:
                    alt_found = True
                    break

            if alt_found:
                avoidable += 1
                if len(examples) < 8:
                    examples.append((ms, a, entry_prev_pos, entry_cur_pos,
                                     entry_prev_zone, entry_cur_zone))
            else:
                forced += 1

    print(f"algo={algo}  {n_scan} harita, {total_deaths} olum")
    print(f"  zone0'dan (tamamen guvenliden) giris: {from_zone0}/{total_deaths}")
    print(f"  KACINILABILIR (alternatif komsu vardi): {avoidable}/{total_deaths}")
    print(f"  ZORUNLU (secenek yoktu): {forced}/{total_deaths}")
    print()
    print("ornekler (harita_tohum, ucak, onceki_poz, girdigi_poz, eski_zon->yeni_zon):")
    for ex in examples:
        print(" ", ex)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="vdn", choices=("vdn", "qmix"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scan", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=C.MAX_STEPS)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    analyze(None, args.algo, args.ckpt, args.scan, args.max_steps, args.seed)
