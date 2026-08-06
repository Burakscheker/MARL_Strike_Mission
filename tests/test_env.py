"""Ortam kabul testleri — Strike_Mission.md §Asama 1.

pytest'siz calisir: python -m tests.test_env

En onemli iki test:
  * test_simulation_matches_analytic — ortamin OLUM ZARI, risk_oracle'in
    ANALITIK olasiligiyla uyusuyor mu? Uyusmuyorsa butun metrikler yalan soyler.
  * test_terminal_obs_keeps_updating — terminal olan ucagin gozlemi guncellenmeye
    devam ediyor mu? MARL-Pathfinding'de bunun bozulmasi VDN'i sessizce IQL'e
    cevirmisti ("en sinsi bug").
"""
import numpy as np

import config as C
from baselines.risk_oracle import (exposure, greedy_path, risk_distance_map,
                                   survival_prob, zone_map)
from env.strike_env import StrikeMissionEnv

_FAILED = []


def check(name: str, cond: bool, detail: str = ""):
    status = "GECTI" if cond else "KALDI"
    print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        _FAILED.append(name)


# ------------------------------------------------------------------ geometri

def test_zone_geometry():
    z = zone_map()
    check("zone sekli", z.shape == (C.GRID_N, C.GRID_N), str(z.shape))
    for i, (r, c) in enumerate(C.RADARS):
        check(f"R{i+1} merkezi ic halkada", z[r, c] == 2)
        # dis halka sinirinda 1, bir disinda 0
        rr = r + C.OUTER_HALF
        check(f"R{i+1} dis sinir", z[rr, c] == 1, f"z={z[rr,c]}")
        if rr + 1 < C.GRID_N:
            check(f"R{i+1} dis sinirin 1 disi", z[rr + 1, c] == 0, f"z={z[rr+1,c]}")
        ri = r + C.INNER_HALF
        check(f"R{i+1} ic sinir", z[ri, c] == 2, f"z={z[ri,c]}")
        check(f"R{i+1} ic sinirin 1 disi", z[ri + 1, c] == 1, f"z={z[ri+1,c]}")
    # kenar genislikleri
    r0, c0 = C.RADARS[0]
    outer_w = int((z[r0, :] >= 1).sum())
    inner_w = int((z[r0, :] == 2).sum())
    check("dis halka genisligi 221", outer_w == 2 * C.OUTER_HALF + 1, str(outer_w))
    check("ic halka genisligi 141", inner_w == 2 * C.INNER_HALF + 1, str(inner_w))


def test_oracle_is_safe_and_optimal():
    d = risk_distance_map()
    p = greedy_path(C.START, C.GOAL, d)
    o, i = exposure(p)
    check("oracle hedefe variyor", p[-1] == C.GOAL, str(p[-1]))
    check("oracle optimal uzunlukta", len(p) - 1 == 2 * (C.GRID_N - 1), str(len(p) - 1))
    check("oracle hic radara girmiyor", o == 0 and i == 0, f"dis={o} ic={i}")
    check("oracle analitik hayatta kalma = 1", abs(survival_prob(p) - 1.0) < 1e-9)


# --------------------------------------------------------------------- gozlem

def test_dims():
    env = StrikeMissionEnv(seed=0)
    obs = env.observations()
    check("OBS_DIM", obs[C.AGENT_1].shape == (C.OBS_DIM,), str(obs[C.AGENT_1].shape))
    check("STATE_DIM", env.state().shape == (C.STATE_DIM,), str(env.state().shape))
    check("gozlemde NaN yok", np.isfinite(obs[C.AGENT_1]).all())


def test_masks():
    env = StrikeMissionEnv(seed=0)
    # B = (0,0) sol ust kose -> YUKARI ve SOL gecersiz
    m = env.action_mask(C.AGENT_1)
    check("kosede YUKARI kapali", m[C.UP] == 0)
    check("kosede SOL kapali", m[C.LEFT] == 0)
    check("kosede SAG acik", m[C.RIGHT] == 1)
    check("aktif ucakta NOOP kapali", m[C.NOOP] == 0)

    env.alive[C.AGENT_2] = False
    m2 = env.action_mask(C.AGENT_2)
    check("terminal ucakta SADECE NOOP", m2[C.NOOP] == 1 and m2.sum() == 1)


def test_terminal_obs_keeps_updating():
    """VDN kredi kanali: A2 terminal olduktan SONRA da gozlemi degismeli.

    A2'nin gozlemindeki 'diger ucaga dx/dy' skalarlari A1 ilerledikce
    degismek ZORUNDA — degismezse Q(obs_2, NOOP) sabit kalir ve VDN'in
    toplamsal ayristirmasi A1'e hicbir gradyan tasimaz.
    """
    env = StrikeMissionEnv(seed=0)
    env.alive[C.AGENT_2] = False               # A2'yi terminal yap
    o0 = env.observe(C.AGENT_2).copy()
    for _ in range(60):
        env.step({C.AGENT_1: C.RIGHT, C.AGENT_2: C.NOOP})
    o1 = env.observe(C.AGENT_2)
    diff = float(np.abs(o1 - o0).max())
    check("terminal ucagin gozlemi guncelleniyor", diff > 1e-6, f"maxfark={diff:.4f}")


# ------------------------------------------------------------- risk dogrulama

def _run_scripted(env, path, seed):
    """Verilen hucre listesini A1'e adim adim oynat; A1 hayatta kaldi mi dondur.

    A2 hemen terminal yapilir ki tek ucagin riski izole olculsun.
    """
    env.reset(seed=seed)
    env.alive[C.AGENT_2] = False
    # DIKKAT: reset() ucagi START'a koyar. Yolun BASINA tasimazsak asagidaki
    # dongu dogru YON dizisini yanlis YERDE oynatir — ilk yazilista tam bu
    # oldu, test "hic olum yok" diyordu cunku ucak radara hic girmiyordu.
    env.pos[C.AGENT_1] = tuple(path[0])
    env.path[C.AGENT_1] = [tuple(path[0])]
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dr, dc = r1 - r0, c1 - c0
        act = C.DIRS.index((dr, dc))
        _, _, done, _ = env.step({C.AGENT_1: act, C.AGENT_2: C.NOOP})
        if not env.alive[C.AGENT_1] or done:
            break
    return env.alive[C.AGENT_1]


def test_simulation_matches_analytic(trials=400):
    """Ortamin zari analitik olasiliga uyuyor mu?

    Kisa ama TEHLIKELI bir yol secilir (R1'in ic halkasindan duz gecis) —
    analitik hayatta kalma orta seviyede olsun ki istatistik anlamli olsun.
    """
    r0, c0 = C.RADARS[0]
    # ic halkanin ustunden altina duz dikey gecis
    path = [(r, c0) for r in range(r0 - C.INNER_HALF, r0 + C.INNER_HALF + 1)]
    analytic = survival_prob(path)

    env = StrikeMissionEnv(max_steps=10_000, seed=0, risk_shaping=False)
    alive = sum(_run_scripted(env, path, seed=1000 + k) for k in range(trials))
    emp = alive / trials
    # 3 sigma bandi
    sd = (analytic * (1 - analytic) / trials) ** 0.5
    ok = abs(emp - analytic) <= 3 * sd + 0.01
    check("simulasyon = analitik (ic halka gecisi)", ok,
          f"analitik={analytic:.4f} olculen={emp:.4f} (+-{3*sd:.4f})")

    # analitik deger, config'in "boydan boya %90 olum" kalibrasyonunu tutuyor mu
    check("ic halka boydan boya ~%90 olum",
          abs((1 - analytic) - C.P_INNER_TOTAL) < 0.02,
          f"olum={1-analytic:.4f} hedef={C.P_INNER_TOTAL}")


def test_safe_path_never_dies(trials=30):
    """Oracle yolunda HIC olum olmamali (p_death=0 hucrelerden geciyor)."""
    d = risk_distance_map()
    path = greedy_path(C.START, C.GOAL, d)
    env = StrikeMissionEnv(max_steps=10_000, seed=0, risk_shaping=False)
    deaths = sum(0 if _run_scripted(env, path, seed=2000 + k) else 1
                 for k in range(trials))
    check("guvenli yolda olum yok", deaths == 0, f"{deaths}/{trials} olum")


# --------------------------------------------------------------------- odul

def test_reward_terminals():
    env = StrikeMissionEnv(max_steps=10_000, seed=0, risk_shaping=False)
    env.reset()
    # ikisini de hedefin bir adim yanina koy
    g = C.GOAL
    env.pos = {C.AGENT_1: (g[0] - 1, g[1]), C.AGENT_2: (g[0], g[1] - 1)}
    _, r, done, info = env.step({C.AGENT_1: C.DOWN, C.AGENT_2: C.RIGHT})
    check("ikisi de ayni anda varinca episode biter", done)
    check("ilk+ikinci varis odulu",
          abs(r - (C.R_STEP + C.R_FIRST_GOAL + C.R_SECOND_GOAL)) < 1e-5,
          f"r={r:.3f}")
    check("both_reached", info["both_reached"])
    check("takim basarisi", info["team_success"])


def test_death_penalty(trials=200):
    """Ic halkaya girmek ~%90 olum getirmeli (per_entry: TEK zar)."""
    r0, c0 = C.RADARS[0]
    deaths = 0
    for k in range(trials):
        env = StrikeMissionEnv(max_steps=10_000, seed=k, risk_shaping=False)
        env.reset()
        env.pos = {C.AGENT_1: (r0, c0), C.AGENT_2: (0, 0)}
        env.step({C.AGENT_1: C.UP, C.AGENT_2: C.RIGHT})
        deaths += int(not env.alive[C.AGENT_1])
    rate = deaths / trials
    check("ic halkaya girisde ~%90 olum", abs(rate - C.P_INNER_TOTAL) < 0.06,
          f"olculen={rate:.3f} hedef={C.P_INNER_TOTAL}")


def test_per_entry_no_accumulation():
    """PATRONUN KURALI: bolgede kalmak EK risk getirmemeli.

    Ic halkanin ortasina koyup 200 adim gezdiriyoruz; ILK adimin zarindan
    sonra hic yeni zar atilmamali. Zar atilsaydi 200 adimda hayatta kalan
    kalmazdi.
    """
    r0, c0 = C.RADARS[0]
    survived = 0
    trials = 200
    for k in range(trials):
        env = StrikeMissionEnv(max_steps=10_000, seed=k, risk_shaping=False)
        env.reset()
        # ILK zari atlamak icin ucagi zaten "icerideymis" gibi baslat
        env.pos = {C.AGENT_1: (r0, c0), C.AGENT_2: (0, 0)}
        env._prev_zone[C.AGENT_1] = 2
        for _ in range(200):
            env.step({C.AGENT_1: C.UP if _ % 2 == 0 else C.DOWN,
                      C.AGENT_2: C.RIGHT})
            if not env.alive[C.AGENT_1]:
                break
        survived += int(env.alive[C.AGENT_1])
    check("ic halkada 200 adim = SIFIR ek risk", survived == trials,
          f"{trials - survived}/{trials} oldu (0 olmali)")


def test_reentry_rolls_again():
    """Cikip tekrar girmek YENI bir zar olmali (yeni angajman)."""
    r0, c0 = C.RADARS[0]
    edge = r0 - C.OUTER_HALF            # dis halkanin ust siniri
    deaths = 0
    trials = 300
    for k in range(trials):
        env = StrikeMissionEnv(max_steps=10_000, seed=k, risk_shaping=False)
        env.reset()
        env.pos = {C.AGENT_1: (edge - 1, c0), C.AGENT_2: (0, 0)}   # disarda
        # gir (zar 1) -> cik -> gir (zar 2)
        for act in (C.DOWN, C.UP, C.DOWN):
            env.step({C.AGENT_1: act, C.AGENT_2: C.RIGHT})
            if not env.alive[C.AGENT_1]:
                break
        deaths += int(not env.alive[C.AGENT_1])
    rate = deaths / trials
    expect = 1 - (1 - C.P_OUTER_TOTAL) ** 2      # iki bagimsiz giris
    check("iki kez girmek = iki zar", abs(rate - expect) < 0.07,
          f"olculen={rate:.3f} beklenen={expect:.3f}")


def test_phi_not_saturated():
    """Phi YOGUN haritada da 0'a kilitlenmemeli — regresyon testi.

    Eskiden `Phi = 1 - min(1, dist/max_man)` idi; dist RISK-mesafesi oldugu
    icin yogun radar dizilisinde max_man'i asiyor ve kirpma Phi'yi 0'a
    sabitliyordu. Olculdu (15 rastgele 40-radar haritasi): dist(B)/max_man
    ortalama 1.29, en kotu 1.75, 14/15 haritada 1'i asiyor — yani oracle
    yolunun ortalama %16.5'inde shaping sinyali TAMAMEN olu idi, ustelik
    tam da baslangicta. Artik olcek harita basina dist_scale.
    """
    import numpy as np

    from baselines.risk_oracle import (RISK_W, build_risk_distance_map,
                                       build_zone_map, direction_costs)

    env = StrikeMissionEnv(seed=0)
    env.pos[C.AGENT_1] = C.START
    phi_b = env._phi(C.AGENT_1)
    env.pos[C.AGENT_1] = C.GOAL
    phi_h = env._phi(C.AGENT_1)
    check("Phi(B) = 0", abs(phi_b) < 1e-6, f"{phi_b:.4f}")
    check("Phi(H) = 1", abs(phi_h - 1.0) < 1e-6, f"{phi_h:.4f}")

    # YOGUN harita: 40 radar. Kirpma devreye girerse Phi yol boyunca uzun sure
    # 0'da kalir; dist_scale ile monoton artmali.
    rng = np.random.default_rng(0)
    radars = [(int(rng.integers(0, C.GRID_N)), int(rng.integers(0, C.GRID_N)))
              for _ in range(40)]
    z = build_zone_map(C.GRID_N, radars, C.OUTER_HALF, C.INNER_HALF)
    d = build_risk_distance_map(zone=z, risk_w=RISK_W, mode=C.HAZARD_MODE,
                                max_iter=120)
    path = greedy_path(C.START, C.GOAL, d,
                       direction_costs(z, RISK_W, C.HAZARD_MODE))
    scale = max(float(d[C.START]), 1.0)
    max_man = 2 * (C.GRID_N - 1)
    check("yogun haritada dist(B) max_man'i asiyor (kirpma tuzagi gercek)",
          float(d[C.START]) > max_man, f"{d[C.START]:.0f} > {max_man}")

    phis = [1.0 - min(1.0, float(d[c]) / scale) for c in path[::50]]
    dead = sum(1 for v in phis if v == 0.0)
    check("dist_scale ile Phi yolun basinda olu degil", dead <= 1, f"{dead} ornek 0")
    check("dist_scale ile Phi monoton artiyor",
          all(b >= a - 1e-6 for a, b in zip(phis, phis[1:])))


def main():
    print("\n=== geometri ===")
    test_zone_geometry()
    test_oracle_is_safe_and_optimal()
    print("\n=== gozlem ===")
    test_dims()
    test_masks()
    test_terminal_obs_keeps_updating()
    test_phi_not_saturated()
    print("\n=== risk modeli ===")
    test_simulation_matches_analytic()
    test_safe_path_never_dies()
    print("\n=== odul / risk kurali ===")
    test_reward_terminals()
    test_death_penalty()
    test_per_entry_no_accumulation()
    test_reentry_rolls_again()

    print("\n" + "=" * 60)
    if _FAILED:
        print(f"KALAN {len(_FAILED)} test: " + ", ".join(_FAILED))
        raise SystemExit(1)
    print("TUM TESTLER GECTI")


if __name__ == "__main__":
    main()
