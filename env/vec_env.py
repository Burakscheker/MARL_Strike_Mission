"""N StrikeMissionEnv'i PARALEL tutan ince sarmalayici — SADECE egitim
rollout'unu GPU'ya batch=1 yerine batch=N vermek icin (bkz. train.py
--n-envs notu). eval/demo/viz hicbirini KULLANMAZ, onlar tek env'de
play_episode_vdn ile calismaya devam eder — davranislari bu dosyadan
ETKILENMEZ.

TASARIM: her alt-ortam kendi RNG'siyle (map_seed VERILMEZ) organik rastgele
harita cekmeye devam eder — StrikeMissionEnv.reset()'in varsayilan davranisi
zaten bu (bkz. oradaki docstring). Yani "her episode taze harita" kurali
tek-env kosuda ne ise burada da AYNEN o.

Bir alt-ortam bittiginde (done=True) o adimda HEMEN resetlenir (standart
"auto-reset" vec-env deseni) — boylece N ortamin hepsi HER round aktif
kalir, kisa süren episode'lar uzun sürenleri beklemez.

BUG (bulundu ve duzeltildi, 2026-08-25, dis inceleme): step() eskiden
next_obs'u HER done=True satirinda (olum VEYA timeout farketmeksizin)
reset SONRASI gozlemle DEGISTIRIYORDU. Olum icin zararsiz (JointReplayBuffer
push_done=True cekiyor, learn()'deki (1-done) carpani next_obs'un ICERIGINI
sifirliyor). AMA TIMEOUT icin push_done=False (bkz. train.py — bu KASITLI,
standart zaman-siniri bootstrap'i: timeout GERCEK terminal degil, deger
fonksiyonu kesilmeseydi ne olacagini tahmin etmeye devam etmeli). Yani
timeout satirlarinda bootstrap TAM OLARAK next_obs'u kullanir — ve next_obs
RASTGELE YENI BIR HARITANIN ilk gozlemiydi, ayni trajektorinin GERCEK son
durumu degil! Bu, tum paralel-rollout egitimlerinde (bu oturumun GPU
kosularinin tamami) her timeout gecisinde CIOP bir bootstrap hedefi
uretiyordu.

DUZELTME: step() artik iki ayri gozlem tutar — done olan alt-ortamlar icin
GERCEK (reset ONCESI) gozlem buffer'a donuyor (dogru bootstrap), self.obs
(bir sonraki turun aktif girdisi) ise reset SONRASI gozlemle guncellenir.
İkisi kasitli olarak FARKLI: self.obs "şu an neredeyim", donen next_obs ise
"bu gecisin GERCEK sonucu neydi".
"""
from __future__ import annotations

import numpy as np

from baselines.risk_oracle import RISK_W, direction_costs, oracle_action
from config import AGENT_1, AGENT_2
from env.strike_env import StrikeMissionEnv


class VecStrikeEnv:
    def __init__(self, n_envs: int, max_steps: int, seed_base: int,
                n_radar: int | None = None, risk_shaping: bool = True,
                death_enabled: bool = True, alert_enabled: bool = False):
        self.n = n_envs
        self.n_radar = n_radar
        self.envs = [StrikeMissionEnv(seed=seed_base + i, radar_random=True,
                                      n_radar=n_radar, max_steps=max_steps,
                                      risk_shaping=risk_shaping,
                                      death_enabled=death_enabled,
                                      alert_enabled=alert_enabled)
                    for i in range(n_envs)]
        # BC-capali RL (oracle-anchored VDN) icin: her alt-ortamin 4-yonlu
        # hareket maliyeti haritasi, o ortamin GUNCEL radar dizilimine gore
        # onbelleklenir (harita degistiginde -reset'te- yeniden hesaplanir).
        # O(n^2) sweep AMA harita basina BIR KEZ; oracle_action() sonrasinda
        # adim basi sadece 4 sozluk okumasi (bkz. baselines/risk_oracle.py).
        self._cost = [None] * n_envs
        self.obs = self.reset_all()

    def reset_all(self, n_radar: int | None = None) -> dict[int, np.ndarray]:
        nr = self.n_radar if n_radar is None else n_radar
        obs1 = [e.reset(n_radar=nr) for e in self.envs]
        for i, e in enumerate(self.envs):
            self._cost[i] = direction_costs(e.zone, RISK_W, e.hazard_mode)
        self.obs = {AGENT_1: np.stack([o[AGENT_1] for o in obs1]),
                   AGENT_2: np.stack([o[AGENT_2] for o in obs1])}
        return self.obs

    def oracle_actions(self) -> tuple[np.ndarray, np.ndarray]:
        """Her alt-ortamin SU ANKI (step'ten ONCE) pozisyonu icin Bellman-
        optimal uzman aksiyonu — BC-capali RL kaybi icin (agents/vdn.py).
        act_batch() ile kullanilan obs'la AYNI ana denk gelmeli, yani step()
        cagrilmadan HEMEN once okunmali (tipki action_masks() gibi).

        BUG (bulundu ve duzeltildi, 2026-08-26, dis inceleme): terminal (olu
        VEYA hedefe varmis) bir ucak icin GERCEK politika maskesi NOOP'a
        kilitliyor, ama oracle_action() pozisyondan bagimsiz HER ZAMAN bir
        YON dondurur. Etiketsiz birakilmazsa BC kaybi ag'i "terminal
        durumda yon sec" diye egitir — gercek davranisla VE TD hedefiyle
        celisir. -1 (etiketsiz) donduruluyor, learn() bunu zaten BC
        kaybindan maskeliyor (bkz. agents/vdn.py)."""
        oa1 = np.empty(self.n, dtype=np.int64)
        oa2 = np.empty(self.n, dtype=np.int64)
        for i, e in enumerate(self.envs):
            oa1[i] = (-1 if e.terminal(AGENT_1)
                      else oracle_action(e.pos[AGENT_1], e.dist, self._cost[i], e.n))
            oa2[i] = (-1 if e.terminal(AGENT_2)
                      else oracle_action(e.pos[AGENT_2], e.dist, self._cost[i], e.n))
        return oa1, oa2

    def states(self) -> np.ndarray:
        """QMIX mixer'i icin GLOBAL STATE, tum alt-ortamlar icin yigilmis
        (n, STATE_DIM). step()'ten ONCE (state) ve SONRA (next_state)
        cagrilir — tipki obs1/obs2 gibi, terminal satirlarda ICERIGI
        onemsizdir (done carpani sifirlar, bkz. modul docstring'i)."""
        return np.stack([e.state() for e in self.envs])

    def action_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """Politika secimi icin — terminal ucakta SADECE NOOP acik (bkz.
        StrikeMissionEnv.action_mask). _act_all'in tek-env davranisiyla
        AYNI: terminal ajanlar ozel-durum GEREKTIRMEZ, mask zaten NOOP'a
        kilitler."""
        m1 = np.stack([e.action_mask(AGENT_1) for e in self.envs])
        m2 = np.stack([e.action_mask(AGENT_2) for e in self.envs])
        return m1, m2

    def step(self, acts1: np.ndarray, acts2: np.ndarray, n_radar: int | None = None,
             need_state: bool = False):
        """Her alt-ortami BIR adim ilerletir; bitenleri HEMEN resetler.

        Doner: TRUE next_obs (dict) — buffer'a PUSH edilecek, reset-ONCESI
        gercek gecis sonucu (bkz. modul docstring'i, timeout-bootstrap bug
        duzeltmesi). self.obs (bir sonraki turun aktif girdisi) ayrica
        guncellenir, reset-SONRASI olabilir. r_team (n,), done (n, bool),
        infos (list), next_phys_mask1/2 (n, N_ACTIONS) — bootstrap icin
        FIZIKSEL maske (play_episode_vdn'deki env.physical_mask ile AYNI rol).

        need_state=True (SADECE QMIX): true_state (n, STATE_DIM) de doner —
        reset-ONCESI GERCEK global state (ayni timeout-bootstrap duzeltmesi
        state icin de gecerli). VDN bunu KULLANMIYOR, varsayilan False ile
        gereksiz state() hesabindan kacinilir (hot path).
        """
        nr = self.n_radar if n_radar is None else n_radar
        true_obs1, true_obs2 = [], []      # GERCEK gecis sonucu -> buffer
        obs1, obs2 = [], []                # bir sonraki tur icin (reset SONRASI olabilir)
        nm1, nm2 = [], []
        true_state = [] if need_state else None
        r_team = np.empty(self.n, dtype=np.float32)
        done = np.empty(self.n, dtype=bool)
        infos = []
        for i, e in enumerate(self.envs):
            o, r, d, info = e.step({AGENT_1: int(acts1[i]), AGENT_2: int(acts2[i])})
            r_team[i] = r
            done[i] = d
            infos.append(info)
            true_obs1.append(o[AGENT_1]); true_obs2.append(o[AGENT_2])
            # BUG (bulundu ve duzeltildi, 2026-08-26, dis inceleme): olum/hedef
            # icin mask onemsiz (done carpani sifirlar) ama TIMEOUT icin
            # push_done=False, yani bu maske GERCEKTEN bootstrap'e giriyor.
            # reset() SONRASI degil, GERCEK son pozisyonun fiziksel maskesi
            # kullanilmali — yoksa Double-DQN sinir-disi bir "en iyi aksiyon"
            # secebilir. physical_mask() done/terminal durumunu zaten
            # dikkate almiyor, reset'ten ONCE cagirmak yeterli.
            nm1.append(e.physical_mask(AGENT_1))
            nm2.append(e.physical_mask(AGENT_2))
            if need_state:
                true_state.append(e.state())   # reset'ten ONCE — GERCEK son state
            if d:
                o = e.reset(n_radar=nr)    # SADECE bir sonraki tur icin — true_obs/nm/state ETKILENMEZ
                self._cost[i] = direction_costs(e.zone, RISK_W, e.hazard_mode)
            obs1.append(o[AGENT_1]); obs2.append(o[AGENT_2])
        self.obs = {AGENT_1: np.stack(obs1), AGENT_2: np.stack(obs2)}
        true_next_obs = {AGENT_1: np.stack(true_obs1), AGENT_2: np.stack(true_obs2)}
        if need_state:
            return (true_next_obs, r_team, done, infos,
                   np.stack(nm1, dtype=np.float32), np.stack(nm2, dtype=np.float32),
                   np.stack(true_state))
        return (true_next_obs, r_team, done, infos,
               np.stack(nm1, dtype=np.float32), np.stack(nm2, dtype=np.float32))

    def set_n_radar(self, n_radar: int):
        self.n_radar = n_radar
