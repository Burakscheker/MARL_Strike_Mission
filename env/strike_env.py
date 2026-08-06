"""Es zamanli iki ucakli radar-kacinma ortami — Strike_Mission.md §1.

MARL-Pathfinding'in MARLGridEnv'inden UC yapisal fark:

 1. ES ZAMANLI akis (orada sirali/turn-based). Iki ucak her timestep'te
    BIRLIKTE hamle eder. Golge NOOP HILESINE GEREK YOK — VDN/QMIX dogal
    haliyle baglaniyor (orada faz mantigi gerekiyordu).
 2. Yasak bolge / carpisma YOK. Ayni hucrede durabilirler, ayni yoldan
    gidebilirler. Tehlike STOKASTIK: radar halkasinda her adimda olum zari.
 3. Terminal ASIMETRIK: bir ucak olur/varirken digeri devam eder. Terminal
    olan ucak haritada kalir, NOOP basar, gozlemi GUNCELLENMEYE DEVAM EDER
    (VDN'in kredi kanali — MARL-Pathfinding'de bunun bozulmasi "en sinsi
    bug"di, tests/test_env.py bunu ayrica dogruluyor).

Odul TEK TAKIM skaleri (VDN/QMIX boyle ister). IQL icin ajan basina ayrisim
info["r_ind"]'de.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from baselines.risk_oracle import risk_distance_map, zone_map
from config import (AGENT_1, AGENT_2, ALERT_DECAY, ALERT_ENABLED, ALERT_MULT,
                    DIRS, GAMMA, GOAL, GRID_N, HAZARD_MODE, INNER_HALF,
                    MAX_STEPS, NOOP, N_ACTIONS, OUTER_HALF, P_DEATH,
                    PATCH_RADIUS, PATCH_SIZE, PATCH_STRIDE, P_INNER_TOTAL,
                    P_OUTER_TOTAL, RADARS, R_ALL_DEAD, R_DEATH, R_FIRST_GOAL,
                    R_RISK_COEF, R_SECOND_GOAL, R_STEP, R_TIMEOUT,
                    SHAPING_COEF, START, STATE_DIM)

Cell = tuple[int, int]

# kanal 0'in hucre degeri: guvenli / dis / ic (bkz. config.py OBS_CHANNELS notu)
DANGER_VALUE = np.array([0.0, 0.5, 1.0], dtype=np.float32)

# gozlem skalar 12-15 (risk-mesafe komsu farki) icin kirpma siniri.
# GEREKCE: guvenli duz bir adimda fark tam +-1.0 — MARL-Pathfinding'in BFS hop
# farkiyla AYNI olcek, transfer edilen agirliklar bu araligi "taniyor". Ic
# halkada ise ham fark ~25'e cikiyor; kirpilmazsa ±1'lik girdilere gore
# egitilmis ilk Linear katmani doygunlasip cop uretir. Kirpma YONU (isareti)
# korur — asil transfer edilen bilgi o.
NEIGHBOR_CLIP = 3.0


class StrikeMissionEnv:
    def __init__(self, n: int = GRID_N, max_steps: int = MAX_STEPS,
                 seed: Optional[int] = None,
                 alert_enabled: bool = ALERT_ENABLED,
                 risk_shaping: bool = True,
                 hazard_mode: str = HAZARD_MODE):
        self.n = n
        self.max_steps = max_steps
        self.alert_enabled = alert_enabled
        self.risk_shaping = risk_shaping
        self.hazard_mode = hazard_mode
        self.rng = np.random.default_rng(seed)

        # Radarlar SABIT -> bu iki harita bir kez hesaplanip onbelleklenir.
        self.zone = zone_map()                       # (n,n) uint8
        self.danger = DANGER_VALUE[self.zone]        # (n,n) float32
        self.dist = risk_distance_map()              # (n,n) float32
        self.p_death = np.asarray(P_DEATH, dtype=np.float64)

        self.goal = GOAL
        self.start = START
        self.max_man = 2 * (n - 1)

        # yerel pencerenin ornekleme ofsetleri (stride'li — bkz. config.py)
        self._off = PATCH_STRIDE * np.arange(-PATCH_RADIUS, PATCH_RADIUS + 1)
        self._coarse_n = (n + PATCH_STRIDE - 1) // PATCH_STRIDE

        self.reset()

    # ------------------------------------------------------------- reset

    def reset(self, seed: Optional[int] = None, config=None) -> dict[int, np.ndarray]:
        """config: (start1, start2, goal) — verilmezse config.py'nin sabitleri.

        Radarlar ve B/H su an SABIT (Burak: "sonradan random yapicaz").
        Imza simdiden config aliyor ki Asama 10'da sampler baglanirken
        egitim dongusunun degismesi gerekmesin.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        s1, s2, g = config if config is not None else (self.start, self.start, self.goal)
        self.goal = g

        # SHAPING OLCEGI — harita basina, max_man DEGIL.
        # BUG (bulundu ve duzeltildi): Phi ve gozlem skalari #11 eskiden
        # `min(1, dist/max_man)` ile normalize ediliyordu, max_man = 2(n-1) =
        # 1998. Ama self.dist RISK-mesafesi: bir ic halkaya girmek
        # RISK_W * 0.9 = 675 adim-esdegeri ekliyor. Radar yogunlastikca
        # dist(B) max_man'i ASIYOR (40 radarli haritada ~3600 olculdu) ve
        # min(1, ...) kirpmasi devreye girip Phi'yi 0'a KILITLIYOR — ajan
        # haritanin buyuk kisminda nereye giderse gitsin ayni potansiyeli
        # goruyor, yani shaping gradyani tam ihtiyac duyulan yerde OLU.
        # (Ayni kirpma skalar #11'i de sabit 1.0'a sabitliyordu: "hedefe ne
        # kadar kaldi" bilgisi gozlemden tamamen siliniyordu.)
        # Cozum: olcegi haritanin KENDI baslangic maliyetinden al. Boylece
        # Phi(B) ~ 0, Phi(H) = 1 HER haritada — radar yogunlugundan bagimsiz.
        # Potential-based shaping HERHANGI bir Phi icin politika-degismezdir
        # (Ng, Harada, Russell 1999), yani bu degisiklik teorik garantiyi
        # bozmaz; sadece sinyali geri getirir.
        self.dist_scale = max(float(self.dist[s1]), float(self.dist[s2]), 1.0)

        self.pos = {AGENT_1: s1, AGENT_2: s2}
        self.path = {AGENT_1: [s1], AGENT_2: [s2]}
        self.alive = {AGENT_1: True, AGENT_2: True}
        self.reached = {AGENT_1: False, AGENT_2: False}
        # Ziyaret izi KABA cozunurlukte tutuluyor: gozlem penceresi stride'li
        # ornekledigi icin 1 hucre genisligindeki bir iz orneklenen noktalara
        # neredeyse hic denk gelmez ve kanal 1 sabit 0 kalirdi (bilgisiz).
        # Kaba izgara = stride x stride blogunda "ugradi mi" (max-pool).
        self._vis = {a: np.zeros((self._coarse_n, self._coarse_n), dtype=np.float32)
                     for a in (AGENT_1, AGENT_2)}
        for a, p in self.pos.items():
            self._vis[a][p[0] // PATCH_STRIDE, p[1] // PATCH_STRIDE] = 1.0

        # per_entry modu icin: ucagin O AN icinde bulundugu radar/bolge
        self._prev_zone = {AGENT_1: 0, AGENT_2: 0}

        self.alert = np.zeros(len(RADARS), dtype=np.int32)   # kalan alarm suresi
        self.t = 0
        self.done = False
        self._timeout = False
        self._first_goal_taken = False
        self.death_step = {AGENT_1: None, AGENT_2: None}
        return self.observations()

    # -------------------------------------------------------- yardimcilar

    def terminal(self, agent: int) -> bool:
        return (not self.alive[agent]) or self.reached[agent]

    def _in_bounds(self, c: Cell) -> bool:
        return 0 <= c[0] < self.n and 0 <= c[1] < self.n

    def _radar_at(self, pos: Cell) -> tuple[int, int]:
        """(radar_index, zone) — hicbirinde degilse (-1, 0). Ic halka DIS'i ezer."""
        r, c = pos
        best = (-1, 0)
        for i, (rr, cc) in enumerate(RADARS):
            dr, dc = abs(r - rr), abs(c - cc)
            if dr <= INNER_HALF and dc <= INNER_HALF:
                return i, 2
            if dr <= OUTER_HALF and dc <= OUTER_HALF:
                best = (i, 1)
        return best

    def physical_mask(self, agent: int) -> np.ndarray:
        """SADECE fiziksel gecerlilik (grid siniri). done/terminal durumunu
        DIKKATE ALMAZ — bootstrap'te next_state'in GERCEK maskesi lazim
        (MARL-Pathfinding'de ayni tuzak, ayni cozum)."""
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        cur = self.pos[agent]
        for a, (dr, dc) in enumerate(DIRS):
            if self._in_bounds((cur[0] + dr, cur[1] + dc)):
                mask[a] = 1.0
        if mask.sum() == 0:
            mask[NOOP] = 1.0
        return mask

    def action_mask(self, agent: int) -> np.ndarray:
        """Politika secimi icin: terminal ucakta SADECE NOOP, digerinde NOOP kapali
        (beklemek asla optimal degil — adim maliyeti var, harita sabit)."""
        if self.done or self.terminal(agent):
            mask = np.zeros(N_ACTIONS, dtype=np.float32)
            mask[NOOP] = 1.0
            return mask
        return self.physical_mask(agent)

    # ------------------------------------------------------------ gozlem

    def _patch(self, grid: np.ndarray, center: Cell, oob: float) -> np.ndarray:
        """center etrafinda PATCH_SIZE x PATCH_SIZE, PATCH_STRIDE araliklarla
        SEYREK orneklenmis pencere. Sinir disi noktalar oob degerini alir."""
        rows = center[0] + self._off
        cols = center[1] + self._off
        rv = (rows >= 0) & (rows < grid.shape[0])
        cv = (cols >= 0) & (cols < grid.shape[1])
        patch = np.full((PATCH_SIZE, PATCH_SIZE), oob, dtype=np.float32)
        if rv.any() and cv.any():
            sub = grid[np.ix_(rows[rv], cols[cv])]
            patch[np.ix_(rv, cv)] = sub
        return patch

    def _coarse_patch(self, grid: np.ndarray, center: Cell, oob: float) -> np.ndarray:
        """Kaba izgaradan (stride cozunurlugunde) pencere — ziyaret izi icin."""
        r0, c0 = center[0] // PATCH_STRIDE, center[1] // PATCH_STRIDE
        idx = np.arange(-PATCH_RADIUS, PATCH_RADIUS + 1)
        rows, cols = r0 + idx, c0 + idx
        rv = (rows >= 0) & (rows < grid.shape[0])
        cv = (cols >= 0) & (cols < grid.shape[1])
        patch = np.full((PATCH_SIZE, PATCH_SIZE), oob, dtype=np.float32)
        if rv.any() and cv.any():
            patch[np.ix_(rv, cv)] = grid[np.ix_(rows[rv], cols[cv])]
        return patch

    def observe(self, agent: int) -> np.ndarray:
        n = self.n
        own, other = self.pos[agent], self.pos[1 - agent]

        # kanal 0: tehlike (sinir disi = 0.0 -> grid disi "guvenli ama gidilemez";
        #          maske zaten oraya gitmeyi engelliyor, sahte tehlike uretme)
        # kanal 1: kendi izi (kaba izgara)
        ch = np.stack([self._patch(self.danger, own, 0.0),
                       self._coarse_patch(self._vis[agent], own, 0.0)])

        d_own = float(self.dist[own])
        nb = []
        for dr, dc in DIRS:
            rr, cc = own[0] + dr, own[1] + dc
            if self._in_bounds((rr, cc)):
                nb.append(float(np.clip(d_own - self.dist[rr, cc],
                                        -NEIGHBOR_CLIP, NEIGHBOR_CLIP)))
            else:
                nb.append(-NEIGHBOR_CLIP)      # grid disi: mumkun en kotu

        man_goal = abs(own[0] - self.goal[0]) + abs(own[1] - self.goal[1])
        man_other = abs(own[0] - other[0]) + abs(own[1] - other[1])

        # SLOT HIZALAMASI: bu 16 skalarin sirasi MARL-Pathfinding'in
        # observe()'undaki sirayla BIREBIR ayni (bkz. config.py N_SCALARS).
        # Checkpoint transferinin anlamli olmasi buna bagli.
        scalars = np.array([
            float(agent),
            float(self.terminal(1 - agent)),          # orada: faz biti
            self.t / self.max_steps,
            own[0] / n, own[1] / n,
            (self.goal[0] - own[0]) / n, (self.goal[1] - own[1]) / n,
            man_goal / self.max_man,
            (other[0] - own[0]) / n, (other[1] - own[1]) / n,
            man_other / self.max_man,
            # "hedefe ne kadar kaldi" — RISK mesafesi, o yuzden olcek de risk
            # tabanli (dist_scale). max_man ile bolununce yogun haritada sabit
            # 1.0'a satüre oluyordu, yani bilgisiz bir sabit girdiye donuyordu.
            min(1.0, d_own / self.dist_scale),
            *nb,
        ], dtype=np.float32)
        return np.concatenate([ch.ravel(), scalars])

    def observations(self) -> dict[int, np.ndarray]:
        return {AGENT_1: self.observe(AGENT_1), AGENT_2: self.observe(AGENT_2)}

    def state(self) -> np.ndarray:
        """QMIX mixer icin global state."""
        n = self.n
        ch = np.stack([self._patch(self.danger, self.pos[AGENT_1], 0.0),
                       self._patch(self.danger, self.pos[AGENT_2], 0.0)])
        n_term = int(self.terminal(AGENT_1)) + int(self.terminal(AGENT_2))
        scalars = np.array([
            self.pos[AGENT_1][0] / n, self.pos[AGENT_1][1] / n,
            self.pos[AGENT_2][0] / n, self.pos[AGENT_2][1] / n,
            self.goal[0] / n, self.goal[1] / n,
            n_term / 2.0,
            self.t / self.max_steps,
        ], dtype=np.float32)
        out = np.concatenate([ch.ravel(), scalars])
        assert out.size == STATE_DIM, (out.size, STATE_DIM)
        return out

    # -------------------------------------------------------------- step

    def _phi(self, agent: int) -> float:
        """Potential-based shaping potansiyeli: 1 = hedefte, 0 = en uzak.
        RISK-FARKINDA mesafeden (Manhattan degil) — yani "hedefe yaklasmak"
        radardan gecerek yaklasmayi ODULLENDIRMEZ.

        Olcek self.dist_scale (harita basina, bkz. reset()); max_man ile
        normalize etmek yogun haritalarda Phi'yi 0'a kilitliyordu."""
        return 1.0 - min(1.0, float(self.dist[self.pos[agent]]) / self.dist_scale)

    def step(self, actions) -> tuple[dict[int, np.ndarray], float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode bitti — reset() cagir.")

        phi_before = {a: self._phi(a) for a in (AGENT_1, AGENT_2)}
        r_team = R_STEP
        r_ind = {AGENT_1: 0.0, AGENT_2: 0.0}
        for a in (AGENT_1, AGENT_2):
            if not self.terminal(a):
                r_ind[a] += R_STEP

        # --- 1) es zamanli hareket
        for agent in (AGENT_1, AGENT_2):
            if self.terminal(agent):
                continue
            act = int(actions[agent])
            if act == NOOP:
                continue                       # maskeli; gelirse yerinde kal
            dr, dc = DIRS[act]
            nxt = (self.pos[agent][0] + dr, self.pos[agent][1] + dc)
            if self._in_bounds(nxt):
                self.pos[agent] = nxt
                self.path[agent].append(nxt)
                self._vis[agent][nxt[0] // PATCH_STRIDE, nxt[1] // PATCH_STRIDE] = 1.0

        self.t += 1
        info: dict = {}

        # --- 2) olum zari (alarm carpani BU adimdan ONCEKI durumu kullanir:
        #        radarin "uyanmasi" bir tik surer — ayni adimda giren ucagin
        #        kendini cezalandirmasini onler, IKINCI ucaga tam isabet eder)
        newly_dead = []
        for agent in (AGENT_1, AGENT_2):
            if self.terminal(agent):
                continue
            ridx, z = self._radar_at(self.pos[agent])
            p = self._hazard(agent, ridx, z)
            if p > 0.0:
                if self.risk_shaping:
                    cost = R_RISK_COEF * p       # beklenen olum maliyetini pesin ode
                    r_team -= cost
                    r_ind[agent] -= cost
                if self.rng.random() < p:
                    self.alive[agent] = False
                    self.death_step[agent] = self.t
                    newly_dead.append(agent)
            self._prev_zone[agent] = z

        for agent in newly_dead:
            r_team += R_DEATH
            r_ind[agent] += R_DEATH

        # --- 3) alarm durumunu GUNCELLE (bir sonraki adim icin)
        if self.alert_enabled:
            np.maximum(self.alert - 1, 0, out=self.alert)
            for agent in (AGENT_1, AGENT_2):
                if self.terminal(agent):
                    continue
                ridx, z = self._radar_at(self.pos[agent])
                if ridx >= 0:
                    self.alert[ridx] = ALERT_DECAY

        # --- 4) hedefe varis
        for agent in (AGENT_1, AGENT_2):
            if self.terminal(agent) or self.pos[agent] != self.goal:
                continue
            self.reached[agent] = True
            if not self._first_goal_taken:
                self._first_goal_taken = True
                r_team += R_FIRST_GOAL          # TAKIM ODULU FULL
                r_ind[agent] += R_FIRST_GOAL
            else:
                r_team += R_SECOND_GOAL
                r_ind[agent] += R_SECOND_GOAL

        # --- 5) terminal kontrolleri
        if all(self.terminal(a) for a in (AGENT_1, AGENT_2)):
            self.done = True
            if not any(self.reached.values()):
                r_team += R_ALL_DEAD            # ikisi de dusuruldu
        elif self.t >= self.max_steps:
            self.done = True
            self._timeout = True
            if not any(self.reached.values()):
                r_team += R_TIMEOUT
                for a in (AGENT_1, AGENT_2):
                    if not self.terminal(a):
                        r_ind[a] += R_TIMEOUT

        # --- 6) potential-based shaping (Ng ve ark. 1999)
        # Phi(terminal) = 0 kosulu: episode'u BITIREN adimda uygulanmaz —
        # policy-invariance garantisi bunu gerektiriyor (MARL-Pathfinding'de
        # ayni gerekce ayrintili belgelendi).
        if not self.done:
            for agent in (AGENT_1, AGENT_2):
                if self.terminal(agent):
                    continue
                shaping = SHAPING_COEF * (GAMMA * self._phi(agent) - phi_before[agent])
                r_team += shaping
                r_ind[agent] += shaping

        if self.done:
            info.update(self._terminal_info())
        info["r_ind"] = r_ind
        info["t"] = self.t
        return self.observations(), float(r_team), self.done, info

    def _hazard(self, agent: int, ridx: int, z: int) -> float:
        """Bu adimda bu ucagin olum olasiligi."""
        if z == 0:
            return 0.0
        if self.hazard_mode == "per_entry":
            # SADECE bolgeye GIRIS adiminda zar (ablation modu)
            if z <= self._prev_zone[agent]:
                return 0.0
            p = P_INNER_TOTAL if z == 2 else P_OUTER_TOTAL
        else:
            p = float(self.p_death[z])
        if self.alert_enabled and ridx >= 0 and self.alert[ridx] > 0:
            p = min(1.0, p * ALERT_MULT)
        return p

    # ----------------------------------------------------------- terminal

    def _terminal_info(self) -> dict:
        from baselines.risk_oracle import exposure, survival_prob
        out = {}
        for a, tag in ((AGENT_1, "1"), (AGENT_2, "2")):
            o, i = exposure(self.path[a], self.zone)
            out[f"len{tag}"] = len(self.path[a]) - 1
            out[f"reached{tag}"] = self.reached[a]
            out[f"alive{tag}"] = self.alive[a]
            out[f"outer{tag}"], out[f"inner{tag}"] = o, i
            # ANALITIK hayatta kalma: gurultusuz "bu yol ne kadar guvenliydi"
            # olcutu. Basari oranini ASLA tek basina raporlama (altin kural).
            out[f"surv{tag}"] = survival_prob(self.path[a], self.zone)
        team = self.reached[AGENT_1] or self.reached[AGENT_2]
        p1, p2 = set(self.path[AGENT_1]), set(self.path[AGENT_2])
        out.update({
            "team_success": team,
            "both_reached": self.reached[AGENT_1] and self.reached[AGENT_2],
            "n_dead": int(not self.alive[AGENT_1]) + int(not self.alive[AGENT_2]),
            "timeout": self._timeout,
            "steps": self.t,
            # iki ucagin yollari ne kadar ortusuyor — koordinasyon gostergesi
            # (alarm kuplaji acilinca DUSMESI beklenir, §Asama 6)
            "route_overlap": len(p1 & p2) / max(1, min(len(p1), len(p2))),
            "outer_total": out["outer1"] + out["outer2"],
            "inner_total": out["inner1"] + out["inner2"],
            "analytic_surv_team": 1.0 - (1.0 - out["surv1"]) * (1.0 - out["surv2"]),
            "path1": tuple(self.path[AGENT_1]),
            "path2": tuple(self.path[AGENT_2]),
        })
        return out

    # ------------------------------------------------------------ render

    def render(self, every: int = 40) -> str:
        """ASCII — 1000x1000 ekrana sigmaz, `every` hucrede bir ornekler."""
        rows = []
        for r in range(0, self.n, every):
            line = []
            for c in range(0, self.n, every):
                cell = (r, c)
                near = [a for a in (AGENT_1, AGENT_2)
                        if abs(self.pos[a][0] - r) < every // 2
                        and abs(self.pos[a][1] - c) < every // 2]
                if len(near) == 2:
                    line.append("*")
                elif near:
                    a = near[0]
                    line.append("+" if not self.alive[a] else str(a + 1))
                elif abs(self.goal[0] - r) < every // 2 and abs(self.goal[1] - c) < every // 2:
                    line.append("H")
                elif self.zone[r, c] == 2:
                    line.append("x")
                elif self.zone[r, c] == 1:
                    line.append("o")
                else:
                    line.append(".")
            rows.append(" ".join(line))
        head = (f"t={self.t}/{self.max_steps}  "
                f"A1={self.pos[AGENT_1]} {'OLDU' if not self.alive[AGENT_1] else ('VARDI' if self.reached[AGENT_1] else 'ucuyor')}  "
                f"A2={self.pos[AGENT_2]} {'OLDU' if not self.alive[AGENT_2] else ('VARDI' if self.reached[AGENT_2] else 'ucuyor')}  "
                f"alarm={self.alert.tolist()}")
        return head + "\n" + "\n".join(rows)
