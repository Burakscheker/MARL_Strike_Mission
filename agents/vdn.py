"""VDN (Value Decomposition Networks) — PLAN §Asama 5.

Q_tot = Q_1(obs_1, a_1) + Q_2(obs_2, a_2)
loss  = (r_team + gamma * Q_tot_target(t+1) - Q_tot(t))^2

IQL'den (agents/dqn.py + iki bagimsiz ajan) TEK mimari fark: iki ajanin da
AYNI timestep'teki (obs, aksiyon) cifti TEK joint transition'da tutuluyor,
boylece TEK TD hatasi ikisine BIRDEN geri yayilir. Golge NOOP (PLAN §2.2)
burada ilk kez gercek anlam kazaniyor: pasif ajanin Q(obs, NOOP)'u da bu
toplama girer ve gradyan alir — IQL'de o ajanin push()'u bile edilmiyordu.

AGIRLIK PAYLASIMI YOK — bilerek: Q_1 ve Q_2 AYRI iki MLP. PLAN ilk yazildiginda
"2 ajan tek ag, daha stabil" varsayilmisti; deneyle TERSI cikti. Tek paylasilan
agla (agent_id tek skaler ayirici) A1'in SALT navigasyon performansi (A2'siz,
Asama 3'un 600 ciftlik testi) ep~1750'de 237/600'e kadar ogrenip sonra
COKUYORDU — hem varsayilan LR/target_update'te hem 3x dusuk LR + 2x yavas
target update'te, 3 ayri tam-olcekli (40k episode) kosuda tekrarlandi. Ayni
testi AYRI iki agla (geri kalani ayni: joint TD hedefi, golge NOOP, ortak
buffer) kosunca ep5000'den itibaren 595-597/600'de ISTIKRARLI kaldi, hic
cokme gorulmedi. Kok neden: TEK ag, iki NITELIKSEL FARKLI rolu (A1 serbest
gezinme / A2 yasak-bolge-kisitli) TEK bir agent_id skaleriyle ayirt etmeye
calisirken yikici parazit (destructive interference) uretiyor. VDN'in ozu
(TEK TD hedefi + toplamsal ayristirma) agirlik paylasimi GEREKTIRMEZ; bunlar
birbirinden bagimsiz iki tasarim karari. Detay: PLAN.md §Asama 5.
"""
import numpy as np
import torch
import torch.nn as nn

from agents.networks import build_qnet, masked_q
from config import (HUBER_BETA, AGENT_1, AGENT_2, EPS_END, EPS_START, GAMMA, GRAD_CLIP,
                    LEARN_EVERY, N_ACTIONS, OBS_DIM, PER_ALPHA, PER_BETA_START,
                    PER_EPS, VDN_BATCH, VDN_BUFFER, VDN_EPS_DECAY_STEPS,
                    VDN_LEARN_START, VDN_LR, VDN_TARGET_UPDATE)


class SumTree:
    """Oncelikli deneyim tekrari (PER, Schaul ve ark. 2016) icin duz-dizi
    toplam agaci: O(log N) orneklem + guncelleme. capacity 2'nin kuvveti
    OLMAK ZORUNDA DEGIL — dugum i'nin cocuklari 2i/2i+1 kurali, "tam" bir
    agac SEKLINE degil SADECE update()'in her dugumu cocuklarinin toplami
    olarak dogru tutmasina dayanir (klasik segment-agaci esdegeri)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.max_priority = 1.0

    def update(self, idx: int, priority: float):
        self.max_priority = max(self.max_priority, priority)
        i = idx + self.capacity
        delta = priority - self.tree[i]
        self.tree[i] = priority
        i //= 2
        while i >= 1:
            self.tree[i] += delta
            i //= 2

    def total(self) -> float:
        return float(self.tree[1])

    def get(self, s: float) -> int:
        """s icin (0 <= s < total()) denk gelen yaprak INDEKSINI (0..capacity-1) dondurur."""
        i = 1
        while i < self.capacity:
            left = 2 * i
            if s <= self.tree[left]:
                i = left
            else:
                s -= self.tree[left]
                i = left + 1
        return i - self.capacity


class JointReplayBuffer:
    """Her satir BIR global timestep: HER IKI ajanin (obs, aksiyon, next_obs,
    next_mask) bilgisi + TEK r_team + TEK done bayragi.

    agents/buffer.py'deki ReplayBuffer'in ikiye katlanmis hali — VDN'in Q_tot
    toplaminin HER iki tarafini da AYNI ornekten (ayni t) beslemesi gerekiyor.
    """

    def __init__(self, capacity: int, obs_dim: int, n_actions: int,
                 rng: np.random.Generator | None = None,
                 prioritized: bool = False, alpha: float = PER_ALPHA):
        self.capacity = capacity
        self.rng = rng or np.random.default_rng()
        self.prioritized = prioritized
        self.alpha = alpha
        self.tree = SumTree(capacity) if prioritized else None
        self.obs1 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.obs2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action1 = np.zeros(capacity, dtype=np.int64)
        self.action2 = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_obs1 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_mask1 = np.zeros((capacity, n_actions), dtype=np.float32)
        self.next_mask2 = np.zeros((capacity, n_actions), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        # OGRETMEN-CAPASI (teacher-anchored VDN, bkz. VDNAgent.learn): o anki
        # pozisyon icin Bellman-optimal uzman aksiyonu. -1 = ETIKET YOK (serial
        # play_episode_vdn hic gondermez) — learn() bu satirlari BC kaybindan
        # maskeler, TD kaybini etkilemez.
        self.oracle_a1 = np.full(capacity, -1, dtype=np.int64)
        self.oracle_a2 = np.full(capacity, -1, dtype=np.int64)
        self._i = 0
        self._full = False

    def __len__(self) -> int:
        return self.capacity if self._full else self._i

    def push(self, obs1, a1, obs2, a2, reward, next_obs1, next_obs2, done,
             next_mask1, next_mask2, oracle_a1: int = -1, oracle_a2: int = -1):
        i = self._i
        self.obs1[i], self.obs2[i] = obs1, obs2
        self.action1[i], self.action2[i] = a1, a2
        self.reward[i] = reward
        self.next_obs1[i], self.next_obs2[i] = next_obs1, next_obs2
        self.done[i] = float(done)
        # Terminal'de next_mask kullanilmaz ama tamamen sifir maske masked_q'da
        # NEG_INF uretir; guvenlik icin 1'lerle doldur (agents/buffer.py ile ayni onlem).
        self.next_mask1[i] = np.ones_like(next_mask1) if done else next_mask1
        self.next_mask2[i] = np.ones_like(next_mask2) if done else next_mask2
        self.oracle_a1[i], self.oracle_a2[i] = oracle_a1, oracle_a2
        if self.prioritized:
            # YENI gecis MAKSIMUM oncelikle girer — henuz TD-hatasi
            # olculmedigi icin "once dene, sonra oncelik duzelt" (Schaul ve
            # ark. standart pratigi), yoksa hic ornekelenmeden buffer'da
            # sessizce eskiyebilir.
            self.tree.update(i, self.tree.max_priority ** self.alpha)
        self._i = (i + 1) % self.capacity
        if self._i == 0:
            self._full = True

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, len(self), size=batch_size)
        return (self.obs1[idx], self.action1[idx], self.obs2[idx], self.action2[idx],
                self.reward[idx], self.next_obs1[idx], self.next_obs2[idx],
                self.done[idx], self.next_mask1[idx], self.next_mask2[idx],
                self.oracle_a1[idx], self.oracle_a2[idx])

    def sample_prioritized(self, batch_size: int, beta: float):
        """Oncelik^alpha ile ORANTILI ornekleme (SumTree) + onyargi-duzeltme
        agirliklari (IS weights, beta ile). Toplam kutle esit `batch_size`
        dilime bolunup her dilimden BIR ornek cekilir (stratified) — saf
        rastgele orneklemeden daha DUSUK varyansli, Schaul ve ark. standart
        pratigi."""
        n = len(self)
        total = self.tree.total()
        segment = total / batch_size
        idx = np.empty(batch_size, dtype=np.int64)
        for k in range(batch_size):
            s = self.rng.uniform(segment * k, segment * (k + 1))
            idx[k] = self.tree.get(min(s, total - 1e-6))
        priorities = self.tree.tree[idx + self.capacity]
        probs = priorities / total
        weights = (n * probs) ** (-beta)
        weights /= weights.max()
        return (self.obs1[idx], self.action1[idx], self.obs2[idx], self.action2[idx],
                self.reward[idx], self.next_obs1[idx], self.next_obs2[idx],
                self.done[idx], self.next_mask1[idx], self.next_mask2[idx],
                self.oracle_a1[idx], self.oracle_a2[idx],
                idx, weights.astype(np.float32))

    def update_priorities(self, idx: np.ndarray, td_errors: np.ndarray):
        """learn() SONRASI cagrilir: yeni |TD-hatasi| -> oncelik^alpha.
        PER_EPS: TD-hatasi TAM SIFIR olsa bile oncelik hicbir zaman 0 olmasin
        (aksi halde o gecis BIR DAHA hic ornekelenmez)."""
        p = (np.abs(td_errors) + PER_EPS) ** self.alpha
        for i, pi in zip(idx, p):
            self.tree.update(int(i), float(pi))


class VDNAgent:
    """Q_1 ve Q_2 icin AYRI iki MLP (agirlik paylasimi YOK — bkz. modul dosya
    stringi). act()/q_value() bu yuzden HANGI ajan oldugunu (agent_id) acikca
    ister; paylasilan tek agda oldugu gibi gozlemden cikarilmiyor.
    """

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS,
                 seed: int = 0, device: str = "cpu",
                 buffer_size: int = VDN_BUFFER, batch_size: int = VDN_BATCH,
                 lr: float = VDN_LR, eps_decay_steps: int = VDN_EPS_DECAY_STEPS,
                 learn_start: int = VDN_LEARN_START,
                 target_update: int = VDN_TARGET_UPDATE,
                 eps_end: float = EPS_END, dueling: bool = False,
                 prioritized: bool = False, al_alpha: float = 0.0):
        self.device = torch.device(device)
        self.n_actions = n_actions
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.batch_size = batch_size
        self.eps_decay_steps = eps_decay_steps
        self.learn_start = learn_start
        self.target_update = target_update
        self.eps_end = eps_end
        # OGRETMEN-CAPASI agirligi (teacher-anchored VDN) — train.py episode
        # basina ayarlar (set_bc_lambda), varsayilan 0.0 = kapali/eski davranis.
        self.bc_lambda = 0.0
        # PER importance-sampling agirligi — train.py episode basina anneal
        # eder (bkz. set_per_beta). prioritized=False'ta hic kullanilmaz.
        self.per_beta = PER_BETA_START
        # ADVANTAGE LEARNING carpani (bkz. config.py AL_ALPHA_DEFAULT + learn()).
        # 0.0 = kapali, learn() eskisiyle BIREBIR AYNI.
        self.al_alpha = float(al_alpha)

        self.online = {AGENT_1: build_qnet(n_actions, dueling=dueling).to(self.device),
                       AGENT_2: build_qnet(n_actions, dueling=dueling).to(self.device)}
        self.target = {AGENT_1: build_qnet(n_actions, dueling=dueling).to(self.device),
                       AGENT_2: build_qnet(n_actions, dueling=dueling).to(self.device)}
        for a in (AGENT_1, AGENT_2):
            self.target[a].load_state_dict(self.online[a].state_dict())
            self.target[a].eval()

        params = (list(self.online[AGENT_1].parameters())
                 + list(self.online[AGENT_2].parameters()))
        self.opt = torch.optim.Adam(params, lr=lr)
        self.buffer = JointReplayBuffer(buffer_size, obs_dim, n_actions, self.rng,
                                        prioritized=prioritized)
        self.steps = 0
        self.eps_progress: float | None = None   # bkz. eps / set_eps_progress

    # ------------------------------------------------------------- politika

    @property
    def eps(self) -> float:
        # bkz. agents/dqn.py'deki ayni metodun notu (episode-tabanli ilerleme,
        # adim-tabanli decay'in egitim uzunluguna gore sessizce bozulmasi).
        frac = (self.eps_progress if self.eps_progress is not None
                else min(1.0, self.steps / self.eps_decay_steps))
        return EPS_START + frac * (self.eps_end - EPS_START)

    def set_eps_progress(self, frac: float | None):
        """bkz. agents/dqn.py'deki ayni metod."""
        self.eps_progress = None if frac is None else min(1.0, max(0.0, frac))

    def set_bc_lambda(self, lam: float):
        """OGRETMEN-CAPASI agirligini ayarla (bkz. learn()). train.py episode
        basina bir decay egrisiyle cagirir — dis inceleme onerisi: baslangicta
        yuksek, egitim ilerledikce azalir ama SIFIRA HEMEN INMEZ (BC'nin
        ogrettigi guvenli rotayi TD guncellemelerinin silmesini onlemek)."""
        self.bc_lambda = max(0.0, lam)

    def set_per_beta(self, beta: float):
        """PER importance-sampling agirliginin ussu — egitim ilerledikce
        PER_BETA_START'tan 1.0'a (tam onyargi duzeltmesi) anneal edilir
        (Schaul ve ark. standart pratigi: erken egitimde biraz onyargi
        kabul edilir, cunku oncelikler henuz guvenilir degildir)."""
        self.per_beta = min(1.0, max(0.0, beta))

    def act(self, agent_id: int, obs: np.ndarray, mask: np.ndarray,
           eps: float | None = None) -> int:
        eps = self.eps if eps is None else eps
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            raise RuntimeError("gecerli aksiyon yok — ortam maskesi hatali")
        if eps > 0.0 and self.rng.random() < eps:
            return int(self.rng.choice(legal))
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            m = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = masked_q(self.online[agent_id](o), m)
            return int(q.argmax(dim=1).item())

    def act_batch(self, agent_id: int, obs: np.ndarray, mask: np.ndarray,
                 eps: float | None = None) -> np.ndarray:
        """act()'in VEKTORIZE hali — N paralel ortam icin TEK forward pass
        (bkz. env/vec_env.py). Epsilon-greedy HER SATIR icin BAGIMSIZ
        uygulanir, ayni RNG akisindan (self.rng) — semantik act() ile AYNI,
        sadece N ornegi tek cagrida isler."""
        eps = self.eps if eps is None else eps
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            m = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
            q = masked_q(self.online[agent_id](o), m)
            greedy = q.argmax(dim=1).cpu().numpy()
        if eps <= 0.0:
            return greedy
        explore = self.rng.random(obs.shape[0]) < eps
        if explore.any():
            acts = greedy.copy()
            for i in np.flatnonzero(explore):
                legal = np.flatnonzero(mask[i])
                acts[i] = self.rng.choice(legal)
            return acts
        return greedy

    def q_value(self, agent_id: int, obs: np.ndarray, action: int) -> float:
        """Tek (obs, aksiyon) icin online Q degeri — PLAN §Asama 5 saglik kontrolu:
        Q(obs_2, NOOP) FAZ A boyunca SABIT KALMAMALI (golge NOOP'un kaniti)."""
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return float(self.online[agent_id](o)[0, action].item())

    # ------------------------------------------------------------- ogrenme

    def push(self, *joint_transition):
        self.buffer.push(*joint_transition)
        self.steps += 1

    def learn(self) -> float | None:
        if len(self.buffer) < self.learn_start or self.steps % LEARN_EVERY != 0:
            return None

        per = self.buffer.prioritized
        if per:
            (obs1, a1, obs2, a2, r, next_obs1, next_obs2, done,
             nm1, nm2, oa1, oa2, per_idx, is_w) = self.buffer.sample_prioritized(
                self.batch_size, self.per_beta)
        else:
            (obs1, a1, obs2, a2, r, next_obs1, next_obs2, done,
             nm1, nm2, oa1, oa2) = self.buffer.sample(self.batch_size)
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=self.device)
        obs1, obs2 = t(obs1), t(obs2)
        next_obs1, next_obs2 = t(next_obs1), t(next_obs2)
        a1, a2 = t(a1, torch.int64), t(a2, torch.int64)
        r, done = t(r), t(done)
        nm1, nm2 = t(nm1), t(nm2)

        q1_all = self.online[AGENT_1](obs1)
        q2_all = self.online[AGENT_2](obs2)
        q1 = q1_all.gather(1, a1.unsqueeze(1)).squeeze(1)
        q2 = q2_all.gather(1, a2.unsqueeze(1)).squeeze(1)
        q_tot = q1 + q2                              # VDN: toplamsal ayristirma

        with torch.no_grad():
            # Double DQN, HER ajan icin KENDI online/target agiyla
            best1 = masked_q(self.online[AGENT_1](next_obs1), nm1).argmax(dim=1, keepdim=True)
            best2 = masked_q(self.online[AGENT_2](next_obs2), nm2).argmax(dim=1, keepdim=True)
            nq1 = self.target[AGENT_1](next_obs1).gather(1, best1).squeeze(1)
            nq2 = self.target[AGENT_2](next_obs2).gather(1, best2).squeeze(1)
            target_val = r + GAMMA * (nq1 + nq2) * (1.0 - done)

            # ADVANTAGE LEARNING (Bellemare ve ark. 2016, bkz. config.py
            # AL_ALPHA_DEFAULT): ALINAN aksiyonun greedy'den geriligi kadar
            # hedefi DUSER — greedy'de fark 0, non-greedy'de action-gap'i
            # ~1/(1-alpha) katina cikarir. V(s)/Q(s,a) TARGET agdan (bootstrap
            # ile tutarli), CARI gozlemde. Depoda cari maske YOK: NOOP her
            # zaman legal, 4 yon sadece grid kenarinda illegal (nadir) — ham
            # max orada V'yi hafif SISIRIR, yani AL duzeltmesi kenarda biraz
            # fazla agresif; kabul edilir bir yanlilik.
            if self.al_alpha > 0.0:
                tq1c = self.target[AGENT_1](obs1)
                tq2c = self.target[AGENT_2](obs2)
                gap1 = tq1c.max(dim=1).values - tq1c.gather(1, a1.unsqueeze(1)).squeeze(1)
                gap2 = tq2c.max(dim=1).values - tq2c.gather(1, a2.unsqueeze(1)).squeeze(1)
                target_val = target_val - self.al_alpha * (gap1 + gap2)

        if per:
            # ONCELIKLI DENEYIM TEKRARI (PER, 2026-08-26): elementwise kayip,
            # IS agirligiyla carpilip ORTALANIR (Schaul ve ark. — sik
            # ornekelenen gecisler gradyanda ONYARGI yaratmasin diye
            # kucultulur). Oncelikler backward'dan SONRA, GERCEK |TD-hatasi|
            # ile guncellenir (bkz. asagida).
            td_elem = nn.functional.smooth_l1_loss(q_tot, target_val, beta=HUBER_BETA,
                                                    reduction="none")
            is_w_t = t(is_w)
            loss_td = (td_elem * is_w_t).mean()
        else:
            loss_td = nn.functional.smooth_l1_loss(q_tot, target_val, beta=HUBER_BETA)

        loss = loss_td
        # OGRETMEN-CAPASI (teacher-anchored VDN, dis inceleme onerisi,
        # 2026-08-26): vanilla BC->RL fine-tune COKTU (ep25'te mission_prob
        # 0.0001) cunku BC cross-entropy'yle egitildigi icin cikis olcegi
        # gercek Q-degerleriyle (ort. 17-37) uyumsuz — normal TD guncellemesi
        # bu olcegi ilk birkac yuz adimda agresifce yeniden kalibre ederken
        # BC'nin ogrendigi ince aksiyon siralamasini siliyordu. Cozum: TD
        # kaybina SUREKLI oracle-capraz-entropi eklemek, boylece ag hem
        # odulden ogrenir hem de her adimda uzman aksiyonundan cok uzaklasamaz.
        # bc_lambda=0.0 (varsayilan) -> eski davranisla BIREBIR AYNI.
        if self.bc_lambda > 0.0:
            oa1_t, oa2_t = t(oa1, torch.int64), t(oa2, torch.int64)
            v1, v2 = oa1_t >= 0, oa2_t >= 0
            loss_bc = obs1.new_zeros(())
            if v1.any():
                loss_bc = loss_bc + nn.functional.cross_entropy(q1_all[v1], oa1_t[v1])
            if v2.any():
                loss_bc = loss_bc + nn.functional.cross_entropy(q2_all[v2], oa2_t[v2])
            loss = loss_td + self.bc_lambda * loss_bc

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.online[AGENT_1].parameters()) + list(self.online[AGENT_2].parameters()),
            GRAD_CLIP)
        self.opt.step()

        if per:
            with torch.no_grad():
                new_td = (q_tot - target_val).abs().cpu().numpy()
            self.buffer.update_priorities(per_idx, new_td)

        if self.steps % self.target_update == 0:
            self.target[AGENT_1].load_state_dict(self.online[AGENT_1].state_dict())
            self.target[AGENT_2].load_state_dict(self.online[AGENT_2].state_dict())
        # loglanan kayip TD-kismi: BC-capali/capasiz kosular karsilastirilabilir kalsin.
        return float(loss_td.item())

    # ------------------------------------------------------------- kayit

    def save(self, path: str):
        torch.save({"online1": self.online[AGENT_1].state_dict(),
                   "online2": self.online[AGENT_2].state_dict(),
                   "steps": self.steps}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online[AGENT_1].load_state_dict(ckpt["online1"])
        self.online[AGENT_2].load_state_dict(ckpt["online2"])
        self.target[AGENT_1].load_state_dict(ckpt["online1"])
        self.target[AGENT_2].load_state_dict(ckpt["online2"])
        self.steps = ckpt.get("steps", 0)
