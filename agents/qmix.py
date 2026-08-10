"""QMIX — PLAN §Asama 7.

VDN'in Q_tot = Q_1 + Q_2 toplamsalligini, GLOBAL STATE'e kosullu monotonik
bir mixing agiyla degistirir:

    Q_tot = Mixer(Q_1(obs_1,a_1), Q_2(obs_2,a_2) | state)

Mixer bir hypernetwork: agirliklarini state'ten URETIR (sabit degil), ve
abs(W) ile NON-NEGATIF tutulur — bu, dQ_tot/dQ_i >= 0 garantisini verir
(IGM: bireysel greedy aksiyonlar = ortak greedy aksiyon). VDN'in ozel hali
(mixer = sabit toplama) QMIX'in bir alt kumesi.

Per-ajan Q aglari (agents/vdn.py'deki gibi) AYRI — agents/vdn.py'nin
dosya-stringinde belgelenen bulgu (paylasimli tek ag, iki niteliksel farkli
rolu tek agent_id bitiyle ayirt etmeye calisirken yikici parazit uretiyordu)
burada da gecerli, aynı riski almiyoruz.

§2.1 hipotezi: bu problemde A2'nin degeri A1'in secimine KOSULLU (A1 kilit-
lediyse A2 ne yaparsa yapsin sifir) — yani odul TOPLAMSAL degil. VDN'in
additivity varsayimi ihlal ediliyor, QMIX'in state-kosullu mixer'i bunu
temsil edebilmeli. Olcum: zor alt kumede QMIX, VDN'i geciyor mu?
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.networks import build_qnet, masked_q
from agents.nstep import SPEC_QMIX, NStepAccumulator
from config import (HUBER_BETA, N_STEP, AGENT_1, AGENT_2, CNN_CHANNELS, EPS_END, EPS_START,
                    GAMMA, GRAD_CLIP, LEARN_EVERY, N_ACTIONS, OBS_DIM,
                    PATCH_SIZE, QMIX_BATCH, QMIX_BUFFER, QMIX_EPS_DECAY_STEPS,
                    QMIX_LEARN_START, QMIX_LR, QMIX_MIXER_EMBED,
                    QMIX_TARGET_UPDATE, STATE_CHANNELS, STATE_DIM,
                    STATE_SCALARS)


class MixerReplayBuffer:
    """agents/vdn.py'deki JointReplayBuffer + GLOBAL STATE (t ve t+1) —
    mixer'in agirlik hypernetwork'u state'i girdi olarak alir."""

    def __init__(self, capacity: int, obs_dim: int, state_dim: int,
                 n_actions: int, rng: np.random.Generator | None = None):
        self.capacity = capacity
        self.rng = rng or np.random.default_rng()
        self.obs1 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.obs2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action1 = np.zeros(capacity, dtype=np.int64)
        self.action2 = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_obs1 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_mask1 = np.zeros((capacity, n_actions), dtype=np.float32)
        self.next_mask2 = np.zeros((capacity, n_actions), dtype=np.float32)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        # n-adim ufku (bkz. agents/nstep.py)
        self.gamma_n = np.zeros(capacity, dtype=np.float32)
        self._i = 0
        self._full = False

    def __len__(self) -> int:
        return self.capacity if self._full else self._i

    def push(self, obs1, a1, obs2, a2, reward, next_obs1, next_obs2, done,
             next_mask1, next_mask2, state, next_state, gamma_n: float = 1.0):
        i = self._i
        self.obs1[i], self.obs2[i] = obs1, obs2
        self.action1[i], self.action2[i] = a1, a2
        self.reward[i] = reward
        self.next_obs1[i], self.next_obs2[i] = next_obs1, next_obs2
        self.done[i] = float(done)
        self.gamma_n[i] = gamma_n
        self.next_mask1[i] = np.ones_like(next_mask1) if done else next_mask1
        self.next_mask2[i] = np.ones_like(next_mask2) if done else next_mask2
        self.state[i] = state
        self.next_state[i] = next_state
        self._i = (i + 1) % self.capacity
        if self._i == 0:
            self._full = True

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, len(self), size=batch_size)
        return (self.obs1[idx], self.action1[idx], self.obs2[idx], self.action2[idx],
                self.reward[idx], self.next_obs1[idx], self.next_obs2[idx],
                self.done[idx], self.next_mask1[idx], self.next_mask2[idx],
                self.state[idx], self.next_state[idx], self.gamma_n[idx])


class QMixer(nn.Module):
    """Hypernetwork tabanli monotonik mixer. 2 ajan icin sabit boyut.

    w1: state -> (2, embed) agirlik, abs() ile non-negatif
    b1: state -> (embed,)   bias, serbest isaret
    w2: state -> (embed, 1) agirlik, abs() ile non-negatif
    b2: state -> (1,)       bias, 2 katmanli kucuk agla (ekstra kapasite)

    BUYUK N NOTU: state_dim=100x100'de 40.002 — hyper_w1/b1/w2/b2'yi bu duz
    vektore DOGRUDAN baglarsan (nn.Linear(40002, ...)) sadece hypernetwork'te
    ~6M parametre olusur (agents/networks.py'nin CNNQNet'i icin ayni sorun
    cozulmustu). Burada da AYNI cozum: kucuk bir CNN + AdaptiveAvgPool ile
    state ONCE kompakt bir vektore (state_embed_dim) indirgenir, hyper_*
    katmanlari BU kompakt vektoru kullanir — parametre sayisi PATCH_SIZE'dan
    bagimsiz kalir.
    """

    def __init__(self, state_dim: int = STATE_DIM, n_agents: int = 2,
                 embed_dim: int = QMIX_MIXER_EMBED,
                 state_channels: int = STATE_CHANNELS, grid_n: int = PATCH_SIZE,
                 state_scalars: int = STATE_SCALARS,
                 conv_channels: tuple[int, ...] = (8, 16), pool_size: int = 4):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.state_channels = state_channels
        self.grid_n = grid_n
        self.spatial_size = state_channels * grid_n * grid_n

        layers = []
        in_c = state_channels
        for out_c in conv_channels:
            layers += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                      nn.ReLU()]
            in_c = out_c
        self.state_conv = nn.Sequential(*layers)
        self.state_pool = nn.AdaptiveAvgPool2d(pool_size)
        state_embed_dim = in_c * pool_size * pool_size + state_scalars

        self.hyper_w1 = nn.Linear(state_embed_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_embed_dim, embed_dim)
        self.hyper_w2 = nn.Linear(state_embed_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, 1))

    def _encode_state(self, state: torch.Tensor) -> torch.Tensor:
        spatial = state[:, :self.spatial_size].view(
            -1, self.state_channels, self.grid_n, self.grid_n)
        scalars = state[:, self.spatial_size:]
        h = self.state_conv(spatial)
        h = self.state_pool(h).flatten(1)
        return torch.cat([h, scalars], dim=1)

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """agent_qs: (batch, n_agents) — her ajanin SECILEN aksiyonun Q degeri.
        state: (batch, state_dim). Donen: (batch,) Q_tot."""
        bsz = agent_qs.size(0)
        s = self._encode_state(state)
        w1 = torch.abs(self.hyper_w1(s)).view(bsz, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(s).view(bsz, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs.view(bsz, 1, self.n_agents), w1) + b1)

        w2 = torch.abs(self.hyper_w2(s)).view(bsz, self.embed_dim, 1)
        b2 = self.hyper_b2(s).view(bsz, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(bsz)


class QMixAgent:
    """Per-ajan AYRI Q agi (VDN'deki gibi) + monotonik mixer. PLAN §Asama 7."""

    def __init__(self, obs_dim: int = OBS_DIM, state_dim: int = STATE_DIM,
                 n_actions: int = N_ACTIONS, seed: int = 0, device: str = "cpu",
                 buffer_size: int = QMIX_BUFFER, batch_size: int = QMIX_BATCH,
                 lr: float = QMIX_LR, eps_decay_steps: int = QMIX_EPS_DECAY_STEPS,
                 learn_start: int = QMIX_LEARN_START,
                 target_update: int = QMIX_TARGET_UPDATE,
                 mixer_embed: int = QMIX_MIXER_EMBED):
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.batch_size = batch_size
        self.eps_decay_steps = eps_decay_steps
        self.learn_start = learn_start
        self.target_update = target_update

        self.online = {AGENT_1: build_qnet(n_actions).to(self.device),
                       AGENT_2: build_qnet(n_actions).to(self.device)}
        self.target = {AGENT_1: build_qnet(n_actions).to(self.device),
                       AGENT_2: build_qnet(n_actions).to(self.device)}
        for a in (AGENT_1, AGENT_2):
            self.target[a].load_state_dict(self.online[a].state_dict())
            self.target[a].eval()

        self.mixer = QMixer(state_dim, 2, mixer_embed).to(self.device)
        self.mixer_target = QMixer(state_dim, 2, mixer_embed).to(self.device)
        self.mixer_target.load_state_dict(self.mixer.state_dict())
        self.mixer_target.eval()

        params = (list(self.online[AGENT_1].parameters())
                 + list(self.online[AGENT_2].parameters())
                 + list(self.mixer.parameters()))
        self.opt = torch.optim.Adam(params, lr=lr)
        self.buffer = MixerReplayBuffer(buffer_size, obs_dim, state_dim, n_actions, self.rng)
        # n-ADIM GETIRI (agents/nstep.py). N_STEP=1 -> eski davranis BIREBIR.
        self.nstep = NStepAccumulator(N_STEP, GAMMA, SPEC_QMIX)
        self.steps = 0
        self.eps_progress: float | None = None   # bkz. eps / set_eps_progress

    @property
    def eps(self) -> float:
        # bkz. agents/dqn.py'deki ayni metodun notu.
        frac = (self.eps_progress if self.eps_progress is not None
                else min(1.0, self.steps / self.eps_decay_steps))
        return EPS_START + frac * (EPS_END - EPS_START)

    def set_eps_progress(self, frac: float | None):
        """bkz. agents/dqn.py'deki ayni metod."""
        self.eps_progress = None if frac is None else min(1.0, max(0.0, frac))

    def act(self, agent_id: int, obs: np.ndarray, mask: np.ndarray,
           eps: float | None = None) -> int:
        eps = self.eps if eps is None else eps
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            raise RuntimeError("gecerli aksiyon yok — ortam maskesi hatali")
        if self.rng.random() < eps:
            return int(self.rng.choice(legal))
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            m = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = masked_q(self.online[agent_id](o), m)
            return int(q.argmax(dim=1).item())

    def q_value(self, agent_id: int, obs: np.ndarray, action: int) -> float:
        """PLAN §Asama 5 saglik kontrolu — QMIX'te de aynen gecerli."""
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return float(self.online[agent_id](o)[0, action].item())

    def push(self, *joint_transition):
        for tr, gamma_n in self.nstep.push(joint_transition):
            self.buffer.push(*tr, gamma_n=gamma_n)
        self.steps += 1

    def end_episode(self):
        """Episode sinirinda kalan kisa pencereleri bosalt (bkz. nstep.py)."""
        for tr, gamma_n in self.nstep.flush():
            self.buffer.push(*tr, gamma_n=gamma_n)

    def learn(self) -> float | None:
        if len(self.buffer) < self.learn_start or self.steps % LEARN_EVERY != 0:
            return None

        (obs1, a1, obs2, a2, r, next_obs1, next_obs2, done,
         nm1, nm2, state, next_state, gamma_n) = self.buffer.sample(self.batch_size)
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=self.device)
        obs1, obs2 = t(obs1), t(obs2)
        next_obs1, next_obs2 = t(next_obs1), t(next_obs2)
        gamma_n = t(gamma_n)
        a1, a2 = t(a1, torch.int64), t(a2, torch.int64)
        r, done = t(r), t(done)
        nm1, nm2 = t(nm1), t(nm2)
        state, next_state = t(state), t(next_state)

        q1 = self.online[AGENT_1](obs1).gather(1, a1.unsqueeze(1)).squeeze(1)
        q2 = self.online[AGENT_2](obs2).gather(1, a2.unsqueeze(1)).squeeze(1)
        q_tot = self.mixer(torch.stack([q1, q2], dim=1), state)

        with torch.no_grad():
            best1 = masked_q(self.online[AGENT_1](next_obs1), nm1).argmax(dim=1, keepdim=True)
            best2 = masked_q(self.online[AGENT_2](next_obs2), nm2).argmax(dim=1, keepdim=True)
            nq1 = self.target[AGENT_1](next_obs1).gather(1, best1).squeeze(1)
            nq2 = self.target[AGENT_2](next_obs2).gather(1, best2).squeeze(1)
            nq_tot = self.mixer_target(torch.stack([nq1, nq2], dim=1), next_state)
            # gamma_n = gamma^k (k = gecisin gercek ufku, bkz. nstep.py)
            target_val = r + gamma_n * nq_tot * (1.0 - done)

        loss = nn.functional.smooth_l1_loss(q_tot, target_val, beta=HUBER_BETA)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.online[AGENT_1].parameters())
            + list(self.online[AGENT_2].parameters())
            + list(self.mixer.parameters()), GRAD_CLIP)
        self.opt.step()

        if self.steps % self.target_update == 0:
            self.target[AGENT_1].load_state_dict(self.online[AGENT_1].state_dict())
            self.target[AGENT_2].load_state_dict(self.online[AGENT_2].state_dict())
            self.mixer_target.load_state_dict(self.mixer.state_dict())
        return float(loss.item())

    def save(self, path: str):
        torch.save({"online1": self.online[AGENT_1].state_dict(),
                   "online2": self.online[AGENT_2].state_dict(),
                   "mixer": self.mixer.state_dict(),
                   "steps": self.steps}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online[AGENT_1].load_state_dict(ckpt["online1"])
        self.online[AGENT_2].load_state_dict(ckpt["online2"])
        self.target[AGENT_1].load_state_dict(ckpt["online1"])
        self.target[AGENT_2].load_state_dict(ckpt["online2"])
        self.mixer.load_state_dict(ckpt["mixer"])
        self.mixer_target.load_state_dict(ckpt["mixer"])
        self.steps = ckpt.get("steps", 0)
