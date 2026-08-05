"""Double DQN + aksiyon maskeleme — PLAN §Asama 3.

Bu ajan tek basina calisir (yasak bolge yok). Amaci MARL'a gecmeden once
DQN makinesinin dogrulugunu kanitlamak: 600 (start, goal) ciftinin HEPSINDE
optimality gap tam 0 olmali. Buradaki bir bug'i sonra VDN'in icinde aramak
gunler yer.
"""
import numpy as np
import torch
import torch.nn as nn

from agents.buffer import ReplayBuffer
from agents.networks import build_qnet, masked_q
from config import (DQN_BATCH, DQN_BUFFER, DQN_EPS_DECAY_STEPS, DQN_LR,
                    DQN_LEARN_START, DQN_TARGET_UPDATE, EPS_END, EPS_START,
                    GAMMA, GRAD_CLIP, LEARN_EVERY, N_ACTIONS, OBS_DIM)


class DQNAgent:
    """Asama 3'te (tek ajan) dogrulugu kanitlanan makine, degistirilmeden.

    Hiperparametreler constructor argumani: Asama 4'ten itibaren AYNI sinif
    IQL'in iki bagimsiz ajani icin de kullaniliyor, farkli buffer/lr/decay
    degerleriyle (IQL_* vs DQN_*) — varsayilanlar Asama 3'un DQN_* setini
    korur, mevcut davranis degismez.
    """

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS,
                 seed: int = 0, device: str = "cpu",
                 buffer_size: int = DQN_BUFFER, batch_size: int = DQN_BATCH,
                 lr: float = DQN_LR, eps_decay_steps: int = DQN_EPS_DECAY_STEPS,
                 learn_start: int = DQN_LEARN_START,
                 target_update: int = DQN_TARGET_UPDATE):
        self.device = torch.device(device)
        self.n_actions = n_actions
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.batch_size = batch_size
        self.eps_decay_steps = eps_decay_steps
        self.learn_start = learn_start
        self.target_update = target_update

        self.online = build_qnet(n_actions).to(self.device)
        self.target = build_qnet(n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, obs_dim, n_actions, self.rng)
        self.steps = 0
        self.eps_progress: float | None = None   # bkz. eps / set_eps_progress

    # ------------------------------------------------------------- politika

    @property
    def eps(self) -> float:
        # eps_progress: egitim donugusunun her episode'da set ettigi 0..1
        # ilerleme. ADIM tabanli decay (asagidaki fallback) egitim uzunlugunu
        # DEGISTIRDIGIMIZDE sessizce yanlis kaliyordu: eps_decay_steps sabit
        # oldugu icin, episode sayisini artirinca epsilon egitimin cok
        # basinda tabana iniyor, azaltinca hic inmiyordu -- ustelik adim/
        # episode orani ajanin ne kadar iyi navige ettigine gore de degisiyor
        # (tavuk-yumurta). Episode tabanli ilerleme bu bagi tamamen koparir.
        frac = (self.eps_progress if self.eps_progress is not None
                else min(1.0, self.steps / self.eps_decay_steps))
        return EPS_START + frac * (EPS_END - EPS_START)

    def set_eps_progress(self, frac: float | None):
        """Egitim donugusu her episode cagirir: frac = ep / (episodes*FLOOR_FRAC),
        1.0'da epsilon tabanda. None verilirse adim-tabanli eski davranisa doner."""
        self.eps_progress = None if frac is None else min(1.0, max(0.0, frac))

    def act(self, obs: np.ndarray, mask: np.ndarray, eps: float | None = None) -> int:
        """Maskeye uyan epsilon-greedy secim. eps=0 -> deterministik greedy."""
        eps = self.eps if eps is None else eps
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            raise RuntimeError("gecerli aksiyon yok — ortam maskesi hatali")
        if self.rng.random() < eps:
            return int(self.rng.choice(legal))
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            m = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = masked_q(self.online(o), m)
            return int(q.argmax(dim=1).item())

    # ------------------------------------------------------------- ogrenme

    def push(self, *transition):
        self.buffer.push(*transition)
        self.steps += 1

    def learn(self) -> float | None:
        if len(self.buffer) < self.learn_start or self.steps % LEARN_EVERY != 0:
            return None

        obs, action, reward, next_obs, done, next_mask = self.buffer.sample(self.batch_size)
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=self.device)
        obs, next_obs = t(obs), t(next_obs)
        action = t(action, torch.int64)
        reward, done, next_mask = t(reward), t(done), t(next_mask)

        q = self.online(obs).gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: aksiyonu ONLINE ag secer, degeri TARGET ag bicer.
            # Tek agla max almak Q'yu sistematik olarak sisiriyor; 5x5'te bile
            # optimality gap'i 0'a oturtmayi engelliyor.
            next_q_online = masked_q(self.online(next_obs), next_mask)
            best = next_q_online.argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs).gather(1, best).squeeze(1)
            target = reward + GAMMA * next_q * (1.0 - done)

        loss = nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), GRAD_CLIP)
        self.opt.step()

        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    # ------------------------------------------------------------- kayit

    def save(self, path: str):
        torch.save({"online": self.online.state_dict(), "steps": self.steps}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["online"])
        self.steps = ckpt.get("steps", 0)
