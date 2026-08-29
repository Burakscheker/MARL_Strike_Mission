"""MAPPO / HAPPO — euzxx/MARL-pathtfinding (mappo_happo dali, agents/ppo.py)
projesinden PORTLANDI (2026-08-28), bizim CNN govdesine/env'imize uyarlandi.

VDN/QMIX'ten (off-policy, TD-hedefli) TEMEL fark: bunlar on-policy PPO —
replay buffer YOK, PPO_ROLLOUT_EPISODES kadar TAM episode toplanir, GAE
(Generalized Advantage Estimation) hesaplanir, PPO_EPOCHS kez minibatch SGD
yapilir, batch ATILIR (bir sonraki rollout GUNCEL politikayla toplanir).

Mimari (CTDE — centralized training, decentralized execution):
  - Actor  : build_qnet() ile AYNI CNN govdesi (agents/networks.py) — cikisi
             artik Q-degeri DEGIL, N_ACTIONS uzerinde politika LOGIT'i.
             Her ajanin KENDI actor'u var (agirlik paylasimi yok — VDN'deki
             ayni gerekce: iki ajanin rolu niteliksel farkli).
  - Critic : TEK merkezi kritik, env.state()'i (QMIX'in mixer'inin kullandigi
             AYNI global gozlem) girdi alir. Egitimde merkezi bilgi kullanip
             calistirmada (act()) sadece yerel gozlemle karar vermek CTDE'nin
             tanimi.

MAPPO/HAPPO farki SADECE actor guncelleme sirasinda (PPOTrainer.update()):
  - MAPPO: iki aktor BAGIMSIZ guncellenir (ayni paylasilan advantage'la).
  - HAPPO: aktorler RASTGELE sirayla guncellenir; ilk guncellenen aktorun
    YENI/ESKI politika ORANI (importance-sampling carpani) SIRADAKI aktorun
    advantage'ina carpilir — Kuba ve ark. 2021 "Trust Region Policy
    Optimisation in Multi-Agent RL" makalesindeki SIRALI monotonik-iyilesme
    garantisi. Bu carpan olmadan HAPPO, MAPPO'dan farksiz olurdu.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from agents.networks import CNNQNet, build_qnet
from config import (AGENT_1, AGENT_2, CNN_CHANNELS, CNN_POOL_SIZE, GAE_LAMBDA,
                    GAMMA, N_ACTIONS, PATCH_SIZE, PPO_ACTOR_LR, PPO_CLIP_COEF,
                    PPO_CRITIC_LR, PPO_EPOCHS, PPO_ENTROPY_START, PPO_ENTROPY_END,
                    PPO_MAX_GRAD_NORM, PPO_MINIBATCH_SIZE, PPO_ROLLOUT_EPISODES,
                    PPO_VALUE_COEF, STATE_CHANNELS, STATE_SCALARS)

_BUF_FIELDS = ("obs", "states", "masks", "actions", "old_logp", "values",
              "advantages", "returns", "alive")


@dataclass
class RolloutBatch:
    """Bir PPO guncellemesi icin toplanmis PPO_ROLLOUT_EPISODES episode'un
    duz (flatten) transition listesi. Her satir bir GLOBAL timestep, ikinci
    boyut (2,) iki ajan icindir (obs/masks/actions/old_logp/alive)."""
    obs: torch.Tensor          # (T, 2, OBS_DIM)
    states: torch.Tensor       # (T, STATE_DIM)
    masks: torch.Tensor        # (T, 2, N_ACTIONS)
    actions: torch.Tensor      # (T, 2)
    old_logp: torch.Tensor     # (T, 2)
    values: torch.Tensor       # (T,) — kritigin O ANKI tahmini (GAE icin)
    advantages: torch.Tensor   # (T,)
    returns: torch.Tensor      # (T,)
    alive: torch.Tensor        # (T, 2) — bkz. _update_actor (olu ajan gradyan almaz)

    def to(self, device):
        return type(self)(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Gecersiz aksiyonlari -inf'e iterek maskeli kategorik dagilim kurar."""
    mask = mask.to(dtype=torch.bool, device=logits.device)
    return Categorical(logits=logits.masked_fill(~mask, -torch.inf))


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, terminated: torch.Tensor,
                bootstrap_value: torch.Tensor, gamma: float, gae_lambda: float):
    """GAE(lambda) — Schulman ve ark. 2016. Tek bir episode'un (T,) dizisi
    icin GERIYE DOGRU hesaplanir. terminated[t]=True ise o adimdan sonra
    bootstrap YOK (gercek terminal); False ise (timeout/kesme) bootstrap_value
    kullanilir — VDN/QMIX'teki AYNI zaman-siniri bootstrap ilkesi (Pardo ve
    ark.), burada tek fark deger fonksiyonunun PPO kritigi olmasi."""
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros((), dtype=rewards.dtype)
    next_value = bootstrap_value.to(dtype=rewards.dtype)
    for t in reversed(range(len(rewards))):
        nonterminal = (~terminated[t]).to(rewards.dtype)
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        advantages[t] = next_advantage
        next_value = values[t]
    return advantages, advantages + values


def clipped_policy_loss(new_logp, old_logp, advantages, clip_coef, factor=None):
    """PPO-clip amac fonksiyonu. factor: HAPPO'nun sirali importance-sampling
    carpani (bkz. modul dosya stringi) — MAPPO'da None (etkisiz)."""
    ratio = (new_logp - old_logp).exp()
    weighted = advantages if factor is None else advantages * factor
    unclipped = ratio * weighted
    clipped = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * weighted
    return -torch.minimum(unclipped, clipped).mean()


class PPOTrainer:
    """MAPPO/HAPPO ortak govdesi. algorithm/update() alt siniflarda tanimli."""
    algorithm = "ppo"

    def __init__(self, seed: int = 0, device: str = "cpu"):
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        # ENTROPI CURRICULUM: baslangicta PPO_ENTROPY_START (yuksek, kesif),
        # train.py set_entropy_progress ile PPO_ENTROPY_END'e (dusuk, kararli)
        # anneal edilir. train.py disi cagrilarda (test/tek-kullanim) sabit
        # START'ta kalir. bkz. config.py PPO_ENTROPY_START notu.
        self.entropy_coef = float(PPO_ENTROPY_START)
        # Aktorler: build_qnet() ile AYNI CNN govdesi — VDN'deki gibi
        # AGIRLIK PAYLASIMI YOK (bkz. agents/vdn.py modul dosya stringi,
        # ayni gerekce burada da gecerli).
        self.actors = nn.ModuleList((build_qnet(N_ACTIONS), build_qnet(N_ACTIONS))).to(self.device)
        # Kritik: AYNI CNNQNet sinifi, farkli girdi/cikti boyutuyla (global
        # state -> tek skaler deger) — mimari kod TEKRARI yok.
        self.critic = CNNQNet(STATE_CHANNELS, PATCH_SIZE, STATE_SCALARS, 1,
                              conv_channels=CNN_CHANNELS, pool_size=CNN_POOL_SIZE).to(self.device)
        self.actor_optimizers = [torch.optim.Adam(a.parameters(), lr=PPO_ACTOR_LR)
                                 for a in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=PPO_CRITIC_LR)
        # ROLLOUT BIRIKTIRICI (bkz. add_episode) — VDN'in replay buffer'inin
        # on-policy karsiligi: TAM episode'lar birikir, rollout_episodes'e
        # ulasinca TEK bir RolloutBatch'e donusturulup ATILIR.
        self.rollout_episodes = PPO_ROLLOUT_EPISODES
        self._buf = {name: [] for name in _BUF_FIELDS}
        self._buf_episodes = 0

    def set_entropy_progress(self, frac: float) -> None:
        """frac 0..1: entropi katsayisini START'tan END'e dogrusal anneal et.
        train.py ep basina cagirir (VDN'in set_eps_progress'iyle AYNI ritim).
        Amac: erken egitimde stokastik politika (rollout cesitli), ilerledikce
        kararli (bkz. config.py PPO_ENTROPY_START notu — euzxx tasarimi)."""
        frac = min(1.0, max(0.0, frac))
        self.entropy_coef = PPO_ENTROPY_START + frac * (PPO_ENTROPY_END - PPO_ENTROPY_START)

    # ------------------------------------------------------------- politika

    @torch.no_grad()
    def act(self, obs: np.ndarray, masks: np.ndarray, state: np.ndarray,
           deterministic: bool = False):
        """obs (2,OBS_DIM), masks (2,N_ACTIONS), state (STATE_DIM,) ->
        (actions (2,) int, log_probs (2,) float, value float)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        masks_t = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions, logps = [], []
        for a, actor in enumerate(self.actors):
            dist = masked_categorical(actor(obs_t[a:a + 1]), masks_t[a:a + 1])
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            actions.append(int(action.item()))
            logps.append(float(dist.log_prob(action).item()))
        value = float(self.critic(state_t).item())
        return np.array(actions, dtype=np.int64), np.array(logps, dtype=np.float32), value

    @torch.no_grad()
    def act_batch(self, obs: np.ndarray, masks: np.ndarray, states: np.ndarray,
                  deterministic: bool = False):
        """act()'in VEKTORIZE hali — N paralel ortam icin TEK forward pass
        (bkz. env/vec_env.py'deki VDN act_batch, AYNI gerekce: batch=1
        dispatch GPU'da CPU'dan yavas). obs (N,2,OBS_DIM), masks
        (N,2,N_ACTIONS), states (N,STATE_DIM) -> actions (N,2) int,
        log_probs (N,2) float, values (N,) float."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        masks_t = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        n = obs.shape[0]
        actions = np.empty((n, 2), dtype=np.int64)
        logps = np.empty((n, 2), dtype=np.float32)
        for a, actor in enumerate(self.actors):
            dist = masked_categorical(actor(obs_t[:, a]), masks_t[:, a])
            act = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            actions[:, a] = act.cpu().numpy()
            logps[:, a] = dist.log_prob(act).cpu().numpy()
        values = self.critic(states_t).squeeze(-1).cpu().numpy()
        return actions, logps, values

    # ------------------------------------------------------------- rollout

    def add_episode(self, ep: dict, terminated_true: bool, bootstrap_value: float,
                    gamma: float = GAMMA, gae_lambda: float = GAE_LAMBDA) -> "RolloutBatch | None":
        """Bir TAMAMLANMIS episode'un ham (obs/masks/actions/old_logp/values/
        alive) listelerini alir, GAE'sini hesaplar, biriktiriciye ekler.
        rollout_episodes'e ulasinca DOLU bir RolloutBatch dondurur ve
        biriktiriciyi SIFIRLAR; aksi halde None (henuz guncelleme yok — VDN'in
        learn()'unun learn_start'a kadar None donmesiyle AYNI cagri sozlesmesi).

        terminated_true: episode GERCEK terminal mi bitti (olum/hedef) yoksa
        timeout/kesme mi (bootstrap_value o zaman gecerli) — VDN/QMIX'teki
        AYNI zaman-siniri bootstrap ayrimi (push_done vs timeout)."""
        rewards = torch.as_tensor(ep["rewards"], dtype=torch.float32)
        values = torch.as_tensor(ep["values"], dtype=torch.float32)
        n = len(rewards)
        terminated = torch.zeros(n, dtype=torch.bool)
        if terminated_true:
            terminated[-1] = True
        bootstrap = torch.tensor(0.0 if terminated_true else bootstrap_value)
        advantages, returns = compute_gae(rewards, values, terminated, bootstrap,
                                          gamma, gae_lambda)
        self._buf["obs"].extend(ep["obs"])
        self._buf["states"].extend(ep["states"])
        self._buf["masks"].extend(ep["masks"])
        self._buf["actions"].extend(ep["actions"])
        self._buf["old_logp"].extend(ep["old_logp"])
        self._buf["values"].extend(ep["values"])
        self._buf["alive"].extend(ep["alive"])
        self._buf["advantages"].extend(advantages.tolist())
        self._buf["returns"].extend(returns.tolist())
        self._buf_episodes += 1
        if self._buf_episodes < self.rollout_episodes:
            return None
        batch = RolloutBatch(
            obs=torch.as_tensor(np.asarray(self._buf["obs"]), dtype=torch.float32),
            states=torch.as_tensor(np.asarray(self._buf["states"]), dtype=torch.float32),
            masks=torch.as_tensor(np.asarray(self._buf["masks"]), dtype=torch.float32),
            actions=torch.as_tensor(np.asarray(self._buf["actions"]), dtype=torch.long),
            old_logp=torch.as_tensor(np.asarray(self._buf["old_logp"]), dtype=torch.float32),
            values=torch.as_tensor(self._buf["values"], dtype=torch.float32),
            advantages=torch.as_tensor(self._buf["advantages"], dtype=torch.float32),
            returns=torch.as_tensor(self._buf["returns"], dtype=torch.float32),
            alive=torch.as_tensor(np.asarray(self._buf["alive"]), dtype=torch.bool),
        )
        self._buf = {name: [] for name in _BUF_FIELDS}
        self._buf_episodes = 0
        return batch

    def _minibatches(self, indices: torch.Tensor):
        shuffled = self.rng.permutation(indices.detach().cpu().numpy())
        for start in range(0, len(shuffled), PPO_MINIBATCH_SIZE):
            yield torch.as_tensor(shuffled[start:start + PPO_MINIBATCH_SIZE], device=self.device)

    def _update_critic(self, batch: RolloutBatch) -> float:
        indices = torch.arange(len(batch.returns), device=self.device)
        losses = []
        for _ in range(PPO_EPOCHS):
            for sample in self._minibatches(indices):
                new_values = self.critic(batch.states[sample]).squeeze(-1)
                old_values = batch.values[sample]
                # DEGER-CLIP (PPO2 standart pratigi): kritigin TEK bir kotu
                # minibatch'te asiri uzaklasmasini onler, VALUE_COEF ile
                # olceklenir.
                clipped_values = old_values + (new_values - old_values).clamp(
                    -PPO_CLIP_COEF, PPO_CLIP_COEF)
                plain_err = (new_values - batch.returns[sample]).square()
                clipped_err = (clipped_values - batch.returns[sample]).square()
                loss = 0.5 * PPO_VALUE_COEF * torch.maximum(plain_err, clipped_err).mean()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), PPO_MAX_GRAD_NORM)
                self.critic_optimizer.step()
                losses.append(float(loss.detach().item()))
        return float(np.mean(losses))

    def _update_actor(self, agent: int, batch: RolloutBatch, factor=None) -> float:
        # OLU ajan gradyan ALMAZ (bkz. VecStrikeEnv.oracle_actions'daki AYNI
        # ilke: terminal duruma yon/aksiyon ogretmenin anlami yok). REACHED
        # (hedefe varmis ama HALA hayatta) ajan DAHIL edilir — tek NOOP'a
        # kilitli forced-aksiyon zararsiz, dislamak fazladan karmasiklik.
        indices = torch.nonzero(batch.alive[:, agent], as_tuple=False).squeeze(-1)
        if len(indices) == 0:
            return 0.0
        advantages = batch.advantages[indices]
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        normalized = torch.zeros_like(batch.advantages)
        normalized[indices] = advantages

        losses = []
        optimizer = self.actor_optimizers[agent]
        actor = self.actors[agent]
        for _ in range(PPO_EPOCHS):
            for sample in self._minibatches(indices):
                dist = masked_categorical(actor(batch.obs[sample, agent]), batch.masks[sample, agent])
                new_logp = dist.log_prob(batch.actions[sample, agent])
                sample_factor = None if factor is None else factor[sample]
                policy_loss = clipped_policy_loss(
                    new_logp, batch.old_logp[sample, agent], normalized[sample],
                    PPO_CLIP_COEF, sample_factor)
                loss = policy_loss - self.entropy_coef * dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), PPO_MAX_GRAD_NORM)
                optimizer.step()
                losses.append(float(loss.detach().item()))
        return float(np.mean(losses))

    # ------------------------------------------------------------- kayit

    def save(self, path: str):
        torch.save({"algorithm": self.algorithm, "seed": self.seed,
                   "entropy_coef": self.entropy_coef,
                   "actors": [a.state_dict() for a in self.actors],
                   "critic": self.critic.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        for actor, sd in zip(self.actors, ckpt["actors"]):
            actor.load_state_dict(sd)
        self.critic.load_state_dict(ckpt["critic"])
        self.entropy_coef = float(ckpt.get("entropy_coef", self.entropy_coef))


class MAPPOTrainer(PPOTrainer):
    algorithm = "mappo"

    def update(self, batch: RolloutBatch) -> dict:
        batch = batch.to(self.device)
        critic_loss = self._update_critic(batch)
        actor_losses = [self._update_actor(a, batch) for a in (AGENT_1, AGENT_2)]
        return {"actor_loss": float(np.mean(actor_losses)), "critic_loss": critic_loss}


class HAPPOTrainer(PPOTrainer):
    algorithm = "happo"

    def __init__(self, seed: int = 0, device: str = "cpu"):
        super().__init__(seed=seed, device=device)
        self.last_update_order: tuple = ()

    def update(self, batch: RolloutBatch) -> dict:
        batch = batch.to(self.device)
        critic_loss = self._update_critic(batch)
        # RASTGELE sira — hangi ajanin "ONCE" guncellenip digerinin
        # advantage'ina carpan uygulayacagini sabitlememek icin (Kuba ve
        # ark.'in onerdigi sekilde).
        self.last_update_order = tuple(int(v) for v in self.rng.permutation(2))
        factor = torch.ones_like(batch.advantages)
        actor_losses = []
        for agent in self.last_update_order:
            actor_losses.append(self._update_actor(agent, batch, factor=factor.detach()))
            with torch.no_grad():
                dist = masked_categorical(self.actors[agent](batch.obs[:, agent]), batch.masks[:, agent])
                new_logp = dist.log_prob(batch.actions[:, agent])
                ratio = (new_logp - batch.old_logp[:, agent]).exp()
                # SIRALI carpan: SADECE bu ajanin HAYATTA oldugu adimlarda
                # guncellenir (aksi halde olu/forced-NOOP adimlarda anlamsiz
                # bir oran carpimina girer).
                factor = factor * torch.where(batch.alive[:, agent], ratio, torch.ones_like(ratio))
        return {"actor_loss": float(np.mean(actor_losses)), "critic_loss": critic_loss}
