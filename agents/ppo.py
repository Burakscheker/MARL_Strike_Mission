from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from config import (
    ACTOR_HIDDEN,
    ACTOR_LR,
    CLIP_COEF,
    CRITIC_HIDDEN,
    CRITIC_LR,
    ENTROPY_COEF,
    MAX_GRAD_NORM,
    MINIBATCH_SIZE,
    N_ACTIONS,
    OBS_DIM,
    PPO_EPOCHS,
    STATE_DIM,
    VALUE_COEF,
)


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    states: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    old_logp: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    terminated: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    alive: torch.Tensor

    def tensors(self):
        return tuple(getattr(self, field.name) for field in fields(self))

    def to(self, device):
        return type(self)(
            **{field.name: getattr(self, field.name).to(device) for field in fields(self)}
        )


def _orthogonal_init(module, gain):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


class Actor(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=ACTOR_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        for layer in self.net[:-1]:
            _orthogonal_init(layer, gain=2**0.5)
        _orthogonal_init(self.net[-1], gain=0.01)

    def forward(self, observation):
        return self.net(observation)


class CentralCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, hidden=CRITIC_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for layer in self.net[:-1]:
            _orthogonal_init(layer, gain=2**0.5)
        _orthogonal_init(self.net[-1], gain=1.0)

    def forward(self, state):
        return self.net(state).squeeze(-1)


def masked_categorical(logits, mask):
    mask = mask.to(dtype=torch.bool, device=logits.device)
    if (~mask).all(dim=-1).any():
        raise ValueError("action mask has no legal action")
    return Categorical(logits=logits.masked_fill(~mask, -torch.inf))


def compute_gae(
    rewards,
    values,
    terminated,
    bootstrap_value,
    gamma,
    gae_lambda,
):
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    next_value = bootstrap_value.to(dtype=rewards.dtype, device=rewards.device)
    for index in reversed(range(len(rewards))):
        nonterminal = (~terminated[index]).to(rewards.dtype)
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    return advantages, advantages + values


def clipped_policy_loss(new_logp, old_logp, advantages, clip_coef, factor=None):
    ratio = (new_logp - old_logp).exp()
    weighted_advantages = advantages if factor is None else advantages * factor
    unclipped = ratio * weighted_advantages
    clipped = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * weighted_advantages
    return -torch.minimum(unclipped, clipped).mean()


class PPOTrainer:
    algorithm = "ppo"

    def __init__(self, seed=0, device="cpu"):
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.actors = nn.ModuleList((Actor(), Actor())).to(self.device)
        self.critic = CentralCritic().to(self.device)
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=ACTOR_LR) for actor in self.actors
        ]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

    @torch.no_grad()
    def act(self, observations, masks, state=None, deterministic=False):
        observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        if state is None:
            state = observations.reshape(-1)
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        actions = []
        log_probabilities = []
        for agent, actor in enumerate(self.actors):
            distribution = masked_categorical(actor(observations[agent]), masks[agent])
            action = distribution.probs.argmax() if deterministic else distribution.sample()
            actions.append(action)
            log_probabilities.append(distribution.log_prob(action))
        value = self.critic(state)
        return (
            torch.stack(actions).cpu().numpy(),
            torch.stack(log_probabilities).cpu().numpy(),
            float(value.item()),
        )

    def _minibatches(self, indices):
        shuffled = self.rng.permutation(indices.detach().cpu().numpy())
        for start in range(0, len(shuffled), MINIBATCH_SIZE):
            yield torch.as_tensor(shuffled[start : start + MINIBATCH_SIZE], device=self.device)

    @staticmethod
    def _check_finite(loss, parameters):
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite PPO loss")
        for parameter in parameters:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("non-finite PPO gradient")

    def _update_critic(self, batch):
        indices = torch.arange(len(batch.rewards), device=self.device)
        losses = []
        for _ in range(PPO_EPOCHS):
            for sample in self._minibatches(indices):
                new_values = self.critic(batch.states[sample])
                old_values = batch.values[sample]
                clipped_values = old_values + (new_values - old_values).clamp(
                    -CLIP_COEF, CLIP_COEF
                )
                plain_error = (new_values - batch.returns[sample]).square()
                clipped_error = (clipped_values - batch.returns[sample]).square()
                loss = 0.5 * VALUE_COEF * torch.maximum(plain_error, clipped_error).mean()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self._check_finite(loss, self.critic.parameters())
                nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
                self.critic_optimizer.step()
                losses.append(float(loss.detach().item()))
        return float(np.mean(losses))

    def _update_actor(self, agent, batch, factor=None):
        indices = torch.nonzero(batch.alive[:, agent], as_tuple=False).squeeze(-1)
        if len(indices) == 0:
            return 0.0
        advantages = batch.advantages[indices]
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        normalized = torch.zeros_like(batch.advantages)
        normalized[indices] = advantages
        losses = []
        optimizer = self.actor_optimizers[agent]
        actor = self.actors[agent]
        for _ in range(PPO_EPOCHS):
            for sample in self._minibatches(indices):
                distribution = masked_categorical(
                    actor(batch.obs[sample, agent]), batch.masks[sample, agent]
                )
                new_logp = distribution.log_prob(batch.actions[sample, agent])
                sample_factor = None if factor is None else factor[sample]
                policy_loss = clipped_policy_loss(
                    new_logp,
                    batch.old_logp[sample, agent],
                    normalized[sample],
                    CLIP_COEF,
                    sample_factor,
                )
                loss = policy_loss - ENTROPY_COEF * distribution.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                self._check_finite(loss, actor.parameters())
                nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                losses.append(float(loss.detach().item()))
        return float(np.mean(losses))


class MAPPOTrainer(PPOTrainer):
    algorithm = "mappo"

    def update(self, batch):
        batch = batch.to(self.device)
        critic_loss = self._update_critic(batch)
        actor_losses = [self._update_actor(agent, batch) for agent in range(2)]
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": critic_loss,
        }


class HAPPOTrainer(PPOTrainer):
    algorithm = "happo"

    def __init__(self, seed=0, device="cpu"):
        super().__init__(seed=seed, device=device)
        self.last_update_order = ()
        self.last_factor = torch.ones(0)

    def update(self, batch):
        batch = batch.to(self.device)
        critic_loss = self._update_critic(batch)
        self.last_update_order = tuple(int(value) for value in self.rng.permutation(2))
        factor = torch.ones_like(batch.advantages)
        actor_losses = []
        for order_index, agent in enumerate(self.last_update_order):
            if order_index == 1:
                self.last_factor = factor.detach().cpu().clone()
            actor_losses.append(self._update_actor(agent, batch, factor=factor.detach()))
            with torch.no_grad():
                distribution = masked_categorical(
                    self.actors[agent](batch.obs[:, agent]), batch.masks[:, agent]
                )
                new_logp = distribution.log_prob(batch.actions[:, agent])
                ratio = (new_logp - batch.old_logp[:, agent]).exp()
                factor = factor * torch.where(batch.alive[:, agent], ratio, 1.0)
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": critic_loss,
        }


def save_checkpoint(path, trainer, config):
    torch.save(
        {
            "algorithm": trainer.algorithm,
            "seed": trainer.seed,
            "actors": [actor.state_dict() for actor in trainer.actors],
            "critic": trainer.critic.state_dict(),
            "actor_optimizers": [
                optimizer.state_dict() for optimizer in trainer.actor_optimizers
            ],
            "critic_optimizer": trainer.critic_optimizer.state_dict(),
            "config": dict(config),
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    trainers = {"mappo": MAPPOTrainer, "happo": HAPPOTrainer}
    try:
        trainer_class = trainers[checkpoint["algorithm"]]
    except KeyError as error:
        raise ValueError("unknown checkpoint algorithm") from error
    trainer = trainer_class(seed=checkpoint["seed"], device=device)
    for actor, state_dict in zip(trainer.actors, checkpoint["actors"]):
        actor.load_state_dict(state_dict)
    trainer.critic.load_state_dict(checkpoint["critic"])
    for optimizer, state_dict in zip(
        trainer.actor_optimizers, checkpoint["actor_optimizers"]
    ):
        optimizer.load_state_dict(state_dict)
    trainer.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    return trainer, checkpoint["config"]
