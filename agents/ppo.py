from dataclasses import dataclass, fields

import torch
from torch import nn
from torch.distributions import Categorical

from config import ACTOR_HIDDEN, CRITIC_HIDDEN, N_ACTIONS, OBS_DIM, STATE_DIM


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
