"""Replay buffer. Asama 3 icin transition-bazli; VDN/QMIX'in episode-bazli
buffer'i Asama 5'te ayrica eklenecek (mixer tam episode ister).
"""
import numpy as np


class ReplayBuffer:
    """Onceden ayrilmis numpy dizileri uzerinde halka tampon."""

    def __init__(self, capacity: int, obs_dim: int, n_actions: int,
                 rng: np.random.Generator | None = None):
        self.capacity = capacity
        self.rng = rng or np.random.default_rng()
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.next_mask = np.zeros((capacity, n_actions), dtype=np.float32)
        self._i = 0
        self._full = False

    def __len__(self) -> int:
        return self.capacity if self._full else self._i

    def push(self, obs, action, reward, next_obs, done, next_mask):
        i = self._i
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        # Terminal gecislerde next_mask kullanilmaz (hedef r'ye esit) ama
        # tamamen sifir maske masked max'ta NEG_INF uretir; 1'lerle doldurup
        # sonra (1-done) ile carpiyoruz. Ikisi birden guvenli olsun.
        self.next_mask[i] = np.ones_like(self.next_mask[i]) if done else next_mask
        self._i = (i + 1) % self.capacity
        if self._i == 0:
            self._full = True

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, len(self), size=batch_size)
        return (self.obs[idx], self.action[idx], self.reward[idx],
                self.next_obs[idx], self.done[idx], self.next_mask[idx])
