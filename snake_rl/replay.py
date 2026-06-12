"""Ring replay buffer of (state, action, reward, next_state, done) tuples."""

from __future__ import annotations

import numpy as np

from snake_rl.env import STATE_DIM


class ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator | None = None):
        self.capacity = capacity
        self.rng = rng or np.random.default_rng()
        self.states = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(self, state, action, reward, next_state, done) -> None:
        i = self._idx
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self._idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self._size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )
