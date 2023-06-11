import random

import torch

from .base import _GLOBAL_DEVICE


class ReplayBuffer:
    def __init__(self, capacity, device=_GLOBAL_DEVICE, state_trans=None):
        self.capacity = capacity
        self.records = []
        self.device = device
        self.state_trans = state_trans

    def __len__(self):
        return len(self.records)

    def append(self, state, action, reward, next_state, done):
        if self.state_trans is not None:
            state = self.state_trans(state)
            next_state = self.state_trans(next_state)

        self.records.append(tuple(
            torch.tensor(value, dtype=dtype).to(self.device)
            for dtype, value in zip(self._DTYPES, (state, action, reward, next_state, done))
        ))
        if len(self.records) > self.capacity:
            self.records = self.records[len(self.records) - self.capacity:]

    _DTYPES = [torch.float32, torch.long, torch.float32, torch.float32, torch.float32]

    def sample(self, batch):
        ids = random.sample(range(len(self.records)), batch)
        values = [self.records[i] for i in ids]

        state, action, reward, next_state, done = [
            torch.stack(list(vs)) for vs in zip(*values)
        ]

        return state, action, reward, next_state, done
