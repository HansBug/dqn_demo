import random

import torch
from torch import optim, nn

from .base import _GLOBAL_DEVICE
from .replay_buffer import ReplayBuffer
from .state import _get_state_trans


class Agent:
    def __init__(self, state_dims, action_dims, replay_buffer_size: int = 10000,
                 batch_size: int = 64, gamma: float = 0.99, lr: float = 1e-3,
                 state_trans: str = 'standalone', device=_GLOBAL_DEVICE):
        self.device = device

        self.batch_size = batch_size
        self.gamma = gamma

        self.action_dims = action_dims
        self.state_trans = _get_state_trans(state_trans)(state_dims, self.device)
        self.replay_buffer = ReplayBuffer(replay_buffer_size, self.device, self._state_map)

        self.model = self.state_trans.create_model(action_dims).to(self.device)
        self.target = self.state_trans.create_model(action_dims).to(self.device)
        self.save_target()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def save_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def _state_map(self, state):
        if self.state_trans is not None:
            return self.state_trans.state_trans(state)
        else:
            return torch.tensor(state).to(self.device)

    def select_action(self, state, eps):
        if random.random() < eps:  # random act
            return torch.randint(0, self.action_dims, ()).item()
        else:
            return self.select_action_without_eps(state)

    def select_action_without_eps(self, state):
        with torch.no_grad():
            state = self._state_map(state).unsqueeze(0)
            return self.model(state)[0].argmax().item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return {}

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        self.optimizer.zero_grad()

        q_values = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values, _ = self.target(next_state).max(dim=1, )
            expected_q_values = reward + self.gamma * next_q_values * (1 - done)

        loss = self.loss_fn(q_values, expected_q_values)
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
