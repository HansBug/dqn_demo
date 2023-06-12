from typing import Tuple

import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_layers=(64, 64)):
        nn.Module.__init__(self)

        self.input_layer = nn.Linear(state_dims, hidden_layers[0])
        self.hidden_layers = [
            nn.Linear(hidden_layers[i], hidden_layers[i + 1])
            for i in range(len(hidden_layers) - 1)
        ]
        self.output_layer = nn.Linear(hidden_layers[-1], action_dims)

        self.net = nn.Sequential(
            self.input_layer,
            *self.hidden_layers,
            self.output_layer,
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, state_xy: Tuple[int, int], action_dims: int):
        nn.Module.__init__(self)
        self.state_xy = state_xy
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, action_dims)

    def forward(self, x):
        # x : Bx1xHxW
        # assert x.shape[-2:] == self.state_xy
        x = torch.relu(self.conv1(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
