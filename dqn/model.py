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
    def __init__(self, state_xy: Tuple[int, int], action_dims: int,
                 kernel_size: int = 3, channels: int = 64, psize: int = 1):
        nn.Module.__init__(self)
        self.state_xy = state_xy
        x, y = self.state_xy

        self.conv1 = nn.Conv2d(
            1, channels, kernel_size=kernel_size,
            padding=max(kernel_size - min(x, y), 0),
            bias=True,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((psize, psize))
        self.fc = nn.Linear(channels * psize * psize, action_dims)

    def forward(self, x):
        # x : Bx1xHxW
        # assert x.shape[-2:] == self.state_xy
        x = torch.relu(self.conv1(x))
        x = self.avgpool(x)
        x = torch.flatten(x, -3)
        x = self.fc(x)
        return x


class CNN2(nn.Module):
    def __init__(self, action_dims: int, psize: int = 1):
        nn.Module.__init__(self)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 64, 3, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((psize, psize))
        self.fc = nn.Linear(64 * psize * psize, action_dims)

    def forward(self, x):
        x = self.cnn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, -3)
        return self.fc(x)


class CNN3(nn.Module):
    def __init__(self, action_dims: int, psize: int = 1):
        nn.Module.__init__(self)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 32, 3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=2, bias=True),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((psize, psize))
        self.fc = nn.Linear(64 * psize * psize, action_dims)

    def forward(self, x):
        x = self.cnn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, -3)
        return self.fc(x)
