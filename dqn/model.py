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
