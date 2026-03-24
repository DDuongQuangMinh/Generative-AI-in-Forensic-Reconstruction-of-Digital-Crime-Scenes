import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        return torch.tanh(self.net(x))


def add_noise(x, noise_level=0.05):
    noise = torch.randn_like(x) * noise_level
    return x + noise, noise