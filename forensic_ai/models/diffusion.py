import torch
import torch.nn as nn

class Denoiser(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, dim)
        )

    def forward(self, x):
        return self.net(x)


def add_noise(x, noise_level=0.1):
    noise = torch.randn_like(x) * noise_level
    return x + noise