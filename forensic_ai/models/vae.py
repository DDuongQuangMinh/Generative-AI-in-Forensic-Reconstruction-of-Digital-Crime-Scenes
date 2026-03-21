import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3, 16)
        self.fc_mu = nn.Linear(16, 2)
        self.fc_logvar = nn.Linear(16, 2)

        self.fc2 = nn.Linear(2, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        h2 = torch.relu(self.fc2(z))
        return self.fc3(h2), mu, logvar