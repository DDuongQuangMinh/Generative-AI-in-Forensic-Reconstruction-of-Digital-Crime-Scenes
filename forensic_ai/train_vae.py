import torch
from torch.utils.data import DataLoader

from models.vae import VAE
from pipeline.dataset import SequenceDataset

# Load dataset
dataset = SequenceDataset("data/processed/lanl_sequences.json")
loader = DataLoader(dataset, batch_size=128, shuffle=True)

mean = torch.load("mean.pt")
std = torch.load("std.pt")

input_dim = next(iter(loader)).shape[1]

vae = VAE(input_dim)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

for epoch in range(9000):
    total_loss = 0

    for x in loader:
        x = x.float()
        x = (x - mean) / std

        recon, mu, logvar = vae(x)

        recon_loss = ((recon - x) ** 2).mean()
        kl = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())

        loss = recon_loss + kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

torch.save(vae.state_dict(), "vae_model.pth")
print("VAE trained")