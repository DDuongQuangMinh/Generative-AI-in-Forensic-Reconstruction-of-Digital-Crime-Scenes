import torch
from torch.utils.data import DataLoader

from models.diffusion import DiffusionModel, add_noise
from pipeline.dataset import SequenceDataset

dataset = SequenceDataset("data/processed/lanl_sequences.json")
loader = DataLoader(dataset, batch_size=128, shuffle=True)

mean = torch.load("mean.pt")
std = torch.load("std.pt")

input_dim = next(iter(loader)).shape[1]

model = DiffusionModel(input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(9000):
    total_loss = 0

    for x in loader:
        x = x.float()
        x = (x - mean) / std

        noisy_x, _ = add_noise(x)

        recon = model(noisy_x)

        loss = ((recon - x) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "diffusion_model.pth")
print("Diffusion trained")