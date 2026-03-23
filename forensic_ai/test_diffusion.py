import torch
from torch.utils.data import DataLoader

from models.diffusion import DiffusionModel, add_noise
from pipeline.dataset import SequenceDataset

dataset = SequenceDataset("data/processed/lanl_sequences.json")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

sample = next(iter(loader))
input_dim = sample.shape[1]

model = DiffusionModel(input_dim)
model.load_state_dict(torch.load("diffusion_model.pth"))
model.eval()

x = sample.float()
x = (x - x.mean()) / (x.std() + 1e-8)

noisy_x, _ = add_noise(x)

recon = model(noisy_x)

print("Original:", x)
print("Noisy:", noisy_x)
print("Reconstructed:", recon)

error = torch.mean((x - recon) ** 2).item()
print("Reconstruction Error:", error)