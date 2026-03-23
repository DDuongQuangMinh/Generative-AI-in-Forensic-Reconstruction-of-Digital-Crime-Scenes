import torch
import numpy as np
from torch.utils.data import DataLoader

from models.vae import VAE
from pipeline.dataset import SequenceDataset

# -----------------------------
# Load data
# -----------------------------
normal_data = SequenceDataset("data/processed/lanl_sequences.json")
attack_data = SequenceDataset("data/processed/redteam_sequences.json")

normal_loader = DataLoader(normal_data, batch_size=1)
attack_loader = DataLoader(attack_data, batch_size=1)

# -----------------------------
# Load model
# -----------------------------
sample = next(iter(normal_loader))
input_dim = sample.shape[1]

vae = VAE(input_dim)
vae.load_state_dict(torch.load("vae_model.pth"))
vae.eval()

# -----------------------------
# Anomaly score = reconstruction error
# -----------------------------
def anomaly_score(x):
    x = x.float()
    x = (x - x.mean()) / (x.std() + 1e-8)

    recon, _, _ = vae(x)

    return torch.mean((x - recon) ** 2).item()

# -----------------------------
# Evaluate
# -----------------------------
normal_scores = []
attack_scores = []

for i, x in enumerate(normal_loader):
    if i > 100: break
    normal_scores.append(anomaly_score(x))

for i, x in enumerate(attack_loader):
    if i > 100: break
    attack_scores.append(anomaly_score(x))

print("Normal avg:", np.mean(normal_scores))
print("Attack avg:", np.mean(attack_scores))