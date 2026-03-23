import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.gan import Generator
from pipeline.dataset import SequenceDataset

# -----------------------------
# Load datasets
# -----------------------------
normal_data = SequenceDataset("data/processed/lanl_sequences.json")
attack_data = SequenceDataset("data/processed/redteam_sequences.json")

normal_loader = DataLoader(normal_data, batch_size=1, shuffle=True)
attack_loader = DataLoader(attack_data, batch_size=1, shuffle=True)

# -----------------------------
# Model setup
# -----------------------------
sample = next(iter(normal_loader))
input_dim = sample.shape[1]
noise_dim = 32

gen = Generator(noise_dim, input_dim)
gen.eval()

# ⚠️ If you saved model:
# gen.load_state_dict(torch.load("generator.pth"))

# -----------------------------
# Anomaly scoring function
# -----------------------------
def anomaly_score(real_seq):
    real_seq = real_seq.squeeze()

    # Try to approximate with generator
    noise = torch.randn(1, noise_dim)
    fake = gen(noise).detach().squeeze()

    # Compute distance
    score = torch.mean((real_seq - fake) ** 2).item()
    return score

# -----------------------------
# Evaluate NORMAL data
# -----------------------------
normal_scores = []

for i, seq in enumerate(normal_loader):
    if i > 100: break
    normal_scores.append(anomaly_score(seq))

# -----------------------------
# Evaluate ATTACK data
# -----------------------------
attack_scores = []

for i, seq in enumerate(attack_loader):
    if i > 100: break
    attack_scores.append(anomaly_score(seq))

print("Normal avg score:", np.mean(normal_scores))
print("Attack avg score:", np.mean(attack_scores))

plt.hist(normal_scores, bins=30, alpha=0.5, label="Normal")
plt.hist(attack_scores, bins=30, alpha=0.5, label="Attack")

plt.legend()
plt.title("Anomaly Score Distribution")
plt.show()