import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from models.gan import Generator
from pipeline.dataset import SequenceDataset

# -----------------------------
# Load dataset
# -----------------------------
dataset = SequenceDataset("data/processed/lanl_sequences.json")
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Get shape
sample = next(iter(loader))
input_dim = sample.shape[1]

noise_dim = 32

# -----------------------------
# Load trained generator
# -----------------------------
gen = Generator(noise_dim, input_dim)

# ⚠️ If you saved model later, load here:
# gen.load_state_dict(torch.load("generator.pth"))

gen.eval()

# -----------------------------
# Get real sample
# -----------------------------
real_sample = sample[0].numpy()

# -----------------------------
# Generate fake sample
# -----------------------------
noise = torch.randn(1, noise_dim)
fake_sample = gen(noise).detach().numpy()[0]

# -----------------------------
# Plot comparison
# -----------------------------
plt.figure(figsize=(12, 5))

plt.plot(real_sample, label="Real", alpha=0.7)
plt.plot(fake_sample, label="Fake (GAN)", alpha=0.7)

plt.title("Real vs Generated Sequence")
plt.legend()
plt.show()

# -----------------------------
# Histogram comparison
# -----------------------------
plt.figure(figsize=(10,5))

plt.hist(real_sample, bins=30, alpha=0.5, label="Real")
plt.hist(fake_sample, bins=30, alpha=0.5, label="Fake")

plt.title("Distribution Comparison")
plt.legend()
plt.show()