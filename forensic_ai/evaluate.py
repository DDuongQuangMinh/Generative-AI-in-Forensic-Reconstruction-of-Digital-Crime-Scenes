import torch
from torch.utils.data import DataLoader

from models.vae import VAE
from models.gan import Generator
from models.diffusion import DiffusionModel

from pipeline.dataset import SequenceDataset
from pipeline.orchestrator import Orchestrator

from evaluation.confidence import (
    vae_confidence,
    gan_confidence,
    diffusion_confidence,
    fuse_scores
)
from evaluation.decision import classify
from evaluation.metrics import compute_metrics, plot_roc

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load datasets
# -----------------------------
normal_data = SequenceDataset("data/processed/lanl_sequences.json")
attack_data = SequenceDataset("data/processed/redteam_sequences.json")

normal_loader = DataLoader(normal_data, batch_size=1)
attack_loader = DataLoader(attack_data, batch_size=1)

# -----------------------------
# Load models
# -----------------------------
sample = next(iter(normal_loader)).to(device)
input_dim = sample.shape[1]

vae = VAE(input_dim).to(device)
vae.load_state_dict(torch.load("vae_model.pth", map_location=device))
vae.eval()

gan = Generator(32, input_dim).to(device)
gan.load_state_dict(torch.load("gan_model.pth", map_location=device))
gan.eval()

diff = DiffusionModel(input_dim).to(device)
diff.load_state_dict(torch.load("diffusion_model.pth", map_location=device))
diff.eval()

orch = Orchestrator(vae, gan, diff)

# -----------------------------
# Evaluation storage
# -----------------------------
y_true = []
y_pred = []
scores = []

THRESHOLD = 0.5  # decision threshold

# -----------------------------
# Process NORMAL data
# -----------------------------
for i, sample in enumerate(normal_loader):
    if i > 200: break

    x = sample.to(device).float()
    x = (x - x.mean()) / (x.std() + 1e-8)

    vae_out, _ = orch.run("metadata", x)
    gan_out, _ = orch.run("sequence", x)
    diff_out, _ = orch.run("binary", x)

    vae_error = torch.mean((x - vae_out) ** 2).item()
    gan_raw = gan_out.mean().item()
    diff_error = torch.mean((x - diff_out) ** 2).item()

    score = fuse_scores(
        vae_confidence(vae_error),
        gan_confidence(gan_raw),
        diffusion_confidence(diff_error)
    )

    y_true.append(0)
    scores.append(score)
    y_pred.append(1 if score > THRESHOLD else 0)

# -----------------------------
# Process ATTACK data
# -----------------------------
for i, sample in enumerate(attack_loader):
    if i > 200: break

    x = sample.to(device).float()
    x = (x - x.mean()) / (x.std() + 1e-8)

    vae_out, _ = orch.run("metadata", x)
    gan_out, _ = orch.run("sequence", x)
    diff_out, _ = orch.run("binary", x)

    vae_error = torch.mean((x - vae_out) ** 2).item()
    gan_raw = gan_out.mean().item()
    diff_error = torch.mean((x - diff_out) ** 2).item()

    score = fuse_scores(
        vae_confidence(vae_error),
        gan_confidence(gan_raw),
        diffusion_confidence(diff_error)
    )

    y_true.append(1)
    scores.append(score)
    y_pred.append(1 if score > THRESHOLD else 0)

# -----------------------------
# Compute metrics
# -----------------------------
precision, recall, f1 = compute_metrics(y_true, y_pred)
roc_auc = plot_roc(y_true, scores)

print("\n=== EVALUATION RESULTS ===")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"AUC: {roc_auc:.3f}")