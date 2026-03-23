import torch
from torch.utils.data import DataLoader

from models.vae import VAE
from models.gan import Generator
from models.diffusion import DiffusionModel

from pipeline.dataset import SequenceDataset
from pipeline.orchestrator import Orchestrator

from evaluation.integrity import hash_data
from evaluation.confidence import (
    vae_confidence,
    gan_confidence,
    diffusion_confidence,
    fuse_scores
)
from evaluation.decision import classify
from evaluation.logger import init_log, log_result

# -----------------------------
# Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Load dataset (batch mode)
# -----------------------------
dataset = SequenceDataset("data/processed/lanl_sequences.json")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

sample = next(iter(loader)).to(device)
input_dim = sample.shape[1]

# -----------------------------
# Load models
# -----------------------------
def load_model(model, path):
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"✅ Loaded {path}")
    except:
        print(f"⚠️ Missing {path}, using untrained model")
    return model.to(device).eval()

vae = load_model(VAE(input_dim), "vae_model.pth")
gan = load_model(Generator(32, input_dim), "gan_model.pth")
diff = load_model(DiffusionModel(input_dim), "diffusion_model.pth")

orch = Orchestrator(vae, gan, diff)

# -----------------------------
# Initialize logging
# -----------------------------
init_log()

# -----------------------------
# Batch processing
# -----------------------------
NUM_SAMPLES = 50  # 🔥 change as needed

for i, sample in enumerate(loader):
    if i >= NUM_SAMPLES:
        break

    x = sample.to(device).float()
    x = (x - x.mean()) / (x.std() + 1e-8)

    # -------------------------
    # Run models
    # -------------------------
    vae_out, _ = orch.run("metadata", x)
    gan_out, _ = orch.run("sequence", x)
    diff_out, _ = orch.run("binary", x)

    # -------------------------
    # Compute scores
    # -------------------------
    vae_error = torch.mean((x - vae_out) ** 2).item()
    vae_score = vae_confidence(vae_error)

    gan_raw = gan_out.mean().item()
    gan_score = gan_confidence(gan_raw)

    diff_error = torch.mean((x - diff_out) ** 2).item()
    diff_score = diffusion_confidence(diff_error)

    final_score = fuse_scores(vae_score, gan_score, diff_score)
    decision = classify(final_score)

    # -------------------------
    # Log result
    # -------------------------
    log_result({
        "vae_error": vae_error,
        "gan_raw": gan_raw,
        "diff_error": diff_error,
        "vae_score": vae_score,
        "gan_score": gan_score,
        "diff_score": diff_score,
        "final_score": final_score,
        "decision": decision
    })

    # -------------------------
    # Print progress
    # -------------------------
    print(f"[{i}] Score: {final_score:.3f} → {decision}")

print("\nBatch processing complete. Results saved to results_log.csv")