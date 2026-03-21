import torch
from models.vae import VAE
from models.gan import Generator
from models.diffusion import Denoiser
from pipeline.orchestrator import Orchestrator
from evaluation.integrity import hash_artifact

vae = VAE()
gan = Generator()
diff = Denoiser()

orch = Orchestrator(vae, gan, diff)

# Example corrupted input
data = torch.randn(1, 3)

# Route as metadata
output, _, _ = orch.route("metadata", data)

hash_value = hash_artifact(output.detach().numpy())

print("Reconstructed:", output)
print("Hash:", hash_value)