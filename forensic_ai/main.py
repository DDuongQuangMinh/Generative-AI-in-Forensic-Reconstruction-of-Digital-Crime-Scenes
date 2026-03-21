import torch
from models.vae import VAE
from models.gan import Generator
from models.diffusion import Denoiser
from pipeline.orchestrator import Orchestrator
from evaluation.integrity import hash_data

vae = VAE()
gan = Generator()
diff = Denoiser()

orch = Orchestrator(vae, gan, diff)

# Example metadata input
x = torch.randn(1, 3)

out, _, _ = orch.run("metadata", x)

print("Output:", out)
print("Hash:", hash_data(out.detach().numpy()))