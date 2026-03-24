import torch

class Orchestrator:
    def __init__(self, vae, gan, diffusion):
        self.vae = vae
        self.gan = gan
        self.diffusion = diffusion

    def run(self, artifact_type, x):
        if artifact_type == "metadata":
            recon, mu, logvar = self.vae(x)
            return recon, {"type": "VAE", "info": "metadata reconstruction"}

        elif artifact_type == "sequence":
            noise = torch.randn(x.size(0), 32).to(x.device)
            fake = self.gan(noise)
            return fake, {"type": "GAN", "info": "sequence generation"}

        elif artifact_type == "binary":
            recon = self.diffusion(x)
            return recon, {"type": "Diffusion", "info": "denoising"}

        else:
            raise ValueError("Unknown artifact type")