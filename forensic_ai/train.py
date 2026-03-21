import torch
from models.vae import VAE
from models.gan import Generator, Discriminator
from models.diffusion import Denoiser

vae = VAE()
gen = Generator()
disc = Discriminator()
diff = Denoiser()

optimizer_vae = torch.optim.Adam(vae.parameters(), lr=1e-3)

# Example training loop (VAE only for brevity)
for epoch in range(10):
    x = torch.randn(32, 3)

    recon, mu, logvar = vae(x)

    recon_loss = ((recon - x) ** 2).mean()
    kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())

    loss = recon_loss + kl_loss

    optimizer_vae.zero_grad()
    loss.backward()
    optimizer_vae.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")