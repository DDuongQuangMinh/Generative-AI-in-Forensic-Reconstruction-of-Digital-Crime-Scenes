import torch
from torch.utils.data import DataLoader

from models.vae import VAE
from models.gan import Generator, Discriminator
from pipeline.dataset import ForensicDataset, SequenceDataset

vae = VAE()
gen = Generator()
disc = Discriminator()

vae_loader = DataLoader(
    ForensicDataset("data/processed/mft.json"),
    batch_size=32,
    shuffle=True
)

gan_loader = DataLoader(
    SequenceDataset("data/processed/lanl_sequences.json"),
    batch_size=64,
    shuffle=True
)

opt_vae = torch.optim.Adam(vae.parameters(), lr=1e-3)
opt_g = torch.optim.Adam(gen.parameters(), lr=1e-3)
opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3)

criterion = torch.nn.BCELoss()

# ---- Train VAE ----
for epoch in range(5):
    for x, y in vae_loader:
        recon, mu, logvar = vae(x)

        recon_loss = ((recon - y) ** 2).mean()
        kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())

        loss = recon_loss + kl

        opt_vae.zero_grad()
        loss.backward()
        opt_vae.step()

    print("VAE Epoch:", epoch, loss.item())

# ---- Train GAN ----
for epoch in range(5):
    for real in gan_loader:
        bs = real.size(0)

        noise = torch.randn(bs, 10)
        fake = gen(noise)

        loss_d = criterion(disc(real), torch.ones(bs,1)) + \
                 criterion(disc(fake.detach()), torch.zeros(bs,1))

        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        loss_g = criterion(disc(fake), torch.ones(bs,1))

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

    print("GAN Epoch:", epoch, loss_g.item())