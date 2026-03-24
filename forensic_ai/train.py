import torch
from torch.utils.data import DataLoader

from models.gan import Generator, Critic
from pipeline.dataset import SequenceDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = SequenceDataset("data/processed/lanl_sequences.json")
loader = DataLoader(dataset, batch_size=64, shuffle=True)

mean = torch.load("mean.pt").to(device)
std = torch.load("std.pt").to(device)

input_dim = next(iter(loader)).shape[1]
noise_dim = 32

gen = Generator(noise_dim, input_dim).to(device)
critic = Critic(input_dim).to(device)

opt_g = torch.optim.Adam(gen.parameters(), lr=1e-4)
opt_c = torch.optim.Adam(critic.parameters(), lr=1e-4)

lambda_gp = 10

def gradient_penalty(real, fake):
    alpha = torch.rand(real.size(0), 1).to(device)
    alpha = alpha.expand_as(real)

    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    pred = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=pred,
        inputs=interpolated,
        grad_outputs=torch.ones_like(pred),
        create_graph=True
    )[0]

    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

for epoch in range(1000):
    for real in loader:
        real = real.float().to(device)
        real = (real - mean) / std

        bs = real.size(0)

        # Train critic
        for _ in range(5):
            noise = torch.randn(bs, noise_dim).to(device)
            fake = gen(noise)

            loss_c = -(critic(real).mean() - critic(fake.detach()).mean())
            gp = gradient_penalty(real, fake)

            loss = loss_c + lambda_gp * gp

            opt_c.zero_grad()
            loss.backward()
            opt_c.step()

        # Train generator
        noise = torch.randn(bs, noise_dim).to(device)
        fake = gen(noise)

        loss_g = -critic(fake).mean()

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

    print(f"Epoch {epoch}")

torch.save(gen.state_dict(), "gan_model.pth")
print("GAN trained")