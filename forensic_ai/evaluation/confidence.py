import numpy as np

# -----------------------------
# Normalize score to [0,1]
# -----------------------------
def normalize(score, min_val, max_val):
    return (score - min_val) / (max_val - min_val + 1e-8)


# -----------------------------
# VAE confidence (higher error = more anomalous)
# -----------------------------
def vae_confidence(error):
    return min(1.0, error * 2)


# -----------------------------
# GAN confidence (lower realism = more anomalous)
# -----------------------------
def gan_confidence(critic_score):
    # assume critic output ~ [-1,1]
    return 1 - (critic_score + 1) / 2


# -----------------------------
# Diffusion confidence
# -----------------------------
def diffusion_confidence(error):
    return min(1.0, error * 1.5)


# -----------------------------
# Final fused score
# -----------------------------
def fuse_scores(vae, gan, diff):
    return (0.5 * vae) + (0.3 * gan) + (0.2 * diff)