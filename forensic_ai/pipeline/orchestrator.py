class Orchestrator:
    def __init__(self, vae, gan, diffusion):
        self.vae = vae
        self.gan = gan
        self.diffusion = diffusion

    def run(self, artifact_type, x):
        if artifact_type == "metadata":
            return self.vae(x)

        elif artifact_type == "sequence":
            return self.gan(x)

        elif artifact_type == "binary":
            return self.diffusion(x)