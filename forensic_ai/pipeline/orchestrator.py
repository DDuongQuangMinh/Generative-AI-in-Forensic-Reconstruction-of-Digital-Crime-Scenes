class Orchestrator:
    def __init__(self, vae, gan, diffusion):
        self.vae = vae
        self.gan = gan
        self.diffusion = diffusion

    def route(self, artifact_type, data):
        if artifact_type == "metadata":
            return self.vae(data)

        elif artifact_type == "sequence":
            return self.gan(data)

        elif artifact_type == "binary":
            return self.diffusion(data)

        else:
            raise ValueError("Unknown artifact type")