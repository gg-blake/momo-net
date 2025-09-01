import torch




class DDPMSampler:
    def __init__(
        self,
        num_training_steps=1000,
        beta_start=0.00085,
        beta_end=0.0120,
    ):
        self.betas = (
            torch.linspace(
                beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32
            )
            ** 2
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, 0)
        self.timesteps = list(range(num_training_steps))

    def apply_noise(self, x: torch.Tensor, noise: torch.Tensor, t: int) -> torch.Tensor:
        return (
            torch.sqrt(self.alpha_bars[t])[:, None, None, None] * x
            + torch.sqrt(1 - self.alpha_bars[t])[:, None, None, None] * noise
        )
        
    def step(self, z_t, model_output, interval: int = 1):
        alpha = self.alphas[self.timestep]
        alpha_bar = self.alpha_bars[self.timestep]
        z_t_1 =  (z_t - (((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * model_output))
        z_t_1 *= 1 / torch.sqrt(alpha)
        self.timestep -= interval
        return z_t_1
        


