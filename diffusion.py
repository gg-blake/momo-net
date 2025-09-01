import torch
from torch import nn
from unet import UNet


class Diffusion(nn.Module):
    def __init__(
        self,
        embed_size: int,
        context_size: int,
        head_size: int,
        head_count: int,
        time_embedding_size: int,
        n_unet_reductions: int = 4,
        num_groups: int = 32,
        **unet_kwargs
    ):
        super().__init__()
        self.time_embedding = nn.Sequential(
            nn.Linear(time_embedding_size, 4 * time_embedding_size),
            nn.SiLU(),
            nn.Linear(time_embedding_size * 4, time_embedding_size),
        )
        in_channels = [embed_size // 2**i for i in range(n_unet_reductions + 1)]
        out_channels = [x * 2 for x in in_channels]
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            context_size=context_size,
            time_embedding_size=time_embedding_size,
            head_size=head_size,
            head_count=head_count,
            num_groups=num_groups,
            **unet_kwargs
        )
        self.output_layer = nn.Sequential(
            nn.GroupNorm(num_groups, out_channels[-1]),
            nn.SiLU(),
            nn.Conv2d(out_channels[-1], embed_size, kernel_size=3, padding=1),
        )

    def forward(
        self,
        latent_image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        time = self.time_embedding(time)
        output = self.unet(latent_image_embeddings, text_embeddings, time)
        output = self.output_layer(output)
        return output
