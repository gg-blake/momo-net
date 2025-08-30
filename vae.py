import torch
from torch import nn
import torch.nn.functional as F
from attention import SelfAttention

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(num_groups, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(num_groups, out_channels)
        self.conv_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual_layer = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = self.residual_layer(x)
        x = self.groupnorm_1(x)
        x = self.forward_conv(x, self.conv_1, self.groupnorm_1)
        x = self.forward_conv(x, self.conv_2, self.groupnorm_2)
        return x + residue
        
    def forward_conv(self, x: torch.Tensor, conv: nn.Conv2d, groupnorm: nn.GroupNorm) -> torch.Tensor:
        x = groupnorm(x)
        x = F.silu(x)
        x = conv(x)
        return x
        
class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int, *attention_args, num_groups: int = 32, **attention_kwargs):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups, channels)
        self.attention = SelfAttention(channels, *attention_args, **attention_kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((B, C, H, W))
        x += residue
        return x

class VAE_Encoder(nn.Module):
    def __init__(self, 
        in_channels: int,
        out_channels: int,
        *attention_args,
        n_layers: int = 3, 
        image_channels: int = 3,
        num_groups: int = 32,
        **attention_kwargs
    ):
        super().__init__()
        current_channels = in_channels
        prev_channels = in_channels
        residual_layers = []
        for i in range(n_layers):
            residual_layers = [*residual_layers, 
                VAE_ResidualBlock(in_channels=prev_channels, out_channels=current_channels),
                VAE_ResidualBlock(in_channels=current_channels, out_channels=current_channels),
                nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding=0) # padding might need to be 1 here...
            ]
            current_channels *= 2
            
        self.ffwd = nn.Sequential(
            nn.Conv2d(image_channels, in_channels, kernel_size=3, padding=1),
            *residual_layers,
            VAE_ResidualBlock(current_channels, current_channels),
            VAE_ResidualBlock(current_channels, current_channels),
            VAE_ResidualBlock(current_channels, current_channels),
            VAE_AttentionBlock(current_channels, *attention_args, **attention_kwargs),
            VAE_ResidualBlock(current_channels, current_channels),
            nn.GroupNorm(num_groups, current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        )
        
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        for module in self.ffwd:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
            
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_var, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        x = mean + stdev * noise
        x *= 0.18215 # constant scale
        return x
        
class VAE_Decoder(nn.Module):
    def __init__(self, 
        in_channels: int,
        out_channels: int,
        *attention_args,
        n_layers: int = 3, 
        image_channels: int = 3,
        num_groups: int = 32,
        **attention_kwargs
    ):
        super().__init__()
        current_channels = in_channels
        prev_channels = in_channels
        residual_layers = []
        for i in range(n_layers):
            residual_layers = [*residual_layers,
                nn.Upsample(scale_factor=2),
                nn.Conv2d(prev_channels, prev_channels, kernel_size=3, padding=1), # padding might need to be 1 here...
                VAE_ResidualBlock(in_channels=prev_channels, out_channels=current_channels),
                VAE_ResidualBlock(in_channels=current_channels, out_channels=current_channels),
                VAE_ResidualBlock(in_channels=current_channels, out_channels=current_channels),
            ]
            current_channels //= 2
            
        self.ffwd = nn.Sequential(
            nn.Conv2d(image_channels, in_channels, kernel_size=3, padding=1),
            *residual_layers,
            VAE_ResidualBlock(current_channels, current_channels),
            VAE_AttentionBlock(current_channels, *attention_args, **attention_kwargs),
            VAE_ResidualBlock(current_channels, current_channels),
            VAE_ResidualBlock(current_channels, current_channels),
            VAE_ResidualBlock(current_channels, current_channels),
            VAE_ResidualBlock(current_channels, current_channels), # maybe one too many...
            nn.GroupNorm(num_groups, current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        )
        
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        for module in self.ffwd:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
            
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_var, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        x = mean + stdev * noise
        x *= 0.18215 # constant scale
        return x
        

class VAE(nn.Module):
    def __init__(self, input_dim: tuple[int, int, int, int]):
        pass