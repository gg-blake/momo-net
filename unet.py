from torch import nn
import torch
from attention import SelfAttention, CrossAttention
import torch.nn.functional as F

class UNetConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embedding_size: int, num_groups: int):
        super().__init__()
        self.enc_conv_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding="same"),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU()
        )
        self.enc_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding="same"),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU()
        )
        self.time_ffwd = nn.Linear(time_embedding_size, out_channels)
        
    def forward(self, feature, time):
        time = F.silu(time)
        time = self.time_ffwd(time)
        conv = self.enc_conv_1(self.enc_conv_0(feature))
        return conv + time.unsqueeze(-1).unsqueeze(-1)

class UNetConvTransposeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embedding_size: int, num_groups: int):
        super().__init__()
        self.dec_up_conv_0 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.dec_conv_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU()
        )
        self.dec_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU()
        )
        self.time_ffwd = nn.Linear(time_embedding_size, out_channels)
        
    def forward(self, x, skip_connection, time):
        time = F.silu(time)
        time = self.time_ffwd(time)
        up = F.interpolate(self.dec_up_conv_0(x), scale_factor=2, mode="nearest")
        cat = torch.cat([skip_connection, up], dim=-3)
        out = self.dec_conv_1(self.dec_conv_0(cat))
        return out + time.unsqueeze(-1).unsqueeze(-1)

class UNetAttentionBlock(nn.Module):
    def __init__(self, channels: int, context_size: int, head_size: int, head_count: int, dropout: float = 0.0, num_groups: int = 32):
        self.group_norm = nn.GroupNorm(num_groups, channels)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layer_norm_0 = nn.LayerNorm(channels)
        self.attention_0 = SelfAttention(channels, head_size, head_count, context_size, dropout)
        self.layer_norm_1 = nn.LayerNorm(channels)
        self.attention_1 = CrossAttention(channels, context_size, head_size, head_count, dropout)
        self.layer_norm_2 = nn.LayerNorm(channels)
        self.linear_geglu_0 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_1 = nn.Linear(4 * channels, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residue_0 = x
        
        # Group Norm + Convolutional Layer
        x = self.group_norm(x)
        x = self.conv_input(x)
        x = x.view((B, C, H * W))
        x = x.transpose(-1, -2)
        residue_1 = x
        
        # Layer Norm + Self Attention w/ Residual Connection (short)
        x = self.layer_norm_0(x)
        x = residue_1 + self.attention_0(x)
        residue_1 = x
        
        # Layer Norm + Cross Attention w/ Residual Connection (short)
        x = self.layer_norm_1(x)
        x = residue_1 + self.attention_1(x, context)
        residue_1 = x
        
        # Layer Norm + Feed Forward w/ Residual (short) and GeGLU Connections
        x = self.layer_norm_2(x)
        x, gate = self.linear_geglu_0(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = residue_1 + self.linear_geglu_1(x)
        x = x.transpose(-1, -2)
        x = x.view((B, C, H, W))
        
        # Convolutional Layer w/ Residual Connection (long)
        return residue_0 + self.conv_output(x)

class UNet(nn.Module):
    def __init__(self, in_channels: list[int], out_channels: list[int], context_size: int, time_embedding_size: int, head_size: int, head_count: int, num_groups: int = 32, dropout: float = 0.0, block_num: int = 0):
        super().__init__()
        self.residual_block_0 = UNetConvBlock(in_channels[block_num], out_channels[block_num], time_embedding_size, num_groups=num_groups)
        self.attention_0 = UNetAttentionBlock(out_channels[block_num], context_size, head_size, head_count, dropout=dropout)
        
        if block_num == len(in_channels) - 1:
            return
        
        self.next_unet = UNet(in_channels, out_channels, context_size, time_embedding_size, head_size, head_count, num_groups=num_groups, dropout=dropout, block_num=block_num+1)
        self.residual_block_1 = UNetConvTransposeBlock(out_channels[block_num], out_channels[block_num], time_embedding_size, num_groups=num_groups)
        self.attention_1 = UNetAttentionBlock(out_channels[block_num], context_size, head_size, head_count, dropout=dropout)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # Left Block / Skip Connection
        x = self.residual_block_0(x, time)
        x = self.attention_0(x, context)
        if self.next_unet is None:
            return x
            
        # Recursive UNet
        pool = nn.MaxPool2d(2)
        next_x = self.next_unet(pool(x))
        
        # Right Block
        x = self.residual_block_1(next_x, x, time)
        x = self.attention_1(x, context)
        return x