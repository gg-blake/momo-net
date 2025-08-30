from torch import nn
import torch
from torchinfo import summary

class Decoder(nn.Module):
    def __init__(self, filter_size_enc: int, filter_size_dec: int, kernel_size: int, num_groups: int, out_channels: int, prev_out_channels: int):
        super().__init__()
        self.dec_up_conv_0 = nn.ConvTranspose2d(prev_out_channels, filter_size_enc, kernel_size=2, stride=2)
        self.dec_conv_0 = nn.Sequential(
            nn.Conv2d(in_channels=filter_size_enc + filter_size_dec, out_channels=filter_size_dec, kernel_size=kernel_size, padding="same"),
            nn.GroupNorm(num_groups, filter_size_dec),
            nn.SiLU()
        )
        
        self.dec_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=filter_size_dec, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
        )
        if out_channels != 1:
            print(filter_size_dec, out_channels, num_groups)
            self.dec_conv_1.add_module(f"dec_conv_1_group_norm", nn.GroupNorm(num_groups, out_channels))
            self.dec_conv_1.add_module(f"dec_conv_1_silu", nn.SiLU())
        
    def forward(self, enc, mid):
        up = self.dec_up_conv_0(mid)
        cat = torch.cat([enc, up], dim=-3)
        out = self.dec_conv_1(self.dec_conv_0(cat))
        return out
        
class Encoder(nn.Module):
    def __init__(self, filter_size_enc: int, kernel_size: int, num_groups: int, in_channels: int):
        super().__init__()
        self.enc_conv_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filter_size_enc, kernel_size=kernel_size, padding="same"),
            nn.GroupNorm(num_groups, filter_size_enc),
            nn.SiLU()
        )
        self.enc_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=filter_size_enc, out_channels=filter_size_enc, kernel_size=kernel_size, padding="same"),
            nn.GroupNorm(num_groups, filter_size_enc),
            nn.SiLU()
        )
        
    def forward(self, x):
        return self.enc_conv_1(self.enc_conv_0(x))
    

class UNet(nn.Module):
    def __init__(self, input_dim: tuple[int, int, int, int]=(1, 1, 512, 512), output_dim: tuple[int, int, int, int]=(1, 2, 512, 512), enc_conv_filters=[64, 128, 256, 512, 1024], dec_conv_filters=[64, 128, 256, 512], strides=[2, 2, 2, 2, 2], kernel_sizes=[3, 3, 3, 3, 3], num_groups: int = 32, layer_num: int = 0):
        super().__init__()
        B, C, H, W = input_dim
        self.layer_num = layer_num
        if layer_num < len(enc_conv_filters):
            self.enc = Encoder(enc_conv_filters[layer_num], kernel_sizes[layer_num], num_groups, input_dim[1] if layer_num == 0 else enc_conv_filters[layer_num-1])
        if layer_num < len(dec_conv_filters):
            self.next = UNet((B, C, H // strides[layer_num], W // strides[layer_num]), output_dim, enc_conv_filters, dec_conv_filters, strides, kernel_sizes, num_groups, layer_num+1)
            self.dec = Decoder(enc_conv_filters[layer_num], dec_conv_filters[layer_num], kernel_sizes[layer_num], num_groups, input_dim[1] if layer_num == 0 else enc_conv_filters[layer_num], self.next.out_channels)
            self.out_channels = dec_conv_filters[layer_num]
        else:
            self.next = None
            self.out_channels = enc_conv_filters[layer_num]
            
    
    def forward(self, x):
        enc = self.enc(x)
        if self.next is None:
            return enc
        
        pool = nn.MaxPool2d(2)
        mid = self.next(pool(enc))
        out = self.dec(enc, mid)
        return out
        
if __name__ == "__main__":
    tensor = torch.rand((64, 1, 28, 28))
    B, C, H, W = tensor.shape
    model = UNet(
        input_dim=(B, C, H, W), 
        output_dim=(B, C, H, W), 
        enc_conv_filters=[64, 128, 256], 
        dec_conv_filters=[64, 128], 
        strides=[2, 2, 2], 
        kernel_sizes=[3, 3, 3], 
        num_groups = 4
    )
    
    out = model(tensor)
    print(out.shape)
    
