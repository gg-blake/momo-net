from torch import nn
import torch
from torchinfo import summary
import math
from mps import get_mps_device

device = get_mps_device()

def parse_size(x: tuple[int, int] | int) -> tuple[int, int]:
    return x if isinstance(x, tuple) else (x, x)



def compute_conv_output_shape(
    input_dim: tuple[int, int, int, int],
    _kernel_size: tuple[int, int] | int = 1,
    _padding: tuple[int, int] | int = 0,
    _stride: tuple[int, int] | int = 1,
    _dilation: tuple[int, int] | int = 1
):
    kernel_size = parse_size(_kernel_size)
    padding = parse_size(_padding)
    stride = parse_size(_stride)
    dilation = parse_size(_dilation)
    h_in, w_in = input_dim[-2:]
    def _compute_conv_output(val_in: int, k: int, p: int, s: int, d: int):
        return math.floor((val_in + (2 * p) - (d * (k - 1)) - 1) / s + 1)
    h_out = _compute_conv_output(h_in, kernel_size[0], padding[0], stride[0], dilation[0])
    w_out = _compute_conv_output(w_in, kernel_size[1], padding[1], stride[1], dilation[1])
    return h_out, w_out

class EncoderBase(nn.Module):
    def __init__(self, input_dim = (1, 1, 28, 28), conv_filters = [32,64,64, 64]
    , conv_kernel_size = [3,3,3,3]
    , conv_strides = [1,2,2,1]
    , z_dim = 2, use_batch_norm: bool = False, use_dropout: bool = False):
        super().__init__()
        self.enc = nn.Sequential()
        count = len(conv_filters)
        prev_shape = input_dim
        for idx, out_channels, kernel_size, stride in zip(range(count), conv_filters, conv_kernel_size, conv_strides):
            in_channels = input_dim[-3]
            if idx != 0:
                in_channels = conv_filters[idx - 1]
                
            padding = 1 # TODO: don't hardcode this
                
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.enc.add_module(f"encoder_conv_{idx}", conv)
            
            # Optional batch normalization
            if use_batch_norm:
                self.enc.add_module(f"batch_norm_{idx}", nn.BatchNorm2d(out_channels))
            
            self.enc.add_module(f"leaky_relu_{idx}", nn.LeakyReLU(0.2))
            
            # Optional dropout
            if use_dropout:
                self.enc.add_module(f"dropout_{idx}", nn.Dropout())
            
            prev_shape = (input_dim[0], out_channels, *compute_conv_output_shape(prev_shape, kernel_size, padding, stride))
        
        self.output_shape = prev_shape
        self.enc.add_module("encoder_flatten", nn.Flatten())

class EncoderAE(EncoderBase):
    def __init__(self, input_dim = (1, 1, 28, 28), conv_filters = [32,64,64, 64]
    , conv_kernel_size = [3,3,3,3]
    , conv_strides = [1,2,2,1]
    , z_dim = 2, use_batch_norm: bool = False, use_dropout: bool = False):
        super().__init__(input_dim, conv_filters, conv_kernel_size, conv_strides, z_dim, use_batch_norm, use_dropout)
        
        flat_size = self.output_shape[-3] * self.output_shape[-2] * self.output_shape[-1]
        self.ffwd = nn.Linear(flat_size, z_dim)
    
    def forward(self, x):
        return self.ffwd(self.enc(x))

class EncoderVAE(EncoderBase):
    def __init__(self, input_dim = (1, 1, 28, 28), conv_filters = [32,64,64, 64]
    , conv_kernel_size = [3,3,3,3]
    , conv_strides = [1,2,2,1]
    , z_dim = 2, use_batch_norm: bool = False, use_dropout: bool = False):
        super().__init__(input_dim, conv_filters, conv_kernel_size, conv_strides, z_dim, use_batch_norm, use_dropout)
        
        # Map to points in multivariate normal distribution in a latent space
        flat_size = self.output_shape[-3] * self.output_shape[-2] * self.output_shape[-1]
        self.mu = nn.Linear(flat_size, z_dim)
        self.log_var = nn.Linear(flat_size, z_dim)

    def forward(self, x):
        flat = self.enc(x)
        mu = self.mu(flat)
        log_var = self.log_var(flat)
        epsilon = torch.randn(mu.shape).to(device)
        return mu + torch.exp(log_var / 2) * epsilon, mu, log_var

class Decoder(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int, int], output_dim = (1, 28, 28), conv_t_filters = [64, 64, 32, 1]
    , conv_t_kernel_size = [3,3,3,3]
    , conv_t_strides = [1,2,2,1]
    , z_dim = 2, use_batch_norm: bool = False, use_dropout: bool = False):
        super().__init__()
        count = len(conv_t_filters)
        self.ffwd = nn.Linear(z_dim, input_shape[-3]*input_shape[-2]*input_shape[-1])
        self.input_shape = input_shape
        prev_shape = input_shape
        H, W = input_shape[-2], input_shape[-1]  # usually the output of the encoder conv stack
        self.dec = nn.Sequential()
        for idx, out_channels, kernel_size, stride in zip(range(count), conv_t_filters, conv_t_kernel_size, conv_t_strides):
            padding = 1 # TODO: don't hardcode this
            
            in_channels = input_shape[-3]
            if idx != 0:
                in_channels = conv_t_filters[idx - 1]
                
            output_padding_h = 0
            output_padding_w = 0
            if idx + 1 == count:
                output_padding_h = output_dim[-2] - ((H - 1) * stride - 2 * padding + kernel_size)
                output_padding_w = output_dim[-1] - ((W - 1) * stride - 2 * padding + kernel_size)
            output_padding = (output_padding_h, output_padding_w)
            conv_t = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding)
            self.dec.add_module(f"decoder_conv_t_{idx}", conv_t)
            
            # Optional batch normalization
            if use_batch_norm:
                self.dec.add_module(f"batch_norm_{idx}", nn.BatchNorm2d(out_channels))
            
            if idx + 1 == count:
                self.dec.add_module("activation_1", nn.Sigmoid())
            else:
                self.dec.add_module(f"leaky_relu_{idx+count}", nn.LeakyReLU(0.2))
            
            # Optional dropout
            if use_dropout:
                self.dec.add_module(f"dropout_{idx}", nn.Dropout())
            
            H = (H - 1) * stride - 2 * padding + kernel_size + output_padding_w
            W = (W - 1) * stride - 2 * padding + kernel_size + output_padding_h

    def forward(self, x):
        ffwd = self.ffwd(x)
        
        reshaped = ffwd.reshape(*self.input_shape)
        return self.dec(reshaped)

"""Variational Autoencoder"""
class VAE(nn.Module):
    def __init__(self, input_dim = (20, 1, 28, 28)
        , encoder_conv_filters = [32,64,64,64]
        , encoder_conv_kernel_size = [3,3,3,3]
        , encoder_conv_strides = [1,2,2,1]
        , decoder_conv_t_filters = [64,64,32,1]
        , decoder_conv_t_kernel_size = [3,3,3,3]
        , decoder_conv_t_strides = [1,2,2,1]
        , z_dim = 2
        , use_batch_norm: bool = False
        , use_dropout: bool = False
        , encoder: type[EncoderBase] = EncoderVAE
        , decoder=Decoder
    ):
        super().__init__()
        self.enc = encoder(input_dim, encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides, z_dim, use_batch_norm=use_batch_norm, use_dropout=use_dropout)
        self.dec = decoder(self.enc.output_shape, input_dim, decoder_conv_t_filters, decoder_conv_t_kernel_size, decoder_conv_t_strides, z_dim, use_batch_norm=use_batch_norm, use_dropout=use_dropout)
    
    def forward(self, x):
        if isinstance(self.enc, EncoderVAE):
            enc, mu, log_var = self.enc(x)
            return self.dec(enc), mu, log_var
        elif isinstance(self.enc, EncoderAE):
            return self.dec(self.enc(x))
   
def vae_r_loss(y_true, y_pred, r_loss_factor):
    r_loss = torch.mean((y_true - y_pred)**2, dim=(1, 2, 3,))
    return r_loss_factor * r_loss
    
def vae_kl_loss(y_true, y_pred, mu, log_var):
    kl_loss = -0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var), dim=1)
    return kl_loss
    
def vae_loss(y_true, y_pred, mu, log_var, r_loss_factor: float=1e-5):
    r_loss = vae_r_loss(y_true, y_pred, r_loss_factor)
    kl_loss = vae_kl_loss(y_true, y_pred, mu, log_var)
    return r_loss + kl_loss
    
def ae_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred)**2, dim=(1, 2, 3)).mean()
          
                
if __name__ == "__main__":
    vae = VAE().to(device)
    tensor = torch.rand(20, 1, 28, 28).to(device)
    out, _, _ = vae(tensor)
    