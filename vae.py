from torch import nn
import torch

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim = (28,28,1)
        , encoder_conv_filters = [32,64,64, 64]
        , encoder_conv_kernel_size = [3,3,3,3]
        , encoder_conv_strides = [1,2,2,1]
        , decoder_conv_t_filters = [64,64,32,1]
        , decoder_conv_t_kernel_size = [3,3,3,3]
        , decoder_conv_t_strides = [1,2,2,1]
        , z_dim = 2):
            self.enc = nn.Sequential()
            count = len(encoder_conv_filters)
            for idx, out_channels, kernel_size, stride in zip(range(count), encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides):
                in_channels = input_dim[2]
                if idx != 0:
                    in_channels = encoder_conv_filters[idx - 1]
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
                
                self.enc.add_module(f"encoder_conv_{idx}", conv)
                self.enc.add_module(f"leaky_relu_{idx}", nn.LeakyReLU())
                
            self.enc.add_module("encoder_flatten", nn.Flatten())
            self.enc.add_module("encoder_linear", nn.Linear(encoder_conv_filters[-1] * encoder_conv_kernel_size[-1] * encoder_conv_kernel_size[-1], z_dim))
                
            self.dec = nn.Sequential()
            self.dec.add_module("decoder_linear", nn.Linear(z_dim, decoder_conv_t_filters[-1] * decoder_conv_t_kernel_size[-1] * decoder_conv_t_kernel_size[-1]))
            self.dec.add_module("decoder_unflatten", nn.Unflatten(1, (decoder_conv_t_filters[-1], decoder_conv_t_kernel_size[-1], decoder_conv_t_kernel_size[-1])))
            
            for idx, out_channels, kernel_size, stride in zip(range(count), encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides):
                in_channels = input_dim[2]
                if idx != 0:
                    in_channels = encoder_conv_filters[idx - 1]
                conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
                self.dec.add_module(f"decoder_conv_t_{idx}", conv)
                self.dec.add_module(f"leaky_relu_{count+idx}", nn.LeakyReLU())
                
            