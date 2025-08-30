# Overview
This is a custom image generation model build from scratch using Pytorch. The architecture is based of the foundational paper behind Stable Diffusion which can be found [here](https://arxiv.org/pdf/2112.10752).

# Architecture Planning
1. Variational Autoencoder (VAE)
    - Encoder
    - Decoder
2. Diffusion UNet
    - FiLM (optional)
    - Time Embeddings
3. CLIP Text Encoder

# Training Planning
1. Train using DDPM