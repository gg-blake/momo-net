import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler
from tokenizer import Tokenizer
from tokenizer_ext import batch_encode, batch_decode

WIDTH = 512
HEIGHT = 512
VAE_N_LAYERS = 3
LATENT_WIDTH = WIDTH // 2**VAE_N_LAYERS
LATENT_HEIGHT = HEIGHT // 2**VAE_N_LAYERS
VAE_EMBED_SIZE = 4
BATCH_SIZE = 1
TIME_EMBED_SIZE = 320

def encode_batch_with_padding(encoding, texts, pad_token_id=0, max_length=None):
    """
    Encode a batch of texts with tiktoken and pad to equal length.
    
    Args:
        encoding (tiktoken.Encoding): tokenizer, e.g. tiktoken.get_encoding("cl100k_base")
        texts (list[str]): input strings
        pad_token_id (int): ID to pad with
        max_length (int or None): max length of output. If None, use longest sequence.
    
    Returns:
        list[list[int]]: padded token IDs
    """
    # Fast encode all texts
    encoded_batch = encoding.encode_batch(texts)
    
    # figure out padding length
    if max_length is None:
        max_length = max(len(seq) for seq in encoded_batch)
    
    # pad
    padded = [
        seq[:max_length] + [pad_token_id] * (max_length - len(seq))
        for seq in encoded_batch
    ]
    
    return padded

def generate(
    prompt: str,
    uncond_prompt: str,
    input_image: torch.Tensor | None = None,
    strength: float = 0.8,
    do_cfg: bool = True,
    cfg_scale: float = 7.5,
    sampler_name: str = "ddpm",
    n_inference_steps: int = 50,
    models={},
    seed: int | None = None,
    device = None,
    idle_device = None,
    tokenizer = None
):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be between 0 and 1")
        
        def to_idle(x):
            if idle_device:
                return x.to(idle_device)
                
            return x
            
            
            
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
            
        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            cond_tokens = encode_batch_with_padding(tokenizer, [prompt], max_length=77)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_context = clip(cond_tokens)
            
            uncond_tokens = encode_batch_with_padding(tokenizer, [uncond_prompt], max_length=77)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)
            
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = encode_batch_with_padding(tokenizer, [prompt], max_length=77)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)
            
        to_idle(clip)
        
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")
            
        latents_shape = (BATCH_SIZE, VAE_EMBED_SIZE, LATENT_HEIGHT, LATENT_WIDTH)
            
        latents = torch.randn(latents_shape, generator=generator, device=device)
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)
            
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0) # adds batch dimension
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
            
            
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            latents = encoder(input_image_tensor, encoder_noise)
            
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            
            to_idle(encoder)
            
        diffusion = models["diffusion"]
        diffusion.to(device)
        
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in timesteps:
            time_embedding = get_time_embedding(timestep, TIME_EMBED_SIZE).to(device)
            model_input = latents
            
            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)
                
            model_output = diffusion(model_input, context, time_embedding)
            
            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2, dim=-1)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
                
            latents = sampler.step(timestep, latents, model_output)
        
        to_idle(diffusion)
        
        decoder = models["decoder"]
        decoder.to(device)
        
        images = decoder(latents)
        to_idle(decoder)
        
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
                
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
        
    return x
    
def get_time_embedding(timestep: int, embed_size: int):
    half_size = embed_size // 2
    freqs = torch.pow(1e4, -torch.arange(start=0, end=half_size, dtype=torch.float32) / half_size)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    