import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import tiktoken
from tqdm import tqdm
from vae import VAE_Encoder, VAE_Decoder
from diffusion import Diffusion
from clip import CLIP
from ddpm import DDPMSampler
from dataset.redcaps.download import load_images
from utils.mps import get_mps_device

WIDTH = 512
HEIGHT = 512
VAE_N_LAYERS = 3
LATENT_WIDTH = WIDTH // 2**VAE_N_LAYERS
LATENT_HEIGHT = HEIGHT // 2**VAE_N_LAYERS
VAE_EMBED_SIZE = 4
BATCH_SIZE = 1
TIME_EMBED_SIZE = 320
TOKEN_SEQUENCE_LENGTH = 77

# ATTENTION HYPERPARAMS
HEAD_SIZE = 512
ATTENTION_EMBED_SIZE = 512
HEAD_COUNT = 8
DROPOUT = 0.1

# CLIP HYPERPARAMS
CLIP_EMBED_SIZE = 768
LAYER_COUNT = 12


def load_images_batch(
    dataset_path: str, batch_size: int, width: int, height: int, type: str = "rgb"
):
    norm = None
    if type == "rgb":
        norm = (0.5, 0.5, 0.5)
    elif type == "grayscale":
        norm = (0.5,)
    else:
        raise ValueError(f"Unrecognized image type: {type}")

    transform = transforms.Compose(
        [
            transforms.Resize((width, height)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm, std=norm),
        ]
    )

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


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
    images: torch.Tensor,
    prompts: list[str],
    neg_prompts: list[str],
    tokenizer: tiktoken.Encoding,
    latent_encoder: VAE_Encoder,
    latent_decoder: VAE_Decoder,
    diffusion: Diffusion,
    clip: CLIP,
    sampler: DDPMSampler,
    device: torch.device,
    idle_device: torch.device,
    do_cfg: bool = True,
    cfg_scale: float = 7.5,
    max_sequence_length: int = 77,
    strength: float = 0.8,
    seed: int | None = None,
):
    images = images.to(device)

    clip = clip.to(device)
    tokens = encode_batch_with_padding(
        tokenizer, prompts, max_length=max_sequence_length
    )
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    context = clip(tokens)
    if do_cfg:
        unconditioned_tokens = encode_batch_with_padding(
            tokenizer, neg_prompts, max_length=max_sequence_length
        )
        unconditioned_tokens = torch.tensor(
            unconditioned_tokens, dtype=torch.long, device=device
        )
        unconditioned_context = clip(unconditioned_tokens)
        context = torch.cat([context, unconditioned_context])

    clip = clip.to(idle_device)
    latent_encoder = latent_encoder.to(device)
    timesteps = sampler.timesteps

    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    latent_shape = (BATCH_SIZE, VAE_EMBED_SIZE, LATENT_HEIGHT, LATENT_WIDTH)
    encoder_noise = torch.randn(latent_shape, generator=generator, device=device)

    latent_images = latent_encoder(images, encoder_noise)

    for timestep in tqdm(timesteps):
        time_embedding = get_time_embedding(timestep, TIME_EMBED_SIZE)
        model_input = latent_images
        if do_cfg:
            model_input = model_input.repeat(2, 1, 1, 1)

        model_output = diffusion(latent_images, context, time_embedding)

        if do_cfg:
            conditioned_output, unconditioned_output = model_output.chunk(2, dim=0)
            model_output = (
                cfg_scale * (conditioned_output - unconditioned_output)
                + conditioned_output
            )

        latent_images = sampler.step(model_input, model_output)

    diffusion = diffusion.to(idle_device)

    latent_decoder = latent_decoder.to(device)
    images = latent_decoder(latent_images)
    images = images.to("cpu", torch.uint8).numpy()
    return images


def get_time_embedding(timestep: int, embed_size: int):
    half_size = embed_size // 2
    freqs = torch.pow(
        10000, -torch.arange(start=0, end=half_size, dtype=torch.float32) / half_size
    )
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


device = get_mps_device()
idle_device = torch.device("cpu")

if __name__ == "__main__":
    dataloader = load_images(500, 1, 512, 512)
    batch = next(iter(dataloader))
    captions = batch["caption"]
    images = batch["image"].squeeze(2)

    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab

    encoder = VAE_Encoder(
        image_channels=3,  # RGB
        in_channels=128,
        out_channels=VAE_EMBED_SIZE,  # 8
        n_layers=VAE_N_LAYERS,  # 3
        num_groups=32,  # Default: 32 (for group norm)
        head_size=HEAD_SIZE,
        head_count=HEAD_COUNT,
        dropout=DROPOUT,
    )

    decoder = VAE_Decoder(
        in_channels=VAE_EMBED_SIZE,
        out_channels=512,
        image_channels=3,  # RGB
        num_groups=32,  # Default: 32 (for group norm)
        head_size=HEAD_SIZE,
        head_count=HEAD_COUNT,
        dropout=DROPOUT,
    )

    diffusion = Diffusion(
        embed_size=4,
        context_size=TOKEN_SEQUENCE_LENGTH,
        head_size=HEAD_SIZE,
        head_count=HEAD_COUNT,
        time_embedding_size=320,
        n_unet_reductions=4,
        num_groups=32,
    )

    clip = CLIP(
        head_size=HEAD_SIZE,
        dropout=DROPOUT,
        head_count=HEAD_COUNT,
        vocab_size=vocab_size,
        embed_size=CLIP_EMBED_SIZE,
        token_count=TOKEN_SEQUENCE_LENGTH,
        layer_count=LAYER_COUNT,
    )

    ddpm_sampler = DDPMSampler()

    new_image = generate(
        images,
        captions,
        captions,
        tokenizer,
        encoder,
        decoder,
        diffusion,
        clip,
        ddpm_sampler,
        device,
        idle_device,
    )

    print(new_image)
