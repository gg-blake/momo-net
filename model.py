import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import random
import os

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

    exit()


device = torch.device("mps")

class ConvDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_size, embedding_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ffwd = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, device=device),
            nn.GroupNorm(group_size, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, device=device),
            nn.GroupNorm(group_size, out_channels),
            nn.SiLU(),
        )
        self.alpha = nn.Linear(embedding_size, out_channels) # shift (FiLM)
        self.beta = nn.Linear(embedding_size, out_channels) # scale (FiLM)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift = self.alpha(t)
        scale = self.beta(t)
        ffwd: torch.Tensor = self.ffwd(x)
        ffwd = ffwd * scale[:, :, None, None]
        ffwd = ffwd + shift[:, :, None, None]
        return ffwd

class ConvUpBlock(ConvDownBlock):
    def __init__(self, in_channels, out_channels, group_size, embedding_size: int):
        super().__init__(in_channels, out_channels, group_size, embedding_size)
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward_up(self, x: torch.Tensor, down_x: torch.Tensor, t: torch.Tensor):
        up_sample = self.up_conv(x)
        height_diff_half = down_x.shape[-2] - up_sample.shape[-2]
        crop_down_x = down_x[:, :, height_diff_half: height_diff_half + up_sample.shape[-2], height_diff_half: height_diff_half + up_sample.shape[-2]]
        cat = torch.cat([crop_down_x, up_sample], dim=-3)
        out = self.forward(cat, t)
        return out
        
    def __call__(self, x: torch.Tensor, down_x: torch.Tensor, t: torch.Tensor):
        return self.forward_up(x, down_x, t)

class UNet(nn.Module):
    def __init__(self, group_size: int, embedding_num: int, embedding_size: int):
        super().__init__()
        self.down_blocks = nn.ModuleList([
            ConvDownBlock(1, 64, group_size, embedding_size),
            *[ConvDownBlock(2**i, 2**i * 2, group_size, embedding_size) for i in range(6, 10)]
        ])
        self.up_blocks = nn.ModuleList([ConvUpBlock(2**(10 - i), 2**(10 - i) // 2, group_size, embedding_size) for i in range(4)])
        self.ffwd_out = nn.Sequential(
            nn.Conv2d(64, 1, 1, device=device),
            nn.SiLU(),
        )
        self.time_embeddings = nn.Embedding(embedding_num, embedding_size)
        

    def forward(self, x: torch.Tensor, idx: torch.Tensor):
        t = self.time_embeddings(idx)
        max_pool = nn.MaxPool2d(2)
        down_outs = []
        for block in self.down_blocks:
            if len(down_outs) == 0:
                down_outs.append(block(x, t))
                continue
                
            out = block(max_pool(down_outs[-1]), t)
            down_outs.append(out)
            
        up_out = None
        for block in self.up_blocks:
            if up_out is None:
                up_out = block(down_outs.pop(), down_outs.pop(), t)
                continue
                
            up_out = block(up_out, down_outs.pop(), t)
            
        logits = self.ffwd_out(up_out)
        return logits
        
total_steps = 32
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, total_steps).to(device)  # e.g., 1e-4 → 0.02
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0).to(device)


group_size = 32
channel_size = 512
embedding_size = 256
embedding_num = total_steps 

learning_rate = 1e-4
batch_size = 10
training_iters = 100

        
def sample(x0, t, noise):
    return (
        torch.sqrt(alpha_bars[t])[:, None, None, None] * x0 +
        torch.sqrt(1 - alpha_bars[t])[:, None, None, None] * noise
    )

def train(iters: int, source_img: torch.Tensor):
    model = UNet(group_size, embedding_num, embedding_size).to(device)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for i in range(iters):
        optimizer.zero_grad()
        t = torch.randint(0, total_steps, (batch_size,)).to(device)
        noise = torch.normal(torch.zeros(source_img.shape), torch.ones(source_img.shape)).to(device)
        x_t = sample(source_img, t, noise).to(device)
        estimated_noise = model(x_t, t)
        loss = torch.nn.functional.mse_loss(noise, estimated_noise)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")
       
def find_all_image_paths(dirname: str):
    png_paths = []
    for dirpath, _, filenames in os.walk(dirname):
        for filename in filenames:
            if filename.lower().endswith('.jpg'):
                png_paths.append(os.path.join(dirpath, filename))
    return png_paths
  
def load_image_grayscale_tensor(path: str):
    grayscale = transforms.Grayscale(num_output_channels=1)
    img = Image.open(path)
    grayscale_img = grayscale(img)
    transform = transforms.ToTensor()
    tensor = transform(grayscale_img).to(device)
    return tensor
        
def load_images_batch(batch_size: int, width: int, height: int):
    paths = find_all_image_paths(os.path.join(os.getcwd(), "archive/images/Images"))
    batch_img_tensor = []
    
    while len(batch_img_tensor) < batch_size:
        idx = random.randint(0, len(paths))
        img_tensor = load_image_grayscale_tensor(paths[idx])
        img_width = img_tensor.shape[2]
        img_height = img_tensor.shape[1]
        if img_width < width or img_height < height: continue
        if img_width > width:
            diff_x = (img_width - width) // 2
            img_tensor = img_tensor[:, :, diff_x:diff_x + width]
        if img_height > height:
            diff_y = (img_height - height) // 2
            img_tensor = img_tensor[:, diff_y:diff_y + height, :]
        
        batch_img_tensor.append(img_tensor)
        
    batch_tensor = torch.stack(batch_img_tensor).to(device)
    return batch_tensor
        



if __name__ == "__main__":
    batch = load_images_batch(batch_size, 512, 512)
    train(training_iters, batch)
    