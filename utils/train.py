from load import load_mnist_data, load_portrait_data
from vae import VAE, EncoderAE, EncoderVAE, vae_loss, ae_loss
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mps import get_mps_device
device = get_mps_device()

batch_num = 32
channel_num = 3
height = 256
width = 256
learning_rate = 1e-4

model = VAE(
    input_dim=(batch_num, channel_num, height, width)
    , encoder_conv_filters = [64,64,64,64]
    , encoder_conv_kernel_size = [3,3,3,3]
    , encoder_conv_strides = [1,2,2,1]
    , decoder_conv_t_filters = [64,64,64,3]
    , decoder_conv_t_kernel_size = [3,3,3,3]
    , decoder_conv_t_strides = [1,2,2,1],
    z_dim=32,
    use_batch_norm=False,
    use_dropout=False,
    encoder=EncoderVAE
).to(device)

def train_mnist(data_loader, num_iters, loss_fn):
    print(f"Training w/ learning rate: {learning_rate}")
    # Example: iterate over batches
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    count = 0
    for y_true, labels in data_loader:
        optimizer.zero_grad()
        if y_true.shape[0] != batch_num:
            break
        
        loss = None
        if isinstance(model.enc, EncoderVAE):
            y_pred, mu, log_var = model(y_true.to(device))
            y_pred = y_pred.to(device)
            mu = mu.to(device)
            log_var = log_var.to(device)
            loss = loss_fn(y_true.to(device), y_pred, mu, log_var).mean()
        elif isinstance(model.enc, EncoderAE):
            y_pred = model(y_true.to(device)).to(device)
            loss = loss_fn(y_true.to(device), y_pred)
        
        if loss is None:
            break
        
        print(f"Loss: {loss.item()}, Iter: {count}")
        loss.backward()
        optimizer.step()
        count += 1
        
        if count >= num_iters:
            break
        
def diagnostics(y_true, y_pred, mu, log_var):
    with torch.no_grad():
        n_pixels = y_true.numel() / y_true.shape[0]  # C*H*W
        per_pixel_mse = torch.mean((y_true - y_pred)**2, dim=(1,2,3))
        per_image_mse = torch.sum((y_true - y_pred)**2, dim=(1,2,3))
        kl_per_image = -0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var), dim=1)
    
        print("per_pixel_mse mean:", per_pixel_mse.mean().item())
        print("per_image_mse mean:", per_image_mse.mean().item())
        print("kl_per_image mean:", kl_per_image.mean().item())
        print("kl_per_pixel mean:", (kl_per_image / n_pixels).mean().item())
        print("mu mean/std:", mu.mean().item(), mu.std().item())
        print("exp(log_var) mean:", torch.exp(log_var).mean().item())
        
def plot_numbers(coords, labels):
    coords = coords.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter, ticks=range(len(set(labels))), label="Label")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Points by Label")
    #plt.autoscale()
    plt.axis("equal")
    plt.show()
    
if __name__ == "__main__":
    data = load_portrait_data(batch_num)
    train_mnist(data, 100, vae_loss)
    '''pts_list = []
    label_list = []
    model.eval()
    for i in range(1000):
        sample, label = next(iter(data))
        pts, _, _ = model.enc(sample.to(device))
        pts_list.append(pts.squeeze())
        label_list.append(label)
    
    print(torch.stack(pts_list).shape, torch.concat(label_list).shape)
    plot_numbers(torch.stack(pts_list), torch.concat(label_list))
    '''
    
    sample, label = next(iter(data))
    sample = sample.to(device)
    pts, mu, log_var = model.enc(sample.to(device))
    out = model.dec(pts).to(device)
    diagnostics(sample, out, mu, log_var)
    plot_numbers(log_var, label)
    plot_numbers(mu, label)