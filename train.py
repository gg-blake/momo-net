from load import load_mnist_data
from autoencoder import UNetAutoEncoder
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

batch_num = 64
channel_num = 1
height = 28
width = 28
learning_rate = 1e-4

model = UNetAutoEncoder(
    input_dim=(batch_num, channel_num, height, width), 
    output_dim=(batch_num, channel_num, height, width), 
    enc_conv_filters=[64, 128, 256], 
    dec_conv_filters=[64, 128], 
    strides=[1, 1, 1], 
    kernel_sizes=[3, 3, 3], 
    num_groups = 4
).to(device)

def train_mnist(data_loader, num_iters):
    # Example: iterate over batches
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    count = 0
    for y_true, labels in data_loader:
        if count >= num_iters:
            break
        optimizer.zero_grad()
        
        y_pred = model(y_true.to(device)).to(device)
        loss = torch.mean((y_true.to(device) - y_pred)**2)
        print(f"Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        count += 1
        

    
if __name__ == "__main__":
    data = load_mnist_data()
    train_mnist(data, 1000)