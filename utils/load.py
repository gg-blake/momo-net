import struct
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_idx_images(filename):
    with open(filename, 'rb') as f:
        # Read metadata
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in {filename}")

        # Read image data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)

    return torch.tensor(data, dtype=torch.float32) / 255.0  # normalize to [0,1]


def load_idx_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in {filename}")

        # Read labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return torch.tensor(labels, dtype=torch.long)
    
def load_mnist_data():
    train_images = load_idx_images("mnist_dataset/train-images-idx3-ubyte/train-images-idx3-ubyte")
    train_labels = load_idx_labels("mnist_dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte")
    
    train_dataset = TensorDataset(train_images.unsqueeze(1), train_labels)  
    # add channel dimension: (N,1,28,28)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    return train_loader


