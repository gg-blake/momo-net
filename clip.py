import torch
import torch.nn as nn
from attention import SelfAttention

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, token_count: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Parameter(torch.zeros(token_count, embed_size))

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.position_embedding
        return x


class CLIP(nn.Module):
    def __init__(
        self,
        head_size: int,
        dropout: float,
        head_count: int = 12,
        vocab_size: int = 200019,
        embed_size: int = 768,
        token_count: int = 64,
        layer_count: int = 12,
    ):
        self.embedding = CLIPEmbedding(vocab_size, embed_size, token_count)
        self.layers = nn.ModuleList(
            [
                CLIPLayer(embed_size, head_size, head_count, dropout)
                for i in range(layer_count)
            ]
        )
        self.layernorm = nn.LayerNorm(embed_size)

    def forward(self, tokens: torch.Tensor) -> torch.FloatTensor:
        state = self.embedding(tokens)
        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)
        return output


class CLIPLayer(nn.Module):
    """
    A class that represents a CLIP layer that can be used in a diffusion model;
    The transformer block consists of a multiheaded attention layer and a feed forward layer;
    The multiheaded attention layer is used to calculate the attention scores of each node in a block of tokens;
    The feed forward layer is used to train the nodes to compute their attention scores individually;
    The structure of the class mirrors the architecture specified in the Attention is All You Need paper (https://arxiv.org/abs/1706.03762)

    Attributes
    ----------
    heads : nn.ModuleList
        The multiheaded attention layers
    proj : nn.Linear
        The linear projection of the outcome of the multiheaded attention layer
    dropout : nn.Dropout
        The dropout layer
    ffwd : nn.Sequential
        The feed forward layer
    layer_norm1 : nn.LayerNorm
        The layer normalization layer for the multiheaded attention layer
    layer_norm2 : nn.LayerNorm
        The layer normalization layer for the feed forward layer
    """

    def __init__(
        self,
        embed_size: int,
        head_size: int,
        head_count: int,
        dropout: float,
    ):
        """
        Parameters
        ----------
        embed_size : int
            The size of the embeddings
        head_size : int
            The size of the heads in the multiheaded attention layer
        head_count : int
            The number of heads in the multiheaded attention layer
        block_size : int
            The number of tokens in a block
        dropout : float
            The rate at which nodes in the network are randomly zeroed out during training to prevent overfitting
        """
        super().__init__()
        # Multiheaded attention (batched attention calculation)
        self.attention = SelfAttention(
            embed_size, head_size, head_count, dropout
        )
        # Linear projection of outcome of multiheaded attention layer
        self.proj = nn.Linear(embed_size, embed_size)
        # Randomly zeros out some of the data to prevent overfitting in training
        self.dropout = nn.Dropout(dropout)
        # Simple multilayered perceptron
        self.ffwd = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            QuickGELU(),  # This activation function is necessary for diffusion (normally is ReLU)
            nn.Linear(4 * embed_size, embed_size),
            self.dropout,
        )
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        """
        Forward pass of the model of a block of tokens; each block consists of a number of tokens from the training/validation data

        Parameters
        ----------
        x : torch.Tensor
            The block of tokens [B x T x C] where B is the batch size, T is the number of tokens in a block, and C is the number of embedding dimensions
        """
        # We want to ensure that our nodes across each batch dimension have mean = 0 and standard deviation = 0 before feeding to the multiheaded attention layer
        # So we want to apply whats called layer normalization
        # Here is the pytorch documentation: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html (LayerNorm)
        layer_norm = self.layer_norm1(x)
        # Both the multiheaded attention layer and feed forward layer add the in features of the layer to the out features
        # This is what is referred to as residual connections, and it solves an issue where increasingly deep networks become hard to train/optimize
        # The paper discussing the benefits of this can be found here: https://arxiv.org/abs/1512.03385 (Deep Residual Learning for Image Recognition)
        x = x + self.attention(layer_norm)
        # We also want to apply layer normalization to our attention output before passing it to the feed forward layer
        # In the original Attention is All You Need paper, layer normalization comes after each layer, but better results come from doing pre-layer normalization
        layer_norm = self.layer_norm2(x)
        # Once all the nodes in the head have their individual attention scores, we need to train the nodes to compute their attention scores individually
        # This is why we feed the data into a multilayered perceptron, which will allow the model to recognize patterns in the data
        x = x + self.ffwd(layer_norm)
        return x
