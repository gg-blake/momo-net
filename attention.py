import torch
import torch.nn as nn
from mps import get_mps_device

device = get_mps_device()

class SelfAttentionHead(nn.Module):
    """
    A class that represents a single attention head of the transformer model architecture;
    The attention head is used to calculate the attention scores of each node in a block of tokens;
    The structure of the class mirrors the architecture specified in the Attention is All You Need paper (https://arxiv.org/abs/1706.03762)

    Attributes
    ----------
    embed_size : int
        The number of embedding dimensions
    head_size : int
        An arbitrary shared size for the query, key, and value weights
    block_size : int
        The number of tokens in a block
    dropout : nn.Dropout
        The dropout layer
    query_weights : nn.Linear
        The linear layer for the query weights
    key_weights : nn.Linear
        The linear layer for the key weights
    value_weights : nn.Linear
        The linear layer for the value weights
    """
    def __init__(self, embed_size, head_size, block_size, dropout: float = 0.0, proj_bias: bool = False):
        """
        Parameters
        ----------
        embed_size : int
            The number of embedding dimensions
        head_size : int
            An arbitrary shared size for the query, key, and value weights
        block_size : int
            The number of tokens in a block
        dropout : float
            The rate at which nodes in the network are randomly zeroed out during training to prevent overfitting
        """
        super().__init__()
        self.embed_size = embed_size # NOTE: Typically, head_size is equal to embed_size but isn't necessary
        self.query_weights = nn.Linear(embed_size, head_size, bias=proj_bias, device=device)
        self.key_weights = nn.Linear(embed_size, head_size, bias=proj_bias, device=device)
        self.value_weights = nn.Linear(embed_size, head_size, bias=proj_bias, device=device)
        self.dropout = nn.Dropout(dropout) # TODO: Use this or remove this
        # We want to apply a mask to the attention scores to prevent the model from cheating during training
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, device=device))) # Lower triangular matrix

    def forward(self, embeddings, masked=True):
        """
        Forward passes a list of embeddings through the attention head and returns the attention scores

        Parameters
        ----------
        embeddings : torch.Tensor
            The embeddings [B x T x C] where B is the batch size, T is the number of tokens in a block, and C is the number of embedding dimensions
        """
        B, T, C = embeddings.shape
        # Queries store the information of what other embeddings have in a particular block
        query = self.query_weights(embeddings)
        # Keys store the information that a particular embedding has relative to other embeddings in a block
        key = self.key_weights(embeddings)
        # By multiplying the keys and queries together, we can allow the embeddings to influence the meaning of other embeddings in the block
        # We need to sqrt(embed_size) to ensure the softmax of wei doesn't get to spiky
        wei = query @ key.transpose(-2, -1) * self.embed_size**-0.5 
        # When training a model, we don't want embeddings that are ahead of an embedding in a block to send information to it (its like cheating in a test)
        # So we will apply a mask to wei
        if masked:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # Then we apply a softmax to make the output on interval [0,1)
        wei = torch.softmax(wei, dim=-1)
        # We don't apply the embeddings directly to wei but instead we apply another backpropagatable linear layer to the embeddings (called value) and then apply wei
        value = self.value_weights(embeddings)
        return wei @ value
        
class SelfAttention(nn.Module):
    def __init__(self, embed_size: int, head_size: int, head_count: int, block_size: int, dropout: float):
        super().__init__()
        # Multiheaded attention (batched attention calculation)
        self.heads = nn.ModuleList([SelfAttentionHead(embed_size, head_size // head_count, block_size, dropout) for _ in range(head_count)])
        # Linear projection of outcome of multiheaded attention layer
        self.proj = nn.Linear(embed_size, embed_size, device=device)
        # Randomly zeros out some of the data to prevent overfitting in training
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper function that forward passes the data through the multiheaded attention layer

        Parameters
        ----------
        x : torch.Tensor
            The block of tokens [B x T x C] where B is the batch size, T is the number of tokens in a block, and C is the number of embedding dimensions
        """
        # We want to apply the multiheaded attention layer to the data so concatenate the outcomes of each head
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        # We want to recombine the outcomes together so we must project it to a layer of the right dimensions 
        # (head_count x embed_size x [embed_size // head_count]) -> (embed_size x embed_size)
        out = self.dropout(out)
        out = self.proj(out)
        return out
        
class CrossAttentionHead(nn.Module):
    """
    Single head of cross-attention: Queries come from decoder embeddings,
    Keys/Values come from encoder embeddings.
    """
    def __init__(self, q_size: int, kv_size: int, head_size: int, dropout: float = 0.0, proj_bias: bool = False):
        super().__init__()
        self.head_size = head_size
        self.query_weights = nn.Linear(q_size, head_size, bias=proj_bias, device=device)
        self.key_weights   = nn.Linear(kv_size, head_size, bias=proj_bias, device=device)
        self.value_weights = nn.Linear(kv_size, head_size, bias=proj_bias, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        queries : torch.Tensor
            Decoder embeddings [B x T_q x C]
        context : torch.Tensor
            Encoder embeddings [B x T_kv x C]
        """
        B, T_q, C = queries.shape
        T_kv = context.shape[1]

        # Linear projections
        Q = self.query_weights(queries)   # [B, T_q, head_size]
        K = self.key_weights(context)     # [B, T_kv, head_size]
        V = self.value_weights(context)   # [B, T_kv, head_size]

        # Attention weights (scaled dot product)
        wei = Q @ K.transpose(-2, -1) * self.embed_size**-0.5  # [B, T_q, T_kv]
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Weighted sum of values
        out = wei @ V  # [B, T_q, head_size]
        return out

class CrossAttention(nn.Module):
    """
    Multiheaded cross-attention layer:  
    Queries come from decoder, Keys/Values come from encoder.
    NOTE: kv_size = block_size
    """
    def __init__(self, q_size: int, kv_size: int, head_size: int, head_count: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList([
            CrossAttentionHead(q_size, kv_size, head_size // head_count, dropout) 
            for _ in range(head_count)
        ])
        self.proj = nn.Linear(q_size, q_size, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        queries : torch.Tensor
            Decoder embeddings [B x T_q x C]
        context : torch.Tensor
            Encoder embeddings [B x T_kv x C]
        """
        # Run each head and concatenate results
        out = torch.cat([head(queries, context) for head in self.heads], dim=-1)
        out = self.dropout(out)
        out = self.proj(out)
        return out
        