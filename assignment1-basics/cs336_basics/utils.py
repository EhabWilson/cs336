import torch
import math
from einops import rearrange, einsum

def softmax(x: torch.Tensor, dim: int):
    x = x - torch.max(x, dim=dim, keepdim=True).values
    x = torch.exp(x)
    x = x / torch.sum(x, dim=dim, keepdim=True)
    return x

def scaled_dot_product_attention(queries, keys, values, mask=None):
    # (batch_size, ..., seq_len, d_k), (batch_size, ..., seq_len, d_v), (seq_len, seq_len)
    seq_len, d_k = keys.shape[-2:]
    x = einsum(queries, keys, "... q d_k, ... k d_k -> ... q k") / math.sqrt(d_k)
    
    if mask is not None:
        x.masked_fill_(mask, -torch.inf)

    x = softmax(x, -1)
    return einsum(x, values, "... q k, ... k d_v -> ... q d_v")