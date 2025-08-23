import torch
import torch.nn as nn
from einops import rearrange, einsum


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=dtype)
        ).to(device)
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3., b=3.)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
    
    def load_state_dict(self, weight: torch.Tensor):
        self.weight.data.copy_(weight)

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.embeds = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), dtype=dtype)
        ).to(device)
        nn.init.trunc_normal_(self.embeds, mean=0, std=1, a=-3., b=3.)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeds[token_ids]
    
    def load_state_dict(self, weight: torch.Tensor):
        self.embeds.data.copy_(weight)