import torch
import torch.nn as nn
from einops import rearrange, einsum


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(
            torch.zeros((out_features, in_features), dtype=dtype)
        ).to(device)
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3., b=3.)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
    
    def load_state_dict(self, weight: torch.Tensor):
        self.weight.data.copy_(weight)