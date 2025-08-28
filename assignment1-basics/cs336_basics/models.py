import torch
import torch.nn as nn
from einops import rearrange, einsum
from collections import OrderedDict
from cs336_basics.utils import *


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=dtype)
        ).to(device)
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3., b=3.)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
    
    # def load_state_dict(self, weights: torch.Tensor):
    #     self.w.data.copy_(weights)

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), dtype=dtype)
        ).to(device)
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3., b=3.)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(d_model, dtype=dtype)
        ).to(device)
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3., b=3.)
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.sum(x**2, dim=-1) / self.d_model + self.eps)
        rms_norm = einsum(x, self.weight, "... d_model, d_model -> ... d_model") / rms[...,None]
        return rms_norm.to(in_type)

class SwiGLU(torch.nn.Module):
    def __init__(self, d_ff, d_model, device=None, dtype=None):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = x1 * torch.sigmoid(x1) * self.w3(x)
        return self.w2(x2)

class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        freq_cis = self._precompute_freq_cis(theta, d_k, max_seq_len).to(device)
        self.register_buffer('freq_cis', freq_cis, persistent=False)

    def _precompute_freq_cis(self, theta, d_k, max_seq_len):
        # cis (cos + i * sin)
        freqs = theta ** (- torch.arange(0, d_k, 2) / d_k)
        t = torch.arange(0, max_seq_len)
        freq_cis = torch.polar(torch.ones(max_seq_len, d_k//2), einsum(t, freqs, "seq_len, half_d -> seq_len half_d"))

        return freq_cis
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor:
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2])

        rotary_matrix = self.freq_cis[token_positions]  # (..., seq_len, d/2)
        x = torch.view_as_complex(x.contiguous().view(*x.shape[:-1], -1, 2)) # (..., seq_len, d/2)

        x = torch.view_as_real(x * rotary_matrix).flatten(-2)
        return x
    
class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, rope=None):
        super().__init__()

        self.output_proj = Linear(d_model, d_model)
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)

        self.rope = rope

        self.d_model = d_model
        self.nh = num_heads

    def forward(self, x, token_positions=None):
        ori_shape = x.shape # (..., seq_len, d_k)
        seq_len = ori_shape[-2]
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)   # (..., seq_len, d_k)
        q = q.reshape(*ori_shape[:-1], self.nh, -1).transpose(-3, -2)    # (..., nh, seq_len, d_k//nh)
        k = k.reshape(*ori_shape[:-1], self.nh, -1).transpose(-3, -2)    # (..., nh, seq_len, d_k//nh)
        v = v.reshape(*ori_shape[:-1], self.nh, -1).transpose(-3, -2)    # (..., nh, seq_len, d_k//nh)

        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        mask = ~torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
        atten = scaled_dot_product_attention(q, k, v, mask=mask)
        atten = atten.transpose(-3, -2).reshape(*ori_shape)

        return self.output_proj(atten)
    
class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope=None):
        super().__init__()

        self.ln1 = RMSNorm(d_model)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, rope)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_ff, d_model)
    
    def forward(self, x, token_positions=None):
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))

        return x

class Transformer(torch.nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)
        
        rope = RoPE(rope_theta, d_model//num_heads, context_length)
        self.layers = nn.Sequential(OrderedDict([
            (str(i), TransformerBlock(d_model, num_heads, d_ff, rope)) for i in range(num_layers)
        ]))

        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.token_embeddings(x)
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)

        return x