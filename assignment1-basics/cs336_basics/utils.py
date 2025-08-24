import torch

def softmax(x: torch.Tensor, dim: int):
    x = x - torch.max(x, dim=dim, keepdim=True).values
    x = torch.exp(x)
    x = x / torch.sum(x, dim=dim, keepdim=True)
    return x