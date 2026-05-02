import torch
from torch import Tensor

def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

def softmax(x: Tensor, dim=-1) -> Tensor:
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)
