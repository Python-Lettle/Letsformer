import torch
from torch import Tensor
from jaxtyping import Float, Bool
import math
from einops import einsum

def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

def softmax(x: Tensor, dim=-1) -> Tensor:
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    
    # 1 计算点积并缩放
    similarity = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    scores = similarity / math.sqrt(d_k)

    # 2 应用 Mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = softmax(scores, dim=-1)

    result = einsum(attention_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")

    return result