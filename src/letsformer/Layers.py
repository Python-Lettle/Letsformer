import math
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange, einsum, reduce
from cs336_basics.functions import silu, scaled_dot_product_attention

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None, weight: Tensor | None = None):
        '''
            构建一个线性变换模块
            参数:
                in_features: int  输入数据的维度
                out_features: int  输出数据的维度
                device: torch.device | None = None  用于存储参数的 device
                dtype: torch.dtype | None = None  参数的数据类型
        '''
        super().__init__()
        # 创建可训练的权重矩阵
        std = math.sqrt(2 / (in_features + out_features))

        if weight:
            self.weights: Float[Tensor, " d_out d_in"] = nn.Parameter(weight, requires_grad=True)
        else:
            self.weights: Float[Tensor, " d_out d_in"] = nn.Parameter(
                torch.nn.init.trunc_normal_(
                    torch.empty(out_features, in_features, device=device),
                    mean = 0,
                    std = std,
                    a = -3*std,
                    b = 3*std,
                ),
                requires_grad=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''对输入应用线性变换'''
        return einsum(x, self.weights, " ... i , o i -> ... o")
        # return x @ self.W.T

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None, weights: Tensor | None = None):
        '''
            Construct an embedding module.
            参数列表:
                vocab_size: int, vocabulary 大小
                d_model: int, embedding 向量维度
                device: torch.device | None = None
                dtype: torch.dtype | None = None
        '''
        super().__init__()
        
        # 创建 embedding 矩阵
        if weights:
            self.weights = nn.Parameter(weights, requires_grad=True)
        else:
            std = 1
            self.weights = nn.Parameter(
                torch.nn.init.trunc_normal_(
                    torch.empty(vocab_size, d_model, device=device),
                    mean = 0,
                    std = std,
                    a = -3*std,
                    b = 3*std,
                ),
                requires_grad=True,
            )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        '''
            查找给定 token ID 的 embedding 向量。
            token_ids: torch.LongTensor (batch_size, sequence_length) 整数 Tensor
            生成一个 (batch_size, sequence_length, d_model) 的向量序列
        '''
        return self.weight[token_ids]




