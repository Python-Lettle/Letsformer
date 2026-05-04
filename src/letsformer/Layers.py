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

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        '''
            Construct the RMSNorm module.
            接收参数：
                d_model: int Hidden dimension of the model
                eps: float = 1e-5 Epsilon value for numerical stability
                device: torch.device | None = None Device to store the parameters on
                dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        # gaim parameter
        self.weights = nn.Parameter(torch.ones(d_model, device=device))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
            处理输入形状为 (batch_size, sequence_length, d_model) 的输入 x，并且返回一个相同形状的 Tensor
        '''
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        rms: Tensor = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        result = x * self.weights / rms
        
        return result.to(in_dtype)

class PWFFN(nn.Module):
    '''
        PWFFN --- Position-Wise Feed-Forward Network
        由 SiLU 激活函数和 GLU 组成的 SwiGLU 前馈网络
    '''
    def __init__(self, d_ff: int, d_model: int, device=None, weights: list[Tensor] | None = None):
        super().__init__()

        if weights:
            self.W1: Float[Tensor, " d_ff d_model"] = nn.Parameter(
                weights[0], requires_grad=True)
            self.W2: Float[Tensor, " d_model d_ff"] = nn.Parameter(
                weights[1], requires_grad=True)
            self.W3: Float[Tensor, " d_ff d_model"] = nn.Parameter(
                weights[2], requires_grad=True)
        else:
            self.W1: Float[Tensor, " d_ff d_model"] = nn.Parameter(
                torch.empty(d_ff, d_model, device=device), requires_grad=True)
            self.W2: Float[Tensor, " d_model d_ff"] = nn.Parameter(
                torch.empty(d_model, d_ff, device=device), requires_grad=True)
            self.W3: Float[Tensor, " d_ff d_model"] = nn.Parameter(
                torch.empty(d_ff, d_model, device=device), requires_grad=True)
    
    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        '''
            FFN(x) = SwiGLU(x, w1, w2, w3) = w2( SiLU(w1 * x) ⊙ w3 * x) )
        '''
        w1x = einsum(self.W1, x, "... d_ff d_model, ... d_model -> ... d_ff")    # Shape: [... d_ff]
        w3x = einsum(self.W3, x, "... d_ff d_model, ... d_model -> ... d_ff")    # Shape: [... d_ff]
        silu_result = silu(w1x)                         # SiLU(w1 * x)  Shape: [... d_ff]
        # FFNx = self.W2 @ (silu.mul(w3x))              # 
        FFNx = einsum(self.W2, silu_result.mul(w3x), "... d_model d_ff, ... d_ff -> ... d_model")

        return FFNx
        
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, token_positions: Int[Tensor, " ... seq_len"], device=None):
        '''
            构建RoPE模块, 并根据需要创建缓冲区
            theta: float,       RoPE 的 theta 值
            d_k: int,           query 向量和 key 向量的维度
            max_seq_len: int,   输入的最大序列长度
            device: torch.device | None = None Device to store the buffer on
        '''
        super().__init__()
        
        self.register_buffer(
            "angle_cache",
            RotaryPositionalEmbedding.init_cache(max_seq_len, d_k, theta), persistent=False
        )

        self.token_positions = token_positions

    @staticmethod
    def init_cache(max_seq_len: int, d_k: int, theta: float) -> tuple[Float[torch.Tensor, "half_dim"], Float[torch.Tensor, "half_dim"]]:
        '''
            初始化 RoPE 缓冲区
            max_seq_len: int,   输入的最大序列长度
            d_k: int,           query 向量和 key 向量的维度
            theta: float,       RoPE 的 theta 值
            device: torch.device | None = None Device to store the buffer on
        '''
        # 计算 theta 值的幂次
        # theta_pow: (d_k,)
        theta_pow = theta ** (-torch.arange(0, d_k, 2) / d_k)

        # 生成 i_range: (max_seq_len, 1)
        i_range = torch.arange(max_seq_len).unsqueeze(-1)

        # 计算 freqs: (max_seq_len, d_k)
        freqs = torch.mul(theta_pow, i_range)       # freqs = theta^( -(2k-2) / d_k)

        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))

    def forward(self, x: Float[Tensor, " ... seq_len d_k"]) -> torch.Tensor:
        '''
            处理一个形状为 (..., seq_len, d_k) 的输入张量，并返回一个相同形状的张量。
            请注意，你应该能够处理具有任意数量的批量维度的 x。
            你应该假设 token 位置是一个形状为 (..., seq_len) 的 Tensor，用于指定 x 在序列维度上的标记位置。
        '''
        # 将输入按照奇偶位置切片
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # 按照 token_positions 获取相应 cos sin
        cos, sin = self.angle_cache[:, self.token_positions, :]

        # 将旋转应用在 x pair 上
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = torch.stack((x1_rot, x2_rot), dim=-1).flatten(-2)
        return result


