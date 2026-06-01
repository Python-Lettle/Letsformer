import math
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange, einsum
from letsformer.functions import silu, scaled_dot_product_attention

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

        if weight is not None:
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
    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None, weights: Float[Tensor, " vocab_size d_model"] | None = None):
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
        if weights is not None:
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
        return self.weights[token_ids]

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
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
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
        seq_len = x.shape[-2]
        # 动态生成位置索引
        token_positions = torch.arange(seq_len, device=x.device)

        # 将输入按照奇偶位置切片
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # 按照 token_positions 获取相应 cos sin
        cos, sin = self.angle_cache[:, token_positions, :]

        # 将旋转应用在 x pair 上
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = torch.stack((x1_rot, x2_rot), dim=-1).flatten(-2)
        return result

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope_embedding=None, device=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k: int = d_model // num_heads
        self.d_v: int = self.d_k

        # 构造多头 Q K V 矩阵, 这里使用一个大矩阵进行矩阵乘法, 效率优于分为多头小矩阵
        self.W_Q = Linear(d_model, d_model, device=device)
        self.W_K = Linear(d_model, d_model, device=device)
        self.W_V = Linear(d_model, d_model, device=device)
        self.W_O = Linear(d_model, d_model, device=device)

        self.rope_embedding = rope_embedding

    def forward(self, X: Float[Tensor, " ... sequence_length d_in"]) -> Float[Tensor, " ... sequence_length d_out"]:
        seq_len = X.shape[-2]

        # 1. 线性投影 得到 Q K V (所有头在一起)
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        # 2. 变换为多头形式 (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        Q = rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        K = rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        V = rearrange(V, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)
        
        # 2.5 对 Q K 应用 RoPE (如果构建类时给出了 RotaryPositionalEmbedding)
        if self.rope_embedding:
            Q = self.rope_embedding(Q)
            K = self.rope_embedding(K)

        # 3. 使用因果编码计算缩放点积注意力
        mask = torch.tril(torch.ones(seq_len, seq_len, device=X.device))

        multi_head_output: Float[Tensor, " ... queries d_v"] = scaled_dot_product_attention(Q, K, V, mask)
        multi_head_output = rearrange(multi_head_output, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")

        output = self.W_O(multi_head_output)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope_embedding=None, device=None) -> None:
        '''
            d_model：  int，表示 Transformer 块输入的维度
            num_heads：int，表示多头自注意力中使用的头的数量
            d_ff：     int，表示位置感知前馈内层的维度。
        '''
        super().__init__()

        self.multihead_attention = MultiHeadSelfAttention(d_model, num_heads, rope_embedding=rope_embedding, device=device)
        self.ffn = PWFFN(d_ff, d_model, device=device)
        self.norm1 = RMSNorm(d_model, device=device)
        self.norm2 = RMSNorm(d_model, device=device)
        
    def forward(self, X: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        # 1. Pre-norm
        _X = self.norm1(X)
        # 2. Causal Multi-Head Self-Attention
        _X = self.multihead_attention(_X)
        # 3. X1 = X + multi_head_output
        X1 = X + _X
        # 4. Pre-norm
        __X = self.norm2(X1)
        # 5. Position-Wise Feed-Forward
        __X = self.ffn(__X)
        # 6. Output = X1 + PWFFN(X1)
        output = X1 + __X
        return output

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        max_seq_len: int,
        batch_size: int,
        weights: dict[str, Tensor] | None = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        '''
        构造 Transformer 模型

        Args:
            vocab_size (int): 
                含义：表示 Transformer 模型的词汇表大小，即模型可识别的唯一 token 数量
                作用：决定嵌入层（Embedding 模块）的维度，每个 token ID 映射为 d_model 维向量
            context_length (int): 
                含义：模型单次处理的最大 token 序列长度
                作用：限制输入序列长度，影响计算复杂度和内存占用（序列越长，计算量越大）
            d_model (int): 
                含义：模型隐藏层的维度（如512、768等）
                作用：决定嵌入向量、注意力机制及前馈网络的维度，贯穿整个 Transformer 架构
            num_layers (int): 
                含义：Transformer 块（Block）的堆叠层数
                作用：增加模型深度，提升表达能力（如4层、12层等）
            num_heads (int): 
                含义：多头注意力机制中的注意力头数量
                作用：并行学习不同表示空间的信息（如16头、20头），每个头的维度为 d_model // num_heads
            d_ff (int): 
                含义：前馈网络（FFN）中间层的维度
                作用：FFN包含两层线性变换，d_ff 通常设为 d_model 的倍数（如8/3 * d_model），用于非线性变换
            rope_theta (float): 
                含义：旋转位置编码（RoPE）的基数参数
                作用：控制位置编码的频率分量，默认值通常为10000，影响位置信息的编码方式
            max_seq_len (int): 
                含义：预计算位置编码的最大序列长度
                作用：RoPE需预计算正弦/余弦值缓冲区，此参数定义缓冲区大小以支持长序列
            batch_size (int): 
                含义：单次训练迭代中输入的样本数
                作用：影响训练效率与内存占用，较大的 batch_size 可提升硬件利用率但增加内存需求
        '''
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.device = device
        self.dtype = dtype

        # 1. Token Embedding 
        self.embedding = Embedding(vocab_size, d_model, weights=None if weights is None else weights["token_embeddings.weight"], device=device, dtype=dtype)

        # 2. Rotary Positional Embedding Layer for Transformer Blocks
        self.d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(rope_theta, self.d_k, context_length, device=device)

        # 3. Transformer Blocks
        self.blocks = nn.ModuleList()
        for num_layer in range(num_layers):
            # 3.1 Instantiate TransformerBlock
            block = TransformerBlock(d_model, num_heads, d_ff, rope_embedding=self.rope, device=device)

            # 3.2 Multi-Head Self-Attention weights
            if weights is not None:
                layer_q_proj_weight = weights[f"layers.{num_layer}.attn.q_proj.weight"]
                layer_k_proj_weight = weights[f"layers.{num_layer}.attn.k_proj.weight"]
                layer_v_proj_weight = weights[f"layers.{num_layer}.attn.v_proj.weight"]
                layer_o_proj_weight = weights[f"layers.{num_layer}.attn.output_proj.weight"]
                out_features, in_features = layer_q_proj_weight.shape

                mha = block.multihead_attention
                mha.W_Q = Linear(in_features, out_features, weight=layer_q_proj_weight)
                mha.W_K = Linear(in_features, out_features, weight=layer_k_proj_weight)
                mha.W_V = Linear(in_features, out_features, weight=layer_v_proj_weight)
                mha.W_O = Linear(in_features, out_features, weight=layer_o_proj_weight)

            # 3.4 Position-Wise Feed-Forward Network weights
                ffn = block.ffn
                state_dict = ffn.state_dict()
                state_dict["W1"] = weights[f"layers.{num_layer}.ffn.w1.weight"]
                state_dict["W2"] = weights[f"layers.{num_layer}.ffn.w2.weight"]
                state_dict["W3"] = weights[f"layers.{num_layer}.ffn.w3.weight"]
                ffn.load_state_dict(state_dict)
            
            # 3.5 RMSNorms weights
                norm1 = block.norm1
                state_dict = norm1.state_dict()
                state_dict["weights"] = weights[f"layers.{num_layer}.ln1.weight"]
                norm1.load_state_dict(state_dict)
            
                norm2 = block.norm2
                state_dict = norm2.state_dict()
                state_dict["weights"] = weights[f"layers.{num_layer}.ln2.weight"]
                norm2.load_state_dict(state_dict)
            
            # 3.6 Add block to list
            self.blocks.append(block)

        # 4. Final Norm
        self.norm = RMSNorm(d_model, eps=1e-6, device=device)
        # Load weights if provided
        if weights is not None:
            state_dict = self.norm.state_dict()
            state_dict["weights"] = weights["ln_final.weight"]
            self.norm.load_state_dict(state_dict)
        
        # 5. Output Linear Layer
        self.output = Linear(d_model, vocab_size, weight=None if weights is None else weights["lm_head.weight"], device=device)

    def forward(self, X: Float[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len vocab_size"]:
        # 1. Token Embedding
        X = self.embedding(X)
        # 2. Transformer Blocks
        for block in self.blocks:
            X = block(X)
        # 3. Final Norm
        X = self.norm(X)
        # 4. Output Embedding
        output = self.output(X)
        return output

    @torch.no_grad()
    def generate(
        self,
        X: torch.Tensor,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: int | None = None,
        eos_token_id: int | None = None,
    ):
        """
        Args:
            X: LongTensor of shape `(1, sequence_length,)` or `(sequence_length, )`.
                Input IDs to condition on when generating.
            max_new_tokens: int
                Maximum number of tokens to generate.
            temperature: float
                Temperature to use during generation.
            top_p: int
                If provided, only sample from the `top_p` vocab items (by probability).
            eos_token_id: int
                If provided, stop generation when we generate this ID.

        Returns: A LongTensor of shape (max_new_tokens,) with the generated model output.
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)

        original_sequence_length = X.size(-1)

        for _ in range(max_tokens):
            # 如果 prompt 超过 context_length，截断 prompt
            X = X[:, -self.context_length :] if X.size(1) > self.context_length else X
            # 计算 logits
            logits = self.forward(X)
            # 获取下一个 token 的 logits
            next_token_logits = logits[:, -1]
            # 应用温度缩放
            temperature_scaled_next_token_logits = next_token_logits / temperature
            # 如果提供了 top-p，只考虑 top-p 个 token
            if top_p:
                topk_values, _ = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_p, temperature_scaled_next_token_logits.size(-1)),
                )
                # 获取 top-p 个 token 中分数最高的 token 的分数
                threshold = topk_values[:, -1]
                top_p_mask = temperature_scaled_next_token_logits < threshold
                temperature_scaled_next_token_logits.masked_fill(top_p_mask, float("-inf"))
            next_token_probabilities = softmax(temperature_scaled_next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probabilities, 1)
            # 遇到 EOS token, 停止生成
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            X = torch.cat((X, next_token_id), dim=-1)
        new_token_ids = X[:, original_sequence_length:]
        return new_token_ids
