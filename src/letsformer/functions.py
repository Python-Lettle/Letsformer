import torch
import math
from einops import einsum
from torch import Tensor
from jaxtyping import Int, Float , Bool
from collections.abc import Iterable

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

def cross_entropy_loss(logits: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """
    计算交叉熵损失

    Args:
        logits (Float[Tensor, "batch_size vocab_size"]): logits[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """    
    # 1. 计算 log_softmax（数值稳定版本）
    # 减去最大值避免指数计算溢出（使用 dim=-1 支持任意维度输入）
    logits_max = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - logits_max
    
    # 计算 log_sum_exp
    exp_logits = torch.exp(shifted_logits)
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)
    log_sum_exp = torch.log(sum_exp)
    
    # 计算 log_softmax: log(exp(shifted_logits) / sum(exp(shifted_logits)))
    log_softmax = shifted_logits - log_sum_exp
    
    # 2. 提取目标类别的对数概率
    # 使用gather从log_softmax中收集目标位置的值
    # targets需要调整形状以便gather操作（在最后一维收集）
    targets_reshaped = targets.unsqueeze(-1)  # 形状从[batch, seq]变为[batch, seq, 1]
    
    # 在最后一维（类别维度）上收集目标值
    target_log_probs = torch.gather(log_softmax, dim=-1, index=targets_reshaped)
    target_log_probs = target_log_probs.squeeze(-1)  # 移除最后一维
    
    # 3. 计算负对数似然并求平均
    target_log_probs = -target_log_probs
    loss = target_log_probs.mean()
    
    return loss

def cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    '''
    余弦学习率调度

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    '''
    if it < warmup_iters:
        # Warm up
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        # Cosine annealing
        return min_learning_rate + (max_learning_rate - min_learning_rate) * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) / 2
    else:
        # Post annealing
        return min_learning_rate


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
):
    '''
    梯度裁剪

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.
    '''
    # 1. 对参数的梯度张量 计算其 l2-norm
    grads = [parameter.grad.view(-1) for parameter in parameters if parameter.grad is not None]
    grads = torch.cat(grads)
    l2_norms = grads.norm().item()

    # 2. 判断是否裁剪
    scale = max_l2_norm / (l2_norms + eps)
    if l2_norms > max_l2_norm:
        # 按比例缩放梯度
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
