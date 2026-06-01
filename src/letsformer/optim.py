from collections.abc import Callable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """
        初始化 AdamW 优化器
        """
        if lr < 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < betas[0] < 1.0 and 0.0 < betas[1] < 1.0:
            raise ValueError("Invalid betas: {}".format(betas))
        if not 0.0 < eps < 1.0:
            raise ValueError("Invalid eps: {}".format(eps))
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay: {}".format(weight_decay))
        
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        
        # 初始化 一阶矩和二阶矩向量
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["t"] = 1
                self.state[p]["m"] = torch.zeros_like(p.data)   # 一阶矩向量
                self.state[p]["v"] = torch.zeros_like(p.data)   # 二阶矩向量
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        # 遍历每个参数组
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            # 遍历该组中的每个参数
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]   # Get state associated with p.
                t = state['t']          # Get iteration number from the state.
                grad = p.grad.data      # Get the gradient of loss with respect to p.

                # 1. 计算第 t 次迭代的调整后的 lr
                lr_t = lr * (math.sqrt(1-betas[1]**t) / (1-betas[0]**t))

                # 2. 应用 weight decay
                if weight_decay > 0:
                    p.data.add_(-lr * weight_decay * p.data)

                # 3. 更新一阶矩和二阶矩
                # state["m"] = betas[0] * state["m"] + (1-betas[0]) * grad
                state["m"].mul_(betas[0]).add_(grad.mul(1-betas[0]))
                # state["v"].mul_(betas[1]).add_(grad.pow(2), value=1-betas[1])
                state["v"].mul_(betas[1]).add_(grad.pow(2).mul(1-betas[1]))

                p.data -= lr_t * state["m"] / (state["v"].sqrt() + eps) # Update weight tensor in-place.
                
                state["t"] = t + 1 # Increment iteration number.
        


        return loss
    
