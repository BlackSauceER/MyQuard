import torch
import torch.nn as nn
from torch.distributions import Normal

from typing import Any, Dict, Optional, Callable, List, Iterable


# 激活函数映射
activation_map = {
    "elu": nn.ELU,
    "celu": nn.CELU,
    "selu": nn.SELU,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}
def activation_mapping(name: str, **kwargs):
    act_cls = activation_map.get(name.lower())
    if act_cls is None:
        raise ValueError(
            f"Invalid activation '{name}', "
            f"choose from {list(activation_map.keys())}"
        )
    return act_cls(**kwargs)

# 构造MLP
def build_mlp(layer_dims: List[int], hidden_activation_function_name: str) -> nn.Sequential:
    linear_num = len(layer_dims) - 1
    layers = []
    for i in range(linear_num):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i + 1 < linear_num:
            layers.append(activation_mapping(hidden_activation_function_name))

    return nn.Sequential(*layers)

# 初始化权重和偏置。我其实不是很懂（
def init_ortho(seq: nn.Sequential, hidden_gain: float, out_gain: float):
    linears = [m for m in seq.modules() if isinstance(m, nn.Linear)]
    for i, m in enumerate(linears):
        gain = out_gain if i == len(linears) - 1 else hidden_gain
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0.0)

# VAE 的重参数化：让随机化能够传递梯度。
def reparameterise(mean, logvar):
    var = torch.exp(logvar * 0.5)
    code_temp = torch.randn_like(var)
    code = mean + var * code_temp
    return code


class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None
        Normal.set_default_validate_args = False

    def update(self, logits):
        std = self.std.to(logits.device).expand_as(logits)
        self.distribution = Normal(logits, std)

    def sample(self):
        samples = self.distribution.sample()
        return samples

    def get_actions_log_prob(self, actions):
        actions_log_prob = self.distribution.log_prob(actions).sum(dim=-1)
        return actions_log_prob

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std

    @property
    def stddev(self):
        return self.distribution.stddev

    @property
    def mean(self):
        return self.distribution.mean