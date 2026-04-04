import math

import torch
import torch.nn as nn

from typing import Any, Dict, Optional, Callable, List, Iterable, Sequence

from rsl_rl.utils.torch_utils import activation_mapping, build_mlp, init_ortho, reparameterise


# Actor-Critic基础组件
class CriticNet(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int] = (512, 256, 128),
                 activation_name: str = "elu",
                 init_weights: bool = True):
        super().__init__()

        # ===== V网络：MLP =====
        self.v_net = build_mlp(
            layer_dims= [input_dim] + list(hidden_dims) + [1],
            hidden_activation_function_name = activation_name
        )

        # 用神奇的技巧初始化权重和偏置
        if init_weights:
            init_ortho(
                self.v_net,
                hidden_gain = math.sqrt(2.0),
                out_gain = 1.0
            )

    def forward(self, state):
        est_v = self.v_net(state)
        return est_v


class ActorNet(nn.Module):
    def __init__(self,
                 input_dim: int,
                 action_dim: int = 12,
                 hidden_dims: Sequence[int] = (512, 256, 128),
                 activation_name: str = "elu",
                 init_weights: bool = True):
        super().__init__()

        # ===== 策略网络：MLP =====
        self.policy_net = build_mlp(
            layer_dims = [input_dim] + list(hidden_dims) + [action_dim],
            hidden_activation_function_name = activation_name
        )

        # 用神奇的技巧初始化权重和偏置
        if init_weights:
            init_ortho(
                self.policy_net,
                hidden_gain = math.sqrt(2.0),
                out_gain = 0.01
            )

    def forward(self, policy_input):
        action_mean = self.policy_net(policy_input)
        return action_mean

