import math

import torch
import torch.nn as nn

from typing import Any, Dict, Optional, Callable, List, Iterable, Sequence

from rsl_rl.utils.torch_utils import activation_mapping, build_mlp, init_ortho, reparameterise


# CENet基础组件
class CEEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: Sequence[int] = (128, 64),
                 activation_name: str = "elu",
                 init_weights: bool = True):
        super().__init__()

        # === 时序感知（历史感知）编码器主干 ===
        self.encoder = build_mlp(
            layer_dims = [input_dim] + list(hidden_dims),
            hidden_activation_function_name = activation_name
        )
        self.encoder.append(activation_mapping(activation_name))

        # === 速度头 ===
        self.vel_head = nn.Linear(hidden_dims[-1], 3)

        # === latent头 ===
        self.latent_mean_head = nn.Linear(hidden_dims[-1], output_dim - 3)
        self.latent_logvar_head = nn.Linear(hidden_dims[-1], output_dim - 3)

        # 用神奇的技巧初始化权重和偏置
        if init_weights:
            init_ortho(self.encoder, math.sqrt(2.0), math.sqrt(2.0))
            nn.init.orthogonal_(self.vel_head.weight, gain=1.0)
            nn.init.orthogonal_(self.latent_mean_head.weight, gain=1.0)
            nn.init.orthogonal_(self.latent_logvar_head.weight, gain=0.01)
            nn.init.constant_(self.latent_logvar_head.bias, -1.0)

    def forward(self, ce_input):
        feature = self.encoder(ce_input)

        vel = self.vel_head(feature)

        latent_mean = self.latent_mean_head(feature)
        latent_logvar = self.latent_logvar_head(feature)
        latent = reparameterise(latent_mean, latent_logvar)

        return vel, latent_mean, latent_logvar, latent

class CEDecoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: Sequence[int] = (64, 128),
                 activation_name: str = "elu",
                 init_weights: bool = True):
        super().__init__()

        # === 解码器：MLP ===
        self.decoder = build_mlp(
            layer_dims = [input_dim] + list(hidden_dims) + [output_dim],
            hidden_activation_function_name = activation_name
        )

        # 用神奇的技巧初始化权重和偏置
        if init_weights:
            init_ortho(
                self.decoder,
                hidden_gain = math.sqrt(2.0),
                out_gain=1.0
            )

    def forward(self, code):
        next_obs = self.decoder(code)
        return next_obs