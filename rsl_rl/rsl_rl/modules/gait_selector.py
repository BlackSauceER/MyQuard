import math

import torch
import torch.nn as nn

from typing import Any, Dict, Optional, Callable, List, Iterable, Sequence

from rsl_rl.utils.torch_utils import build_mlp, init_ortho


class SelectorActorNet(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 gait_num: int,
                 phase_cmd_len: int,
                 hidden_dims: Sequence[int],
                 activation: str = "elu",
                 init_weights: bool = True):
        super().__init__()

        # Shared feature extractor.
        backbone_dims = [obs_dim, *hidden_dims]
        self.backbone = build_mlp(
            backbone_dims,
            hidden_activation_function_name=activation,
        )

        last_dim = hidden_dims[-1]

        # Head 1: discrete gait selection logits.
        self.gait_head = nn.Linear(last_dim, gait_num)

        # Head 2: continuous phase-generator command mean.
        self.phase_cmd_head = nn.Linear(last_dim, phase_cmd_len)

        if init_weights:
            self._init_weights()

    def forward(self, obs):
        feat = self.backbone(obs)
        gait_logits = self.gait_head(feat)
        phase_cmd_mean = self.phase_cmd_head(feat)
        return gait_logits, phase_cmd_mean

    def _init_weights(self):
        # Hidden backbone: normal PPO-style orthogonal init.
        init_ortho(self.backbone, hidden_gain=math.sqrt(2), out_gain=math.sqrt(2))

        # Gait logits head: small output, initial softmax close to uniform.
        nn.init.orthogonal_(self.gait_head.weight, gain=0.01)
        nn.init.constant_(self.gait_head.bias, 0.0)

        # Phase command head: small output, initial phase command close to zero.
        nn.init.orthogonal_(self.phase_cmd_head.weight, gain=0.01)
        nn.init.constant_(self.phase_cmd_head.bias, 0.0)