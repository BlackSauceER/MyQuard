# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import Tuple, Union


class VecEnv(ABC):
    num_envs: int
    num_observations: int
    num_privileged_obs: int
    num_actions: int
    num_cenet_lantent: int
    num_obs_hist: int
    max_episode_length: int | torch.Tensor
    episode_length_buf: torch.Tensor
    rew_buf: torch.Tensor
    reset_buf: torch.Tensor
    extras: dict
    device: torch.device | str
    # cfg: dict | object

    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        pass
    @abstractmethod
    def reset(self, env_ids: Union[list, torch.Tensor]):
        pass
    @abstractmethod
    def get_observations(self) -> torch.Tensor:
        pass
    @abstractmethod
    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        pass
