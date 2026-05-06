import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical
from typing import Sequence, Tuple

from rsl_rl.modules import CriticNet, SelectorActorNet
from rsl_rl.utils.torch_utils import build_mlp, init_ortho, MultivariateGaussianDiagonalCovariance



class GaitSelector(nn.Module):
    """
    Gait selector module.

    Input:
        obs: [B, obs_dim]

    Output/action:
        gait_id: discrete gait choice, e.g. walk/trot/gallop
        phase_cmd: continuous phase-generator command

    Network:
        shared backbone -> gait logits head
                        -> phase command mean head

    PPO action distribution:
        gait_id   ~ Categorical(logits)
        phase_cmd ~ Normal(mean, diag_std)
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        gait_num: int,
        phase_cmd_len: int,
        hidden_dims: Sequence[int] = (256, 256, 64),
        critic_hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 0.5,
        init_weights: bool = True,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.gait_num = gait_num
        self.phase_cmd_len = phase_cmd_len

        # critic输入state，估计value
        self.critic = CriticNet(
            input_dim=state_dim,
            hidden_dims=critic_hidden_dims,
            activation_name=activation,
            init_weights=init_weights
        )

        # Actor
        self.actor = SelectorActorNet(
            obs_dim=obs_dim,
            gait_num=gait_num,
            phase_cmd_len=phase_cmd_len,
            hidden_dims=hidden_dims,
            activation=activation,
            init_weights=init_weights
        )

        # Continuous action distribution for phase command.
        self.phase_cmd_distribution = MultivariateGaussianDiagonalCovariance(
            phase_cmd_len,
            init_noise_std,
        )

        # Discrete distribution cache.
        self.gait_distribution = None


    def forward(self, obs: Tensor, state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Deterministic forward.

        Returns:
            gait_logits: [B, gait_num]
            phase_cmd_mean: [B, phase_cmd_len]
        """
        value = self.critic(state)
        gait_logits, phase_cmd_mean = self.actor(obs)
        return gait_logits, phase_cmd_mean, value

    def run_critic(self, state) -> Tensor:
        value = self.critic(state)
        return value

    def run_policy(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        gait_logits, phase_cmd_mean = self.actor(obs)
        self.gait_distribution = Categorical(logits=gait_logits)
        self.phase_cmd_distribution.update(phase_cmd_mean)
        gait_id = self.gait_distribution.sample()
        phase_cmd = self.phase_cmd_distribution.sample()
        return gait_id, phase_cmd

    def update_distribution(self, obs: Tensor):
        """
        Build current action distributions.
        Call before sample/log_prob/entropy.
        """
        gait_logits, phase_cmd_mean = self.actor(obs)
        self.gait_distribution = Categorical(logits=gait_logits)
        self.phase_cmd_distribution.update(phase_cmd_mean)

    def act_inference(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Deterministic action for play/eval.

        Returns:
            gait_id: [B], argmax gait
            phase_cmd_mean: [B, phase_cmd_len]
        """
        gait_logits, phase_cmd_mean = self.actor(obs)
        gait_id = torch.argmax(gait_logits, dim=-1)
        return gait_id, phase_cmd_mean

    def get_action_log_prob(self, gait_id: Tensor, phase_cmd: Tensor) -> Tensor:
        """
        PPO log probability of hybrid action.

        Args:
            gait_id: [B]
            phase_cmd: [B, phase_cmd_len]

        Returns:
            log_prob: [B]
        """
        gait_log_prob = self.gait_distribution.log_prob(gait_id)
        phase_log_prob = self.phase_cmd_distribution.get_actions_log_prob(phase_cmd)
        return gait_log_prob + phase_log_prob

    def entropy(self) -> Tensor:
        """
        PPO entropy of hybrid action.

        Returns:
            entropy: [B]
        """
        gait_entropy = self.gait_distribution.entropy()
        phase_entropy = self.phase_cmd_distribution.entropy().sum(dim=-1)
        return gait_entropy + phase_entropy

    def get_gait_probs(self) -> Tensor:
        """
        Debug / logging.

        Returns:
            probs: [B, gait_num]
        """
        return self.gait_distribution.probs

    def get_phase_cmd_mean(self) -> Tensor:
        return self.phase_cmd_distribution.mean

    def get_phase_cmd_sigma(self) -> Tensor:
        return self.phase_cmd_distribution.stddev

    def get_action_mean(self) -> Tensor:
        return self.get_phase_cmd_mean()

    def get_action_sigma(self) -> Tensor:
        return self.get_phase_cmd_sigma()

    def actor_critic_params(self, recurse: bool = True):
        return (
                list(self.actor.parameters(recurse=recurse))
                + list(self.critic.parameters(recurse=recurse))
                + [self.phase_cmd_distribution.std]
        )

    def enforce_minimum_phase_std(self, min_std: Tensor):
        self.phase_cmd_distribution.enforce_minimum_std(min_std)