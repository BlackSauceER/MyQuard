import torch
import torch.nn as nn
from torch import Tensor

from typing import Any, Dict, Optional, Callable, List, Iterable, Sequence

from rsl_rl.modules import ActorNet, CriticNet, CEEncoder, CEDecoder
from rsl_rl.utils.torch_utils import MultivariateGaussianDiagonalCovariance


class DreamWaQ(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 state_dim: int,
                 action_dim: int,
                 history_len: int,
                 latent_dim: int,
                 init_weights: bool = True,
                 init_noise_std: float = 1.0,
                 actor_hidden_dims: Sequence[int] = (512, 256, 128),
                 critic_hidden_dims: Sequence[int] = (512, 256, 128),
                 ce_encoder_hidden_dims: Sequence[int] = (128, 64),
                 ce_decoder_hidden_dims: Sequence[int] = (64, 128),
                 activation: str = "elu"):
        super().__init__()

        # === Actor-Critic ===
        # actor输入obs，输出action mean
        self.actor = ActorNet(
            input_dim = obs_dim + 3 + latent_dim,
            action_dim = action_dim,
            hidden_dims = actor_hidden_dims,
            activation_name = activation,
            init_weights = init_weights
        )
        # critic输入state，估计value
        self.critic = CriticNet(
            input_dim = state_dim,
            hidden_dims = critic_hidden_dims,
            activation_name = activation,
            init_weights = init_weights
        )

        # CENet
        # encoder输入时序感知，输出v和z
        self.ce_encoder = CEEncoder(
            input_dim =obs_dim * history_len,
            output_dim = latent_dim + 3,    # z和v
            hidden_dims = ce_encoder_hidden_dims,
            activation_name = activation,
            init_weights = init_weights
        )
        # decoder输入v和z，预测下一时间步观测。
        self.ce_decoder = CEDecoder(
            input_dim = latent_dim + 3,     # z和v
            output_dim = obs_dim,
            hidden_dims = ce_decoder_hidden_dims,
            activation_name = activation,
            init_weights = init_weights
        )

        self.action_distribution = MultivariateGaussianDiagonalCovariance(action_dim, init_noise_std)

    def forward(self, obs, history, state):
        vel, latent_mean, latent_logvar, latent = self.ce_encoder(history)
        code = torch.cat((vel, latent), dim=-1)
        next_obs = self.ce_decoder(code)

        policy_input = torch.cat((obs, code.detach()), dim=-1)   # obs.shape == (B, obs_dim) 且 code.shape == (B, latent_dim+3)
        action_mean = self.actor(policy_input)
        value = self.critic(state)

        self.action_distribution.update(action_mean)
        action = self.action_distribution.sample()

        return vel, latent_mean, latent_logvar, latent, next_obs, action, value

    def run_critic(self, state) -> Tensor:
        value = self.critic(state)
        return value

    def run_policy(self, obs, history) -> Tensor:
        vel, latent_mean, latent_logvar, latent = self.ce_encoder(history)
        code = torch.cat((vel, latent), dim=-1)
        policy_input = torch.cat((obs, code.detach()), dim=-1)  # obs.shape == (B, obs_dim) 且 code.shape == (B, latent_dim+3)
        action_mean = self.actor(policy_input)
        self.action_distribution.update(action_mean)
        action = self.action_distribution.sample()
        return action

    def run_cenet(self, history):
        vel, latent_mean, latent_logvar, latent = self.ce_encoder(history)
        code = torch.cat((vel, latent), dim=-1)
        next_obs = self.ce_decoder(code)
        return vel, latent_mean, latent_logvar, next_obs
    
    def update_distribution(self, obs, history):
        with torch.no_grad():
            vel, latent_mean, latent_logvar, latent = self.ce_encoder(history)
        code = torch.cat((vel, latent), dim=-1)
        policy_input = torch.cat((obs, code.detach()), dim=-1)  # obs.shape == (B, obs_dim) 且 code.shape == (B, latent_dim+3)
        action_mean = self.actor(policy_input)
        self.action_distribution.update(action_mean)
        
    def act_inference(self, obs, history):
        vel, latent_mean, latent_logvar, latent = self.ce_encoder(history)
        code = torch.cat((vel, latent), dim=-1)
        policy_input = torch.cat((obs, code.detach()), dim=-1)  # obs.shape == (B, obs_dim) 且 code.shape == (B, latent_dim+3)
        action_mean = self.actor(policy_input)
        return action_mean

    def get_action_log_prob(self, action):
        return self.action_distribution.get_actions_log_prob(action)

    def get_action_mean(self):
        return self.action_distribution.mean

    def get_action_sigma(self):
        return self.action_distribution.stddev

    def entropy(self):
        return self.action_distribution.entropy().sum(dim=-1)

    def actor_critic_params(self, recurse: bool = True):
        return list(self.actor.parameters(recurse)) + list(self.critic.parameters(recurse)) + [self.action_distribution.std]

    def cenet_params(self, recurse: bool = True):
        return list(self.ce_encoder.parameters(recurse)) + list(self.ce_decoder.parameters(recurse))