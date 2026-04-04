# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from rsl_rl.models import DreamWaQ
from rsl_rl.storage import RolloutBuffer

class PPO:
    def __init__(self, 
                 model: DreamWaQ, 
                 learning_rate: float = 1e-3,
                 ce_learning_rate: float = 1e-3,
                 entropy_coef: float = 0.01,
                 kl_weight: float = 1.0,
                 init_std: float = 1.0,
                 gamma: float = 0.99,
                 lamda: float = 0.95,
                 clip_param: float = 0.2,
                 use_clipped_value_loss: bool = True,
                 num_mini_batches: int = 4,
                 num_learning_epochs: int = 5,
                 schedule: str = 'fixed',
                 desired_kl: float = 0.01,
                 value_loss_coef: float = 1.0,
                 num_adaptation_module_substeps = 1,
                 max_grad_norm = 1,
                 device='cpu'):
        # === 基础配置 ===
        self.device = device
        # === 算法超参数 ===
        self.learning_rate = learning_rate
        self.ce_learning_rate = ce_learning_rate
        self.entropy_coef = entropy_coef
        self.kl_weight = kl_weight
        self.init_std = init_std
        self.gamma = gamma
        self.lamda = lamda
        self.clip_param = clip_param
        self.use_clipped_value_loss = use_clipped_value_loss
        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.value_loss_coef = value_loss_coef
        self.num_adaptation_module_substeps = num_adaptation_module_substeps
        self.max_grad_norm = max_grad_norm
        # === PPO运行组件 ===
        self.model = model.to(device)
        self.buffer = None
        self.optimizer_policy = optim.Adam(self.model.actor_critic_params(), lr=self.learning_rate)
        self.optimizer_cenet = optim.Adam(self.model.cenet_params(), lr=self.ce_learning_rate)
        self.transition = RolloutBuffer.Transition()

    def init_storage(self, num_envs, num_transitions_per_env, obs_shape, state_shape, obs_history_shape, action_shape):
        self.buffer = RolloutBuffer(num_envs, num_transitions_per_env, obs_shape, state_shape, obs_history_shape, action_shape, self.device)
    
    def switch_to_test(self):
        self.model.eval()

    def switch_to_train(self):
        self.model.train()

    def record_before_act(self, obs, obs_history, state):
        # 计算动作、价值。
        action = self.model.run_policy(obs, obs_history)
        actions_log_prob = self.model.get_action_log_prob(action)
        action_mean = self.model.get_action_mean()
        action_sigma = self.model.get_action_sigma()

        value = self.model.run_critic(state)
        # 存入动作与价值
        self.transition.values = value.detach()
        self.transition.actions = action.detach()
        self.transition.actions_log_prob = actions_log_prob.detach()
        self.transition.action_mean = action_mean.detach()
        self.transition.action_sigma = action_sigma.detach()
        # 这些信息需要在动作执行前存入暂存器（状态转移对的前一部分）
        self.transition.observations = obs.detach()
        self.transition.privileged_observations = state.detach()
        self.transition.observation_histories = obs_history.detach()
        # 从state取出速度，存入暂存器
        base_vel = state[:,45:48]
        self.transition.base_vel = base_vel.detach()
        return self.transition.actions

    def record_after_act(self, rewards, dones, next_obs, infos):
        # 这些信息需要在动作执行后存入（状态转移对的后一部分）
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.next_observations = next_obs
        # Bootstrapping on time-outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # 将完整的状态转移对信息存入缓存器
        self.buffer.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(self, state):
        last_values = self.model.run_critic(state).detach()
        self.buffer.compute_returns(last_values, self.gamma, self.lamda)

    def update(self):
        mean_value_loss = 0
        mean_entropy_loss = 0
        mean_surrogate_loss = 0
        mean_recons_loss = 0
        mean_vel_loss = 0
        mean_kld_loss = 0

        generator = self.buffer.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, state_batch, obs_history_batch, actions_batch, \
                target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, base_vel_batch, dones_batch, next_obs_batch in generator:
            # ====== 训练策略 ======
            self.model.update_distribution(obs_batch, obs_history_batch)
            actions_log_prob_batch = self.model.get_action_log_prob(actions_batch)  # 这里要取旧策略对数概率
            mu_batch = self.model.get_action_mean()
            action_sigma_batch = self.model.get_action_sigma()
            entropy_batch = self.model.entropy()
            value_batch = self.model.run_critic(state_batch)
            
            # batch 维度统一
            value_batch = value_batch.squeeze(-1)
            target_values_batch = target_values_batch.squeeze(-1)
            returns_batch = returns_batch.squeeze(-1)
            advantages_batch = advantages_batch.squeeze(-1)
            old_actions_log_prob_batch = old_actions_log_prob_batch.squeeze(-1)
            dones_batch = dones_batch.squeeze(-1)

            # KL
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(action_sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(action_sigma_batch)) - 0.5, dim=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif self.desired_kl / 2.0 > kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer_policy.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
            surrogate = -advantages_batch * ratio
            surrogate_clipped = -advantages_batch * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                               1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + \
                                (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                          self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss_policy = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer_policy.zero_grad()
            loss_policy.backward()
            policy_params = [p for group in self.optimizer_policy.param_groups for p in group["params"]]
            nn.utils.clip_grad_norm_(policy_params, self.max_grad_norm)
            self.optimizer_policy.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_batch.mean().item()

            # ====== 训练CENet ======
            for epoch in range(self.num_adaptation_module_substeps):
                self.optimizer_cenet.zero_grad()

                est_vel_batch, latent_mean_batch, latent_logvar_batch, est_next_obs_batch = self.model.run_cenet(obs_history_batch)
                recons_loss = F.mse_loss(est_next_obs_batch, next_obs_batch, reduction='none').mean(-1)
                vel_loss = F.mse_loss(est_vel_batch, base_vel_batch, reduction='none').mean(-1)
                kld_loss = -0.5 * torch.sum(1 + latent_logvar_batch - latent_mean_batch ** 2 - latent_logvar_batch.exp(), dim=1)
                loss_cenet = recons_loss + vel_loss + self.kl_weight * kld_loss
                # 用 dones_batch==0 做 mask，只在“非终止转移”上训练重建/预测（避免跨 episode 的 next_obs 造成错误监督）
                valid = dones_batch == 0
                # 防止因mini-batch刚好全是终止样本而loss NaN
                if valid.any():
                    loss_cenet = loss_cenet[valid].mean()
                    loss_cenet.backward()
                    cenet_params = [p for group in self.optimizer_cenet.param_groups for p in group["params"]]
                    nn.utils.clip_grad_norm_(cenet_params, self.max_grad_norm)
                    self.optimizer_cenet.step()

                    with torch.no_grad():
                        mean_recons_loss += recons_loss[valid].mean().item()
                        mean_vel_loss += vel_loss[valid].mean().item()
                        mean_kld_loss += kld_loss[valid].mean().item()
                if not valid.any():
                    continue

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_recons_loss /= (num_updates * self.num_adaptation_module_substeps)
        mean_vel_loss /= (num_updates * self.num_adaptation_module_substeps)
        mean_kld_loss /= (num_updates * self.num_adaptation_module_substeps)

        self.buffer.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy_loss, \
            mean_recons_loss, mean_vel_loss, mean_kld_loss
