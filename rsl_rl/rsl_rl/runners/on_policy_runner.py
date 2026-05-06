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

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import SingleGaitPPO, GaitSelectorPPO
from rsl_rl.models import DreamWaQ, GaitSelector
from rsl_rl.env import VecEnv


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 mode: str,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.selector_cfg = train_cfg["selector"]
        self.device = device
        self.env = env
        self.mode = mode

        # 初始化策略模型和PPO
        if self.mode == "gait":
            policy = DreamWaQ(
                obs_dim = self.env.num_observations,
                ce_obs_dim = self.env.num_ce_observations,
                state_dim = self.env.num_privileged_obs,
                action_dim = self.env.num_actions,
                history_len = self.env.num_obs_hist,
                latent_dim = 16,
                **self.policy_cfg
            ).to(self.device)

            self.alg = SingleGaitPPO(
                policy,
                device=self.device,
                **self.alg_cfg
            )

            self.num_steps_per_env = self.cfg["num_steps_per_env"]
            self.alg.init_storage(
                self.env.num_envs,
                self.num_steps_per_env,
                [self.env.num_observations],
                [self.env.num_ce_observations],
                [self.env.num_privileged_obs],
                [self.env.num_obs_hist * self.env.num_ce_observations],
                [self.env.num_actions]
            )

        elif self.mode == "selector":
            # TODO
            env.init_single_gait_policy(self.cfg["experiment_name"])

            policy = GaitSelector(
                obs_dim=env.num_selector_observations,
                state_dim=env.num_privileged_obs,
                gait_num=3,
                phase_cmd_len=3,
                **self.selector_cfg
            )

            self.alg = GaitSelectorPPO(
                policy,
                device=self.device,
                **self.alg_cfg
            )

            self.num_steps_per_env = self.cfg["num_steps_per_env"]
            self.alg.init_storage(
                self.env.num_envs,
                self.num_steps_per_env,
                [self.env.num_selector_observations],
                [self.env.num_privileged_obs],
                [3],
                3,
            )

        else:
            raise ValueError("Invalid mode! It must be provided as \"gait\" or \"selector\"")

        # Log
        self.save_interval = self.cfg["save_interval"]
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.mode == "gait":
            self.learn_gait(num_learning_iterations, init_at_random_ep_len)
        elif self.mode == "selector":
            self.learn_selector(num_learning_iterations, init_at_random_ep_len)
        else:
            raise ValueError("Invalid mode! It must be provided as \"gait\" or \"selector\"")

    def learn_selector(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length)
            )

        # 初始化 selector obs
        self.env.compute_observations()
        self.env.compute_selector_observations()

        obs = self.env.selector_obs_buf
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs

        obs = obs.to(self.device)
        critic_obs = critic_obs.to(self.device)

        self.alg.switch_to_train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)

        cur_reward_sum = torch.zeros(
            self.env.num_envs,
            dtype=torch.float,
            device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs,
            dtype=torch.float,
            device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.record_before_act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    self.alg.record_after_act(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                        if len(new_ids) > 0:
                            rewbuffer.extend(cur_reward_sum[new_ids].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids].cpu().numpy().tolist())

                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_entropy_loss, \
                mean_recons_loss, mean_vel_loss, mean_kld_loss = self.alg.update()

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def learn_gait(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs, obs_hist = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, obs_hist = \
        obs.to(self.device), critic_obs.to(self.device), obs_hist.to(self.device)
        self.alg.switch_to_train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    
                    actions = self.alg.record_before_act(obs, obs_hist, critic_obs)
                    #prev_critic_obs = critic_obs
                    # print("######prev_critic_obs =====",prev_critic_obs)
                    obs, privileged_obs, obs_hist, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, obs_hist, rewards, dones = \
                    obs.to(self.device), critic_obs.to(self.device), obs_hist.to(self.device), rewards.to(self.device), dones.to(self.device)
                    # print("######prev_critic_obs =====",prev_critic_obs[0,0],'\n',"#####critic_obs =====",critic_obs[0,0])
                    # print("######obs_hist =====",obs_hist[180,0],'\n',"#####obs =====",obs[0,0])
                    ce_obs = obs[:, -self.env.num_ce_observations:]
                    self.alg.record_after_act(rewards, dones, ce_obs, infos)    # 这里记录的应该是ce_obs
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss, mean_entropy_loss, \
            mean_recons_loss, mean_vel_loss, mean_kld_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.model.get_action_sigma().mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/Policy/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/Policy/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/Policy/entropy', locs['mean_entropy_loss'], locs['it'])
        self.writer.add_scalar('Loss/CENet/recons', locs['mean_recons_loss'], locs['it'])
        self.writer.add_scalar('Loss/CENet/vel_est', locs['mean_vel_loss'], locs['it'])
        self.writer.add_scalar('Loss/CENet/kl', locs['mean_kld_loss'], locs['it'])
        self.writer.add_scalar('Loss/Policy/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Loss/CENet/learning_rate', self.alg.ce_learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Recons loss:':>{pad}} {locs['mean_recons_loss']:.4f}\n"""
                          f"""{'Vel Est loss:':>{pad}} {locs['mean_vel_loss']:.4f}\n"""
                          f"""{'KL loss:':>{pad}} {locs['mean_kld_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Recons loss:':>{pad}} {locs['mean_recons_loss']:.4f}\n"""
                          f"""{'Vel Est loss:':>{pad}} {locs['mean_vel_loss']:.4f}\n"""
                          f"""{'KL loss:':>{pad}} {locs['mean_kld_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        save_dict = {
            "model_state_dict": self.alg.model.state_dict(),
            "optimizer_state_dict_policy": self.alg.optimizer_policy.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }

        if hasattr(self.alg, "optimizer_cenet"):
            save_dict["optimizer_state_dict_cenet"] = self.alg.optimizer_cenet.state_dict()

        torch.save(save_dict, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.model.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer_policy.load_state_dict(loaded_dict["optimizer_state_dict_policy"])
            if hasattr(self.alg, "optimizer_cenet") and "optimizer_state_dict_cenet" in loaded_dict:
                self.alg.optimizer_cenet.load_state_dict(loaded_dict["optimizer_state_dict_cenet"])

        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.model.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.model.to(device)
        return self.alg.model.act_inference
