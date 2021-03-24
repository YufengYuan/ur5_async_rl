import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from models import EncoderModel, ActorModel, CriticModel
import copy
from utils import random_augment
import time
import os



class SacRadAgent:
    """SAC+AE algorithm."""
    def __init__(
        self,
        obs_shape,
        state_shape,
        action_shape,
        device,
        training_steps,
        net_params,
        discount=0.99,
        init_temperature=0.1,
        alpha_lr=1e-3,
        actor_lr=1e-3,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_tau=0.005,
        rad_offset=(4, 4),
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.rad_offset = rad_offset
        self.training_steps = training_steps

        # modify obs_shape when rad_offset is used
        # TODO: better way to do this?
        if len(obs_shape) == 3:
            obs_shape = list(obs_shape)
            obs_shape[1] -= 2 * rad_offset[0]
            obs_shape[2] -= 2 * rad_offset[1]
            obs_shape = tuple(obs_shape)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr

        self.action_dim = action_shape[0]
        # nn models
        #self.encoder = EncoderModel(obs_shape, self.rl_latent_dim).to(device)

        self.actor = ActorModel(obs_shape, state_shape, action_shape[0], net_params).to(device)

        self.critic = CriticModel(obs_shape, state_shape, action_shape[0], net_params).to(device)

        self.critic_target = copy.deepcopy(self.critic) # also copies the encoder instance

        if hasattr(self.actor.encoder, 'convs'):
            self.actor.encoder.convs = self.critic.encoder.convs

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self.num_updates = 0

        # optimizers
        self.init_optimizers()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def share_memory(self):
        self.actor.share_memory()
        self.critic.share_memory()
        self.critic_target.share_memory()
        self.log_alpha.share_memory_()

    def init_optimizers(self):
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr, betas=(0.9, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, betas=(0.9, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.alpha_lr, betas=(0.5, 0.999)
        )



    @property
    def alpha(self):
        return self.log_alpha.exp()

    def sample_action(self, obs, state, deterministic=False):
        if obs is not None:
            c, h, w = obs.shape
            obs = obs[:, self.rad_offset[0]: h - self.rad_offset[0], self.rad_offset[1]: w - self.rad_offset[1]]

        with torch.no_grad():
            if obs is not None:
                obs = torch.FloatTensor(obs).to(self.device)
                obs = obs.unsqueeze(0)
            if state is not None:
                state = torch.FloatTensor(state).to(self.device)
                state = state.unsqueeze(0)
            mu, pi, _, _ = self.actor(
                obs, state,  compute_pi=True, compute_log_pi=False
            )
            if deterministic:
                return mu.cpu().data.numpy().flatten()
            else:
                return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, state, action, reward, next_obs, next_state, not_done):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs, next_state)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_state, policy_action)
            target_V = torch.min(target_Q1, target_Q2) \
                       - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, state, action, detach_encoder=False)

        # Ignore terminal transitions to enable infinite bootstrap
        # TODO: disable infinite bootstrap in environments other than DM_control
        critic_loss = torch.mean(
            (current_Q1 - target_Q) ** 2 * not_done + (current_Q2 - target_Q) ** 2 * not_done
             #(current_Q1 - target_Q) ** 2 + (current_Q2 - target_Q) ** 2
        )
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        critic_stats = {
            'train_critic/loss': critic_loss.item()
        }

        return critic_stats

    def update_actor_and_alpha(self, obs, state, L=None, step=None):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, state ,detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, state, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        actor_stats = {
            'train_actor/loss': actor_loss.item(),
            'train_actor/target_entropy': self.target_entropy.item(),
            'train_actor/entropy': entropy.mean().item(),
            'train_alpha/loss': alpha_loss.item(),
            'train_alpha/value': self.alpha.item(),
        }
        return actor_stats

    def update(self, replay_buffer):
        # regular update of SAC_RAD, sequentially augment data and train
        obs, state, action, reward, next_obs, next_state, not_done = replay_buffer.sample()
        if obs is not None:
            obs = random_augment(obs, self.rad_offset)
            next_obs = random_augment(next_obs, self.rad_offset)
        stats = self.update_critic(obs, state, action, reward, next_obs, next_state, not_done)
        if self.num_updates % self.actor_update_freq == 0:
            actor_stats = self.update_actor_and_alpha(obs, state)
            stats = {**stats, **actor_stats}
        if self.num_updates % self.critic_target_update_freq == 0:
            self.soft_update_target()
        stats['train/batch_reward'] = reward.mean().item()
        return stats

    # TODO: modify asynchronous augmnentation and update for multimodal observations
    @staticmethod
    def async_data_augment(agent, input_queue, output_queue):
        # asynchronously augment data and convert it to tensor on another process
        device = agent.device
        while True:
            obs, state, action, reward, next_obs, next_state, not_done = input_queue.get()
            if obs is not None:
                obs = utils.random_augment(obs, agent.rad_offset, numpy=True)
                next_obs = utils.random_augment(next_obs, agent.rad_offset, numpy=True)
            obs = torch.as_tensor(obs, device=device).float()
            state = torch.as_tensor(state, device=device)
            action = torch.as_tensor(action, device=device)
            reward = torch.as_tensor(reward, device=device)
            next_obs = torch.as_tensor(next_obs, device=device).float()
            next_state = torch.as_tensor(next_state, device=device)
            not_done = torch.as_tensor(not_done, device=device)
            output_queue.put(obs, state, action, reward, next_obs, next_state, not_done)

    #@staticmethod
    def async_update(self, tensor_queue, output_quque):
        # asynchronously update actor critic on another process
        while True:
            obs, state, action, reward, next_obs, next_state, not_done = tensor_queue.get()

            stats = self.update_critic(obs, state, action, reward, next_obs, next_state, not_done)
            if self.num_updates % self.actor_update_freq == 0:
                actor_stats = self.update_actor_and_alpha(obs, state)
                stats = {**stats, **actor_stats}
            if self.num_updates % self.critic_target_update_freq == 0:
                self.soft_update_target()
            stats['train/batch_reward'] = reward.mean().item()
            stats['train/num_updates'] = self.num_updates
            output_quque.put(stats)
            self.num_updates += 1

        #critic_loss = agent.update_critic(obs, state, action, reward, next_obs, next_state, not_done)

        #    if agent.num_updates % agent.actor_update_freq == 0:
        #        actor_loss, target_entropy, entropy, \
        #        alpha_loss, alpha =agent.update_actor_and_alpha(obs, state)
        #    if agent.num_updates % agent.critic_target_update_freq == 0:
        #        agent.soft_update_target()
        #    # return training statistics to main process
        #    if agent.num_updates % agent.actor_update_freq == 0 and \
        #       agent.num_updates % agent.critic_target_update_freq == 0:
        #        output_quque.put(
        #            reward.mean().item(), critic_loss, actor_loss, \
        #            target_entropy, entropy, alpha_loss, \
        #            alpha, agent.num_updates
        #            )


    def soft_update_target(self):
        utils.soft_update_params(
            self.critic.Q1, self.critic_target.Q1, self.critic_tau
        )
        utils.soft_update_params(
            self.critic.Q2, self.critic_target.Q2, self.critic_tau
        )
        utils.soft_update_params(
            self.critic.encoder, self.critic_target.encoder,
            self.encoder_tau
        )

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
