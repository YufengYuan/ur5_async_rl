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
        action_shape,
        device,
        training_steps,
        discount=0.99,
        init_temperature=0.1,
        alpha_lr=1e-3,
        actor_lr=1e-3,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_tau=0.005,
        critic_target_update_freq=2,
        rl_latent_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        rad_offset=4,
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.rl_latent_dim = rl_latent_dim
        self.rad_offset = rad_offset
        self.training_steps = training_steps

        # modify obs_shape when rad_offset is used
        obs_shape = list(obs_shape)
        obs_shape[1] -= 2 * rad_offset
        obs_shape[2] -= 2 * rad_offset
        obs_shape = tuple(obs_shape)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr

        self.action_dim = action_shape[0]
        # nn models
        self.encoder = EncoderModel(obs_shape, self.rl_latent_dim).to(device)

        self.actor = ActorModel(self.encoder, action_shape[0]).to(device)

        self.critic = CriticModel(self.encoder, action_shape[0]).to(device)

        self.critic_target = copy.deepcopy(self.critic) # also copies the encoder instance

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

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        #for param in self.critic.parameters():
        #    print(param[0, 0, 0])
        #    break
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) \
                       - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)
            #target_Q = 0

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=False)

        # Ignore terminal transitions to enable infinite bootstrap
        # TODO: disable infinite bootstrap in environments other than DM_control
        critic_loss = torch.mean(
            (current_Q1 - target_Q) ** 2 * not_done + (current_Q2 - target_Q) ** 2 * not_done
             #(current_Q1 - target_Q) ** 2 + (current_Q2 - target_Q) ** 2
        )
        if L and step:
            L.log('train_critic/loss', critic_loss, step)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def update_actor_and_alpha(self, obs, L=None, step=None):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

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

        if L and step:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
            L.log('train_actor/entropy', entropy.mean(), step)
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)

        return actor_loss.item(), self.target_entropy, \
               entropy.mean().item(), alpha_loss.item(), \
               self.alpha.item()


    def update(self, replay_buffer, L=None, step=None):
        # regular update of SAC_RAD, sequentially augment data and train
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs = random_augment(obs)
        next_obs = random_augment(next_obs)
        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_target()

    @staticmethod
    def async_data_augment(input_queue, output_queue, device):
        # asynchronously augment data and convert it to tensor on another process
        while True:
            obs, action, reward, next_obs, not_done = input_queue.get()
            obs = utils.random_augment(obs, numpy=True)
            next_obs = utils.random_augment(next_obs, numpy=True)
            obs = torch.as_tensor(obs, device=device).float()
            action = torch.as_tensor(action, device=device)
            reward = torch.as_tensor(reward, device=device)
            next_obs = torch.as_tensor(next_obs, device=device).float()
            not_done = torch.as_tensor(not_done, device=device)
            output_queue.put(obs, action, reward, next_obs, not_done)

    @staticmethod
    def async_update(agent, tensor_queue, output_quque):
        # asynchronously update actor critic on another process
        while True:
            obs, action, reward, next_obs, not_done = tensor_queue.get()

            critic_loss = agent.update_critic(obs, action, reward, next_obs, not_done)

            if agent.num_updates % agent.actor_update_freq == 0:
                actor_loss, target_entropy, entropy, \
                alpha_loss, alpha =agent.update_actor_and_alpha(obs)
            if agent.num_updates % agent.critic_target_update_freq == 0:
                agent.soft_update_target()
            # return training statistics to main process
            if agent.num_updates % agent.actor_update_freq == 0 and \
               agent.num_updates % agent.critic_target_update_freq == 0:
                output_quque.put(
                    reward.mean().item(), critic_loss, actor_loss, \
                    target_entropy, entropy, alpha_loss, \
                    alpha, agent.num_updates
                    )
            agent.num_updates += 1


    def soft_update_target(self):
        utils.soft_update_params(
            self.critic.Q1, self.critic_target.Q1, self.critic_tau
        )
        utils.soft_update_params(
            self.critic.Q2, self.critic_target.Q2, self.critic_tau
        )
        utils.soft_update_params(
            self.critic.conv2latent, self.critic_target.conv2latent,
            self.encoder_tau
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
