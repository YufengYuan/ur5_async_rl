import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random
import copy
import skimage
import torch.multiprocessing as mp


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs

def random_augment(obses, rad_height, rad_width):
    n, c, h, w = obses.shape
    _h = h - 2 * rad_height
    _w = w - 2 * rad_width
    w1 = torch.randint(0, rad_width + 1, (n,))
    h1 = torch.randint(0, rad_height + 1, (n,))
    cropped_obses = torch.empty((n, c, _h, _w), device=obses.device).float()
    for i, (obs, w11, h11) in enumerate(zip(obses, w1, h1)):
        cropped_obses[i][:] = obs[:, h11:h11 + _h, w11:w11 + _w]
    return cropped_obses

def evaluate(env, agent, num_episodes, L, step, args):
    for i in range(num_episodes):
        obs = env.reset()
        #video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done:
            with eval_mode(agent):
                obs = obs[:, args.rad_offset: args.image_size + args.rad_offset, args.rad_offset: args.image_size + args.rad_offset]
                action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            #video.record(env)
            episode_reward += reward

        #video.save('%d.mp4' % step)
        L.log('eval/episode_reward', episode_reward, step)
    L.dump(step)


class BufferQueue(object):
    """Queue to transfer arbitrary number of data between processes"""
    def __init__(self, num_items, max_size=10, start_method='spawn'):
        self.max_size = max_size
        ctx = mp.get_context(start_method)
        self.queues = [ctx.Queue(max_size) for _ in range(num_items)]

    def put(self, *items):
            for queue, item in zip(self.queues, items):
                queue.put(item)

    def get(self):
        return [queue.get() for queue in self.queues]


class RadReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, state_shape, action_shape, capacity, batch_size, rad_offset, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        #obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.ignore_obs = True
        self.ignore_state = True
        if obs_shape[-1] != 0:
            self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
            self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
            self.ignore_obs = False
            self.rad_h = round(rad_offset * obs_shape[1])
            self.rad_w = round(rad_offset * obs_shape[2])
        if state_shape[-1] != 0:
            self.states = np.empty((capacity, *state_shape), dtype=np.float32)
            self.next_states = np.empty((capacity, *state_shape), dtype=np.float32)
            self.ignore_state = False
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, state, action, reward, next_obs, next_state, done):
        if not self.ignore_obs:
            self.obses[self.idx] = obs
            self.next_obses[self.idx] = next_obs
        if not self.ignore_state:
            self.states[self.idx]= state
            self.next_states[self.idx]= next_state
        self.actions[self.idx]= action
        self.rewards[self.idx]= reward
        self.not_dones[self.idx]= not done

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
        if self.ignore_obs:
            obses = None
            next_obses = None
        else:
            obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
            next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
            obses = random_augment(obses, self.rad_h, self.rad_w)
            next_obses = random_augment(next_obses, self.rad_h, self.rad_w)
        if self.ignore_state:
            states = None
            next_states = None
        else:
            states = torch.as_tensor(self.states[idxs], device=self.device).float()
            next_states = torch.as_tensor(self.next_states[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, states, actions, rewards, next_obses, next_states, not_dones

import time
import threading
import cv2 as cv
class AsyncRadReplayBuffer(RadReplayBuffer):

    def __init__(self, obs_shape, state_shape, action_shape, capacity, batch_size, rad_offset,
                 device, input_queue, output_queue, init_step, max_update_freq, sync_queue):
        super(AsyncRadReplayBuffer, self).__init__(obs_shape, state_shape, action_shape, capacity, batch_size,
                                                rad_offset, device)
        self.init_step = init_step
        self._step = 0
        self._send_counter = 0
        self._max_update_freq = max_update_freq
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.sync_queue = sync_queue
        self.start_thread()

    def start_thread(self):
        threading.Thread(target=self.recv_from_env).start()
        threading.Thread(target=self.send_to_update).start()

    def recv_from_env(self):
        while True:
            self.add(*self.input_queue.get())
            self._step += 1

    def send_to_update(self):
        while True:
            if self._send_counter > (self._step - self.init_step) * self._max_update_freq:
                time.sleep(0.1)
            else:
                if self.sync_queue is not None:
                    self.sync_queue.get()
                self.output_queue.put(tuple(self.sample()))
                self._send_counter += 1


    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.states[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.next_states[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)



    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.states[start:end] = payload[1]
            self.next_obses[start:end] = payload[2]
            self.next_states[start:end] = payload[3]
            self.actions[start:end] = payload[4]
            self.rewards[start:end] = payload[5]
            self.not_dones[start:end] = payload[6]
            self.idx = end

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)



