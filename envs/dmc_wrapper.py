
import numpy as np
import gym
import os
from collections import deque
from dmc2gym.wrappers import DMCWrapper

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

class DMCEnv(DMCWrapper):

    def __init__(self, name, seed=0, image_width=84, image_height=84, frame_stack=3, frame_skip=4):

        # Available envs are
        domain_name, task_name, mode = name.split('_', 2)
        from_pixels = True if mode == 'pixel' or mode == 'multi' else False
        if mode == 'multi':
            assert domain_name == 'reacher', f'{domain_name} is not implemented for multimodal observation space!'
        super(DMCEnv, self).__init__(domain_name, task_name,
                                     {'random': seed}, False,
                                     from_pixels=from_pixels,
                                     height=image_height,
                                     width=image_width,
                                     frame_skip=frame_skip)
        self.mode = mode
        self.domain_name = domain_name
        self.task_name = task_name
        self.frame_stack = frame_stack
        self.frames = deque([], maxlen=frame_stack)
        self.rewards = deque([], maxlen=frame_stack)
        # Modify the observation_space for frame stack
        if self.mode == 'state':
            self._observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(0,),
                dtype=self.observation_space.dtype
            )
        elif self.mode == 'pixel':
            obs_shape = self._observation_space.shape
            self._observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=((obs_shape[0] * frame_stack,) + obs_shape[1:]),
                dtype=self.observation_space.dtype
            )
            self._state_space = gym.spaces.Box(0, 1, shape=(0,), dtype=np.float32)
        elif self.mode == 'multi':
            #low = np.concatenate([self._state_space.low[:2], self._state_space.low[-2:]])
            #high = np.concatenate([self._state_space.high[:2], self._state_space.high[-2:]])
            self._state_space = gym.spaces.Box(0, 1, shape=(4,), dtype=np.float32)
            obs_shape = self._observation_space.shape
            self._observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=((obs_shape[0] * frame_stack,) + obs_shape[1:]),
                dtype=self.observation_space.dtype
            )

    # Override step method to discard target information if image is used
    def step(self, action):
        obs, reward, done, extra = super().step(action)
        if self.mode == 'state':
            state = obs
            obs = None
        elif self.mode == 'pixel':
            self.frames.append(obs)
            assert len(self.frames) == self.frame_stack
            obs = np.concatenate(self.frames, axis=0)
            state = None
        if self.mode == 'multi':
            state = np.concatenate([extra['internal_state'][:2], extra['internal_state'][-2:]])
            self.frames.append(obs)
            assert len(self.frames) == self.frame_stack
            obs = np.concatenate(self.frames, axis=0)
        return obs, state, reward, done, extra

    # Override reset method to discard target information if image is used
    def reset(self):
        obs = super().reset()
        if self.mode == 'state':
            state = obs
            obs = None
        elif self.mode == 'pixel':
            for _ in range(self.frame_stack):
                self.frames.append(obs)
            assert len(self.frames) == self.frame_stack
            obs = np.concatenate(self.frames, axis=0)
            state = None
        if self.mode == 'multi':

            internal_state = self._env.physics.get_state().copy()
            state = np.concatenate([internal_state[:2], internal_state[-2:]])
            for _ in range(self.frame_stack):
                self.frames.append(obs)
            #self.frames.append(obs)
            assert len(self.frames) == self.frame_stack
            obs = np.concatenate(self.frames, axis=0)
        return obs, state

