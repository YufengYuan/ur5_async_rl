import numpy as np
import torch
import argparse
import os
import math
#import gym
import sys
import random
import time
import json
import copy
import dmc2gym
import  threading
from sac_rad import SacRadAgent
import utils
from logger import Logger
from queue import Queue
import torch.multiprocessing as mp
#import multiprocessing as mp
from utils import BufferQueue
import multiprocessing
from utils import evaluate
from envs.dmc_wrapper import DMCEnv
#from configs.dmc_config import config
from configs.ur5_config import config
from envs.visual_ur5_reacher.reacher_env import ReacherEnv
from senseact.utils import tf_set_seeds, NormalizedEnv
import cv2 as cv

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--setup', default='Visual-UR5')
    parser.add_argument('--ip', default='129.128.159.210', type=str)
    parser.add_argument('--camera_id', default=1, type=int)
    parser.add_argument('--image_width', default=160, type=int)
    parser.add_argument('--image_height', default=90, type=int)
    parser.add_argument('--target_type', default='reaching', type=str)
    parser.add_argument('--random_action_repeat', default=1, type=int)
    parser.add_argument('--agent_action_repeat', default=1, type=int)
    parser.add_argument('--image_history', default=3, type=int)
    parser.add_argument('--joint_history', default=1, type=int)
    parser.add_argument('--ignore_joint', default=False, action='store_true')
    parser.add_argument('--episode_length', default=4.0, type=float)
    parser.add_argument('--dt', default=0.04, type=float)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)
    # train
    parser.add_argument('--init_step', default=1000, type=int)
    parser.add_argument('--env_step', default=100000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--async', default=False, action='store_true')
    parser.add_argument('--max_update_freq', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    # misc
    parser.add_argument('--seed', default=9, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    # parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_model_freq', default=10000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--device', default='', type=str)
    parser.add_argument('--lock', default=False, action='store_true')
    args = parser.parse_args()
    return args

from envs.ur5_wrapper import UR5Wrapper

def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)
    env = UR5Wrapper(
        setup=args.setup,
        ip=args.ip,
        seed=args.seed,
        camera_id=args.camera_id,
        image_width=args.image_width,
        image_height=args.image_height,
        target_type=args.target_type,
        image_history=args.image_history,
        joint_history=args.joint_history,
        episode_length=args.episode_length,
        dt=args.dt,
        ignore_joint=args.ignore_joint,
    )
    if not args.async:
        version = 'SACv0'
    elif args.async and args.lock:
        version = 'SACv1'
    elif args.async:
        version = 'SACv2'
    else:
        raise NotImplementedError('Not supported mode!')
    args.work_dir += f'/results/{version}_{args.target_type}_' \
                     f'dt={args.dt}_bs={args.batch_size}_' \
                     f'dim={args.image_width}*{args.image_height}_{args.seed}/'
    #utils.make_dir(args.work_dir)
    #print(args.work_dir)
    # video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    #buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    #with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
    #    json.dump(vars(args), f, sort_keys=True, indent=4)

    if args.device is '':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    agent = SacRadAgent(
        obs_shape=env.observation_space.shape,
        state_shape=env.state_space.shape,
        action_shape=env.action_space.shape,
        device=device,
        training_steps=args.env_step // args.agent_action_repeat,
        net_params=config,
        discount=args.discount,
        init_temperature=args.init_temperature,
        alpha_lr=args.alpha_lr,
        actor_lr=args.actor_lr,
        actor_update_freq=args.actor_update_freq,
        critic_lr=args.critic_lr,
        critic_tau=args.critic_tau,
        critic_target_update_freq=args.critic_target_update_freq,
        encoder_tau=args.encoder_tau,
        rad_offset=args.rad_offset,
    )

    #L = Logger(args.work_dir, use_tb=args.save_tb)

    #agent.load(model_dir, 100000)
    settings = [
        './results/SACv0_tracking_dt=0.08_bs=128_dim=160*90',
        #'./results/SACv0_reaching_dt=0.125_bs=128_dim=320*180',
        './results/SACv0_tracking_dt=0.2_bs=512_dim=160*90',
        #'./results/SACv1_reaching_dt=0.04_bs=128_dim=320*180',
        './results/SACv1_tracking_dt=0.04_bs=512_dim=160*90',
        './results/SACv2_tracking_dt=0.04_bs=128_dim=160*90',
        #'./results/SACv2_reaching_dt=0.04_bs=128_dim=320*180',
        './results/SACv2_tracking_dt=0.04_bs=512_dim=160*90',
    ]
    #settings = [
    #    './results/SACv0_tracking_dt=0.125_bs=128_dim=320*180',
    #    './results/SACv1_tracking_dt=0.04_bs=128_dim=320*180',
    #    './results/SACv2_tracking_dt=0.04_bs=128_dim=320*180',
    #]
    #steps = [50000, 20000, 20000, 100000, 100000, 100000, 100000, 100000]
    steps = [50000, 20000, 100000, 100000, 100000]
    #steps = [20000, 100000, 100000]
    seeds = [0, 1, 2, 3, 4]
    #seeds = [0]
    for setting, step in zip(settings, steps):
        ret = [0, 0, 0, 0, 0]
        #ret = [0]
        for seed in seeds:
            agent.load(setting + f'_{seed}/model', step)
            episode, episode_reward, episode_step, done = 0, 0, 0, False
            obs, state = env.reset()
            for _ in range(10):
                while not done:
                    with utils.eval_mode(agent):
                        action = agent.sample_action(obs, state, True)
                    next_obs, next_state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    obs = next_obs
                    state = next_state
                obs, state = env.reset()
                done = False
                #episode_step += 1
            ret[seed] = episode_reward / 10
        print(f'{setting}: episode_return: {np.mean(ret)}+-{np.std(ret)}')
    env.terminate()

if __name__ == '__main__':
    main()
