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
    parser.add_argument('--env', default='Visual-UR5')
    parser.add_argument('--ip', default='129.128.159.210', type=str)
    parser.add_argument('--camera_id', default=0, type=int)
    parser.add_argument('--image_width', default=160, type=int)
    parser.add_argument('--image_height', default=120, type=int)
    parser.add_argument('--action_repeat', default=4, type=int)
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
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--async', default=False, action='store_true')
    parser.add_argument('--max_update_freq', default=2, type=int)
    #parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    #parser.add_argument('--no_eval', default=False, action='store_true')
    #parser.add_argument('--eval_freq', default=5000, type=int)
    #parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
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
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_model_freq', default=-1, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--device', default='', type=str)
    args = parser.parse_args()
    return args

from envs.ur5_wrapper import UR5Wrapper

def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)
    setup, target_type = args.env.split('_')
    env = UR5Wrapper(
        setup = setup,
        ip = args.ip,
        seed = args.seed,
        camera_id = args.camera_id,
        image_width = args.image_width,
        image_height = args.image_height,
        target_type = target_type,
        image_history = args.image_history,
        joint_history = args.joint_history,
        episode_length = args.episode_length,
        dt = args.dt,
        ignore_joint = args.ignore_joint,
    )
    mode = 'async' if args.async else 'sync'
    args.work_dir += f'/results/{args.env}_{mode}_{args.seed}/'
    utils.make_dir(args.work_dir)
    #video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    if args.device is '':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    agent = SacRadAgent(
        obs_shape=env.observation_space.shape,
        state_shape=env.state_space.shape,
        action_shape=env.action_space.shape,
        device=device,
        training_steps=args.env_step // args.action_repeat,
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

    L = Logger(args.work_dir, use_tb=args.save_tb)

    if args.async:
        agent.share_memory()

        # easily transfer step information to 'async_recv_data'

        def recv_from_update(buffer_queue, L, stop):
            while True:
                if stop():
                    break
                stat_dict = buffer_queue.get()
                for k, v in stat_dict.items():
                    L.log(k, v, step)

        # initialize processes in 'spawn' mode, required by CUDA runtime
        ctx = mp.get_context('spawn')

        MAX_QSIZE = 10
        input_queue = ctx.Queue(MAX_QSIZE)
        output_queue = ctx.Queue(MAX_QSIZE)
        tensor_queue = ctx.Queue(MAX_QSIZE)

        # initialize data augmentation process
        replay_buffer_process = ctx.Process(target=utils.AsyncRadReplayBuffer,
                                args=(
                                    env.observation_space.shape,
                                    env.state_space.shape,
                                    env.action_space.shape,
                                    args.replay_buffer_capacity,
                                    args.batch_size,
                                    args.rad_offset,
                                    device,
                                    input_queue,
                                    tensor_queue,
                                    args.init_step,
                                    args.max_update_freq)
                                )
        replay_buffer_process.start()
        # initialize SAC update process
        update_process = ctx.Process(target=agent.async_update,
                               args=(tensor_queue, output_queue))
        update_process.start()
        # flag for whether stop threads
        stop = False
        # initialize training statistics receiving thread
        stat_recv_thread = threading.Thread(target=recv_from_update, args=(output_queue, L, lambda: stop))
        #threads.append(t_recv)
        stat_recv_thread.start()
    else:
        replay_buffer = utils.RadReplayBuffer(
            obs_shape=env.observation_space.shape,
            state_shape=env.state_space.shape,
            action_shape=env.action_space.shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            rad_offset=args.rad_offset,
            device=device
        )

    episode, episode_reward, episode_step, done = 0, 0, 0, True
    start_time = time.time()
    obs, state = env.reset()
    for step in range(args.env_step + 1 + args.init_step):
        if done and step > 0:
            L.log('train/duration', time.time() - start_time, step)
            L.log('train/episode_reward', episode_reward, step)
            start_time = time.time()
            L.dump(step)
            obs, state = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            L.log('train/episode', episode, step)
            # wait image and joint update after reset
            #time.sleep(1.0)
            #if args.save_model_freq > 0 and step > 0 and step % args.save_model_freq == 0:
            #    agent.save(model_dir, step)

        # sample action for data collection
        if step < args.init_step:
            if step % args.action_repeat == 0:
                action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                if step % args.action_repeat == 0:
                    action = agent.sample_action(obs, state)

        # step in the environment
        next_obs, next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        if args.async:
            input_queue.put((obs, state, action, reward, next_obs, next_state, done))
        else:
            replay_buffer.add(obs, state, action, reward, next_obs, next_state, done)
            if step >= args.init_step:
                agent.update(replay_buffer)
        obs = next_obs
        state = next_state
        episode_step += 1

        # Terminate all threads and processes once done
        if step == args.env_step + 1 + args.init_step  and args.async:
            stop = True
            stat_recv_thread.join()
            replay_buffer_process.terminate()
            update_process.terminate()


if __name__ == '__main__':
    main()
