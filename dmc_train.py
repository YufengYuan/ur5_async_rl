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
from configs.dmc_config import config

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='reacher')
    parser.add_argument('--task_name', default='easy')
    parser.add_argument('--env', default='reacher_easy_state')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--rad_offset', default=4, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='sac_rad', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--env_steps', default=500000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    #parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--no_eval', default=False, action='store_true')
    parser.add_argument('--eval_freq', default=5000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    # encoder
    parser.add_argument('--latent_dim', default=50, type=int)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    #parser.add_argument('--num_layers', default=4, type=int)
    #parser.add_argument('--num_filters', default=32, type=int)

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
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--device', default='', type=str)
    parser.add_argument('--async', default=False, action='store_true')
    parser.add_argument('--desc', default='', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

    #env = dmc2gym.make(
    #    domain_name=args.domain_name,
    #    task_name=args.task_name,
    #    seed=args.seed,
    #    visualize_reward=False,
    #    from_pixels=True,
    #    height=args.image_size + 2 * args.rad_offset,
    #    width=args.image_size + 2 * args.rad_offset,
    #    frame_skip=args.action_repeat
    #)
    #env = utils.FrameStack(env, k=args.frame_stack)
    env = DMCEnv(
        name=args.env,
        seed=args.seed,
        image_height=args.image_size + 2 * args.rad_offset,
        image_width=args.image_size + 2 * args.rad_offset,
        frame_stack=args.frame_stack,
        frame_skip=args.action_repeat
    )
    eval_env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed + 100,
        visualize_reward=False,
        from_pixels=True,
        height=args.image_size + 2 * args.rad_offset,
        width=args.image_size + 2 * args.rad_offset,
        frame_skip=args.action_repeat
    )
    #eval_env.seed(args.seed + 100)
    eval_env = utils.FrameStack(eval_env, k=args.frame_stack)

    #mode = 'async' if args.async else 'sync'
    #args.work_dir = f'results/sac_{args.desc}_{args.domain_name}_{args.task_name}_{args.seed}/'
    args.work_dir = f'results/sac_{args.desc}_{args.env}_{args.seed}/'
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

    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        state_shape=env.state_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device
    )

    agent = SacRadAgent(
        obs_shape=env.observation_space.shape,
        state_shape=env.state_space.shape,
        action_shape=env.action_space.shape,
        device=device,
        training_steps=args.env_steps // args.action_repeat,
        net_params=config,
        discount=args.discount,
        init_temperature=args.init_temperature,
        alpha_lr=args.alpha_lr,
        actor_lr=args.actor_lr,
        actor_update_freq=args.actor_update_freq,
        critic_lr=args.critic_lr,
        critic_tau=args.critic_tau,
        critic_target_update_freq=args.critic_target_update_freq,
        rl_latent_dim=args.latent_dim,
        encoder_tau=args.encoder_tau,
        rad_offset=args.rad_offset,
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    training_steps = args.env_steps // args.action_repeat
    for step in range(training_steps + 1):
        if done:
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # evaluate agent periodically
            # TODO: modify evaluation function to enable evaluation for multimodal observations
            if step % args.eval_freq == 0 and not args.no_eval:
                L.log('eval/episode', episode, step)
                evaluate(eval_env, agent, args.num_eval_episodes, L, step, args)
                if args.save_model:
                    agent.save(model_dir, step)
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)

            L.log('train/episode_reward', episode_reward, step)

            obs, state = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs, state)
                #action = agent.sample_action(
                #    obs[:, args.rad_offset: args.image_size + args.rad_offset,
                #        args.rad_offset: args.image_size + args.rad_offset]
                #)


        # run training update
        if args.async and step == args.init_steps:
            agent.share_memory()
            input_queue = BufferQueue(5, 10)
            output_queue = BufferQueue(8, 10)
            tensor_queue = utils.BufferQueue(5, 10)
            # easily transfer step information to 'async_recv_data'
            def async_send_data(replay_buffer, buffer_queue, stop):
                while True:
                    if stop():
                        break
                    buffer_queue.put(*replay_buffer.sample_numpy())

            def async_recv_data(buffer_queue, L, stop):
                while True:
                    if stop():
                        break
                    batch_reward, critic_loss, actor_loss, \
                    target_entropy, entropy, alpha_loss, \
                    alpha, num_updates = buffer_queue.get()
                    #step = cur_step[0]
                    L.log('train_critic/loss', critic_loss, step)
                    L.log('train/batch_reward', batch_reward, step)
                    L.log('train/entropy', entropy, step)
                    L.log('train_actor/loss', actor_loss, step)
                    L.log('train_actor/target_entropy', target_entropy, step)
                    L.log('train_actor/entropy', entropy, step)
                    L.log('train_alpha/loss', alpha_loss, step)
                    L.log('train_alpha/value', alpha, step)
                    L.log('train/step_update_ratio', num_updates / (step - args.init_steps + 1), step)
                    #print(num_updates, step)

            processes = []
            threads = []
            # initialize data augmentation process
            p_augment = mp.Process(target=SacRadAgent.async_data_augment, args=(input_queue, tensor_queue, agent.device))
            processes.append(p_augment)
            p_augment.start()
            # initialize SAC update process
            p_update = mp.Process(target=SacRadAgent.async_update, args=(
                agent, tensor_queue, output_queue
            ))
            processes.append(p_update)
            p_update.start()
            # flag for whether stop threads
            stop = False
            # initialize transition sending thread
            t_send = threading.Thread(target=async_send_data, args=(replay_buffer, input_queue, lambda : stop))
            threads.append(t_send)
            t_send.start()
            # initialize training statistics receiving thread
            t_recv = threading.Thread(target=async_recv_data, args=(output_queue, L, lambda : stop))
            threads.append(t_recv)
            t_recv.start()
        # update regularly
        elif not args.async and step >= args.init_steps:
            agent.update(replay_buffer, L, step)
        # step in the environment
        next_obs, next_state, reward, done, _ = env.step(action)
        #done_bool = episode_step >= env._max_episode_steps or done
        done_bool = done
        episode_reward += reward
        replay_buffer.add(obs, state, action, reward, next_obs, next_state, done_bool)
        obs = next_obs
        state = next_state
        episode_step += 1

        # Terminate all threads and processes once done
        if step == training_steps and args.async:
            stop = True
            for t in threads:
                t.join()
            for p in processes:
                p.terminate()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
