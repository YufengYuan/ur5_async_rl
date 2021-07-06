# Asynchronous Reinforcement Learning for UR5 Robotic Arm

This is the implementation for asynchronous reinforcement learning for UR5 robotic arm. This repo consists of two parts, the vision-based UR5 environment, which is based on the [SenseAct](https://github.com/kindredresearch/SenseAct) framework, and a asynchronous learning architecture for Soft-Actor-Critic. (Our implementation of SAC is partly borrowed from [here](https://sites.google.com/view/sac-ae/home))

### Trained results:
| ![UR-Reacher-2](figs/reaching.GIF) <br> Reaching | ![UR-Reacher-6](figs/tracking.GIF) <br /> Tracking |
| --- | --- |

### Installation:
If you have the physical setup shown above (an UR5 robotic arm, an USB camera, and a monitor which are all connected to a Linux workstaion with GPU support), follow the steps below to install
1. install [SenseAct](https://github.com/kindredresearch/SenseAct/blob/master/README.md#installation) framework
2. `git clone https://github.com/YufengYuan/ur5_async_rl`
3. `cd ur5_async_rl`
4. `pip3 install -r requirements.txt`

## Instructions
To train an agent with asynchronous learning updates 
on the our `reaching` task, run:
```
python3 ur5_train.py \
      --target_type reaching \
      --seed 0 \
      --batch_size 128 \ 
      --episode_length 4 \ 
      --dt 0.04 \ 
      --image_width 160 \ 
      --image_height 90 \ 
      --init_step 100 \
      --async
```

where `--async` is the flag to enable asynchronous learning updates, without it, all computations will be processed sequentially. You may also need to change UR5 `ip` and `camera_id` in `env/ur5_wrapper.py` so that the system can properly detect the hardware.

The console output is available in a form:
```
| train | E: 1 | S: 1000 | D: 0.8 s | R: 0.0000 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | NUM: 0.0000
```
a training entry decodes as:
```
E - total number of episodes 
S - total number of environment steps
D - duration in seconds of 1 episode
R - episode reward
BR - average reward of sampled batch
ALOSS - average loss of actor
CLOSS - average loss of critic
NUM - number of gradient updates performed so far
```
