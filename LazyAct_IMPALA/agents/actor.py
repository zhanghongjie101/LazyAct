import os, sys
#sys.path.append("..")
import time
import numpy as np
import six
import copy

import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import torch.multiprocessing as mp
from multiprocessing import Process
from collections import deque
#from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import environment_creator
from atari_emulator import AtariEmulator
import random
from collections import OrderedDict
from torchstat import stat
from torch import quantization

backend='fbgemm'

def _action(*entries):
  return np.array(entries, dtype=np.intc)

def softmax(x, dim):
    x -= np.max(x, axis= dim, keepdims=True)
    f_x = np.exp(x) / np.sum(np.exp(x), axis=dim, keepdims=True)
    return f_x

class Actor(object):
    """
    Args:
    """
    def __init__(self, args, q_trace, comp_pth, rollouts, train_steps, actor_name=None):
        self.args = args
        self.q_trace = q_trace
        self.comp_pth = comp_pth
        self.rollouts = rollouts
        self.train_steps = train_steps
        self.actor_name = actor_name

    def performing(self, rank, stage=0):
        if rank == 0:
            self.summary_writer = tf.summary.FileWriter(os.path.join(self.args.debugging_folder, self.args.game + '/actor'))
        torch.set_num_threads(1)
        torch.backends.quantized.engine = backend
        print('Build Environment for {}'.format(self.actor_name))
        self.env = AtariEmulator(rank, self.args)
        torch.manual_seed(self.args.seed)
        obs = self.env.get_initial_state()
        done = False
        total_reward = 0.
        total_episode_length = 0
        num_episodes = 0

        iterations = 0
        timesteps = 0
        cnt = 0
        pred_time = []
        skips = []
        curr_model = None
        skip_curr = 0.0
        while self.train_steps.value<=self.args.total_num_steps:
            if cnt % 100 == 0:
                while True:
                    try:
                        curr_model = torch.load(self.comp_pth, map_location=torch.device('cpu'))
                        break
                    except:
                        time.sleep(0.01)

            cnt += 1
            self.rollouts.init()

            for step in range(self.args.num_steps):
                shared_state = obs.transpose((2,0,1)).astype(np.float32)/255.
                x = torch.from_numpy(shared_state).unsqueeze(0).to('cpu')
                start_time = time.time()
                #predict skip
                skip_policies, hiddens = curr_model(x)
                network_output_skip = Fnn.softmax(skip_policies, dim=-1)
                log_skip = Fnn.log_softmax(skip_policies, dim=-1)
                if stage == 0:
                    skip = torch.zeros([1,1],dtype=torch.long).to('cpu')
                else:
                    skip = network_output_skip.multinomial(num_samples=1)
                skips.append(skip.item())
                skip_log_prob = log_skip[0, skip.item()]
                skip_tensor = torch.zeros([1,curr_model.num_skips]).to('cpu')
                skip_tensor[torch.arange(1).long(), skip.long()] = 1.0
                #predict action
                policies, _, _ = curr_model(hiddens, skip_tensor)
                network_output_pi = Fnn.softmax(policies, dim=-1)
                log_pi = Fnn.log_softmax(policies, dim=-1)
                action = network_output_pi.multinomial(num_samples=1)
                action_log_prob = log_pi[0, action.item()]
                end_time = time.time()
                during_time = end_time-start_time
                pred_time.append(during_time)
                
                exe_action = np.zeros(curr_model.num_actions)
                exe_action[action.item()] = 1.0
                self.rollouts.obs[step] = shared_state
                self.rollouts.actions[step, 0] = action.item()
                self.rollouts.skip[step, 0] = skip.item()
                self.rollouts.action_log_probs[step, 0] = action_log_prob.item()
                self.rollouts.skip_log_probs[step, 0] = skip_log_prob.item()
                
                #skip n steps
                clip_reward = 0.0
                cost_reward = 0.0
                for it in range(skip.item()+1):
                    timesteps += 1
                    obs, reward, done = self.env.next(exe_action)
                    if timesteps >= 4000: done = True
                    total_reward += reward
                    cost = 0.0 if it > 0 else 0.1
                    clip_reward = clip_reward + np.power(self.args.gamma, it) * (np.clip(reward, -1.0, 1.0))
                    cost_reward = cost_reward + np.power(self.args.gamma, it) * cost
                    if done:
                        break
                self.rollouts.rewards[step, 0] = clip_reward
                self.rollouts.costs[step, 0] = cost_reward
                self.rollouts.masks[step, 0] = 0.0 if done else 1.0
                skip_curr = np.sum(skips)/timesteps
                if done:
                    num_episodes += 1
                    total_episode_length += 1
                    obs = self.env.get_initial_state()
                    print("env: "+str(rank)+" episode reward: "+str(total_reward))
                    avg_time = np.mean(pred_time)
                    avg_skip = np.mean(skips)
                    skip_ratio = np.sum(skips)/timesteps
                    pred_time = []
                    skips = []
                    
                    if rank == 0:
                        episode_summary = tf.Summary(value=[tf.Summary.Value(tag='actor/reward', simple_value=total_reward),
                                                            tf.Summary.Value(tag='actor/pred_time', simple_value=avg_time),
                                                            tf.Summary.Value(tag='actor/skips', simple_value=avg_skip),
                                                            tf.Summary.Value(tag='actor/skip_ratio', simple_value=skip_ratio),
                                                            tf.Summary.Value(tag='actor/episode_length', simple_value=timesteps),])
                        self.summary_writer.add_summary(episode_summary, self.train_steps.value)
                        self.summary_writer.flush()
                    iterations += 1
                    total_reward = 0
                    num_episodes = 0
                    timesteps = 0

            self.rollouts.obs[-1] = obs.transpose((2,0,1)).astype(np.float32)/255.
            self.rollouts.skip[-1, 0] = 0
            self.q_trace.put((self.rollouts.obs, self.rollouts.actions, self.rollouts.skip, self.rollouts.rewards, self.rollouts.costs, \
                    self.rollouts.action_log_probs, self.rollouts.skip_log_probs, self.rollouts.masks, skip_curr))