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

class Evaluator(object):
    """
    Args:
    """
    def __init__(self, args, comp_pth):
        self.args = args
        self.comp_pth = comp_pth

    def performing(self, alpha):
        torch.set_num_threads(1)
        torch.backends.quantized.engine = backend
        self.env = AtariEmulator(0, self.args)
        torch.manual_seed(self.args.seed)
        
        pred_time = []
        skips = []
        episode_lengthes = []
        curr_model = torch.load(self.comp_pth, map_location=torch.device('cpu'))
        test_id = 0
        scores = []
        while test_id<20:
            obs = self.env.get_initial_state()
            done = False
            total_reward = 0.
            total_skip_reward = 0.
            episode_length = 0
            skip_step = 0
            test_id += 1
            while not done:
                shared_state = obs.transpose((2,0,1)).astype(np.float32)/255.
                x = torch.from_numpy(shared_state).unsqueeze(0).to('cpu')
                start_time = time.time()
                #predict skip
                skip_policies, hiddens = curr_model(x)
                network_output_skip = Fnn.softmax(skip_policies, dim=-1)
                skip = network_output_skip.multinomial(num_samples=1)
                skip_step += skip.item()
                skip_tensor = torch.zeros([1,curr_model.num_skips]).to('cpu')
                skip_tensor[torch.arange(1).long(),skip.long()] = 1.0
                #predict action
                policies, _, _ = curr_model(hiddens, skip_tensor)
                network_output_pi = Fnn.softmax(policies, dim=-1)
                action = network_output_pi.multinomial(num_samples=1)
                end_time = time.time()
                during_time = end_time-start_time
                pred_time.append(during_time)
                
                exe_action = np.zeros(curr_model.num_actions)
                exe_action[action.item()] = 1.0
                
                #skip n steps
                clip_reward = 0.0
                for it in range(skip.item()+1):
                    episode_length += 1
                    obs, reward, done = self.env.next(exe_action)
                    total_reward += reward
                    clip_reward = clip_reward + np.power(self.args.gamma, it) * (np.clip(reward, -1.0, 1.0))
                    if done:
                        break
                clip_reward = clip_reward + skip.item()*alpha
                total_skip_reward += clip_reward
                if done:
                    scores.append(total_reward)
                    episode_lengthes.append(episode_length)
                    skips.append(skip_step)
        avg_score = np.mean(scores)
        avg_skip = np.mean(np.array(skips)/np.array(episode_lengthes))
        print("alpha="+str(alpha)+" avg score:"+str(avg_score)+'==> avg skip:'+str(avg_skip))