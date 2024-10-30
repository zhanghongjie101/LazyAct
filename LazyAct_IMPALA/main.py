import os, sys
from pathlib import Path
import argparse
import shutil
#import yaml
from copy import deepcopy

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import SimpleQueue, Queue, Value, Lock
from agents.learner_stage0 import Learner as LearnerStage0
from agents.learner import Learner as LearnerStage1
from agents.quantizer import Quantizer
from agents.actor import Actor
from misc.q_manager import QManager
from misc.storage import RolloutStorage
import environment_creator
from models import SkipCNN as AtariCNN

import tensorflow as tf
import my_optim
import time
import tensorboard


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-g', default='pong', help='Name of game', dest='game')
    parser.add_argument('-ec', '--emulator_counts', default=8, type=int, help="The amount of emulators per agent. Default is 32.", dest="emulator_counts")
    parser.add_argument('-qc', '--q_counts', default=4, type=int, help="The amount of q_manager. Default is 5.", dest="q_counts")
    parser.add_argument('--rom_path', default='./atari_roms', help='Directory where the game roms are located (needed for ALE environment)', dest="rom_path")
    parser.add_argument('-df', '--debugging_folder', default='logs/', type=str, help="Folder where to save the debugging information.", dest="debugging_folder")
    parser.add_argument('-pj', '--project', default='impala', type=str, help="Folder of current project.", dest="project")
    parser.add_argument('-rs', '--random_start', default=True, type=bool, help="Whether or not to start with 30 noops for each env. Default True", dest="random_start")
    parser.add_argument('--single_life_episodes', default=False, type=bool, help="If True, training episodes will be terminated when a life is lost (for games)", dest="single_life_episodes")
    parser.add_argument('-v', '--visualize', default=False, type=bool, help="0: no visualization of emulator; 1: all emulators, for all actors, are visualized; 2: only 1 emulator (for one of the actors) is visualized", dest="visualize")
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--lam_lr', type=float, default=0.001, help='Lambda Learning rate')
    parser.add_argument('--init_lam', type=float, default=0.0, help='initial Lambda')
    parser.add_argument('-lra', '--lr_annealing_steps', default=200000, type=int, help="Nr. of global steps during which the learning rate will be linearly annealed towards zero", dest="lr_annealing_steps")
    parser.add_argument('--num_steps', type=int, default=5,
                      help='Number of Steps to learn')
    parser.add_argument('--total_num_steps', type=int, default=150000,
                      help='Number of Steps to learn')
    parser.add_argument('--seed', type=int, default=2022,
                      help='Random seed')
    parser.add_argument('--coef_hat', type=float, default=1.0)
    parser.add_argument('--rho_hat', type=float, default=10.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='discount rate')
    #parser.add_argument('--coef_skip', type=float, default=0.01)
    parser.add_argument('--entropy_coef', type=float, default=0.02)
    parser.add_argument('--value_loss_coef', type=float, default=0.25)
    parser.add_argument('--max_grad_norm', type=float, default=3.)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--skip_ratio', type=float, default=0.85)
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        mp.set_start_method('forkserver', force=True)
        print("forkserver init")
    except RuntimeError:
        pass
    
    model_dir = "saved_models"
    model_filename = "model.pt"
    model_filepath = os.path.join(model_dir, model_filename)

    quant_model_filename = "quant.pt"
    quant_model_filepath = os.path.join(model_dir, quant_model_filename)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    processes = []
    q_trace = Queue(maxsize=300)
    q_batch = Queue(maxsize=2)
    for q in range(args.q_counts):
        q_manager = QManager(args, q_trace, q_batch)
        p = mp.Process(target=q_manager.listening)
        p.start()
        processes.append(p)

    envs = []
    actors = []

    env_creator = environment_creator.EnvironmentCreator(args)
    env = env_creator.create_environment(0)
    obs = env.get_initial_state()
    obs_shape = obs.shape
    obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
    print('Observation Space: ', obs_shape)
    action_space = env_creator.num_actions
    args.action_space = action_space
    train_steps = Value('i', 0)
    
    if os.path.exists(model_filepath):
        print('exist.............................................................')
        actor_critic = torch.load(model_filepath, map_location=args.device)
        #actor_critic.random_set()
    else:
        actor_critic = AtariCNN(action_space)
    actor_critic.to(args.device)  
    actor_critic.share_memory()
    
    if args.stage == 0:
        learner = LearnerStage0(args, q_batch, actor_critic, model_filepath, train_steps)
    elif args.stage == 1:
        learner = LearnerStage1(args, q_batch, actor_critic, model_filepath, train_steps)

    for i in range(args.emulator_counts):
        print('Build Actor {:d}'.format(i))
        rollouts = RolloutStorage(args.num_steps,
                                  1,
                                  obs_shape,
                                  action_space)

        actor_name = 'actor_' + str(i)
        actor = Actor(args, q_trace, quant_model_filepath, rollouts, train_steps, actor_name)
        actors.append(actor)

    print('Run processes')
    
    quantizer = Quantizer(args, train_steps, model_filepath, quant_model_filepath)

    for rank, a in enumerate(actors):
        p = mp.Process(target=a.performing, args=(rank, args.stage))
        p.start()
        processes.append(p)

    p = mp.Process(target=learner.learning)
    p.start()
    processes.append(p)

    p = mp.Process(target=quantizer.quant_and_save)
    p.start()
    processes.append(p)

    for p in processes:
        p.join()
