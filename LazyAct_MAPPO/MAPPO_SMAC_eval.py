import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_smac import MAPPO_SMAC
from smac.env import StarCraft2Env
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class Runner_MAPPO_SMAC:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.args.device = device
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = StarCraft2Env(map_name=self.env_name, seed=self.seed, replay_dir=args.replay_dir)
        self.env_info = self.env.get_env_info()
        self.args.N = self.env_info["n_agents"]  # The number of agents
        self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.args.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
        self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.args.cost = -0.1  # balance the reward and cost
        self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("skip_dim={}".format(self.args.skip_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        # Create N agents
        self.agent_n = MAPPO_SMAC(self.args)
        #'''
        if args.load_prev:
            try:
                self.agent_n.load_model(self.env_name, self.number-1)
                print('load prev model')
            except:
                print('faild load prev model')
        #'''
        self.replay_buffer = ReplayBuffer(self.args)
        self.win_rates = []  # Record the win rates
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)

    def run(self, ):
        self.run_episode_smac(False)
        self.env.close()

    def run_episode_smac(self, evaluate=False):
        win_tag = False
        episode_reward = 0
        episode_reward_cost = 0
        skips = np.zeros((self.args.N, ))
        skip_ratio = []
        prev_a_n = None
        mask_rec = []
        action_rec = []
        self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            obs_n = self.env.get_obs()  # obs_n.shape=(N,obs_dim)
            s = self.env.get_state()  # s.shape=(state_dim,)
            avail_a_n = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            skip_mask = np.zeros((self.args.N, ))
            skip_n, skip_logprob_n, a_n, a_logprob_n = self.agent_n.choose_action(obs_n, avail_a_n, evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents
            if prev_a_n is None: prev_a_n = a_n.copy()
            for idx in range(self.args.N):
                if skips[idx] > 0:
                    skip_mask[idx] = 1.0
                    skips[idx] -= 1
                    a_n[idx] = prev_a_n[idx] if avail_a_n[idx][prev_a_n[idx]] == 1 else np.array(avail_a_n[idx]).argmax()
                else:
                    prev_a_n[idx] = a_n[idx]
                    skips[idx] = skip_n[idx]
            s = np.append(s,skip_mask.sum()/self.args.N)
            v_n, c_n = self.agent_n.get_value(s, obs_n)  # Get the state values (V(s)) of N agents
            #print('--------------------------------------')
            #print(skip_mask)
            #print(a_n)
            mask_rec.append(skip_mask)
            action_rec.append(a_n)
            r, done, info = self.env.step(a_n)  # Take a step
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            
            episode_reward += r
            costs = (self.args.N - skip_mask.sum()) * self.args.cost
            episode_reward_cost += (r + self.agent_n.alpha * costs)
            skip_ratio.append(skip_mask.sum()/self.args.N)

            if done:
                break
        #self.env.save_replay()
        print(win_tag)
        #print(np.mean(skip_ratio))
        mask_rec = np.array(mask_rec)
        action_rec = np.array(action_rec)
        mask_rec = np.transpose(mask_rec)
        time = np.arange(mask_rec.shape[1])  # 横坐标  
        id_ = np.arange(mask_rec.shape[0])
        
        colors = ['gray', 'white']  
        cmap = ListedColormap(colors)  
  
        # 使用imshow函数绘制矩阵  
        plt.figure(figsize=(50, 5))
        plt.imshow(mask_rec, cmap=cmap, aspect='auto')  
        # 设置横纵坐标的标签  
        plt.xticks(range(len(time)), time)  
        #plt.yticks(range(len(id_)), id_)  
  
        # 显示网格线  
        #plt.grid(True, which='both', color='black', linewidth=0.5)  
  
          
        plt.savefig('vis.jpg', dpi=250)
        return win_tag, episode_reward, episode_reward_cost, np.mean(skip_ratio), episode_step + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in SMAC environment")
    parser.add_argument("--max_train_steps", type=int, default=int(2e7), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=32, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")
    parser.add_argument("--replay_dir", type=str, default="replay", help="Save replay dir")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--skip_dim", type=int, default=3, help="skip dim")
    parser.add_argument("--number", type=int, default=1, help="skip dim")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=True, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_agent_specific", type=float, default=True, help="Whether to use agent specific global state.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")
    parser.add_argument('--alpha_lr', type=float, default=0.02, help='Lambda Learning rate')
    parser.add_argument('--init_alpha', type=float, default=0.0, help='initial Lambda')
    parser.add_argument("--skip_ratio", type=float, default=0.1, help="cost factor")
    parser.add_argument("--skip_train", action='store_true', help="train with skip?")
    parser.add_argument("--load_prev", action='store_true', help="load prev model?")

    args = parser.parse_args()
    env_names = ['3m', '8m', '25m', '2s3z', '3s5z', '3s5z_vs_3s6z', '3s_vs_3z', '5m_vs_6m', '8m_vs_9m']
    env_index = 2
    runner = Runner_MAPPO_SMAC(args, env_name=env_names[env_index], number=args.number, seed=1)
    runner.run()
